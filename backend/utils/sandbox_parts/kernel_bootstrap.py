"""Trusted kernel launch, readiness, and bootstrap source builders."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from queue import Empty as QueueEmpty
from typing import Dict

from jupyter_client import KernelManager

from .shared import _to_thread_drained, logger


def _expose_celltypist_cache(folder: Path) -> None:
    """Link the host's CellTypist model cache into the session view.

    Read-only for the kernel in effect: the links point at the data volume and
    a model the kernel downloads itself lands as a real file next to them. An
    optional resource must never stop a kernel, so every failure is a warning.
    """
    try:
        from backend.services import celltypist_models
        celltypist_models.link_cache_into(folder)
    except Exception as e:
        logger.warning(f"celltypist cache not exposed to the kernel: {e}")


def build_kernel_launch_environment(sandbox, *, bindings) -> Dict[str, str]:
    """Build an isolated environment before the kernel process starts.

    ``sitecustomize`` applies the irreversible process-level boundary, so
    cache and runtime locations must already point at per-session roots.
    """
    Path = bindings.Path
    os = bindings.os
    json = bindings.json
    sys = bindings.sys
    startup_dir = Path(__file__).resolve().parents[1] / "kernel_startup"
    session_tmp = sandbox._session_tmp.resolve()
    workspace = sandbox.working_dir.absolute()
    session_root_path = workspace.parent
    state_dir = sandbox.notebook_state.state_dir.absolute()
    if workspace.is_symlink() or session_root_path.is_symlink():
        raise RuntimeError("kernel workspace/session root must not be a symlink")
    if state_dir.is_symlink():
        raise RuntimeError("kernel state directory must not be a symlink")
    session_root = session_root_path.resolve(strict=True)
    resolved_workspace = workspace.resolve(strict=True)
    resolved_state_dir = state_dir.resolve(strict=True)
    if resolved_workspace.parent != session_root:
        raise RuntimeError("kernel workspace escaped its session root")
    isolated_dirs = {
        "HOME": session_tmp / "home",
        "XDG_CACHE_HOME": session_tmp / "xdg-cache",
        "XDG_CONFIG_HOME": session_tmp / "xdg-config",
        "XDG_DATA_HOME": session_tmp / "xdg-data",
        "MPLCONFIGDIR": session_tmp / "matplotlib",
        "IPYTHONDIR": session_tmp / "ipython",
        "JUPYTER_CONFIG_DIR": session_tmp / "jupyter-config",
        "JUPYTER_DATA_DIR": session_tmp / "jupyter-data",
        "JUPYTER_RUNTIME_DIR": session_tmp / "jupyter-runtime",
        "NUMBA_CACHE_DIR": session_tmp / "numba-cache",
        "JOBLIB_TEMP_FOLDER": session_tmp / "joblib",
        "DASK_TEMPORARY_DIRECTORY": session_tmp / "dask",
        # celltypist creates data/models under this folder when imported, so
        # it must be a write root; the host's model cache is linked into it.
        "CELLTYPIST_FOLDER": session_tmp / "celltypist",
        "TMPDIR": session_tmp,
        "TEMP": session_tmp,
        "TMP": session_tmp,
    }
    for path in set(isolated_dirs.values()):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.chmod(0o700)
    _expose_celltypist_cache(isolated_dirs["CELLTYPIST_FOLDER"])

    env = os.environ.copy()
    env.update({name: str(path) for name, path in isolated_dirs.items()})
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(startup_dir) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONSAFEPATH"] = "1"
    env["DASK_DATAFRAME__QUERY_PLANNING"] = (
        bindings._dask_query_planning_setting()
    )
    env["SCAI_KERNEL_WRITE_ROOTS"] = json.dumps(
        [str(resolved_workspace), str(resolved_state_dir), str(session_tmp)]
    )
    env["SCAI_KERNEL_REQUIRE_LANDLOCK"] = (
        "1" if sys.platform == "linux" else "0"
    )
    return env

def install_isolated_kernel_launcher(
    sandbox, manager: KernelManager, *, bindings
) -> None:
    """Wrap the kernelspec with a trusted pre-IPython isolation stage."""
    Path = bindings.Path
    sys = bindings.sys
    original_argv = list(manager.kernel_spec.argv)
    if not original_argv:
        raise RuntimeError("Python kernelspec has no launch command")
    python_aliases = {
        "python",
        f"python{sys.version_info[0]}",
        f"python{sys.version_info[0]}.{sys.version_info[1]}",
    }
    if original_argv[0] in python_aliases:
        original_argv[0] = sys.executable
    if Path(original_argv[0]).resolve() != Path(sys.executable).resolve():
        raise RuntimeError("ScAI sandbox requires a Python kernelspec")
    if "-P" not in original_argv[1:]:
        original_argv.insert(1, "-P")

    launcher = (
        Path(__file__).resolve().parents[1]
        / "kernel_startup"
        / "launcher.py"
    )
    manager.kernel_spec.argv = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(launcher),
        *original_argv,
    ]

async def wait_for_ready(sandbox, *, bindings):
    """Wait for kernel to be ready"""
    while True:
        try:
            sandbox.kc.kernel_info()
            msg = await bindings._to_thread_drained(
                sandbox.kc.get_shell_msg, timeout=5
            )
            if msg['msg_type'] == 'kernel_info_reply':
                break
        except bindings.QueueEmpty:
            # ZMQ poll timed out — kernel not ready yet
            await bindings.asyncio.sleep(0.5)
        except Exception as e:
            bindings.logger.warning(f"Kernel not ready yet: {e}")
            await bindings.asyncio.sleep(0.5)

_COMMON_BOOTSTRAP_TEMPLATE = """
# === MANDATORY: Set ALL random seeds for reproducibility ===
import random
import os
import sys
from pathlib import Path

# Private aliases for the trusted wrappers installed further down. Analysis
# cells share this namespace and routinely rebind `Path` -- `from
# matplotlib.path import Path` is ordinary in code that draws connectors --
# so a wrapper that resolves `Path` or `os` by its public name stops working
# the moment a cell does. path_setup_code re-imports the same two names for
# the I/O write guard.
import os as _sandbox_os
import pathlib as _pathlib

if (
    os.environ.get('SCAI_KERNEL_REQUIRE_LANDLOCK') == '1'
    and not os.environ.get('SCAI_KERNEL_LANDLOCK_ACTIVE')
):
    raise RuntimeError('required kernel filesystem isolation is not active')

# Define the workspace Path before any matplotlib wrappers refer to it.  Do not
# chdir, create directories, or install the I/O sandbox until path_setup_code.
_WORKSPACE_DIR = Path(__SCAI_WORKSPACE_REPR__)
# ``-P`` keeps the workspace off Python's startup path so it cannot shadow the
# trusted sitecustomize module. Restore ordinary notebook import semantics only
# after the native boundary is already active.
if str(_WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_DIR))

SEED = 42

# === Isolate Numba cache per session to prevent concurrent cache corruption ===
_numba_cache = Path(__SCAI_SESSION_TMP_REPR__) / 'numba_cache'
_numba_cache.mkdir(parents=True, exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = str(_numba_cache)
# Prevent Numba from spawning extra threads that compete for cache locks
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

# Python built-in random
random.seed(SEED)

# Environment variable for hash seed
os.environ['PYTHONHASHSEED'] = str(SEED)

# NumPy
import numpy as np
np.random.seed(SEED)

# Try to set PyTorch seeds (if available)
try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Process-backed DataLoader workers require flat POSIX semaphores in the
    # container-wide /dev/shm. Sharing that directory would defeat per-session
    # write isolation, so fail immediately with an actionable alternative.
    _scai_original_dataloader_init = torch.utils.data.DataLoader.__init__
    if not getattr(_scai_original_dataloader_init, '_scai_thread_guard', False):
        def _scai_isolated_dataloader_init(self, *args, **kwargs):
            if len(args) > 5:
                _scai_num_workers = args[5]
            else:
                _scai_num_workers = kwargs.get('num_workers', 0)
            if int(_scai_num_workers or 0) > 0:
                raise RuntimeError(
                    'Process-backed PyTorch DataLoader workers are unavailable '
                    'inside an isolated ScAI kernel because /dev/shm is not '
                    'shared across sessions. Set num_workers=0.'
                )
            return _scai_original_dataloader_init(self, *args, **kwargs)
        _scai_isolated_dataloader_init._scai_thread_guard = True
        torch.utils.data.DataLoader.__init__ = _scai_isolated_dataloader_init
except ImportError:
    pass

# Try to set TensorFlow seeds (if available)
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
except ImportError:
    pass

# Initialize environment
import warnings
warnings.filterwarnings('ignore')

# Trusted publication-wrapper output baselines. These underscore-prefixed
# values are installed before user code, excluded from runtime persistence,
# and survive ordinary notebook namespace pollution between cells.
import builtins as _scai_bootstrap_builtins
import json as _scai_bootstrap_json

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import dill for session persistence
import dill

# Set matplotlib defaults for publication-quality figures
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'savefig.dpi': 100,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.grid': False,
    'axes.linewidth': 1.2,
    'pdf.fonttype': 42,   # Vectorized fonts for editable PDFs
    'ps.fonttype': 42,
})

# Capture the true matplotlib.pyplot.close/show baselines before any wrapper
# installs stubs over them (e.g. _TPL_NOOP_INSTALL).  Placed before the
# Combined Monkey-patch below to avoid the _WORKSPACE_DIR NameError that
# makes lines 781+ dead code in the kernel.
_original_plt_close = plt.close
_original_plt_show = plt.show
_original_plt_savefig = plt.savefig

# === Combined Monkey-patch: Figure.savefig with PNG->JPG + path redirect ===
import matplotlib.figure as _mpl_figure
_original_savefig = _mpl_figure.Figure.savefig

_FIGURES_DIR = _WORKSPACE_DIR / 'figures'
_WORKSPACE_DIR_RESOLVED = _WORKSPACE_DIR.resolve()

def _combined_savefig(self, fname, *args, **kwargs):
    '''Combined patch: PNG->JPG conversion + workspace root redirect.'''
    # Step 1: Handle PNG->JPG conversion (if fname is a path string)
    _fname_str = (
        str(fname) if isinstance(fname, (str, _sandbox_os.PathLike)) else None
    )
    _needs_jpg_convert = _fname_str and _fname_str.lower().endswith('.png')

    if _needs_jpg_convert:
        try:
            _base, _ = _sandbox_os.path.splitext(_fname_str)
            _jpg_path = _base + '.jpg'
            # Modify kwargs only; let the final return handle saving
            kwargs = dict(kwargs)
            kwargs.pop('metadata', None)
            kwargs['format'] = 'jpeg'
            kwargs.setdefault('facecolor', 'white')
            if 'pil_kwargs' not in kwargs:
                kwargs['pil_kwargs'] = {}
            kwargs['pil_kwargs'].setdefault('quality', 85)
            kwargs['pil_kwargs'].setdefault('optimize', True)
            print(
                f"[Auto-convert] {_sandbox_os.path.basename(_fname_str)}"
                f" -> {_sandbox_os.path.basename(_jpg_path)}"
            )
            # Only update fname, do NOT save here — let final return handle it
            fname = _jpg_path
        except Exception as _err:
            # Fallback: save as original PNG if JPEG conversion fails
            print(f"[Auto-convert] Warning: {_err}, saving as PNG")

    # Step 2: Redirect workspace root -> figures/ (only for path strings)
    _redirect_failed = False
    if isinstance(fname, (str, _sandbox_os.PathLike)):
        try:
            _p = _pathlib.Path(str(fname)).resolve()
            if _p.parent == _WORKSPACE_DIR_RESOLVED:
                _new = _FIGURES_DIR / _p.name
                print(f"[savefig-redirect] {_p.name} -> figures/")
                fname = str(_new)
            elif not str(_p).startswith(str(_WORKSPACE_DIR_RESOLVED)):
                # Path resolved outside the workspace root — log for diagnostics (3c)
                print(
                    f"[savefig-redirect] skip: '{fname}' -> resolved to {_p} "
                    f"(workspace is {_WORKSPACE_DIR_RESOLVED}, "
                    f"cwd is {_sandbox_os.getcwd()})"
                )
        except Exception as _redir_err:
            print(f"[savefig-redirect] failed: {_redir_err}, saving to original location")
            _redirect_failed = True

    try:
        return _original_savefig(self, fname, *args, **kwargs)
    except Exception as _save_err:
        # Last resort: if redirect caused issues, retry with original fname
        if not _redirect_failed:
            raise
        print(f"[savefig] redirect recovery failed: {_save_err}, trying original path")
        _orig_path_str = str(fname)
        if str(_FIGURES_DIR) in _orig_path_str:
            _relative = _orig_path_str.replace(str(_FIGURES_DIR), "", 1)
            _recovery = str(_WORKSPACE_DIR / _relative.lstrip("/"))
        else:
            _recovery = _orig_path_str
        return _original_savefig(self, _recovery, *args, **kwargs)

_mpl_figure.Figure.savefig = _combined_savefig

# Capture the true matplotlib.pyplot.close baseline before any wrapper installs
# stubs over it. Used by per-execution wrappers to defensively force-restore
# when a previous run left a stale no-op behind (e.g. legacy unmarked stubs
# from sessions started before the unwind-on-install fix).
_original_plt_close = plt.close

# === Monkey-patch plt.show() to capture figures as display_data ===
# With Agg backend, plt.show() is a no-op — figures are silently lost.
# This patch saves all open figures as inline PNG images via IPython.display,
# so the Jupyter kernel emits display_data messages that get captured.
import io as _io
_original_plt_show = plt.show  # Baseline before any SCAI patches (symmetric with _original_plt_close)

def _capturing_show(*args, **kwargs):
    '''Capture all open figures as inline images, then close them.'''
    import matplotlib.pyplot as _plt
    _fignums = _plt.get_fignums()
    if not _fignums:
        return
    try:
        from IPython.display import display, Image as IPImage
    except ImportError:
        _original_plt_show(*args, **kwargs)
        return
    for _fig_num in _fignums:
        _fig = _plt.figure(_fig_num)
        _buf = _io.BytesIO()
        _original_savefig(_fig, _buf, format='png', bbox_inches='tight', dpi=100, facecolor='white')
        _buf.seek(0)
        display(IPImage(data=_buf.read(), format='png'))
        _buf.close()
    _plt.close('all')

plt.show = _capturing_show  # Baseline capturing_show; referenced by defensive restore in _TPL_NOOP_INSTALL

# Keep publication-finalization capabilities outside IPython's mutable user_ns.
# The startup module accepts this tuple once and exposes no mutation API.
import scai_kernel_trust as _scai_kernel_trust
_scai_kernel_shell = get_ipython()
_scai_kernel_trust.capture_runtime(
    builtins_map=_scai_bootstrap_builtins.dict(
        _scai_bootstrap_builtins.vars(_scai_bootstrap_builtins)
    ),
    print_fn=_scai_bootstrap_builtins.print,
    json_dumps=_scai_bootstrap_json.dumps,
    stdout=sys.stdout,
    get_parent_header=_scai_kernel_shell.get_parent,
    set_stdout_parent=sys.stdout.set_parent,
    plt_close=_original_plt_close,
    plt_show=_capturing_show,
    plt_get_fignums=plt.get_fignums,
    plt_gcf=plt.gcf,
    figure_savefig=_original_savefig,
    runtime_savefig=_combined_savefig,
)
del _scai_kernel_shell
del _scai_kernel_trust

# Try importing common bioinformatics packages
try:
    import scanpy as sc
    sc.settings.verbosity = 1
    sc.settings.set_figure_params(dpi=100, frameon=False)
    # Set scanpy random state
    sc.settings.seed = SEED
except ImportError:
    pass

sys.modules["scai_pydeseq2_compat"].install()
sys.modules["scai_scanpy_compat"].install()
sys.modules["scai_arboreto_compat"].install()

try:
    import anndata as ad
except ImportError:
    pass

try:
    import seaborn as sns
    sns.set_theme(style="ticks", font_scale=1.2)
except ImportError:
    pass

# Dask's process-backed LocalCluster also requires container-wide POSIX
# semaphores. Force local clusters onto Worker threads; remote Client addresses
# are unaffected. This preserves local Dask/arboreto execution without opening
# /dev/shm to every session.
try:
    import distributed as _scai_distributed
    _scai_local_cluster_class = _scai_distributed.LocalCluster
    _scai_original_local_cluster_init = _scai_local_cluster_class.__init__
    if not getattr(_scai_original_local_cluster_init, '_scai_thread_guard', False):
        def _scai_threaded_local_cluster_init(self, *args, **kwargs):
            _scai_cluster_args = list(args)
            if len(_scai_cluster_args) > 3:
                _scai_cluster_args[3] = False
            else:
                kwargs['processes'] = False
            _scai_worker_class = (
                _scai_cluster_args[21]
                if len(_scai_cluster_args) > 21
                else kwargs.get('worker_class')
            )
            if (
                _scai_worker_class is not None
                and isinstance(_scai_worker_class, type)
                and issubclass(_scai_worker_class, _scai_distributed.Nanny)
            ):
                if len(_scai_cluster_args) > 21:
                    _scai_cluster_args[21] = _scai_distributed.Worker
                else:
                    kwargs['worker_class'] = _scai_distributed.Worker
            return _scai_original_local_cluster_init(
                self, *_scai_cluster_args, **kwargs
            )
        _scai_threaded_local_cluster_init._scai_thread_guard = True
        _scai_local_cluster_class.__init__ = _scai_threaded_local_cluster_init
except ImportError:
    pass

# Helper function to ensure reproducibility in function calls
def get_seed():
    return 42

# Helper function to check and fix NaN values in AnnData
def check_and_fix_nan(adata, fix=False):
    '''
    Check for NaN values in adata.X and optionally fix them.
    MUST be called before PCA or any dimensionality reduction!
\x20\x20\x20\x20
    Args:
        adata: AnnData object
        fix: If True, replace NaN with 0. Defaults to False so inspection does
             not mutate analysis data.
\x20\x20\x20\x20
    Returns:
        Number of NaN values found
    '''
    import scipy.sparse as sp
\x20\x20\x20\x20
    if sp.issparse(adata.X):
        nan_count = np.isnan(adata.X.data).sum()
        if fix and nan_count > 0:
            adata.X.data[np.isnan(adata.X.data)] = 0.0
            print(f"Fixed {nan_count} NaN values in sparse matrix")
    else:
        nan_count = np.isnan(adata.X).sum()
        if fix and nan_count > 0:
            adata.X = np.where(np.isnan(adata.X), 0.0, adata.X)
            print(f"Fixed {nan_count} NaN values in dense matrix")
\x20\x20\x20\x20
    if nan_count == 0:
        print("No NaN values found - data is clean")
    elif not fix:
        print(f"Found {nan_count} NaN values; no changes made (pass fix=True to replace them)")
\x20\x20\x20\x20
    return nan_count

print(f"Environment initialized with SEED={SEED} for reproducibility")
print("Helper functions available: get_seed(), check_and_fix_nan(adata, fix=False)")
"""


def build_common_bootstrap_source(sandbox) -> str:
    working_dir_abs = str(sandbox.working_dir.absolute())
    return _COMMON_BOOTSTRAP_TEMPLATE.replace(
        '__SCAI_SESSION_TMP_REPR__', repr(str(sandbox._session_tmp))
    ).replace('__SCAI_WORKSPACE_REPR__', repr(working_dir_abs))


def build_path_setup_source(sandbox) -> str:
    working_dir_abs = str(sandbox.working_dir.absolute())
    state_dir_abs = str(sandbox.notebook_state.state_dir)
    path_setup_code = f"""
# === Set working directory paths ===
import os
from pathlib import Path

# Redirect tempfile to per-session subdir
os.environ['TMPDIR'] = {repr(str(sandbox._session_tmp))}
os.environ['TEMP'] = {repr(str(sandbox._session_tmp))}
os.environ['TMP'] = {repr(str(sandbox._session_tmp))}

_WORKSPACE_DIR = Path({repr(working_dir_abs)})
os.chdir(_WORKSPACE_DIR)

# Create standard subdirectories
(_WORKSPACE_DIR / 'figures').mkdir(exist_ok=True)
(_WORKSPACE_DIR / 'tables').mkdir(exist_ok=True)

# Set scanpy figdir to absolute path (prevent relative path issues)
try:
    import scanpy as sc
    sc.settings.figdir = str(_WORKSPACE_DIR / 'figures')
except Exception as _sc_err:
    print(f"[sandbox] scanpy figdir setup skipped: {{_sc_err}}")

# === Runtime I/O sandbox: intercept writes outside allowed directories ===
import builtins as _builtins
import io as _sandbox_io
import os as _sandbox_os
import pathlib as _pathlib

_ORIGINAL_OPEN = _builtins.open
_ORIGINAL_IO_OPEN = _sandbox_io.open
_ORIGINAL_OS_OPEN = os.open
_ORIGINAL_MKDIR = _pathlib.Path.mkdir
_ORIGINAL_OS_MKDIR = os.mkdir
_ORIGINAL_PATH_OPEN = _pathlib.Path.open
_ORIGINAL_PATH_TOUCH = _pathlib.Path.touch

# Leaf directories where write operations are allowed. Reads remain available
# outside these roots so uploaded inputs and installed libraries still work.
_ALLOWED_WRITE_ROOTS = [
    {repr(working_dir_abs)},   # Analysis outputs and user-visible artifacts
    {repr(state_dir_abs)},     # Trusted runtime persistence cells
    {repr(str(sandbox._session_tmp))},   # Per-session temporary files
]

# The guard shares its namespace with the analysis code it guards, so it
# must never resolve `Path` or `os` by their public names. A figure cell
# doing `from matplotlib.path import Path` rebinds `Path` to a vertex-array
# constructor; every path resolution then raises, the guard reads that as
# "outside the workspace", and the kernel loses the ability to write anything
# at all -- including the trusted runtime-persistence cell, which surfaces as
# a runtime_persistence_failure with a workspace message that names the state
# directory the guard itself allows. The underscore-prefixed aliases below are
# bound before any user code runs and are excluded from the persisted
# namespace, so they stay reachable no matter what a cell imports.
def _resolve_guard_path(filepath, dir_fd=None):
    '''Resolve a path, including paths relative to an open directory fd.'''
    if isinstance(filepath, int):
        try:
            return _pathlib.Path(
                _sandbox_os.readlink(f'/proc/self/fd/{{filepath}}')
            ).resolve()
        except Exception:
            return None
    try:
        raw_path = _sandbox_os.fspath(filepath)
        if isinstance(raw_path, bytes):
            raw_path = _sandbox_os.fsdecode(raw_path)
        candidate = _pathlib.Path(raw_path)
    except (TypeError, ValueError) as e:
        print(f"[sandbox] Path resolution failed for {{filepath!r}}: {{e}}")
        return None
    if not candidate.is_absolute() and dir_fd not in (
        None, -1, getattr(_sandbox_os, 'AT_FDCWD', -100)
    ):
        try:
            candidate = _pathlib.Path(
                _sandbox_os.readlink(f'/proc/self/fd/{{dir_fd}}')
            ) / candidate
        except Exception:
            return None
    try:
        return candidate.resolve()
    except Exception as e:
        print(f"[sandbox] Path resolution failed for {{filepath!r}}: {{e}}")
        return None

def _is_write_allowed(filepath, dir_fd=None):
    '''Check if writing to this path is allowed.'''
    target = _resolve_guard_path(filepath, dir_fd=dir_fd)
    if target is None:
        return False
    resolved = str(target)
    # Check allowed roots
    for root in _ALLOWED_WRITE_ROOTS:
        r = root.rstrip('/')
        if resolved == r or resolved.startswith(r + '/'):
            return True
    if resolved == _sandbox_os.devnull:
        return True
    return False

def _mode_requests_write(mode):
    '''Return whether a high-level file mode can mutate filesystem content.'''
    return isinstance(mode, str) and any(flag in mode for flag in 'wax+')

def _os_flags_request_write(flags):
    '''Return whether os.open flags request write/create/truncate semantics.'''
    if not isinstance(flags, int):
        return False
    if (
        flags & getattr(_sandbox_os, 'O_ACCMODE', 3)
    ) != getattr(_sandbox_os, 'O_RDONLY', 0):
        return True
    mutation_flags = (
        getattr(_sandbox_os, 'O_CREAT', 0)
        | getattr(_sandbox_os, 'O_TRUNC', 0)
        | getattr(_sandbox_os, 'O_APPEND', 0)
        | getattr(_sandbox_os, 'O_TMPFILE', 0)
    )
    return bool(flags & mutation_flags)

def _sandboxed_open(file, mode='r', *args, **kwargs):
    '''Sandboxed open() that blocks writes outside allowed directories.'''
    # Integer file descriptors are already-open resources (e.g. the inter-process
    # pipes multiprocessing/dask use under GRNBoost2). The access decision was made
    # when they were opened, and they are not path-resolvable — only path-like
    # targets are subject to the write check.
    if isinstance(file, int):
        return _ORIGINAL_OPEN(file, mode, *args, **kwargs)
    if _mode_requests_write(mode):
        if not _is_write_allowed(file):
            raise PermissionError(
                f"\\n[Sandbox] Write blocked: {{file}}\\n"
                f"Files must be saved within the session workspace.\\n"
                f"  - CSV/Excel -> {{_WORKSPACE_DIR}}/tables/\\n"
                f"  - H5AD      -> {{_WORKSPACE_DIR}}/\\n"
                f"  - Figures are saved automatically by the system."
            )
    return _ORIGINAL_OPEN(file, mode, *args, **kwargs)

def _sandboxed_path_open(self, mode='r', *args, **kwargs):
    '''Sandboxed Path.open() with the same mode policy as builtins.open().'''
    if _mode_requests_write(mode) and not _is_write_allowed(self):
        raise PermissionError(
            f"\\n[Sandbox] Write blocked: {{self}}\\n"
            f"Files must be saved within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_PATH_OPEN(self, mode, *args, **kwargs)

def _sandboxed_os_open(path, flags, mode=0o777, *, dir_fd=None):
    '''Sandboxed os.open() including low-level update and creation flags.'''
    if _os_flags_request_write(flags) and not _is_write_allowed(path, dir_fd=dir_fd):
        raise PermissionError(
            f"\\n[Sandbox] Write blocked: {{path}}\\n"
            f"Files must be saved within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_OS_OPEN(path, flags, mode, dir_fd=dir_fd)

def _sandboxed_path_touch(self, *args, **kwargs):
    '''Sandboxed Path.touch() that blocks creation outside allowed areas.'''
    if not _is_write_allowed(self):
        raise PermissionError(
            f"\\n[Sandbox] touch blocked: {{self}}\\n"
            f"Files must be saved within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_PATH_TOUCH(self, *args, **kwargs)

def _sandboxed_mkdir(self, *args, **kwargs):
    '''Sandboxed Path.mkdir() that blocks directory creation outside allowed areas.'''
    if not _is_write_allowed(self):
        raise PermissionError(
            f"\\n[Sandbox] mkdir blocked: {{self}}\\n"
            f"Directories must be within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_MKDIR(self, *args, **kwargs)

def _sandboxed_os_mkdir(path, mode=0o777, *, dir_fd=None):
    '''Sandboxed os.mkdir() used by os.makedirs and tempfile helpers.'''
    if not _is_write_allowed(path, dir_fd=dir_fd):
        raise PermissionError(
            f"\\n[Sandbox] mkdir blocked: {{path}}\\n"
            f"Directories must be within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_OS_MKDIR(path, mode, dir_fd=dir_fd)

_ORIGINAL_WRITE_TEXT = _pathlib.Path.write_text
_ORIGINAL_WRITE_BYTES = _pathlib.Path.write_bytes

def _sandboxed_write_text(self, *args, **kwargs):
    '''Sandboxed Path.write_text() that blocks writes outside allowed areas.'''
    if not _is_write_allowed(self):
        raise PermissionError(
            f"\\n[Sandbox] Write blocked: {{self}}\\n"
            f"Files must be saved within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_WRITE_TEXT(self, *args, **kwargs)

def _sandboxed_write_bytes(self, *args, **kwargs):
    '''Sandboxed Path.write_bytes() that blocks writes outside allowed areas.'''
    if not _is_write_allowed(self):
        raise PermissionError(
            f"\\n[Sandbox] Write blocked: {{self}}\\n"
            f"Files must be saved within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_WRITE_BYTES(self, *args, **kwargs)

_builtins.open = _sandboxed_open
_sandbox_io.open = _sandboxed_open
open = _sandboxed_open
os.open = _sandboxed_os_open
os.mkdir = _sandboxed_os_mkdir
_pathlib.Path.mkdir = _sandboxed_mkdir
_pathlib.Path.open = _sandboxed_path_open
_pathlib.Path.touch = _sandboxed_path_touch
_pathlib.Path.write_text = _sandboxed_write_text
_pathlib.Path.write_bytes = _sandboxed_write_bytes

# === Sandbox file/directory deletion ===
_ORIGINAL_OS_REMOVE = os.remove
_ORIGINAL_OS_UNLINK = os.unlink
_ORIGINAL_OS_RMDIR = os.rmdir
_ORIGINAL_PATH_UNLINK = _pathlib.Path.unlink
_ORIGINAL_PATH_RMDIR = _pathlib.Path.rmdir

def _sandboxed_os_remove(path, *args, **kwargs):
    '''Sandboxed os.remove() that blocks deletes outside allowed directories.'''
    if not _is_write_allowed(path):
        raise PermissionError(
            f"\\n[Sandbox] Delete blocked: {{path}}\\n"
            f"Deletes must be within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_OS_REMOVE(path, *args, **kwargs)

def _sandboxed_os_unlink(path, *args, **kwargs):
    '''Sandboxed os.unlink() that blocks deletes outside allowed directories.'''
    if not _is_write_allowed(path):
        raise PermissionError(
            f"\\n[Sandbox] Delete blocked: {{path}}\\n"
            f"Deletes must be within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_OS_UNLINK(path, *args, **kwargs)

def _sandboxed_os_rmdir(path, *args, **kwargs):
    '''Sandboxed os.rmdir() that blocks rmdir outside allowed directories.'''
    if not _is_write_allowed(path):
        raise PermissionError(
            f"\\n[Sandbox] rmdir blocked: {{path}}\\n"
            f"Directory removal must be within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_OS_RMDIR(path, *args, **kwargs)

def _sandboxed_path_unlink(self, *args, **kwargs):
    '''Sandboxed Path.unlink() that blocks deletes outside allowed directories.'''
    if not _is_write_allowed(self):
        raise PermissionError(
            f"\\n[Sandbox] Delete blocked: {{self}}\\n"
            f"Deletes must be within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_PATH_UNLINK(self, *args, **kwargs)

def _sandboxed_path_rmdir(self, *args, **kwargs):
    '''Sandboxed Path.rmdir() that blocks rmdir outside allowed directories.'''
    if not _is_write_allowed(self):
        raise PermissionError(
            f"\\n[Sandbox] rmdir blocked: {{self}}\\n"
            f"Directory removal must be within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_PATH_RMDIR(self, *args, **kwargs)

os.remove = _sandboxed_os_remove
os.unlink = _sandboxed_os_unlink
os.rmdir = _sandboxed_os_rmdir
_pathlib.Path.unlink = _sandboxed_path_unlink
_pathlib.Path.rmdir = _sandboxed_path_rmdir

print(f"Working directory set to: {{os.getcwd()}}")
print(f"I/O sandbox active: writes restricted to session directory")
"""
    return path_setup_code
