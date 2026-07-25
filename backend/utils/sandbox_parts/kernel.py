# Jupyter kernel lifecycle: start, readiness, restart, stop.
# Moved verbatim out of backend/utils/sandbox.py; composed into LocalJupyterSandbox.
"""
Local Jupyter Kernel Executor with State Persistence

Uses dill.dump_session/load_session to save and restore the complete Python environment.
Supports pause/resume by saving and restoring the full session state.
"""

import os
import re
import json
import base64
import asyncio
import time
import hmac
import hashlib
import logging
from queue import Empty as QueueEmpty
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_output
from jupyter_client import KernelManager

from backend.utils.crypto import get_integrity_secret, sign_hmac_sha256
from backend.config._config import DATA_BASE_DIR


# ---------------------------------------------------------------------------
# Session pkl HMAC integrity helpers (host-side only)
# ---------------------------------------------------------------------------

from .shared import (  # noqa: F401
    os,
    re,
    json,
    base64,
    asyncio,
    time,
    hmac,
    hashlib,
    logging,
    QueueEmpty,
    Optional,
    Dict,
    Any,
    List,
    Tuple,
    datetime,
    Path,
    nbformat,
    new_notebook,
    new_code_cell,
    new_output,
    KernelManager,
    get_integrity_secret,
    sign_hmac_sha256,
    DATA_BASE_DIR,
    IOPUB_POLL_CAP,
    _sig_path,
    _pkl_hmac,
    _pkl_hmac_stream,
    _write_pkl_signature,
    _verify_pkl_integrity,
    _integrity_dir,
    _path_is_within,
    compute_remaining_timeout,
    logger,
    _is_kernel_protocol_error,
    DANGEROUS_PATTERNS,
    _is_safe_os_system,
    _is_safe_subprocess,
    _is_safe_importlib,
    _is_safe_file_write,
    _EXCEPTION_CHECKS,
    _DANGEROUS_AST_FUNCS,
    _DANGEROUS_AST_ATTRS,
    _DANGEROUS_AST_MODULES,
    _SUBPROCESS_EXEC_FUNCS,
    _validate_code_safety_ast,
    validate_code_safety,
    extract_code_from_markdown,
    CodeExecutionResult,
    NotebookState,
    _TPL_NOOP_INSTALL,
)


class KernelMixin:
    """Jupyter kernel lifecycle: start, readiness, restart, stop."""

    def _init_exec_lock(self):
        """Single lock serializing every coroutine that drives the kernel client
        sockets (execute + save_session). Two coroutines reading the same ZMQ
        socket concurrently triggers libzmq abort (see src/msg.cpp warning)."""
        self._exec_lock = asyncio.Lock()

    async def start(self) -> dict:
        """Start the Jupyter kernel

        Returns:
            dict with keys:
                - started: bool - whether kernel was started (False if already running)
                - session_restored: bool - whether previous session was restored
                - restore_failed: bool - whether restore was attempted but failed
        """
        # Snapshot V2: unresolved recovery blocks kernel startup entirely.
        from backend.services.session_store import assert_recovery_clear
        assert_recovery_clear(session_id=self.session_id)

        if self.km is not None:
            return {
                "started": False,
                "session_restored": self._session_restored,
                "restore_failed": self._session_restore_failed
            }

        logger.info(f"Starting Jupyter kernel for session {self.session_id}")

        # Create kernel manager
        self.km = KernelManager(kernel_name='python3')
        self.km.start_kernel(cwd=str(self.working_dir))

        # Get client
        self.kc = self.km.client()
        self.kc.start_channels()

        # Wait for kernel to be ready
        try:
            await asyncio.wait_for(
                self._wait_for_ready(),
                timeout=30
            )
        except asyncio.TimeoutError:
            raise RuntimeError("Kernel failed to start within timeout")

        # Initialize with common imports and settings
        await self._initialize_kernel()

        # A freshly started+initialized kernel is alive regardless of any prior
        # dead state; reset before the best-effort restore below.
        self._kernel_dead = False

        # Try to restore previous session
        await self._restore_session()

        return {
            "started": True,
            "session_restored": self._session_restored,
            "restore_failed": self._session_restore_failed
        }

    async def _wait_for_ready(self):
        """Wait for kernel to be ready"""
        while True:
            try:
                self.kc.kernel_info()
                msg = await asyncio.to_thread(self.kc.get_shell_msg, timeout=5)
                if msg['msg_type'] == 'kernel_info_reply':
                    break
            except QueueEmpty:
                # ZMQ poll timed out — kernel not ready yet
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"Kernel not ready yet: {e}")
                await asyncio.sleep(0.5)

    async def _initialize_kernel(self):
        """Initialize kernel with common setup"""
        if self._initialized:
            return
        
        # NOTE: This is a plain string (not f-string) because the code contains
        # dict literals ({...}), inner f-strings, and other braces that conflict
        # with f-string parsing.  The only dynamic value (session_id) is injected
        # via str.replace() at the end.
        init_code = """
# === MANDATORY: Set ALL random seeds for reproducibility ===
import random
import os
from pathlib import Path

SEED = 42

# === Isolate Numba cache per session to prevent concurrent cache corruption ===
_numba_cache = Path('__SESSION_TMP__') / 'numba_cache'
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

# === Combined Monkey-patch: Figure.savefig with PNG->JPG + path redirect ===
import matplotlib.figure as _mpl_figure
_original_savefig = _mpl_figure.Figure.savefig

_FIGURES_DIR = _WORKSPACE_DIR / 'figures'
_WORKSPACE_DIR_RESOLVED = _WORKSPACE_DIR.resolve()

def _combined_savefig(self, fname, *args, **kwargs):
    '''Combined patch: PNG->JPG conversion + workspace root redirect.'''
    # Step 1: Handle PNG->JPG conversion (if fname is a path string)
    _fname_str = str(fname) if isinstance(fname, (str, os.PathLike)) else None
    _needs_jpg_convert = _fname_str and _fname_str.lower().endswith('.png')

    if _needs_jpg_convert:
        try:
            _base, _ = os.path.splitext(_fname_str)
            _jpg_path = _base + '.jpg'
            # Modify kwargs only; let the final return handle saving
            kwargs = dict(kwargs)
            kwargs['format'] = 'jpeg'
            kwargs.setdefault('facecolor', 'white')
            if 'pil_kwargs' not in kwargs:
                kwargs['pil_kwargs'] = {}
            kwargs['pil_kwargs'].setdefault('quality', 85)
            kwargs['pil_kwargs'].setdefault('optimize', True)
            print(f"[Auto-convert] {os.path.basename(_fname_str)} -> {os.path.basename(_jpg_path)}")
            # Only update fname, do NOT save here — let final return handle it
            fname = _jpg_path
        except Exception as _err:
            # Fallback: save as original PNG if JPEG conversion fails
            print(f"[Auto-convert] Warning: {_err}, saving as PNG")

    # Step 2: Redirect workspace root -> figures/ (only for path strings)
    _redirect_failed = False
    if isinstance(fname, (str, os.PathLike)):
        try:
            _p = Path(str(fname)).resolve()
            if _p.parent == _WORKSPACE_DIR_RESOLVED:
                _new = _FIGURES_DIR / _p.name
                print(f"[savefig-redirect] {_p.name} -> figures/")
                fname = str(_new)
            elif not str(_p).startswith(str(_WORKSPACE_DIR_RESOLVED)):
                # Path resolved outside the workspace root — log for diagnostics (3c)
                print(f"[savefig-redirect] skip: '{fname}' -> resolved to {_p} (workspace is {_WORKSPACE_DIR_RESOLVED}, cwd is {os.getcwd()})")
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

# Try importing common bioinformatics packages
try:
    import scanpy as sc
    sc.settings.verbosity = 1
    sc.settings.set_figure_params(dpi=100, frameon=False)
    # Set scanpy random state
    sc.settings.seed = SEED
except ImportError:
    pass

try:
    import anndata as ad
except ImportError:
    pass

try:
    import seaborn as sns
    sns.set_theme(style="ticks", font_scale=1.2)
except ImportError:
    pass

# Helper function to ensure reproducibility in function calls
def get_seed():
    return 42

# Helper function to check and fix NaN values in AnnData
def check_and_fix_nan(adata, fix=True):
    '''
    Check for NaN values in adata.X and optionally fix them.
    MUST be called before PCA or any dimensionality reduction!
    
    Args:
        adata: AnnData object
        fix: If True, replace NaN with 0
    
    Returns:
        Number of NaN values found
    '''
    import scipy.sparse as sp
    
    if sp.issparse(adata.X):
        nan_count = np.isnan(adata.X.data).sum()
        if fix and nan_count > 0:
            adata.X.data = np.nan_to_num(adata.X.data, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"Fixed {nan_count} NaN values in sparse matrix")
    else:
        nan_count = np.isnan(adata.X).sum()
        if fix and nan_count > 0:
            adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"Fixed {nan_count} NaN values in dense matrix")
    
    if nan_count == 0:
        print("No NaN values found - data is clean")
    
    return nan_count

print(f"Environment initialized with SEED={SEED} for reproducibility")
print("Helper functions available: get_seed(), check_and_fix_nan(adata)")
""".replace('__SESSION_ID__', self.session_id).replace(
            '__SESSION_TMP__', str(self._session_tmp)
        )
        result = await self._execute_unlocked(init_code, save_to_history=False, skip_safety_check=True)
        if result.output:
            logger.debug("Kernel init stdout: %s", result.output[:1000])

        # Set working directory paths and install I/O sandbox (requires f-string with self.working_dir)
        working_dir_abs = str(self.working_dir.absolute())
        state_dir_abs = str(self.notebook_state.state_dir)
        # Session root is the parent of workspace (e.g. sessions/{id}/)
        session_root_abs = str(self.working_dir.absolute().parent)
        path_setup_code = f"""
# === Set working directory paths ===
import os
from pathlib import Path

# Redirect tempfile to per-session subdir
os.environ['TMPDIR'] = {repr(str(self._session_tmp))}
os.environ['TEMP'] = {repr(str(self._session_tmp))}
os.environ['TMP'] = {repr(str(self._session_tmp))}

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
import pathlib as _pathlib

_ORIGINAL_OPEN = _builtins.open
_ORIGINAL_MKDIR = _pathlib.Path.mkdir

# Directories where write operations are allowed
_ALLOWED_WRITE_ROOTS = [
    {repr(session_root_abs)},   # Session root (workspace/, states/, uploads/, reports/)
    {repr(str(self._session_tmp))},   # Per-session temporary files
]

# System paths (pip installs, font cache, etc.) — always allowed
_SYSTEM_WRITE_PREFIXES = [
    '/usr/', '/opt/', '/var/',
    os.path.expanduser('~/.cache'),
    os.path.expanduser('~/.local'),
    os.path.expanduser('~/.config'),
]

def _is_write_allowed(filepath):
    '''Check if writing to this path is allowed.'''
    try:
        resolved = str(Path(filepath).resolve())
    except Exception as e:
        print(f"[sandbox] Path resolution failed for {{filepath!r}}: {{e}}")
        return False  # Can't resolve -> deny by default for safety
    # Check allowed roots
    for root in _ALLOWED_WRITE_ROOTS:
        r = root.rstrip('/')
        if resolved == r or resolved.startswith(r + '/'):
            return True
    # Check system paths
    for prefix in _SYSTEM_WRITE_PREFIXES:
        p = prefix.rstrip('/')
        if resolved == p or resolved.startswith(p + '/'):
            return True
    return False

def _sandboxed_open(file, mode='r', *args, **kwargs):
    '''Sandboxed open() that blocks writes outside allowed directories.'''
    # Integer file descriptors are already-open resources (e.g. the inter-process
    # pipes multiprocessing/dask use under GRNBoost2). The access decision was made
    # when they were opened, and they are not path-resolvable — only path-like
    # targets are subject to the write check.
    if isinstance(file, int):
        return _ORIGINAL_OPEN(file, mode, *args, **kwargs)
    if isinstance(mode, str) and any(m in mode for m in ('w', 'a', 'x')):
        if not _is_write_allowed(file):
            raise PermissionError(
                f"\\n[Sandbox] Write blocked: {{file}}\\n"
                f"Files must be saved within the session workspace.\\n"
                f"  - CSV/Excel -> {{_WORKSPACE_DIR}}/tables/\\n"
                f"  - H5AD      -> {{_WORKSPACE_DIR}}/\\n"
                f"  - Figures are saved automatically by the system."
            )
    return _ORIGINAL_OPEN(file, mode, *args, **kwargs)

def _sandboxed_mkdir(self, *args, **kwargs):
    '''Sandboxed Path.mkdir() that blocks directory creation outside allowed areas.'''
    if not _is_write_allowed(self):
        raise PermissionError(
            f"\\n[Sandbox] mkdir blocked: {{self}}\\n"
            f"Directories must be within the session workspace: {{_WORKSPACE_DIR}}"
        )
    return _ORIGINAL_MKDIR(self, *args, **kwargs)

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
_pathlib.Path.mkdir = _sandboxed_mkdir
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
        path_result = await self._execute_unlocked(path_setup_code, save_to_history=False, skip_safety_check=True)
        if path_result.output:
            logger.debug("Kernel path-setup stdout: %s", path_result.output[:1000])

        self._initialized = True
        logger.info("Kernel initialized")

    async def _restart_kernel(self) -> None:
        """Restart the kernel, then best-effort restore the session.

        Kernel-level failure (restart/ready/init) marks the sandbox dead. A
        restore failure (missing/corrupt dill dump) leaves the kernel fully
        usable, so it must NOT set _kernel_dead. Reentrancy is guarded because
        _initialize_kernel runs code through execute(), which can itself
        trigger a restart on a protocol error.
        """
        if self._restarting:
            logger.warning(
                f"[{self.session_id}] _restart_kernel reentered; skipping nested restart"
            )
            return
        self._restarting = True
        try:
            try:
                await asyncio.to_thread(self.km.restart_kernel)

                # Rebuild client connection.  Brief sleep to let any
                # in-flight ZMQ reader threads finish their current poll
                # cycle before we close the channels underneath them.
                await asyncio.sleep(0.3)

                if self.kc is not None:
                    try:
                        self.kc.stop_channels()
                    except Exception as e:
                        logger.debug(f"Failed to stop kernel channels during recovery: {e}")
                self.kc = self.km.client()
                self.kc.start_channels()

                # Drain stale messages that may have arrived on the new
                # channels before the kernel was fully initialised (e.g.
                # kernel-info broadcasts).  These carry the new session key
                # so they won't trigger "Invalid Signature", but leaving
                # them in the queue can confuse the first execute().
                try:
                    for _ in range(10):
                        await asyncio.to_thread(self.kc.get_iopub_msg, timeout=0.1)
                except QueueEmpty:
                    pass  # queue drained

                # Wait for kernel to be ready
                await asyncio.wait_for(self._wait_for_ready(), timeout=30)

                # Re-initialize the kernel
                self._initialized = False
                await self._initialize_kernel()
            except Exception as restart_error:
                logger.error(f"Failed to restart kernel: {restart_error}")
                self._kernel_dead = True
                return

            # Kernel is healthy again.
            self._kernel_dead = False

            # Restore is best-effort: a corrupt dump must not brick a live kernel.
            try:
                await self._restore_session()
            except Exception as restore_error:
                logger.error(
                    f"[{self.session_id}] Session restore failed after kernel restart "
                    f"(kernel remains usable): {restore_error}"
                )

            logger.info(f"Kernel restarted for session {self.session_id}")
        finally:
            self._restarting = False

    async def stop(self):
        """Stop the kernel and save state. Always shuts down kernel even if save fails."""
        # Try to save session before stopping, but don't let failure prevent kernel cleanup
        try:
            await self.await_pending_saves()
            await self.save_session()
            self.notebook_state.save()
        except Exception as e:
            logger.exception(f"Failed to save session during stop for {self.session_id}")
        finally:
            if self.kc is not None:
                try:
                    self.kc.stop_channels()
                except Exception as e:
                    logger.exception(f"Error stopping kernel channels for {self.session_id}")
                self.kc = None

            if self.km is not None:
                try:
                    self.km.shutdown_kernel(now=True)
                except Exception as e:
                    logger.exception(f"Error shutting down kernel for {self.session_id}")
                self.km = None
                self._initialized = False

            logger.info(f"Kernel stopped for session {self.session_id}")

    async def restart(self):
        """Restart kernel and restore state"""
        await self.stop()
        await self.start()
