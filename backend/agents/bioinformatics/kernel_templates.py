# Builders for the Python source that is executed inside the Jupyter kernel: resource guards, autosave wrappers, and figure wrappers.
# Moved verbatim out of backend/agents/bioinformatics_agent.py.
"""
Bioinformatics Agent - Handles snRNA-seq analysis with iterative code execution
Supports autonomous technical steps and PI consultation for biological decisions
With step-level checkpointing for recovery from interruptions
"""
from typing import List
import logging
import re
import ast
import textwrap


from ...utils.sandbox import _TPL_NOOP_INSTALL
import inspect
from backend.services import ora_kernel

_ORA_KERNEL_SRC = inspect.getsource(ora_kernel)

# Configure logger
logger = logging.getLogger(__name__)

MAX_WORKSPACE_H5AD_FILES = 2


_HIGH_MEMORY_ANALYSIS_RE = re.compile(
    r"\b(?:scv|scvelo)\s*\.\s*(?:pp\s*\.\s*moments|tl\s*\.\s*(?:velocity|velocity_graph))\s*\(|"
    r"\bcellrank\b|\bsc\s*\.\s*pp\s*\.\s*scale\s*\(|\.X\s*\.\s*toarray\s*\(",
    re.IGNORECASE,
)

def _high_memory_referenced_names(code: str) -> List[str]:
    """Return AnnData arguments for graph-building or dense-memory operations."""
    if not _HIGH_MEMORY_ANALYSIS_RE.search(code):
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    cellrank_aliases = {"cellrank"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "cellrank":
                    cellrank_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("cellrank"):
            for alias in node.names:
                cellrank_aliases.add(alias.asname or alias.name)

    def _dotted_name(node: ast.AST) -> str:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))

    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = _dotted_name(node.func)
        is_scvelo = bool(re.fullmatch(
            r"(?:scv|scvelo)\.(?:pp\.moments|tl\.(?:velocity|velocity_graph))",
            func_name,
            re.IGNORECASE,
        ))
        root = func_name.split(".", 1)[0]
        is_cellrank = root in cellrank_aliases
        is_scale = bool(re.fullmatch(r"sc\.pp\.scale", func_name, re.IGNORECASE))
        if is_scale and any(
            kw.arg == "zero_center"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is False
            for kw in node.keywords
        ):
            is_scale = False
        is_toarray = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "toarray"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "X"
            and isinstance(node.func.value.value, ast.Name)
        )
        if not (is_scvelo or is_cellrank or is_scale or is_toarray):
            continue
        if is_toarray:
            target = node.func.value.value
        else:
            target = node.args[0] if node.args else next(
                (kw.value for kw in node.keywords if kw.arg in {"adata", "data"}),
                None,
            )
        if isinstance(target, ast.Name):
            names.add(target.id)
    return sorted(names)

def _build_resource_guard_preamble(code: str) -> str:
    """Build a kernel-side memory preflight for high-memory operations.

    The guard measures referenced AnnData matrices and actual host/cgroup
    headroom immediately before user code runs. It fails early with actionable
    guidance instead of letting the kernel be OOM-killed and losing the cell.
    """
    names = _high_memory_referenced_names(code)
    if not names:
        return ""
    names_repr = repr(names)
    dense_transform = bool(re.search(r"(?:\bsc\s*\.\s*pp\s*\.\s*scale\s*\(|\.X\s*\.\s*toarray\s*\()", code, re.I))
    operation_factor = 2 if dense_transform else (
        8 if re.search(r"\bcellrank\b", code, re.I) else (
            6 if re.search(r"velocity_graph", code, re.I) else 4
        )
    )
    return f"""
# ScAI resource preflight for high-memory analysis.
def _scai_matrix_bytes(_value, _seen):
    if _value is None or id(_value) in _seen:
        return 0
    _seen.add(id(_value))
    _total = 0
    _has_sparse_parts = False
    for _part_name in ('data', 'indices', 'indptr'):
        _part = getattr(_value, _part_name, None)
        if _part is not None and id(_part) not in _seen:
            _has_sparse_parts = True
            _seen.add(id(_part))
            _total += int(getattr(_part, 'nbytes', 0) or 0)
    if not _has_sparse_parts:
        _total = int(getattr(_value, 'nbytes', 0) or 0)
    return _total

def _scai_adata_bytes(_adata):
    _seen = set()
    _total = _scai_matrix_bytes(getattr(_adata, 'X', None), _seen)
    for _mapping_name in ('layers', 'obsp', 'varp', 'obsm', 'varm'):
        _mapping = getattr(_adata, _mapping_name, None)
        if _mapping is not None:
            try:
                for _value in _mapping.values():
                    _total += _scai_matrix_bytes(_value, _seen)
            except Exception:
                pass
    return _total

def _scai_memory_headroom():
    _limits = []
    try:
        with open('/proc/meminfo', 'r', encoding='utf-8') as _meminfo:
            for _line in _meminfo:
                if _line.startswith('MemAvailable:'):
                    _limits.append(int(_line.split()[1]) * 1024)
                    break
    except Exception:
        pass
    for _max_path, _cur_path in (
        ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
        ('/sys/fs/cgroup/memory/memory.limit_in_bytes',
         '/sys/fs/cgroup/memory/memory.usage_in_bytes'),
    ):
        try:
            with open(_max_path, 'r', encoding='utf-8') as _f:
                _maximum = _f.read().strip()
            with open(_cur_path, 'r', encoding='utf-8') as _f:
                _current = int(_f.read().strip())
            if _maximum != 'max':
                _maximum_int = int(_maximum)
                if 0 < _maximum_int < (1 << 60):
                    _limits.append(max(0, _maximum_int - _current))
        except Exception:
            pass
    return min(_limits) if _limits else None

_scai_guard_names = {names_repr}
_scai_guard_candidates = []
for _scai_guard_name in _scai_guard_names:
    _scai_guard_obj = globals().get(_scai_guard_name)
    if (_scai_guard_obj is not None
            and hasattr(_scai_guard_obj, 'n_obs')
            and hasattr(_scai_guard_obj, 'n_vars')
            and hasattr(_scai_guard_obj, 'X')):
        _scai_guard_candidates.append((_scai_guard_name, _scai_guard_obj))

if _scai_guard_candidates:
    # Measure each candidate once and reuse the winner's byte count; the
    # earlier form re-traversed X/layers/obsp/varp/obsm/varm for the chosen
    # AnnData a second time just to recompute the same number.
    _scai_guard_name, _scai_guard_adata, _scai_guard_bytes = max(
        ((_n, _o, _scai_adata_bytes(_o)) for _n, _o in _scai_guard_candidates),
        key=lambda _item: _item[2],
    )
    _scai_guard_bytes = max(1, _scai_guard_bytes)
    _scai_guard_n_obs = int(getattr(_scai_guard_adata, 'n_obs', 0) or 0)
    _scai_guard_n_vars = int(getattr(_scai_guard_adata, 'n_vars', 0) or 0)
    _scai_guard_dense_bytes = _scai_guard_n_obs * _scai_guard_n_vars * 8 if {dense_transform!r} else 0
    _scai_guard_graph_bytes = (
        0 if {dense_transform!r}
        else _scai_guard_n_obs * min(_scai_guard_n_obs, 5000) * 8
    )
    _scai_guard_required = max(
        512 * 1024**2,
        _scai_guard_bytes * {operation_factor} + _scai_guard_graph_bytes + _scai_guard_dense_bytes,
    )
    _scai_guard_headroom = _scai_memory_headroom()
    if _scai_guard_dense_bytes > int(1.5 * 1024**3):
        raise MemoryError(
            'ScAI resource preflight blocked a full-matrix dense transformation: '
            f'{{_scai_guard_name}} would require at least '
            f'{{_scai_guard_dense_bytes / 1024**3:.1f}} GiB for one dense matrix, '
            'which would make runtime snapshots unsafe. Subset to relevant features/HVGs, '
            'use zero_center=False for sparse scaling, or keep the matrix sparse.'
        )
    if (_scai_guard_headroom is not None
            and _scai_guard_required > int(_scai_guard_headroom * 0.70)):
        raise MemoryError(
            'ScAI resource preflight blocked a high-memory operation: '
            f'{{_scai_guard_name}} has {{_scai_guard_n_obs:,}} cells and the operation '
            f'is estimated to need {{_scai_guard_required / 1024**3:.1f}} GiB, but only '
            f'{{_scai_guard_headroom / 1024**3:.1f}} GiB headroom is available. '
            'Subset/downsample the relevant lineage, remove unused AnnData copies, '
            'or use DPT/PAGA before retrying.'
        )
    print(
        f'[ScAI resource preflight] {{_scai_guard_name}}: '
        f'{{_scai_guard_n_obs:,}} cells; estimated additional budget '
        f'{{_scai_guard_required / 1024**3:.1f}} GiB; '
        f'headroom {{(_scai_guard_headroom or 0) / 1024**3:.1f}} GiB'
    )
"""

# Reusable sandbox template snippet: safely close all matplotlib figures
_TPL_SAFE_CLOSE_ALL = """\
try:
    import matplotlib.pyplot as plt
    # plt.close 可能是被中断的 cell 残留的 SCAI no-op 桩，此时 plt.close('all')
    # 会静默无效，figure 在内核里只增不减，每次执行都被重存/重显示一遍 ->
    # 报告里同一张图出现上百次。改用内核初始化时捕获的基线 close 绕过 no-op 桩。
    _scai_real_close = globals().get('_original_plt_close', plt.close)
    _scai_real_close('all')
except Exception:
    pass"""

# Reusable sandbox template snippet: re-import matplotlib AFTER user code and
# restore plt.close / plt.show to the immutable baselines captured at kernel
# init. User-generated code runs in the persistent global namespace and may
# rebind the ``plt`` alias (e.g. ``plt = 'shadowed'``); cleanup must never
# trust that mutable name, so it re-binds a fresh module reference instead.
# f-string safe: contains no { or }.
_TPL_PLT_RESTORE_BASELINE = """\
import matplotlib.pyplot as _scai_cleanup_plt
try:
    _scai_cleanup_plt.close = globals().get(
        '_original_plt_close', _scai_cleanup_plt.close
    )
except Exception:
    pass
try:
    _scai_cleanup_plt.show = globals().get('_capturing_show') or globals().get(
        '_original_plt_show', _scai_cleanup_plt.show
    )
except Exception:
    pass"""

def _build_exec_autosave_wrapper(code: str, figures_dir: str, timestamp: str) -> str:
    """Build the execute_code auto-save wrapper template.

    Only saves/displays figures newly created during THIS execution (by
    snapshotting fignums before user code and only processing the diff).
    Prevents re-dumping stale figures accumulated in the kernel — the root
    cause of the same image appearing 100+ times in reports.
    """
    figures_dir_repr = repr(str(figures_dir))
    user_source_repr = repr(code)
    resource_guard = textwrap.indent(_build_resource_guard_preamble(code), '    ')
    restore_baseline = textwrap.indent(_TPL_PLT_RESTORE_BASELINE, '    ')
    return f'''
import matplotlib.pyplot as plt
import matplotlib.figure as _mpl_figure
from pathlib import Path
from IPython.display import display

# Close any stale figures from previous executions to prevent poisoning
{_TPL_SAFE_CLOSE_ALL}

# Set PDF vectorized fonts
plt.rcParams.update({{
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}})

# Create figures directory
_figures_dir = Path({figures_dir_repr})
_figures_dir.mkdir(exist_ok=True)

# Track figures explicitly saved by user code: map fig_number -> list of saved paths
_explicitly_saved_figs = {{}}
# Unwind any leftover tracking wrappers from prior executions (e.g. a previous
# cell raised before its finally restored, or a dill session restore brought
# back a stale closure). Without this, each new wrap stacks one recursion level
# deeper and eventually hits Python's recursion limit.
_unwrap_guard = 0
while getattr(_mpl_figure.Figure.savefig, '_scai_tracking', False) and _unwrap_guard < 64:
    _mpl_figure.Figure.savefig = getattr(
        _mpl_figure.Figure.savefig, '_scai_prev', _mpl_figure.Figure.savefig
    )
    _unwrap_guard += 1
# Defensive: legacy sessions started before the marker-unwind fix have stacked
# unmarked closures that the loop above can't peel. Force-restore to the known
# baseline (_combined_savefig from sandbox init) if it's not already there.
try:
    if '_combined_savefig' in globals() and _mpl_figure.Figure.savefig is not _combined_savefig:
        _mpl_figure.Figure.savefig = _combined_savefig
        print("[Auto-save] Reset Figure.savefig to baseline (legacy stack cleared)")
except Exception:
    pass
_prev_fig_savefig = _mpl_figure.Figure.savefig
def _tracking_savefig(self, fname=None, *_a, **_kw):
    try:
        _explicitly_saved_figs.setdefault(self.number, [])
        if fname is not None:
            _explicitly_saved_figs[self.number].append(str(fname))
    except Exception as _track_err:
        print(f"[Auto-save] Debug: figure tracking failed: {{_track_err}}")
    return _prev_fig_savefig(self, fname, *_a, **_kw)
_tracking_savefig._scai_tracking = True
_tracking_savefig._scai_prev = _prev_fig_savefig
_mpl_figure.Figure.savefig = _tracking_savefig

{_TPL_NOOP_INSTALL}

# Snapshot figures that already existed before user code. Only figures newly
# created by THIS execution are saved/displayed below — even if stale figures
# linger (e.g. a prior cell left plt.close as a no-op), they are never re-dumped.
_pre_fignums = set(plt.get_fignums())

# Execute user code (auto-save runs in finally so PDFs are always supplemented,
# even when the user code raises midway through)
try:
{resource_guard}
    _scai_user_source = {user_source_repr}
    exec(compile(_scai_user_source, '<scai-user-code>', 'exec'), globals(), globals())
finally:
{restore_baseline}
    import matplotlib.figure as _scai_cleanup_figure

    # Restore Figure.savefig to the immutable baseline captured when the
    # sandbox kernel was initialized (close/show already restored above).
    try:
        _scai_cleanup_figure.Figure.savefig = globals().get(
            '_combined_savefig', _scai_cleanup_figure.Figure.savefig
        )
    except Exception:
        pass

    # Detect and save only figures created during THIS execution (diff vs snapshot)
    try:
        _fig_nums = [_n for _n in _scai_cleanup_plt.get_fignums() if _n not in _pre_fignums]
        for _fig_num in _fig_nums:
            try:
                _fig = _scai_cleanup_plt.figure(_fig_num)

                # Display figure to trigger capture via display_data message (always)
                display(_fig)

                _fig_id = f"fig_{timestamp}_{{_fig_num}}"

                # For figures already saved by user code, skip JPG but supplement PDF if missing
                if _fig_num in _explicitly_saved_figs:
                    _saved_paths = _explicitly_saved_figs[_fig_num]
                    # Emit [Auto-saved] for explicitly saved images so they get tracked
                    for _sp in _saved_paths:
                        _sp_lower = _sp.lower()
                        if _sp_lower.endswith(('.jpg', '.jpeg')):
                            print(f"[Auto-saved] JPG: {{_sp}}")
                        elif _sp_lower.endswith('.png'):
                            print(f"[Auto-saved] PNG: {{_sp}}")
                    _has_pdf = any(_p.lower().endswith('.pdf') for _p in _saved_paths)
                    if not _has_pdf and _saved_paths:
                        # Supplement PDF for every image path that doesn't already have one
                        _seen_pdf_targets = set()
                        for _src_str in _saved_paths:
                            _src_path = Path(_src_str)
                            if _src_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                                continue
                            _pdf_supplement = _src_path.with_suffix('.pdf')
                            if str(_pdf_supplement) in _seen_pdf_targets:
                                continue
                            _seen_pdf_targets.add(str(_pdf_supplement))
                            try:
                                _fig.savefig(str(_pdf_supplement), dpi=300, bbox_inches='tight', format='pdf')
                                print(f"[Auto-save] Supplemented PDF: {{_pdf_supplement}}")
                            except Exception as _pdf_err:
                                print(f"[Auto-save] Warning: could not supplement PDF for {{_src_path}}: {{_pdf_err}}")
                    else:
                        print(f"[Auto-save] Skipped figure {{_fig_num}} (already saved by code with PDF)")
                    continue

                _jpg_path = _figures_dir / f"{{_fig_id}}.jpg"
                _pdf_path = _figures_dir / f"{{_fig_id}}.pdf"

                # Save JPG (for web display — smaller file size than PNG)
                try:
                    _fig.savefig(_jpg_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none',
                                 format='jpeg', pil_kwargs={{'quality': 85, 'optimize': True}})
                    print(f"[Auto-saved] JPG: {{_jpg_path}}")
                except Exception as _jpg_err:
                    print(f"[Auto-save] Warning: could not save JPG for figure {{_fig_num}}: {{_jpg_err}}")

                # Save PDF (vector format for publication quality download)
                try:
                    _fig.savefig(_pdf_path, dpi=300, bbox_inches='tight', format='pdf')
                    print(f"[Auto-saved] PDF: {{_pdf_path}}")
                except Exception as _pdf_err:
                    print(f"[Auto-save] Warning: could not save PDF for figure {{_fig_num}}: {{_pdf_err}}")
            except Exception as _save_err:
                print(f"[Auto-save] Warning: could not save figure {{_fig_num}}: {{_save_err}}")
                try:
                    _scai_cleanup_plt.close(_fig_num)
                except Exception as _close_err:
                    print(f"[Auto-save] Debug: plt.close({{_fig_num}}) failed: {{_close_err}}")
    except Exception as _autosave_block_err:
        print(f"[Auto-save] Warning: figure auto-save failed: {{_autosave_block_err}}")

    # Close all figures to free memory
    try:
        _scai_cleanup_plt.close('all')
    except Exception:
        pass
'''

def _build_publication_figure_wrapper(
    code: str, figures_dir: str, jpg_path: str, pdf_path: str
) -> str:
    """Build the publication-figure execution wrapper (JPG + PDF auto-save).

    Shared by generate_publication_figure and its revision / auto-retry paths
    so every figure-execution path uses one cleanup + save template. The
    revised/retry paths previously inlined drifted copies that bypassed the
    ``_TPL_NOOP_INSTALL`` stub-unwind (a naive ``plt.close = lambda``); they
    now go through the same machinery as the initial render. Includes the
    velocity/fate memory preflight so the resource guard is not skipped on
    figure code (it is a no-op unless the code runs scv.tl.velocity / cellrank).
    """
    figures_dir_repr = repr(str(figures_dir))
    jpg_path_repr = repr(str(jpg_path))
    pdf_path_repr = repr(str(pdf_path))
    user_source_repr = repr(code)
    restore_baseline = textwrap.indent(_TPL_PLT_RESTORE_BASELINE, '    ')
    resource_guard = textwrap.indent(_build_resource_guard_preamble(code), '    ')
    return f'''
import matplotlib.pyplot as plt
from pathlib import Path

# Close any stale figures from previous executions to prevent poisoning
{_TPL_SAFE_CLOSE_ALL}

# Set publication-quality parameters
plt.rcParams.update({{
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'axes.grid': False,
    'axes.linewidth': 1.2,
    'pdf.fonttype': 42,  # Ensure fonts are vectorized in PDF
    'ps.fonttype': 42,
}})

# Create figures directory
_figures_dir = {figures_dir_repr}
Path(_figures_dir).mkdir(exist_ok=True)

{_TPL_NOOP_INSTALL}

# Execute the plotting code (save runs in finally so JPG+PDF are always written,
# even if user code raises midway through)
try:
{resource_guard}
    _scai_user_source = {user_source_repr}
    exec(compile(_scai_user_source, '<scai-user-code>', 'exec'), globals(), globals())
finally:
{restore_baseline}

    # Get current figure and save (both JPG and PDF)
    if _scai_cleanup_plt.get_fignums():
        try:
            fig = _scai_cleanup_plt.gcf()
            _jpg_path = {jpg_path_repr}
            _pdf_path = {pdf_path_repr}

            try:
                fig.savefig(_jpg_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none',
                            format='jpeg', pil_kwargs={{'quality': 85, 'optimize': True}})
                print(f"Saved JPG: {{_jpg_path}}")
            except Exception as _jpg_err:
                print(f"[Auto-save] Warning: could not save JPG: {{_jpg_err}}")
            try:
                fig.savefig(_pdf_path, dpi=300, bbox_inches='tight', format='pdf')
                print(f"Saved PDF: {{_pdf_path}}")
            except Exception as _pdf_err:
                print(f"[Auto-save] Warning: could not save PDF: {{_pdf_err}}")
        except Exception as _save_err:
            print(f"[Auto-save] Warning: could not save figure: {{_save_err}}")
    else:
        print("[Auto-save] No figures found — skipping save")

    try:
        _scai_cleanup_plt.close('all')
    except Exception:
        pass
'''
