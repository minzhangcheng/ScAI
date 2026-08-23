# Builders for the Python source that is executed inside the Jupyter kernel: autosave wrappers and figure wrappers.
# Moved verbatim out of backend/agents/bioinformatics_agent.py.
"""
Bioinformatics Agent - Handles snRNA-seq analysis with iterative code execution
Supports autonomous technical steps and PI consultation for biological decisions
With step-level checkpointing for recovery from interruptions
"""
import logging
import re
import textwrap


from ...utils.sandbox import _TPL_NOOP_INSTALL
from ...services.figure_content import FIGURE_CONTENT_SRC
from ...services.figure_layout import LAYOUT_LINT_SRC

# Configure logger
logger = logging.getLogger(__name__)


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
    _scai_user_source = {user_source_repr}
    # Keep the exact source behind the compiled filename so traceback
    # rendering and later diagnostics retain the user's original line.
    try:
        import linecache as _scai_linecache
        _scai_linecache.cache['<scai-user-code>'] = (
            len(_scai_user_source),
            None,
            _scai_user_source.splitlines(True),
            '<scai-user-code>',
        )
    except Exception:
        pass
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
                    # Scanpy accepts the requested suffix but may write only a
                    # vector file such as ``dotplot__x.jpg.pdf``. Recover one
                    # raster companion for that explicitly tracked save.
                    _has_raster = any(
                        _p.lower().endswith(('.jpg', '.jpeg', '.png'))
                        for _p in _saved_paths
                    )
                    if not _has_raster and _saved_paths:
                        _seen_jpg_targets = set()
                        for _src_str in _saved_paths:
                            _src_path = Path(_src_str)
                            if _src_path.suffix.lower() != '.pdf':
                                continue
                            if Path(_src_path.stem).suffix.lower() in ('.jpg', '.jpeg', '.png'):
                                _jpg_supplement = _src_path.with_name(_src_path.stem)
                            else:
                                _jpg_supplement = _src_path.with_suffix('.jpg')
                            if str(_jpg_supplement) in _seen_jpg_targets:
                                continue
                            _seen_jpg_targets.add(str(_jpg_supplement))
                            try:
                                _fig.savefig(
                                    str(_jpg_supplement),
                                    dpi=150,
                                    bbox_inches='tight',
                                    facecolor='white',
                                    edgecolor='none',
                                    format='jpeg',
                                    pil_kwargs={{'quality': 85, 'optimize': True}},
                                )
                                print(f"[Auto-saved] JPG: {{_jpg_supplement}}")
                            except Exception as _jpg_supp_err:
                                print(
                                    "[Auto-save] Warning: could not supplement JPG "
                                    f"for {{_src_path}}: {{_jpg_supp_err}}"
                                )
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

    The wrapper also runs the renderer-level layout linter (``layout_lint``)
    on the finished figure before saving, and surfaces matplotlib's own layout
    warnings, which the kernel's global ``warnings.filterwarnings('ignore')``
    would otherwise swallow. Neither can prevent the figure from being written.
    """
    figures_dir_repr = repr(str(figures_dir))
    jpg_path_repr = repr(str(jpg_path))
    pdf_path_repr = repr(str(pdf_path))
    user_source_repr = repr(code)
    layout_lint_src_repr = repr(LAYOUT_LINT_SRC)
    return f'''
import matplotlib.pyplot as plt
import matplotlib.figure as _scai_local_figure
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

# Execute plotting and finalization in one function scope. Trusted references
# remain lexical locals until the authoritative lint event is emitted as the
# final line, after warnings, both saves, and cleanup.
def _scai_execute_publication_source():
    import builtins as _scai_local_builtins
    import json as _scai_local_json
    import matplotlib.figure as _scai_local_figure
    import matplotlib.pyplot as _scai_local_plt
    import linecache as _scai_local_linecache
    import sys as _scai_local_sys
    import warnings as _scai_local_warnings

    _scai_persistent_ns = _scai_local_builtins.globals()
    try:
        import scai_kernel_trust as _scai_trust_module
    except _scai_local_builtins.ModuleNotFoundError:
        # Host-side wrapper unit tests do not start through the kernel launcher.
        # Production kernels always have this module and fail closed if its
        # bootstrap capture is absent.
        _scai_runtime_trust = None
    else:
        _scai_runtime_trust = _scai_trust_module.get_runtime_trust()

    if _scai_runtime_trust is None:
        _scai_pristine_builtins = _scai_local_builtins.dict(
            _scai_local_builtins.vars(_scai_local_builtins)
        )
        _scai_trusted_print = _scai_local_builtins.print
        _scai_trusted_dumps = _scai_local_json.dumps
        _scai_trusted_stdout = _scai_local_sys.stdout
        _scai_get_parent_header = lambda: None
        _scai_set_stdout_parent = lambda _parent: None
        _scai_trusted_get_fignums = _scai_local_plt.get_fignums
        _scai_trusted_gcf = _scai_local_plt.gcf
        # _TPL_NOOP_INSTALL runs before this function. Its `_orig_*` values are
        # the pre-stub callables; the live module attributes are the temporary
        # no-ops and must not become the host-test fallback trust baseline.
        _scai_trusted_close = _scai_persistent_ns.get(
            '_orig_close', _scai_local_plt.close
        )
        _scai_trusted_show = _scai_persistent_ns.get(
            '_orig_show', _scai_local_plt.show
        )
        _scai_trusted_savefig = _scai_local_figure.Figure.savefig
        _scai_runtime_savefig = _scai_local_figure.Figure.savefig
        _scai_trusted_plt_savefig = _scai_persistent_ns.get(
            '_original_plt_savefig', _scai_local_plt.savefig
        )
    else:
        _scai_pristine_builtins = _scai_runtime_trust.builtins_map['dict'](
            _scai_runtime_trust.builtins_map
        )
        _scai_trusted_print = _scai_runtime_trust.print_fn
        _scai_trusted_dumps = _scai_runtime_trust.json_dumps
        _scai_trusted_stdout = _scai_runtime_trust.stdout
        _scai_get_parent_header = _scai_runtime_trust.get_parent_header
        _scai_set_stdout_parent = _scai_runtime_trust.set_stdout_parent
        _scai_trusted_get_fignums = _scai_runtime_trust.plt_get_fignums
        _scai_trusted_gcf = _scai_runtime_trust.plt_gcf
        _scai_trusted_close = _scai_runtime_trust.plt_close
        _scai_trusted_show = _scai_runtime_trust.plt_show
        _scai_trusted_savefig = _scai_runtime_trust.figure_savefig
        _scai_runtime_savefig = _scai_runtime_trust.runtime_savefig
        _scai_trusted_plt_savefig = _scai_persistent_ns.get(
            '_original_plt_savefig', _scai_local_plt.savefig
        )

    _scai_trusted_exec = _scai_pristine_builtins['exec']
    _scai_trusted_compile = _scai_pristine_builtins['compile']
    _scai_trusted_str = _scai_pristine_builtins['str']
    _scai_trusted_len = _scai_pristine_builtins['len']
    _scai_trusted_list = _scai_pristine_builtins['list']
    _scai_trusted_set = _scai_pristine_builtins['set']
    _scai_trusted_any = _scai_pristine_builtins['any']
    _scai_trusted_getattr = _scai_pristine_builtins['getattr']
    _scai_trusted_exception = _scai_pristine_builtins['Exception']
    _scai_content_ns = {{'__builtins__': _scai_pristine_builtins}}
    _scai_trusted_exec(
        _scai_trusted_compile(
            {repr(FIGURE_CONTENT_SRC)}, '<scai-figure-content>', 'exec'
        ),
        _scai_content_ns,
        _scai_content_ns,
    )
    _scai_census_figure_content = _scai_content_ns[
        '_scai_census_figure_content'
    ]
    # ipykernel assigns an execute_request parent only to whatever object is
    # currently installed as sys.stdout. A prior cell may have replaced that
    # object, leaving the bootstrap-owned OutStream associated with an older
    # request. Re-parent it before user code and retain that parent in a local
    # so later user mutations cannot redirect the authoritative event.
    _scai_current_parent = _scai_get_parent_header()
    _scai_set_stdout_parent(_scai_current_parent)
    _scai_local_sys.stdout = _scai_trusted_stdout

    def _scai_one_line(_value):
        return _scai_trusted_str(_value).replace('\\r', ' ').replace('\\n', ' ')

    _scai_lint_ns = {{'__builtins__': _scai_pristine_builtins}}
    _scai_trusted_exec(
        _scai_trusted_compile(
            {layout_lint_src_repr}, '<scai-layout-lint>', 'exec'
        ),
        _scai_lint_ns,
        _scai_lint_ns,
    )
    _scai_trusted_lint = _scai_lint_ns['_scai_lint_figure_layout']
    _scai_authoritative_event = None
    _scai_captured_warnings = []

    try:
        with _scai_local_warnings.catch_warnings(record=True) as _scai_captured_warnings:
            _scai_local_warnings.simplefilter('always')
            _scai_user_source = {user_source_repr}
            _scai_local_linecache.cache['<scai-user-code>'] = (
                _scai_trusted_len(_scai_user_source),
                None,
                _scai_user_source.splitlines(True),
                '<scai-user-code>',
            )
            _scai_original_figure_savefig = _scai_local_figure.Figure.savefig
            _scai_original_plt_savefig = _scai_local_plt.savefig
            def _scai_noop_user_figure_savefig(*_args, **_kwargs):
                return None
            def _scai_noop_user_plt_savefig(*_args, **_kwargs):
                return None
            _scai_local_figure.Figure.savefig = _scai_noop_user_figure_savefig
            _scai_local_plt.savefig = _scai_noop_user_plt_savefig
            _scai_trusted_exec(
                _scai_trusted_compile(
                    _scai_user_source, '<scai-user-code>', 'exec'
                ),
                _scai_persistent_ns,
                _scai_persistent_ns,
            )
    finally:
        # Restore process/module state before measuring or saving. User code
        # can mutate these shared objects even when it cannot reach our locals.
        try:
            _scai_local_builtins.print = _scai_trusted_print
            _scai_local_json.dumps = _scai_trusted_dumps
            _scai_set_stdout_parent(_scai_current_parent)
            _scai_local_sys.stdout = _scai_trusted_stdout
            _scai_local_plt.close = _scai_trusted_close
            _scai_local_plt.show = _scai_trusted_show
            _scai_local_plt.savefig = _scai_trusted_plt_savefig
            _scai_local_plt.get_fignums = _scai_trusted_get_fignums
            _scai_local_plt.gcf = _scai_trusted_gcf
            _scai_local_figure.Figure.savefig = _scai_runtime_savefig
        except _scai_trusted_exception:
            pass

        _scai_fignums = []
        _scai_fig = None
        try:
            _scai_fignums = _scai_trusted_get_fignums()
            if _scai_fignums:
                _scai_fig = _scai_trusted_gcf()
                try:
                    _scai_content = _scai_census_figure_content(_scai_fig)
                    if _scai_content is None:
                        _scai_content_event = (
                            '__SCAI_FIGURE_CONTENT_ERROR__ '
                            'figure could not be inspected'
                        )
                    else:
                        _scai_content_event = (
                            '__SCAI_FIGURE_CONTENT__'
                            + _scai_trusted_dumps(_scai_content)
                        )
                except _scai_trusted_exception as _scai_content_err:
                    _scai_content_event = (
                        '__SCAI_FIGURE_CONTENT_ERROR__ '
                        + _scai_one_line(_scai_content_err)
                    )
                _scai_trusted_print(
                    _scai_content_event,
                    file=_scai_trusted_stdout,
                )
                _scai_layout_v = _scai_trusted_lint(_scai_fig)
                if _scai_layout_v is None:
                    _scai_authoritative_event = (
                        '__SCAI_LAYOUT_LINT_ERROR__ figure could not be measured'
                    )
                else:
                    _scai_authoritative_event = (
                        '__SCAI_LAYOUT_LINT__' + _scai_trusted_dumps(_scai_layout_v)
                    )
            else:
                _scai_authoritative_event = (
                    '__SCAI_LAYOUT_LINT_ERROR__ no figure to measure'
                )
        except _scai_trusted_exception as _scai_lint_err:
            _scai_authoritative_event = (
                '__SCAI_LAYOUT_LINT_ERROR__ ' + _scai_one_line(_scai_lint_err)
            )

        try:
            # Report matplotlib warnings before the verdict. A warning may
            # contain newlines or even sentinel text, so flatten it first.
            _scai_layout_kw = (
                'layout', 'constrained', 'not compatible', 'glyph',
                'font family', 'clip', 'overlap',
            )
            _scai_seen_warn = _scai_trusted_set()
            for _scai_w in _scai_trusted_list(_scai_captured_warnings or [])[:200]:
                _scai_wmsg = _scai_one_line(
                    _scai_trusted_getattr(_scai_w, 'message', '') or ''
                )
                if not _scai_trusted_any(
                    _k in _scai_wmsg.lower() for _k in _scai_layout_kw
                ):
                    continue
                if _scai_wmsg in _scai_seen_warn:
                    continue
                _scai_seen_warn.add(_scai_wmsg)
                _scai_trusted_print(
                    '[Layout] %s: %s' % (
                        _scai_trusted_getattr(
                            _scai_trusted_getattr(_scai_w, 'category', None),
                            '__name__',
                            'Warning',
                        ),
                        _scai_wmsg,
                    ),
                    file=_scai_trusted_stdout,
                )
                if _scai_trusted_len(_scai_seen_warn) >= 5:
                    break

            if _scai_trusted_len(_scai_fignums) > 1:
                _scai_trusted_print(
                    '[Layout] Warning: %d figures are open but only the current '
                    'one is saved. Draw every panel into a single figure.'
                    % _scai_trusted_len(_scai_fignums),
                    file=_scai_trusted_stdout,
                )

            if _scai_fig is not None:
                _jpg_path = {jpg_path_repr}
                _pdf_path = {pdf_path_repr}
                try:
                    _scai_trusted_savefig(
                        _scai_fig,
                        _jpg_path,
                        dpi=150,
                        bbox_inches='tight',
                        facecolor='white',
                        edgecolor='none',
                        format='jpeg',
                        pil_kwargs={{'quality': 85, 'optimize': True}},
                    )
                    _scai_trusted_print(
                        f'Saved JPG: {{_jpg_path}}', file=_scai_trusted_stdout
                    )
                except _scai_trusted_exception as _jpg_err:
                    _scai_trusted_print(
                        '[Auto-save] Warning: could not save JPG: '
                        + _scai_one_line(_jpg_err),
                        file=_scai_trusted_stdout,
                    )
                try:
                    _scai_trusted_savefig(
                        _scai_fig,
                        _pdf_path,
                        dpi=300,
                        bbox_inches='tight',
                        format='pdf',
                    )
                    _scai_trusted_print(
                        f'Saved PDF: {{_pdf_path}}', file=_scai_trusted_stdout
                    )
                except _scai_trusted_exception as _pdf_err:
                    _scai_trusted_print(
                        '[Auto-save] Warning: could not save PDF: '
                        + _scai_one_line(_pdf_err),
                        file=_scai_trusted_stdout,
                    )
            else:
                _scai_trusted_print(
                    '[Auto-save] No figures found — skipping save',
                    file=_scai_trusted_stdout,
                )
        except _scai_trusted_exception as _save_err:
            _scai_trusted_print(
                '[Auto-save] Warning: could not save figure: '
                + _scai_one_line(_save_err),
                file=_scai_trusted_stdout,
            )
        finally:
            try:
                _scai_trusted_close('all')
            except _scai_trusted_exception:
                pass
            # This is deliberately the final output event from the wrapper.
            # Any user/warning/savefig sentinel necessarily precedes it.
            if _scai_authoritative_event is not None:
                _scai_trusted_print(
                    _scai_authoritative_event,
                    file=_scai_trusted_stdout,
                )

_scai_execute_publication_source()
'''
