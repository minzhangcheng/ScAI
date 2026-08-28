"""Pure sections for the trusted publication-figure kernel wrapper."""
from __future__ import annotations


def _publication_wrapper_setup(
    figures_dir_repr: str,
    safe_close_all: str,
    noop_install: str,
    figure_content_src: str,
) -> str:
    _TPL_SAFE_CLOSE_ALL = safe_close_all
    _TPL_NOOP_INSTALL = noop_install
    FIGURE_CONTENT_SRC = figure_content_src
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

'''


def _publication_wrapper_execution(
    user_source_repr: str,
    layout_lint_src_repr: str,
    jpg_path_repr: str,
    pdf_path_repr: str,
) -> str:
    return f'''    _scai_lint_ns = {{'__builtins__': _scai_pristine_builtins}}
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


def build_publication_figure_wrapper(
    code: str,
    figures_dir: str,
    jpg_path: str,
    pdf_path: str,
    *,
    safe_close_all: str,
    noop_install: str,
    layout_lint_src: str,
    figure_content_src: str,
) -> str:
    return _publication_wrapper_setup(
        repr(str(figures_dir)),
        safe_close_all,
        noop_install,
        figure_content_src,
    ) + _publication_wrapper_execution(
        repr(code),
        repr(layout_lint_src),
        repr(str(jpg_path)),
        repr(str(pdf_path)),
    )
