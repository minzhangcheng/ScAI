
import pytest

from backend.agents.bioinformatics_agent import _build_exec_autosave_wrapper


@pytest.fixture(autouse=True)
def _restore_host_pyplot_functions():
    """Kernel wrapper tests must not leave their no-op stubs in pytest."""
    import matplotlib.pyplot as pyplot

    real_close = pyplot.close
    while getattr(real_close, "_scai_noop", False):
        real_close = getattr(real_close, "_scai_prev", real_close)
    real_show = pyplot.show
    while getattr(real_show, "_scai_noop", False):
        real_show = getattr(real_show, "_scai_prev", real_show)
    pyplot.close = real_close
    pyplot.show = real_show
    try:
        yield
    finally:
        pyplot.close = real_close
        pyplot.show = real_show
        real_close("all")


def test_autosave_cleanup_reimports_matplotlib_after_user_namespace_mutation(tmp_path):
    wrapper = _build_exec_autosave_wrapper(
        "plt = 'shadowed by generated code'",
        str(tmp_path),
        "test_namespace_cleanup",
    )
    namespace = {}
    exec(wrapper, namespace, namespace)

    import matplotlib.pyplot as pyplot

    assert callable(pyplot.close)
    assert "import matplotlib.pyplot as _scai_cleanup_plt" in wrapper


def test_autosave_cleanup_restores_capturing_show(tmp_path):
    import matplotlib.pyplot as pyplot

    previous_show = pyplot.show

    def capturing_show(*args, **kwargs):
        return None

    namespace = {
        "_capturing_show": capturing_show,
        "_original_plt_show": previous_show,
    }
    try:
        exec(_build_exec_autosave_wrapper(
            "print('done')", str(tmp_path), "test_show_cleanup"
        ), namespace, namespace)
        assert pyplot.show is capturing_show
    finally:
        pyplot.show = previous_show
