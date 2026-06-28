"""Figure deduplication / figure cleanup regression tests."""
import glob
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_safe_close_all_closes_through_noop_stub(tmp_path):
    """Even when plt.close is overridden by a SCAI no-op stub,
    _TPL_SAFE_CLOSE_ALL must close all figures."""
    from backend.utils.sandbox import LocalJupyterSandbox
    from backend.agents.bioinformatics_agent import _TPL_SAFE_CLOSE_ALL

    sb = LocalJupyterSandbox(
        session_id="test-safe-close",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
        timeout=120,
    )
    await sb.start()
    try:
        # Simulate a no-op close stub left by an interrupted cell, and open a figure
        setup = await sb.execute(
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "def _noop(*a, **k):\n"
            "    return None\n"
            "_noop._scai_noop = True\n"
            "plt.close = _noop\n"
            "plt.figure(); plt.plot([1, 2, 3])\n"
            "print('OPEN_BEFORE', plt.get_fignums())"
        )
        assert setup.success, setup.error
        assert "OPEN_BEFORE [1]" in setup.output

        r = await sb.execute(
            _TPL_SAFE_CLOSE_ALL + "\nprint('OPEN_AFTER', plt.get_fignums())"
        )
        assert r.success, r.error
        assert "OPEN_AFTER []" in r.output
    finally:
        await sb.stop()


@pytest.mark.asyncio
async def test_exec_wrapper_saves_only_new_figures(tmp_path):
    """3 consecutive executions each drawing 1 figure should produce exactly 3
    fig_*.jpg files; never re-save old figures (1+2+3=6)."""
    from backend.utils.sandbox import LocalJupyterSandbox
    from backend.agents.bioinformatics_agent import _build_exec_autosave_wrapper

    workspace = tmp_path / "workspace"
    figures_dir = workspace / "figures"
    sb = LocalJupyterSandbox(
        session_id="test-exec-wrapper",
        state_dir=str(tmp_path / "states"),
        working_dir=str(workspace),
        timeout=120,
    )
    await sb.start()
    try:
        plot_code = (
            "import matplotlib.pyplot as plt\n"
            "plt.figure()\n"
            "plt.plot([1, 2, 3])\n"
        )
        for i in range(3):
            wrapped = _build_exec_autosave_wrapper(
                plot_code, str(figures_dir), f"ts{i}"
            )
            r = await sb.execute(wrapped)
            assert r.success, r.error

        jpgs = sorted(Path(p).name for p in glob.glob(str(figures_dir / "fig_*.jpg")))
        assert len(jpgs) == 3, f"expected 3 unique new figures, got {len(jpgs)}: {jpgs}"
    finally:
        await sb.stop()


def test_exec_wrapper_filters_by_pre_fignums():
    """The wrapper template must snapshot fignums before user code and only
    process the diff (preventing re-saving stale figures)."""
    from backend.agents.bioinformatics_agent import _build_exec_autosave_wrapper

    code = _build_exec_autosave_wrapper("pass", "/tmp/figs", "ts0")
    assert "_pre_fignums = set(plt.get_fignums())" in code
    assert "if _fig_num not in _pre_fignums" in code or "not in _pre_fignums" in code
