"""Figure deduplication / figure cleanup regression tests."""
import glob
from pathlib import Path

import pytest


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
            r = await sb.execute(wrapped, skip_safety_check=True)
            assert r.success, r.error

        jpgs = sorted(Path(p).name for p in glob.glob(str(figures_dir / "fig_*.jpg")))
        assert len(jpgs) == 3, f"expected 3 unique new figures, got {len(jpgs)}: {jpgs}"
    finally:
        await sb.stop()
