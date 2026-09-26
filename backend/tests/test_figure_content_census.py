"""Tests for renderer-owned figure content and publication side effects."""

from __future__ import annotations

import os

import pytest

from backend.utils.error_parsing import strip_ansi


@pytest.mark.skipif(
    not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
    reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
)
@pytest.mark.asyncio
async def test_live_s3_renderer_contracts(tmp_path):
    from backend.agents.bioinformatics.kernel_templates import (
        _build_exec_autosave_wrapper,
        _build_publication_figure_wrapper,
    )
    from backend.utils.sandbox import LocalJupyterSandbox

    workspace = tmp_path / "workspace"
    figures = workspace / "figures"
    sandbox = LocalJupyterSandbox(
        session_id="s3-live-renderer",
        state_dir=str(tmp_path / "states"),
        working_dir=str(workspace),
        timeout=120,
    )
    await sandbox.start()
    try:
        forced_png = figures / "forced.png"
        forced_jpg = figures / "forced.jpg"
        forced_conversion = await sandbox.execute(
            "import matplotlib.pyplot as plt\n"
            "from pathlib import Path\n"
            f"_forced_png = Path({str(forced_png)!r})\n"
            f"_forced_jpg = Path({str(forced_jpg)!r})\n"
            "_forced_seen = {}\n"
            "_forced_original = _original_savefig\n"
            "def _forced_probe(fig, filename, *args, **kwargs):\n"
            "    _forced_seen.update(kwargs)\n"
            "    return _forced_original(fig, filename, *args, **kwargs)\n"
            "_original_savefig = _forced_probe\n"
            "try:\n"
            "    _forced_fig, _forced_ax = plt.subplots()\n"
            "    _forced_ax.plot([1, 2], [2, 1])\n"
            "    _forced_fig.savefig(_forced_png, metadata={'drop': 'me'}, dpi=77, facecolor='yellow')\n"
            "finally:\n"
            "    _original_savefig = _forced_original\n"
            "assert _forced_jpg.is_file()\n"
            "assert not _forced_png.is_file()\n"
            "assert 'metadata' not in _forced_seen\n"
            "assert _forced_seen['dpi'] == 77\n"
            "assert _forced_seen['facecolor'] == 'yellow'\n",
            records_history=False,
            skip_safety_check=True,
        )
        assert forced_conversion.success, forced_conversion.error

        scanpy_code = (
            "import anndata as ad\n"
            "import numpy as np\n"
            "import scanpy as sc\n"
            "adata = ad.AnnData(np.array([[1.0, 2.0], [2.0, 1.0]]))\n"
            "adata.var_names = ['x', 'y']\n"
            "sc.settings.figdir = r'" + str(figures) + "'\n"
            "sc.pl.scatter(adata, x='x', y='y', show=False, save='_markers.jpg')\n"
        )
        scanpy_wrapper = _build_exec_autosave_wrapper(
            scanpy_code, str(figures), "live"
        )
        scanpy_result = await sandbox.execute(
            scanpy_wrapper, records_history=False, skip_safety_check=True
        )
        assert scanpy_result.success, scanpy_result.error
        recovered = figures / "scatter_markers.jpg"
        assert recovered.is_file(), sorted(p.name for p in figures.iterdir())
        assert (figures / "scatter_markers.jpg.pdf").is_file()

        side = workspace / "user-side.jpg"
        publication_code = (
            "import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot([1, 2, 3], [3, 2, 1])\n"
            f"fig.savefig({str(side)!r})\n"
            f"plt.savefig({str(workspace / 'user-side-2.jpg')!r})\n"
            "plt.show()\n"
            "plt.close('all')\n"
        )
        publication_wrapper = _build_publication_figure_wrapper(
            publication_code,
            str(figures),
            str(figures / "canonical.jpg"),
            str(figures / "canonical.pdf"),
        )
        publication_result = await sandbox.execute(
            publication_wrapper, records_history=False, skip_safety_check=True
        )
        assert publication_result.success, publication_result.error
        assert (figures / "canonical.jpg").is_file()
        assert (figures / "canonical.pdf").is_file()
        assert not side.exists()
        assert not (workspace / "user-side-2.jpg").exists()
        identity = await sandbox.execute(
            "import matplotlib.pyplot as plt\n"
            "import matplotlib.figure as _figure\n"
            "assert _figure.Figure.savefig is _combined_savefig\n"
            "assert plt.close is _original_plt_close\n"
            "assert plt.show is _capturing_show\n"
            "assert plt.savefig is _original_plt_savefig\n",
            records_history=False,
            skip_safety_check=True,
        )
        assert identity.success, identity.error

        exception_side_a = workspace / "exception-side-a.jpg"
        exception_side_b = workspace / "exception-side-b.jpg"
        exception_code = (
            "import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot([1, 2], [2, 1])\n"
            f"fig.savefig({str(exception_side_a)!r})\n"
            f"plt.savefig({str(exception_side_b)!r})\n"
            "plt.show()\n"
            "plt.close('all')\n"
            "raise RuntimeError('live known render failure')\n"
        )
        exception_wrapper = _build_publication_figure_wrapper(
            exception_code,
            str(figures),
            str(figures / "exception-canonical.jpg"),
            str(figures / "exception-canonical.pdf"),
        )
        exception_result = await sandbox.execute(
            exception_wrapper, records_history=False, skip_safety_check=True
        )
        assert not exception_result.success
        assert "RuntimeError" in exception_result.error
        assert "<scai-user-code>" in exception_result.error
        assert "raise RuntimeError('live known render failure')" in strip_ansi(
            exception_result.error
        )
        assert not exception_side_a.exists()
        assert not exception_side_b.exists()
        assert (figures / "exception-canonical.jpg").is_file()
        assert (figures / "exception-canonical.pdf").is_file()
        exception_identity = await sandbox.execute(
            "import matplotlib.pyplot as plt\n"
            "import matplotlib.figure as _figure\n"
            "assert _figure.Figure.savefig is _combined_savefig\n"
            "assert plt.close is _original_plt_close\n"
            "assert plt.show is _capturing_show\n"
            "assert plt.savefig is _original_plt_savefig\n",
            records_history=False,
            skip_safety_check=True,
        )
        assert exception_identity.success, exception_identity.error
    finally:
        await sandbox.stop(save=False)
