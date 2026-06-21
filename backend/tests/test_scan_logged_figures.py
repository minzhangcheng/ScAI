"""Regression: report scanner must collect figures recorded in
AnalysisStep.saved_figure_filenames even when they live in results/<dir>/figures/
subdirectories (scai-data-5 bug: 34 figures generated, only 2 reached report)."""
import os
from pathlib import Path
from types import SimpleNamespace

from backend.services.report.figures import FiguresMixin


class _Harness(FiguresMixin):
    def __init__(self, workspace_dir: Path, figures_dir: Path):
        self.workspace_dir = Path(workspace_dir)
        self.figures_dir = Path(figures_dir)


def _png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)


def test_collects_subdir_figure_from_saved_filenames(tmp_path):
    ws = tmp_path / "workspace"
    fig_out = tmp_path / "report" / "figures"
    fig_out.mkdir(parents=True)
    _png(ws / "results" / "round3" / "figures" / "fig7_dpt_trajectory_collapse.png")
    (ws / "figures").mkdir(parents=True, exist_ok=True)

    harness = _Harness(ws, fig_out)
    step = SimpleNamespace(
        saved_figure_filenames=["fig7_dpt_trajectory_collapse.png"]
    )

    figs = harness._scan_existing_figures(analysis_log=[step])

    names = {f["filename"] for f in figs}
    assert "fig7_dpt_trajectory_collapse.png" in names
    assert (fig_out / "fig7_dpt_trajectory_collapse.png").exists()


def test_logged_figure_does_not_duplicate_workspace_root_scan(tmp_path):
    ws = tmp_path / "workspace"
    fig_out = tmp_path / "report" / "figures"
    fig_out.mkdir(parents=True)
    _png(ws / "figures" / "fig_root.png")
    harness = _Harness(ws, fig_out)
    step = SimpleNamespace(saved_figure_filenames=["fig_root.png"])

    figs = harness._scan_existing_figures(analysis_log=[step])

    stems = [f["filename"] for f in figs]
    assert stems.count("fig_root.png") == 1


def test_missing_logged_file_is_skipped(tmp_path):
    ws = tmp_path / "workspace"
    fig_out = tmp_path / "report" / "figures"
    fig_out.mkdir(parents=True)
    (ws / "figures").mkdir(parents=True)
    harness = _Harness(ws, fig_out)
    step = SimpleNamespace(saved_figure_filenames=["nonexistent.png"])

    figs = harness._scan_existing_figures(analysis_log=[step])

    assert figs == []


def test_dict_form_of_analysis_step(tmp_path):
    ws = tmp_path / "workspace"
    fig_out = tmp_path / "report" / "figures"
    fig_out.mkdir(parents=True)
    _png(ws / "results" / "figures" / "fig_dict.png")
    (ws / "figures").mkdir(parents=True, exist_ok=True)

    harness = _Harness(ws, fig_out)
    step_dict = {"saved_figure_filenames": ["fig_dict.png"]}

    figs = harness._scan_existing_figures(analysis_log=[step_dict])

    names = {f["filename"] for f in figs}
    assert "fig_dict.png" in names
