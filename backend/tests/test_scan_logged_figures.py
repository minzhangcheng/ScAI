"""Regression: report scanner must collect figures recorded in
AnalysisStep.saved_figure_filenames even when they live in results/<dir>/figures/
subdirectories (scai-data-5 bug: 34 figures generated, only 2 reached report)."""
import os
import hashlib
from pathlib import Path
from types import SimpleNamespace

from backend.models.schemas import AnalysisState
from backend.services.report import artifacts
from backend.services.report.figure_discovery import ensure_legacy_artifact_inventory
from backend.services.report.figures import FiguresMixin
from backend.tests.figure_image_helpers import valid_pdf_bytes, write_valid_image


class _Harness(FiguresMixin):
    def __init__(self, workspace_dir: Path, figures_dir: Path):
        self.workspace_dir = Path(workspace_dir)
        self.figures_dir = Path(figures_dir)


def _png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_valid_image(path, min_bytes=2200)


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


def test_registry_v1_analysis_log_cannot_admit_unregistered_disk_file(tmp_path):
    ws = tmp_path / "workspace"
    fig_out = tmp_path / "report" / "figures"
    fig_out.mkdir(parents=True)
    _png(ws / "results" / "figures" / "logged_only.png")
    harness = _Harness(ws, fig_out)

    figures = harness._scan_existing_figures(
        analysis_log=[{"saved_figure_filenames": ["logged_only.png"]}],
        publication_figures=[],
        artifact_registry_version=1,
        legacy_artifacts=[],
    )

    assert figures == []


def test_frozen_legacy_pdf_report_copy_uses_verified_bytes_after_source_swap(
    tmp_path, monkeypatch,
):
    ws = tmp_path / "workspace"
    fig_out = tmp_path / "report" / "figures"
    fig_out.mkdir(parents=True)
    image = ws / "figures" / "legacy.png"
    _png(image)
    pdf = image.with_suffix(".pdf")
    frozen_pdf = valid_pdf_bytes(color=(10, 20, 30))
    replacement_pdf = valid_pdf_bytes(color=(220, 20, 20))
    pdf.write_bytes(frozen_pdf)
    state = AnalysisState(id="legacy-session", user_id="u")
    inventory = ensure_legacy_artifact_inventory(state, ws)
    assert inventory[0].pdf_sha256 == hashlib.sha256(frozen_pdf).hexdigest()

    real_read = artifacts._read_regular_file_nofollow
    swapped = False

    def swap_pdf_after_verified_read(path):
        nonlocal swapped
        payload = real_read(path)
        if not swapped and Path(str(path)) == pdf:
            swapped = True
            pdf.write_bytes(replacement_pdf)
        return payload

    monkeypatch.setattr(
        artifacts, "_read_regular_file_nofollow", swap_pdf_after_verified_read
    )
    harness = _Harness(ws, fig_out)
    figures = harness._scan_existing_figures(
        publication_figures=[],
        artifact_registry_version=state.artifact_registry_version,
        legacy_artifacts=state.legacy_artifacts,
    )

    assert swapped
    assert len(figures) == 1
    report_pdf = Path(figures[0]["pdf_path"])
    assert report_pdf.read_bytes() == frozen_pdf
    assert pdf.read_bytes() == replacement_pdf
