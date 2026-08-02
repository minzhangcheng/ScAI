"""Regression: supervisor _collect_figure_paths must resolve figures from
results/<dir>/figures/ subdirectories via rglob (same bug as scai-data-5)."""
import base64
import hashlib
from pathlib import Path

from backend.agents.supervisor_agent import SupervisorAgent
from backend.models.schemas import (
    AnalysisState,
    FigureReviewStatus,
    PublicationFigure,
)
from backend.tests.figure_image_helpers import valid_image_bytes, write_valid_image


def _png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_valid_image(path)


def test_collect_figure_paths_finds_subdir_figure(tmp_path):
    ws = tmp_path / "workspace"
    _png(ws / "results" / "round3" / "figures" / "fig_trajectory.png")
    (ws / "figures").mkdir(parents=True, exist_ok=True)

    agent = SupervisorAgent.__new__(SupervisorAgent)
    records = [{"saved_figure_filenames": ["fig_trajectory.png"]}]

    figure_map = agent._collect_figure_paths(records, ws)

    assert "fig_trajectory.png" in figure_map
    assert figure_map["fig_trajectory.png"].exists()


def test_collect_figure_paths_no_duplicate_with_root_scan(tmp_path):
    ws = tmp_path / "workspace"
    _png(ws / "figures" / "fig_root.png")

    agent = SupervisorAgent.__new__(SupervisorAgent)
    records = [{"saved_figure_filenames": ["fig_root.png"]}]

    figure_map = agent._collect_figure_paths(records, ws)

    assert "fig_root.png" in figure_map
    assert len([k for k in figure_map if Path(k).stem == "fig_root"]) == 1


def test_collect_figure_paths_missing_file_skipped(tmp_path):
    ws = tmp_path / "workspace"
    (ws / "figures").mkdir(parents=True)

    agent = SupervisorAgent.__new__(SupervisorAgent)
    records = [{"saved_figure_filenames": ["nonexistent.png"]}]

    figure_map = agent._collect_figure_paths(records, ws)

    assert "nonexistent.png" not in figure_map


def test_supervision_filename_cannot_bypass_registered_render_hash(tmp_path):
    ws = tmp_path / "workspace"
    figures = ws / "figures"
    figures.mkdir(parents=True)
    path = figures / "reviewed.jpg"
    original = write_valid_image(path, min_bytes=2200)
    preview = valid_image_bytes("JPEG", color=(90, 40, 10))
    figure = PublicationFigure(
        name="reviewed",
        description="Reviewed render",
        figure_type="other",
        png_path=str(path),
        pdf_path="",
        png_base64=base64.b64encode(preview).decode(),
        render_sha256=hashlib.sha256(original).hexdigest(),
        preview_sha256=hashlib.sha256(preview).hexdigest(),
        review_status=FigureReviewStatus.APPROVED,
    )
    state = AnalysisState(user_id="test-user", publication_figures=[figure])

    replacement = write_valid_image(
        path, color=(220, 10, 10), min_bytes=2200
    )
    agent = SupervisorAgent.__new__(SupervisorAgent)
    figure_map = agent._collect_figure_paths(
        [{"saved_figure_filenames": ["reviewed.jpg"]}], ws, state=state
    )

    assert len(figure_map) == 1
    resolved = next(iter(figure_map.values()))
    assert resolved.read_bytes() == preview
    assert resolved.read_bytes() != replacement


def test_shared_helper_finds_deep_figure(tmp_path):
    from backend.utils.files import resolve_figure_in_workspace

    ws = tmp_path / "workspace"
    _png(ws / "results" / "fibrosis" / "figures" / "fig_deep.png")

    result = resolve_figure_in_workspace("fig_deep.png", ws)
    assert result is not None
    assert result.name == "fig_deep.png"


def test_shared_helper_returns_none_for_missing(tmp_path):
    from backend.utils.files import resolve_figure_in_workspace

    ws = tmp_path / "workspace"
    ws.mkdir()

    result = resolve_figure_in_workspace("missing.png", ws)
    assert result is None


def test_shared_helper_respects_seen_stems(tmp_path):
    from backend.utils.files import resolve_figure_in_workspace

    ws = tmp_path / "workspace"
    _png(ws / "figures" / "fig_dup.png")

    result = resolve_figure_in_workspace("fig_dup.png", ws, seen_stems={"fig_dup"})
    assert result is None
