"""Static report variants recover image bytes owned by artifact records."""

import base64
import hashlib
import os
from pathlib import Path

from backend.models.schemas import (
    AnalysisState,
    AnalysisStep,
    ExplorationReport,
    FigureReviewStatus,
    PublicationFigure,
    RoundResult,
)
from backend.services.report.variants_parts import detailed, exploration
from backend.tests.figure_image_helpers import valid_image_bytes, write_valid_image


def _state_with_overwritten_record(tmp_path: Path, session_id: str):
    workspace = tmp_path / session_id / "workspace"
    figures_dir = workspace / "figures"
    figures_dir.mkdir(parents=True)
    original_path = figures_dir / "reviewed_result.jpg"
    original = write_valid_image(original_path, min_bytes=2200)
    preview = valid_image_bytes("JPEG", color=(30, 90, 140))
    figure = PublicationFigure(
        name="Reviewed result",
        description="Record-owned result figure",
        figure_type="heatmap",
        png_path=str(original_path),
        pdf_path="",
        png_base64=base64.b64encode(preview).decode(),
        render_sha256=hashlib.sha256(original).hexdigest(),
        preview_sha256=hashlib.sha256(preview).hexdigest(),
        pdf_sha256="2" * 64,
        review_status=FigureReviewStatus.APPROVED,
        round_number=1,
    )
    state = AnalysisState(
        id=session_id,
        name="Variant recovery",
        user_id="test-user",
        current_round=1,
        publication_figures=[figure],
        analysis_log=[AnalysisStep(
            step_number=1,
            step_type="execute_code",
            round_number=1,
            saved_figure_filenames=[original_path.name],
        )],
    )
    replacement = write_valid_image(
        original_path, color=(220, 20, 20), min_bytes=2200
    )
    return state, workspace, preview, replacement


def test_exploration_variant_recovers_record_preview(monkeypatch, tmp_path):
    state, workspace, preview, replacement = _state_with_overwritten_record(
        tmp_path, "exploration-recovery"
    )
    reports = tmp_path / "exploration-reports"
    embedded_payloads = []

    monkeypatch.setattr(
        "backend.config.get_session_workspace", lambda _session_id: workspace
    )
    monkeypatch.setattr(
        "backend.config.get_session_reports", lambda _session_id: reports
    )

    def _capture(path_value, **_kwargs):
        embedded_payloads.append(Path(path_value).read_bytes())
        return "data:image/jpeg;base64,AA=="

    monkeypatch.setattr(exploration, "compress_image_to_base64", _capture)

    exploration.generate_exploration_report_html(
        state,
        ExplorationReport(),
        RoundResult(round_number=1, phase="exploration"),
    )

    assert embedded_payloads == [preview]
    assert replacement not in embedded_payloads


def test_detailed_variant_recovers_record_preview(monkeypatch, tmp_path):
    state, workspace, preview, replacement = _state_with_overwritten_record(
        tmp_path, "detailed-recovery"
    )
    embedded_payloads = []

    monkeypatch.setattr(
        "backend.config.get_session_workspace", lambda _session_id: workspace
    )

    def _capture(path_value, **_kwargs):
        embedded_payloads.append(Path(path_value).read_bytes())
        return "data:image/jpeg;base64,AA=="

    monkeypatch.setattr(detailed, "compress_image_to_base64", _capture)

    html = detailed.generate_detailed_report_segment_html(
        state,
        "exploration_complete",
        "Exploration complete",
        round_result=RoundResult(round_number=1, phase="exploration"),
    )

    assert 'data-figure-key="reviewed_result"' in html
    assert embedded_payloads == [preview]
    assert replacement not in embedded_payloads


def test_detailed_variant_checks_candidates_beyond_the_first_five(monkeypatch, tmp_path):
    workspace = tmp_path / "beyond-five" / "workspace"
    figures = workspace / "figures"
    figures.mkdir(parents=True)
    for index in range(5):
        path = figures / f"unrelated_{index}.jpg"
        write_valid_image(path, min_bytes=2200)
        os.utime(path, (100 - index, 100 - index))
    target = figures / "round_result.jpg"
    write_valid_image(target, min_bytes=2200)
    os.utime(target, (1, 1))
    state = AnalysisState(
        id="beyond-five",
        name="Beyond five",
        user_id="u",
        current_round=1,
        analysis_log=[AnalysisStep(
            step_number=1,
            step_type="execute_code",
            round_number=1,
            saved_figure_filenames=["round_result.jpg"],
        )],
    )
    monkeypatch.setattr(
        "backend.config.get_session_workspace", lambda _session_id: workspace
    )
    monkeypatch.setattr(
        detailed,
        "compress_image_to_base64",
        lambda *_args, **_kwargs: "data:image/jpeg;base64,AA==",
    )

    html = detailed.generate_detailed_report_segment_html(
        state,
        "exploration_complete",
        "Exploration complete",
        round_result=RoundResult(round_number=1, phase="exploration"),
    )

    assert 'data-figure-key="round_result"' in html
    assert html.count('class="figure-item"') == 1


def test_detailed_variant_omits_empty_gallery_shell(monkeypatch, tmp_path):
    workspace = tmp_path / "empty-gallery" / "workspace"
    (workspace / "figures").mkdir(parents=True)
    write_valid_image(workspace / "figures" / "unrelated.jpg", min_bytes=2200)
    state = AnalysisState(
        id="empty-gallery",
        name="Empty gallery",
        user_id="u",
        current_round=1,
    )
    monkeypatch.setattr(
        "backend.config.get_session_workspace", lambda _session_id: workspace
    )

    html = detailed.generate_detailed_report_segment_html(
        state,
        "exploration_complete",
        "Exploration complete",
        round_result=RoundResult(round_number=1, phase="exploration"),
    )

    assert "Generated Figures" not in html
    assert 'class="figure-gallery"' not in html
