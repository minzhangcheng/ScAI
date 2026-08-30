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
from backend.models.schema_parts.session import DetailedReportSegment
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


def test_detailed_report_discloses_unverified_technical_evidence(monkeypatch, tmp_path):
    state = AnalysisState(
        id="evidence-disclosure",
        name="Evidence disclosure",
        user_id="u",
        current_round=2,
        round_results=[RoundResult(
            round_number=2,
            bioinformatics_evidence_verified=False,
            evidence_execution_ids=[],
            evidence_notes=["Bioinformatics evidence reconciliation was unavailable."],
        )],
        plan_evidence_reconciliations=[{
            "round_number": 2,
            "status": "unverified",
            "overall_reason": "Supervisor reconciliation timed out.",
            "links": [
                {"plan_step_id": 1, "status": "unverified"},
                {"plan_step_id": 2, "status": "unverified"},
            ],
        }],
    )
    reports = tmp_path / "reports"
    monkeypatch.setattr(
        "backend.config.get_session_workspace", lambda _session_id: tmp_path / "missing"
    )
    monkeypatch.setattr(
        "backend.config.get_session_reports", lambda _session_id: reports
    )

    segment_html = detailed.generate_detailed_report_segment_html(
        state,
        "innovation_round_2",
        "Innovation round 2",
        round_result=state.round_results[0],
    )

    assert "Technical Evidence Observations" in segment_html
    assert "not verified" in segment_html
    assert "2 unverified plan steps" in segment_html
    assert "do not block this report" in segment_html

    state.detailed_report_segments = [DetailedReportSegment(
        phase="innovation_round_2",
        title="Stored innovation round 2",
        round_number=2,
        content_html=(
            '<div class="report-segment"><p class="segment-timestamp">old</p>'
            '<div class="segment-section"><h3>PI Synthesis</h3>'
            '<p>Strong conclusion.</p></div></div>'
        ),
    )]
    report_path = detailed.generate_combined_detailed_report_html(state)
    combined = Path(report_path).read_text(encoding="utf-8")

    assert "Strong conclusion." in combined
    assert "Technical Evidence Observations" in combined
    assert combined.index("Technical Evidence Observations") < combined.index(
        "Strong conclusion."
    )
