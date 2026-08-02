"""Annotation-context figures remain bound to registered render bytes."""

import base64
import hashlib
from pathlib import Path

from backend.models.schemas import (
    AnalysisState,
    AnalysisStep,
    FigureReviewStatus,
    PublicationFigure,
)
from backend.services.report.variants_parts import annotation as annotation_report
from backend.tests.figure_image_helpers import valid_image_bytes, write_valid_image


def test_annotation_step_filename_cannot_bypass_registered_hash(
    monkeypatch, tmp_path
):
    workspace = tmp_path / "workspace"
    reports = tmp_path / "reports"
    figures = workspace / "figures"
    figures.mkdir(parents=True)
    path = figures / "annotation_panel.jpg"
    original = write_valid_image(path, min_bytes=2200)
    preview = valid_image_bytes("JPEG", color=(80, 40, 20))
    figure = PublicationFigure(
        name="annotation_panel",
        description="Reviewed annotation panel",
        figure_type="barplot",
        png_path=str(path),
        pdf_path="",
        png_base64=base64.b64encode(preview).decode(),
        render_sha256=hashlib.sha256(original).hexdigest(),
        preview_sha256=hashlib.sha256(preview).hexdigest(),
        review_status=FigureReviewStatus.APPROVED,
    )
    state = AnalysisState(
        id="annotation-integrity",
        user_id="test-user",
        publication_figures=[figure],
        analysis_log=[
            AnalysisStep(
                step_number=1,
                step_type="execute_code",
                code_description="Annotation panel",
                saved_figure_filenames=["annotation_panel.jpg"],
            )
        ],
    )
    replacement = write_valid_image(
        path, color=(220, 10, 10), min_bytes=2200
    )
    embedded_payloads = []

    monkeypatch.setattr(
        "backend.config.get_session_workspace", lambda session_id: workspace
    )
    monkeypatch.setattr(
        "backend.config.get_session_reports", lambda session_id: reports
    )

    def _capture(path_value, **kwargs):
        embedded_payloads.append(Path(path_value).read_bytes())
        return "data:image/jpeg;base64,AA=="

    monkeypatch.setattr(annotation_report, "compress_image_to_base64", _capture)

    annotation_report.generate_annotation_context_report_html(state, None, [])

    assert embedded_payloads == [preview]
    assert replacement not in embedded_payloads
