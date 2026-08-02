"""Partial report status survives generation and the terminal WS event."""

from types import SimpleNamespace

from backend.services.report.generator import (
    report_completion_fields,
    report_partial_reason,
    unreported_figure_names,
)
from backend.workflow.reporting_parts.final_report import detailed_report_event_payload


def test_fallback_generator_marks_approved_partial_report_partial():
    assert report_completion_fields(
        SimpleNamespace(allow_partial_report=True)
    ) == ("partial", "Partial report approved")
    assert report_completion_fields(
        SimpleNamespace(allow_partial_report=False)
    ) == ("completed", "Complete")
    assert report_partial_reason(
        SimpleNamespace(allow_partial_report=True)
    ) == "approved_partial"


def test_fallback_generator_reports_artifact_coverage_gap():
    state = SimpleNamespace(allow_partial_report=False)
    missing = unreported_figure_names(
        [
            {"filename": "included.png"},
            {"filename": "missing.png"},
        ],
        [{"png_path": "/report/figures/included.png"}],
    )

    assert missing == ["missing.png"]
    assert report_completion_fields(state, missing) == (
        "partial", "Complete with unreported figures: missing.png",
    )
    assert report_partial_reason(state, missing) == "figure_coverage"


def test_detailed_report_event_preserves_report_status():
    payload = detailed_report_event_payload({
        "id": "report-1",
        "title": "Partial report",
        "status": "partial",
        "current_step": "Complete with figure coverage gaps: direction_1",
        "sections": [{"id": "methods"}],
        "figures": [],
        "tables": [],
    })

    assert payload["status"] == "partial"
    assert payload["current_step"].startswith("Complete with figure coverage gaps")
