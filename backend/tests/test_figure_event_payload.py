"""Live figure-review events carry compact gallery metadata and provenance."""

from datetime import datetime
from types import SimpleNamespace

from backend.workflow.callbacks import compact_figure_event_payload


def test_compact_figure_event_payload_keeps_provenance_without_preview():
    figure = SimpleNamespace(
        id="fig-1",
        name="planned_panel",
        description="Result panel requested by the approved plan",
        figure_type="heatmap",
        png_path="/workspace/figures/planned_panel.jpg",
        pdf_path="/workspace/figures/planned_panel.pdf",
        png_base64="do-not-send-this-large-preview",
        review_status="pending",
        pi_feedback=None,
        revision_count=2,
        plan_step_id=7,
        plan_id="plan-v3",
        purpose="result",
        round_number=4,
        diagnostic=False,
        layout_violations=[{"severity": "block", "kind": "text_over_text"}],
        created_at=datetime(2026, 8, 2, 10, 0, 0),
    )

    payload = compact_figure_event_payload(
        figure,
        review_status="revision_requested",
        pi_feedback="Increase label size.",
    )

    assert payload["id"] == "fig-1"
    assert payload["review_status"] == "revision_requested"
    assert payload["plan_step_id"] == 7
    assert payload["plan_id"] == "plan-v3"
    assert payload["purpose"] == "result"
    assert payload["round"] == 4
    assert payload["pi_feedback"] == "Increase label size."
    assert payload["layout_blocking"] is True
    assert "png_base64" not in payload
