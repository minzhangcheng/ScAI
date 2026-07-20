"""All images fail -> status='error', not hardcoded 'completed'."""
from backend.agents.pathology_agent import _summarize_status


def test_all_images_failed_is_error():
    detailed = [{"image_id": "1", "error": "vision timeout", "findings": []}]
    assert _summarize_status(detailed, all_findings=[]) == "error"


def test_partial_success_is_partial():
    detailed = [{"image_id": "1", "error": "x", "findings": []},
                {"image_id": "2", "findings": ["fibrosis grade 2"]}]
    assert _summarize_status(detailed, all_findings=["fibrosis grade 2"]) == "partial"


def test_degraded_tile_coverage_is_partial():
    detailed = [{
        "image_id": "1",
        "status": "partial",
        "findings": ["finding from validated subset"],
    }]
    assert _summarize_status(detailed, all_findings=detailed[0]["findings"]) == "partial"


def test_all_success_is_completed():
    detailed = [{"image_id": "1", "findings": ["normal"]}]
    assert _summarize_status(detailed, all_findings=["normal"]) == "completed"
