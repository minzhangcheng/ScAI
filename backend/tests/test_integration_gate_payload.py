from backend.models.schemas import AnalysisState
from backend.workflow.helpers import build_integration_gate_payload


def _state(**kw):
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_payload_shape_and_url_when_report_present():
    s = _state(integration_cycle=2,
               integration_reports=["latest analysis"],
               integration_gate_candidates=[{"direction": "x"}],
               detailed_report_html_path="/some/path.html")
    p = build_integration_gate_payload(s, "sess-1")
    assert p == {
        "cycle": 2,
        "integration_analysis": "latest analysis",
        "candidates": [{"direction": "x"}],
        "detailed_report_url": "/api/session/sess-1/detailed-report/html",
        "html_url": None,
    }


def test_url_none_without_report_path_and_empty_analysis():
    s = _state(integration_cycle=1, integration_reports=[],
               integration_gate_candidates=[])
    p = build_integration_gate_payload(s, "sess-1")
    assert p["detailed_report_url"] is None
    assert p["integration_analysis"] == ""
