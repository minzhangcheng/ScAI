from backend.models.schemas import AnalysisState


def test_integration_report_html_path_default():
    state = AnalysisState(id="test", name="Test", user_id="test")
    assert state.integration_report_html_path is None


def test_integration_report_url_in_api_dict():
    state = AnalysisState(id="test-session", name="Test", user_id="test")
    state.integration_report_html_path = "/some/path/integration_report.html"
    d = state.to_api_dict()
    assert d.get("integration_report_url") == "/api/session/test-session/integration-report/html"


def test_integration_report_url_none_when_no_path():
    state = AnalysisState(id="test-session", name="Test", user_id="test")
    d = state.to_api_dict()
    assert d.get("integration_report_url") is None
