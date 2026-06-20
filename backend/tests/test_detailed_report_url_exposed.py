"""to_api_dict 必须按是否存在 detailed_report_html_path 暴露 detailed_report_url。"""
from backend.models.schemas import AnalysisState


def test_detailed_report_url_present_when_path_set():
    s = AnalysisState(id="abc", user_id="test")
    s.detailed_report_html_path = "/data/sessions/abc/reports/detailed_report.html"
    d = s.to_api_dict()
    assert d["detailed_report_url"] == "/api/session/abc/detailed-report/html"


def test_detailed_report_url_none_when_path_unset():
    s = AnalysisState(id="abc", user_id="test")
    d = s.to_api_dict()
    assert d["detailed_report_url"] is None
