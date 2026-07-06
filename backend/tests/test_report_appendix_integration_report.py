from backend.models.schemas import AnalysisState
from backend.services.report.download_section import render_related_reports_downloads_html


def test_related_downloads_renders_integration_report_link():
    html = render_related_reports_downloads_html(
        {"integration_report": "integration_report.html"},
        notebook_filename="",
        literature_report="",
    )

    assert "Integration Report" in html
    assert "integration_report.html" in html


def test_analysis_state_with_integration_report_has_expected_related_metadata():
    state = AnalysisState(id="appendix-session", name="Appendix", user_id="u1")
    state.integration_report_html_path = "/data/sessions/appendix-session/reports/integration_report.html"

    related = {
        "integration_report": (
            "integration_report.html"
            if getattr(state, "integration_report_html_path", None) else None
        )
    }

    html = render_related_reports_downloads_html(
        related,
        notebook_filename="",
        literature_report="",
    )
    assert "Integration Report" in html
    assert "integration_report.html" in html


def test_related_report_metadata_copies_integration_report_into_branch_output(tmp_path):
    from backend.services.report.download_section import build_related_reports_metadata

    state = AnalysisState(id="appendix-session", name="Appendix", user_id="u1")
    main_reports = tmp_path / "reports"
    branch_reports = main_reports / "branches" / "branch-a"
    main_reports.mkdir(parents=True)
    branch_reports.mkdir(parents=True)
    source = main_reports / "integration_report.html"
    source.write_text("<html>integration</html>", encoding="utf-8")
    state.integration_report_html_path = str(source)

    related = build_related_reports_metadata(
        state,
        {"annotation_record": None, "direction_records": []},
        reports_dir=branch_reports,
    )

    assert related["integration_report"] == "integration_report.html"
    assert (branch_reports / "integration_report.html").read_text(encoding="utf-8") == "<html>integration</html>"
