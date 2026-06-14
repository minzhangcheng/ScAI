from backend.services.report.regenerator import _build_appendix_from_report


def _appendix(**extra):
    base = {
        "related_reports": {"exploration_report": "exploration_report.html"},
        "notebook_filename": "sess123.ipynb",
    }
    base.update(extra)
    return {"appendix": base}


def test_link_present_when_literature_report_set():
    html = _build_appendix_from_report(_appendix(literature_report="literature_report.html"), "en")
    assert 'href="literature_report.html"' in html
    assert "Literature Research Report" in html


def test_link_absent_when_not_set():
    html = _build_appendix_from_report(_appendix(), "en")
    assert "literature_report.html" not in html
    assert "Literature Research Report" not in html