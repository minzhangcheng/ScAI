from backend.services.report.download_section import render_related_reports_downloads_html


def test_renders_all_items_styled():
    related = {
        "annotation_record": "annotation_record.html",
        "direction_records": [{"filename": "direction_record_00_round1.html",
                               "label": "Round 1 \u00b7 innovation"}],
    }
    html = render_related_reports_downloads_html(
        related, notebook_filename="sess.ipynb", literature_report="lit.html")
    assert "Related Reports &amp; Downloads" in html or "Related Reports & Downloads" in html
    assert "annotation_record.html" in html
    assert "direction_record_00_round1.html" in html
    assert "Round 1 \u00b7 innovation" in html
    assert "sess.ipynb" in html


def test_empty_returns_empty_string():
    assert render_related_reports_downloads_html({}, notebook_filename="", literature_report="") == ""
