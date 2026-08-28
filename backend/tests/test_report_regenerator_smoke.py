"""
Smoke tests for report HTML regenerator.
"""
import pytest
import json
import sys
from pathlib import Path

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_report_regenerator_smoke(tmp_path):
    """
    Construct minimal report.json (single section, no figures) in tmp_path;
    call generate_html_report; assert report.html exists, contains title,
    placeholder replacements, and HTML length > 1000.
    """
    # Import here to avoid side-effects at module load
    try:
        from backend.services.report.regenerator import generate_html_report
    except ImportError as e:
        pytest.skip(f"Cannot import generate_html_report: {e}")

    # Create minimal report.json
    report_data = {
        "title": "Test Analysis Report",
        "abstract": "This is a test abstract with **bold** content.",
        "conclusions": "Test conclusions here.",
        "generated_at": "2024-01-01T00:00:00",
        "status": "partial",
        "partial_reason": "figure_coverage",
        "partial_reasons": ["figure_coverage"],
        "figureless_sections": ["overview"],
        "quality_disclosures": {
            "schema_version": 1,
            "status": "qualified",
            "rounds": [],
            "post_execution_reviews": [],
            "annotation": {
                "confirmation": {"status": "not_applicable"},
                "validation": {"status": "not_run"},
                "cell_counts": {"known_clusters": 0, "unknown_clusters": 0},
            },
            "blocking_issues": [],
            "cautions": ["Evidence is qualified."],
        },
        "sections": [
            {
                "id": "overview",
                "title": "Overview",
                "analysis_type": "general",
                "content": "This is the overview section with some content.",
                "figures": [],
                "tables": []
            }
        ]
    }

    report_json_path = tmp_path / "report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f)

    # Create empty figures directory (regenerator expects it)
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()

    # Call generate_html_report
    html_path = generate_html_report(tmp_path)

    # Assert report.html exists
    assert html_path.exists(), f"Expected {html_path} to exist"

    # Read HTML content
    html_content = html_path.read_text(encoding="utf-8")

    # Assert HTML length > 1000
    assert len(html_content) > 1000, f"HTML content too short: {len(html_content)} bytes"

    # Assert title appears in HTML
    assert "Test Analysis Report" in html_content, "Title not found in HTML"

    # Assert some placeholder replacements occurred (structural HTML present)
    assert "<html" in html_content.lower(), "HTML structure missing"
    assert "<div" in html_content, "HTML div elements missing"
    assert "<h1" in html_content, "HTML h1 element missing"
    assert "Detailed evidence qualifications" in html_content
    assert "overview" in html_content
    assert "Evidence is qualified" in html_content
    assert html_content.index("Conclusions") < html_content.index(
        "Detailed evidence qualifications"
    )
