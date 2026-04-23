"""Verify DetailedReportGenerator has all expected methods after mixin composition."""
from backend.services.report import DetailedReportGenerator


def test_empty_analysis_summary_exists():
    assert hasattr(DetailedReportGenerator, "_empty_analysis_summary")
    result = DetailedReportGenerator._empty_analysis_summary()
    assert result == {
        "analysis_approach": "",
        "methodology_overview": "",
        "workflow_summary": "",
        "key_results": "",
        "figure_descriptions": {},
        "parameters_used": {},
    }


def test_all_pi_review_methods_exist():
    """Spot-check that key PIReviewMixin methods are accessible on the composed class."""
    for name in [
        "_pi_synthesize_review",
        "_pi_comprehensive_review",
        "_pi_review_analysis",
    ]:
        assert hasattr(DetailedReportGenerator, name), f"Missing method: {name}"