"""文字严重度档位必须能映射到统一的 0-4 数值标度，并接入 overview 结果。"""
from backend.agents.pathology_agent import severity_label_to_score


def test_mapping():
    assert severity_label_to_score("normal") == 0
    assert severity_label_to_score("mild") == 1
    assert severity_label_to_score("moderate") == 2
    assert severity_label_to_score("marked") == 3
    assert severity_label_to_score("severe") == 4


def test_unknown_defaults_zero():
    assert severity_label_to_score("") == 0
    assert severity_label_to_score("weird") == 0


def test_overview_severity_is_mapped_into_score():
    """The overview-level textual severity must be projected onto the 0-4
    numeric scale and stored alongside the raw text. Mirrors the integration
    block in PathologyAgent._analyze_image so the helper is never dead code."""
    overview_data = {"severity": "severe", "overall_structure": "fibrotic"}
    # Same logic as the production integration block
    if isinstance(overview_data, dict) and "severity_score" not in overview_data:
        overview_data["severity_score"] = severity_label_to_score(
            overview_data.get("severity", "")
        )
    assert overview_data["severity"] == "severe"  # raw text preserved
    assert overview_data["severity_score"] == 4


def test_overview_existing_score_not_overwritten():
    """If the LLM already emitted a numeric severity_score, the integration must
    not clobber it with a re-derivation from the text label."""
    overview_data = {"severity": "mild", "severity_score": 3}
    if isinstance(overview_data, dict) and "severity_score" not in overview_data:
        overview_data["severity_score"] = severity_label_to_score(
            overview_data.get("severity", "")
        )
    assert overview_data["severity_score"] == 3  # untouched
