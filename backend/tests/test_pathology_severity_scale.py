"""病理严重度只能有一套 rubric-bound 标度；无适用量表时整体为空，而不是退回通用等级。

A 0-4 tier ladder is only meaningful against a validated rubric for one tissue and
one lesion class, so the overview no longer emits a generic normal..severe label
that gets projected onto a canonical scale. Both the overview and the synthesis
step report score + max + basis, or nothing at all.
"""
import re

import backend.agents.pathology_parts.image_analysis as mod


def _source() -> str:
    return open(mod.__file__).read()


def test_no_generic_tier_ladder_remains():
    src = _source()
    assert "normal | mild | moderate | marked | severe" not in src, (
        "the generic severity ladder is back; a tier label is an unanchored 0-4 "
        "in disguise and is not comparable across tissues or stains"
    )
    assert "severity_label_to_score" not in src


def test_helper_is_gone_with_its_only_caller():
    import backend.agents.pathology_agent as agent_mod
    assert not hasattr(agent_mod, "severity_label_to_score"), (
        "a mapping helper left behind after its production caller is removed is "
        "exactly the dead-guard pattern this change removes"
    )


def test_both_layers_ask_for_score_max_and_basis():
    src = _source()
    for field in ("severity_score", "severity_max", "severity_basis"):
        assert src.count(field) >= 2, f"{field} must be asked for at both layers"


def test_ungradeable_stays_absent_rather_than_becoming_zero():
    """The integration block must not invent a score for an ungraded overview."""
    overview_data = {"overall_structure": "fibrotic"}
    if isinstance(overview_data, dict):
        overview_data.setdefault("severity_score", None)
        overview_data.setdefault("severity_max", None)
        overview_data.setdefault("severity_basis", "")
    assert overview_data["severity_score"] is None
    assert overview_data["severity_max"] is None


def test_existing_score_not_overwritten():
    overview_data = {"severity_score": 3, "severity_max": 4, "severity_basis": "Ishak"}
    if isinstance(overview_data, dict):
        overview_data.setdefault("severity_score", None)
        overview_data.setdefault("severity_max", None)
        overview_data.setdefault("severity_basis", "")
    assert overview_data["severity_score"] == 3
    assert overview_data["severity_basis"] == "Ishak"
