"""Stage1→stage2 transition must not leak stage1 pi_instructions into stage2 planning."""
from unittest.mock import MagicMock


def _build_stage_transition_state():
    return {
        "pi_instructions": {
            "bioinformatics_tasks": ["stage1: QC", "stage1: clustering"],
            "pathology_tasks": ["stage1: HE review"],
            "biochemistry_tasks": ["stage1: ALT/AST"],
            "literature_topics": ["stage1: liver QC"],
            "overview": "stage1 exploration",
            "merge_strategy": "merge similar hepatocytes",
        },
        "planning_stage": "stage1",
        "plan_approved": True,
    }


def test_stage1_to_stage2_clears_task_lists_in_pi_instructions():
    from backend.workflow.agents import _reset_pi_instructions_for_stage2

    state = _build_stage_transition_state()
    new_instructions = _reset_pi_instructions_for_stage2(state["pi_instructions"])

    assert new_instructions["bioinformatics_tasks"] == []
    assert new_instructions["pathology_tasks"] == []
    assert new_instructions["biochemistry_tasks"] == []
    assert new_instructions["literature_topics"] == []
    assert new_instructions["overview"] == ""


def test_stage1_to_stage2_preserves_round_global_fields():
    from backend.workflow.agents import _reset_pi_instructions_for_stage2

    state = _build_stage_transition_state()
    state["pi_instructions"]["cluster_comparison_strategy"] = "compare treated vs control"
    new_instructions = _reset_pi_instructions_for_stage2(state["pi_instructions"])

    assert new_instructions["merge_strategy"] == "merge similar hepatocytes"
    assert new_instructions["cluster_comparison_strategy"] == "compare treated vs control"


def test_reset_returns_new_dict():
    from backend.workflow.agents import _reset_pi_instructions_for_stage2

    state = _build_stage_transition_state()
    original = state["pi_instructions"]
    new_instructions = _reset_pi_instructions_for_stage2(original)

    assert new_instructions is not original
    assert original["bioinformatics_tasks"] == ["stage1: QC", "stage1: clustering"]