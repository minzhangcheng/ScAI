from typing import Optional


SNAPSHOT_RESUME_TARGET = {
    "pre_round_wait": {"kind": "semantic_gate", "gate": "round_wait", "node": "wait_for_user_input"},
    "pre_direction": {"kind": "semantic_gate", "gate": "direction_selection", "node": "wait_for_user_input"},
    "pre_annotation": {"kind": "semantic_gate", "gate": "annotation_confirmation", "node": None},
    "pre_plan_review": {"kind": "graph_node", "gate": "plan_review", "node": "user_review_gate"},
    "pre_execution_full": {"kind": "graph_node", "gate": None, "node": "bioinformatics_analysis"},
    "pre_execution_stage1": {"kind": "graph_node", "gate": None, "node": "bioinformatics_analysis"},
    "pre_execution_stage2": {"kind": "graph_node", "gate": None, "node": "bioinformatics_analysis"},
    "post_report": {"kind": "terminal", "gate": None, "node": None},
}


def stable_restore_branch_id(snapshot_id: str, branch_label: str | None) -> str:
    return f"restore-{snapshot_id}"


def stable_restore_branch_label(snapshot_id: str, branch_label: Optional[str]) -> str:
    if branch_label and branch_label != "main":
        return branch_label
    return f"branch-{snapshot_id[:8]}"
