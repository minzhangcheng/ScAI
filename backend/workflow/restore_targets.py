from typing import Optional

from ..constants import (
    GATE_ANNOTATION_CONFIRMATION,
    GATE_DIRECTION_SELECTION,
    GATE_INTEGRATION_SELECTION,
    GATE_ROUND_WAIT,
)


SNAPSHOT_RESUME_TARGET = {
    "pre_round_wait": {"kind": "semantic_gate", "gate": GATE_ROUND_WAIT, "node": "wait_for_user_input"},
    "pre_direction": {"kind": "semantic_gate", "gate": GATE_DIRECTION_SELECTION, "node": "wait_for_user_input"},
    "pre_annotation": {"kind": "semantic_gate", "gate": GATE_ANNOTATION_CONFIRMATION, "node": None},
    "pre_plan_review": {"kind": "graph_node", "gate": "plan_review", "node": "user_review_gate"},
    "pre_major_revision": {"kind": "semantic_gate", "gate": "plan_revision", "node": "planning_dialogue"},
    "pre_execution_full": {"kind": "graph_node", "gate": None, "node": "bioinformatics_analysis"},
    "pre_execution_stage1": {"kind": "graph_node", "gate": None, "node": "bioinformatics_analysis"},
    "pre_execution_stage2": {"kind": "graph_node", "gate": None, "node": "bioinformatics_analysis"},
    "pre_integration_gate": {"kind": "semantic_gate", "gate": GATE_INTEGRATION_SELECTION, "node": "wait_for_user_input"},
    "post_report": {"kind": "terminal", "gate": None, "node": None},
}


def stable_restore_branch_id(snapshot_id: str, branch_label: str | None) -> str:
    return f"restore-{snapshot_id}"


def stable_restore_branch_label(snapshot_id: str, branch_label: Optional[str]) -> str:
    if branch_label and branch_label != "main":
        return branch_label
    return f"branch-{snapshot_id[:8]}"
