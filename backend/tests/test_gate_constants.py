from backend.constants import (
    GATE_ROUND_WAIT,
    GATE_DIRECTION_SELECTION,
    GATE_INTEGRATION_SELECTION,
    GATE_ANNOTATION_CONFIRMATION,
    GATE_PLAN_REVISION,
    SEMANTIC_WAIT_GATES,
)


def test_gate_constant_values():
    assert GATE_ROUND_WAIT == "round_wait"
    assert GATE_DIRECTION_SELECTION == "direction_selection"
    assert GATE_INTEGRATION_SELECTION == "integration_selection"
    assert GATE_ANNOTATION_CONFIRMATION == "annotation_confirmation"


def test_semantic_wait_gates_membership():
    assert SEMANTIC_WAIT_GATES == frozenset({
        "round_wait",
        "direction_selection",
        "integration_selection",
        "annotation_confirmation",
        "plan_revision",
    })
    assert GATE_ROUND_WAIT in SEMANTIC_WAIT_GATES
    assert GATE_DIRECTION_SELECTION in SEMANTIC_WAIT_GATES
    assert GATE_INTEGRATION_SELECTION in SEMANTIC_WAIT_GATES
    assert GATE_ANNOTATION_CONFIRMATION in SEMANTIC_WAIT_GATES
    assert GATE_PLAN_REVISION in SEMANTIC_WAIT_GATES


def test_restore_targets_use_constants():
    from backend.workflow.restore_targets import SNAPSHOT_RESUME_TARGET
    assert SNAPSHOT_RESUME_TARGET["pre_round_wait"]["gate"] == GATE_ROUND_WAIT
    assert SNAPSHOT_RESUME_TARGET["pre_direction"]["gate"] == GATE_DIRECTION_SELECTION
    assert SNAPSHOT_RESUME_TARGET["pre_integration_gate"]["gate"] == GATE_INTEGRATION_SELECTION
    assert SNAPSHOT_RESUME_TARGET["pre_annotation"]["gate"] == GATE_ANNOTATION_CONFIRMATION
