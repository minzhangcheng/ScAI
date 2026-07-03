from backend.workflow.restore_targets import SNAPSHOT_RESUME_TARGET


def test_pre_integration_gate_restore_target():
    t = SNAPSHOT_RESUME_TARGET["pre_integration_gate"]
    assert t["kind"] == "semantic_gate"
    assert t["node"] == "wait_for_user_input"
    assert t["gate"] == "integration_selection"
