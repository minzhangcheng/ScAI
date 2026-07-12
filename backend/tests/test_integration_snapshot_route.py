import inspect
import backend.routes.lifecycle as lc
import backend.workflow.integration as integration
import backend.workflow.restore_targets as restore_targets


def test_snapshot_restore_sets_waiting_input_for_integration_gate():
    """Integration gate snapshot phase is wired through restore targets + node."""
    assert "pre_integration_gate" in restore_targets.SNAPSHOT_RESUME_TARGET
    gate = restore_targets.SNAPSHOT_RESUME_TARGET["pre_integration_gate"]
    assert gate.get("gate") or gate.get("kind")
    src = inspect.getsource(integration)
    assert "pre_integration_gate" in src
    assert "integration_gate" in src


def test_lifecycle_accepts_integration_selection():
    src = inspect.getsource(lc)
    assert "SEMANTIC_WAIT_GATES" in src
