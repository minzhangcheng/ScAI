import inspect
import backend.routes.snapshots as snaps
import backend.routes.lifecycle as lc


def test_snapshot_restore_sets_waiting_input_for_integration_gate():
    src = inspect.getsource(snaps)
    assert 'phase == "pre_integration_gate"' in src
    assert "integration_gate" in src
    assert "WebSocketMessage(" in src


def test_lifecycle_accepts_integration_selection():
    src = inspect.getsource(lc)
    assert "SEMANTIC_WAIT_GATES" in src
