import inspect
import backend.routes.interaction as ix
import backend.routes.websockets as ws
from backend.routes.websocket_parts import state_sync


def test_finalize_no_workflow_does_not_pause():
    """finalize_from_integration must NOT call _pause_and_restart_ws (which
    sets status=PAUSED) — that prevents final_report_requested from being
    consumed. Instead it should set GENERATING_REPORT for report-only path."""
    src = inspect.getsource(ix.finalize_from_integration)
    assert "_pause_and_restart_ws" not in src, (
        "finalize_from_integration must not call _pause_and_restart_ws — "
        "it sets status=PAUSED which prevents final_report_requested consumption"
    )
    assert "GENERATING_REPORT" in src


def test_needs_report_only_honors_final_report_requested():
    """needs_report_only must be True when final_report_requested is set,
    even if current_round < effective_max_rounds (integration loop extends
    beyond the original max_rounds)."""
    src = inspect.getsource(state_sync._needs_report_only)
    assert "final_report_requested" in src, (
        "needs_report_only must check final_report_requested"
    )


def test_in_integration_loop_excludes_final_report_requested():
    """_in_integration_loop must return False when final_report_requested=True
    — the user explicitly chose to write report, so the session is NOT 'in
    the integration loop' anymore."""
    src = inspect.getsource(state_sync._in_integration_loop)
    assert "final_report_requested" in src
