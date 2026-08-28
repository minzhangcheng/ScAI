"""P1-8/P1-9: verify session-lock plumbing across mutating routes and WS creation.

This is a static-source test: the goal is to ensure that mutating endpoints
acquire the per-session lock around their read-modify-save sequence, and that
the WebSocket workflow creation holds the lock through the full create-or-observe
decision (closing the TOCTOU race).
"""
import re
import inspect

import backend.state as state_mod
import backend.routes.interaction as ix
import backend.routes.lifecycle as lc
import backend.routes.websockets as ws
import backend.routes.snapshots as snap
from backend.routes.lifecycle_parts import interventions, run_resume
from backend.routes.websocket_parts import analysis_connection


def _extract_async_fn(src: str, name: str) -> str:
    """Return the source of an async function `name` from src, until the next top-level def."""
    pattern = rf"async def {name}\b.*?(?=\n(?:async )?def |\Z)"
    m = re.search(pattern, src, re.DOTALL)
    assert m, f"could not locate async def {name}"
    return m.group(0)


def test_get_session_lock_exists():
    """P1-8/9: utility to acquire per-session lock must exist."""
    assert hasattr(state_mod, "get_session_lock"), \
        "P1-8/9: state.get_session_lock(session_id) must be defined"


def test_release_session_lock_slot_exists():
    """P1-8/9: best-effort release helper for session-deletion paths."""
    assert hasattr(state_mod, "release_session_lock_slot"), \
        "P1-8/9: state.release_session_lock_slot(session_id) must be defined"


def test_choose_direction_uses_session_lock():
    src = inspect.getsource(ix)
    body = _extract_async_fn(src, "choose_direction")
    assert "get_session_lock" in body, \
        "P1-9: choose_direction must acquire get_session_lock"


def test_confirm_annotations_uses_session_lock():
    src = inspect.getsource(ix)
    body = _extract_async_fn(src, "confirm_annotations")
    assert "get_session_lock" in body, \
        "P1-9: confirm_annotations must acquire get_session_lock"


def test_send_human_input_uses_session_lock():
    src = inspect.getsource(ix)
    body = _extract_async_fn(src, "send_human_input")
    assert "get_session_lock" in body, \
        "P1-9: send_human_input must acquire get_session_lock"


def test_llm_failure_action_uses_session_lock():
    src = inspect.getsource(interventions)
    body = _extract_async_fn(src, "llm_failure_action")
    assert "get_session_lock" in body, \
        "P1-9: llm_failure_action must acquire get_session_lock"


def test_pause_analysis_uses_session_lock():
    src = inspect.getsource(run_resume)
    body = _extract_async_fn(src, "pause_analysis")
    assert "get_session_lock" in body, \
        "P1-9: pause_analysis must acquire get_session_lock"


def test_resume_analysis_uses_session_lock():
    src = inspect.getsource(run_resume)
    body = _extract_async_fn(src, "resume_analysis")
    assert "get_session_lock" in body, \
        "P1-9: resume_analysis must acquire get_session_lock"


def test_stop_analysis_uses_session_lock():
    src = inspect.getsource(run_resume)
    body = _extract_async_fn(src, "stop_analysis")
    assert "get_session_lock" in body, \
        "P1-9: stop_analysis must acquire get_session_lock"


def test_websocket_creation_holds_lock():
    """P1-8: the create-or-observe decision in the WS handler must run inside
    the per-session lock so two concurrent connections cannot both decide to
    create a workflow.

    The fix should acquire get_session_lock(session_id) before the shared
    graph/report activity classifier and reserve ``initializing_sessions``
    before releasing it.
    """
    src = inspect.getsource(analysis_connection)
    body = _extract_async_fn(src, "websocket_endpoint")
    # The handler must obtain a session-scoped lock, and it must be used to
    # guard the workflow-creation decision (not just the post-creation writes).
    assert body.count("get_session_lock") >= 1, \
        "P1-8: websocket_endpoint must call get_session_lock for the create-or-observe gate"
    # The lock acquisition must come before the workflow_running decision —
    # otherwise the TOCTOU race remains.
    lock_pos = body.find("get_session_lock")
    decision_pos = body.find("_session_workflow_activity(session_id)")
    assert lock_pos != -1 and decision_pos != -1
    assert lock_pos < decision_pos, \
        "P1-8: get_session_lock must be acquired BEFORE the workflow_running check"
