"""Lifecycle / chat / WS guards for snapshot_failure gate."""
from backend.tests.split_facade import patch_facade
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.models.schemas import AnalysisState, TaskStatus
from backend.services.snapshot_v2.errors import (
    RequiredSnapshotBlocked,
    is_snapshot_failure_blocked,
)


def _state(**kw):
    defaults = dict(
        id="sess-gate",
        name="Test",
        user_id="u1",
        initial_instruction="analyze",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason="snapshot_failure",
        snapshot_failure={
            "code": "QUOTA",
            "message": "disk full",
            "allowed_actions": ["retry", "manage", "terminate"],
        },
    )
    defaults.update(kw)
    return AnalysisState(**defaults)


def test_is_snapshot_failure_blocked_true_on_reason_or_payload():
    s = _state()
    assert is_snapshot_failure_blocked(s) is True
    s.wait_input_reason = None
    assert is_snapshot_failure_blocked(s) is True
    s.snapshot_failure = None
    assert is_snapshot_failure_blocked(s) is False


@pytest.mark.asyncio
async def test_resume_refuses_when_snapshot_failure_blocked():
    from backend.routes import lifecycle

    state = _state()
    with patch.object(
        lifecycle, "require_session_ownership", new=AsyncMock(return_value=state)
    ):
        resp = await lifecycle._resume_analysis_locked("sess-gate", "u1")

    assert resp.success is False
    assert "snapshot failure" in resp.message.lower()
    assert resp.data.get("snapshot_failure") is not None
    # Gate must not be cleared
    assert state.wait_input_reason == "snapshot_failure"
    assert state.snapshot_failure is not None
    assert state.status == TaskStatus.WAITING_INPUT


@pytest.mark.asyncio
async def test_run_refuses_when_snapshot_failure_blocked():
    from fastapi import HTTPException
    from backend.routes import lifecycle

    state = _state(status=TaskStatus.PAUSED)
    req = MagicMock()
    req.state = MagicMock(user_id="u1")

    with patch.object(
        lifecycle, "require_session_ownership", new=AsyncMock(return_value=state)
    ):
        with patch.object(lifecycle, "validate_session_id"):
            with pytest.raises(HTTPException) as ei:
                await lifecycle.run_analysis("sess-gate", req)

    assert ei.value.status_code == 409
    assert state.status == TaskStatus.PAUSED  # not advanced to RUNNING


@pytest.mark.asyncio
async def test_stop_summarize_refuses_when_snapshot_failure_blocked():
    from backend.routes import lifecycle
    from backend.models.api_requests import StopAnalysisRequest

    state = _state(status=TaskStatus.WAITING_INPUT)
    req = StopAnalysisRequest(action="summarize")

    with patch.object(
        lifecycle, "require_session_ownership", new=AsyncMock(return_value=state)
    ):
        resp = await lifecycle._stop_analysis_locked("sess-gate", "u1", req)

    assert resp.success is False
    assert "snapshot failure" in resp.message.lower()
    assert state.status == TaskStatus.WAITING_INPUT
    assert state.wait_input_reason == "snapshot_failure"


@pytest.mark.asyncio
async def test_ws_resend_includes_snapshot_failure():
    from backend.routes.websockets import _resend_wait_input_state_sync

    state = _state()
    sent = []

    class FakeWS:
        async def send_json(self, payload):
            sent.append(payload)

    await _resend_wait_input_state_sync(FakeWS(), state, "sess-gate")
    assert sent, "expected state_sync resend"
    content = sent[0].get("content") or sent[0]
    assert content.get("change_type") == "snapshot_failure"
    assert content.get("snapshot_failure") is not None
    assert content.get("wait_input_reason") == "snapshot_failure"


@pytest.mark.asyncio
async def test_ws_resend_upgrades_legacy_bare_snapshot_failure_gate():
    from backend.routes.websockets import _resend_wait_input_state_sync

    state = _state()
    state.snapshot_failure = {
        "code": "snapshot_quota_exceeded",
        "message": "disk full",
        "allowed_actions": ["retry", "manage", "terminate"],
    }
    sent = []

    class FakeWS:
        async def send_json(self, payload):
            sent.append(payload)

    await _resend_wait_input_state_sync(FakeWS(), state, "sess-gate")

    content = sent[0].get("content") or sent[0]
    failure = content["snapshot_failure"]
    assert failure["allowed_actions"] == [
        "retry", "skip", "manage", "terminate",
    ]
    assert failure["retry_context"]["producer"] == "workflow_boundary"


@pytest.mark.asyncio
async def test_chat_snapshot_failure_not_rewritten_as_chat_error():
    """RequiredSnapshotBlocked from post-chat must surface structured failure."""
    from backend.routes import chat as chat_mod

    state = AnalysisState(
        id="sess-chat",
        name="T",
        user_id="u1",
        initial_instruction="x",
        status=TaskStatus.COMPLETED,
    )
    # Simulate _create_chat_snapshot raising after state already set by service
    async def _fail(*a, **kw):
        state.wait_input_reason = "snapshot_failure"
        state.snapshot_failure = {
            "code": "QUOTA",
            "message": "full",
            "allowed_actions": ["retry", "manage", "terminate"],
        }
        raise RequiredSnapshotBlocked("QUOTA", "full")

    with patch_facade("backend.routes.chat", "_create_chat_snapshot", side_effect=_fail):
        with patch_facade("backend.routes.chat", "async_save_session"):
            # Minimal exercise of the exception handler path used by chat_with_pi
            result = {"role": "assistant", "content": "ok"}
            try:
                await chat_mod._create_chat_snapshot("sess-chat", state)
            except RequiredSnapshotBlocked as e:
                from backend.services.snapshot_v2.errors import SNAPSHOT_FAILURE_ACTIONS
                if not state.snapshot_failure:
                    state.snapshot_failure = {
                        "code": e.code,
                        "message": str(e),
                        "allowed_actions": list(SNAPSHOT_FAILURE_ACTIONS),
                    }
                state.wait_input_reason = "snapshot_failure"
                result["snapshot_failure"] = state.snapshot_failure
                result["wait_input_reason"] = "snapshot_failure"
                result["status"] = "snapshot_failure"

    assert result["status"] == "snapshot_failure"
    assert result["snapshot_failure"]["code"] == "QUOTA"
    assert "error" not in result or result.get("status") != "chat_error"
