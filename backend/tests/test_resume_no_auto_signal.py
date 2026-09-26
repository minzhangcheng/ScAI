"""P0-5: Resume from PAUSED must NOT auto-signal HITL gates.

When a session was paused while in WAITING_INPUT or WAITING_ANNOTATION,
generic /resume should restore the waiting status and only call
signal_resume(). It must NOT call signal_continue() or
signal_annotation_confirmed() — that would silently bypass the gate,
treating the user as if they had submitted direction/annotation choices.
"""
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from backend.models.schemas import AnalysisState, TaskStatus


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting_status", [
    TaskStatus.WAITING_INPUT, TaskStatus.WAITING_ANNOTATION, TaskStatus.WAITING_PLAN_REVIEW,
])
async def test_resume_from_pause_preserves_gate_without_auto_signals(monkeypatch, waiting_status):
    from backend.routes import lifecycle

    state = AnalysisState(user_id="u", initial_instruction="x")
    state.status = TaskStatus.PAUSED
    state.pre_pause_status = waiting_status
    state.user_chosen_direction = None

    request = SimpleNamespace(state=SimpleNamespace(user_id="u"))
    workflow = MagicMock()
    workflow.signal_resume = MagicMock()
    workflow.signal_continue = MagicMock()
    workflow.signal_annotation_confirmed = MagicMock()
    workflow.signal_plan_review = MagicMock()

    async def _fake_require_session_ownership(session_id: str, user_id: str):
        return state

    monkeypatch.setattr(lifecycle, "require_session_ownership", _fake_require_session_ownership)
    monkeypatch.setattr(lifecycle, "_update_session_cache", AsyncMock())
    monkeypatch.setattr(lifecycle, "async_save_session", AsyncMock())
    monkeypatch.setattr(lifecycle, "_pause_and_restart_ws", AsyncMock())
    monkeypatch.setattr(lifecycle, "workflows", {"s1": workflow})

    resp = await lifecycle.resume_analysis("s1", request)

    assert resp.success is True
    workflow.signal_resume.assert_called_once()
    workflow.signal_continue.assert_not_called()
    workflow.signal_annotation_confirmed.assert_not_called()
    if waiting_status == TaskStatus.WAITING_PLAN_REVIEW:
        workflow.signal_plan_review.assert_not_called()
    assert state.status == waiting_status
    if waiting_status != TaskStatus.WAITING_PLAN_REVIEW:
        assert state.pre_pause_status is None
    if waiting_status == TaskStatus.WAITING_INPUT:
        assert not state.user_chosen_direction


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["llm_failure", "agent_disabled", "marker_db_missing"])
async def test_generic_resume_preserves_active_error_gate(monkeypatch, reason):
    from backend.routes import lifecycle

    state = AnalysisState(user_id="u", status=TaskStatus.WAITING_INPUT,
                          wait_input_reason=reason)
    before = state.model_dump(mode="json")
    workflow = MagicMock()
    monkeypatch.setattr(lifecycle, "require_session_ownership", AsyncMock(return_value=state))
    monkeypatch.setattr(lifecycle, "workflows", {"s1": workflow})
    save = AsyncMock()
    monkeypatch.setattr(lifecycle, "async_save_session", save)
    response = await lifecycle.resume_analysis(
        "s1", SimpleNamespace(state=SimpleNamespace(user_id="u")))
    assert response.success
    assert response.data["needs_reconnect"] is False
    assert state.model_dump(mode="json") == before
    workflow.signal_continue.assert_not_called()
    workflow.signal_resume.assert_not_called()
    save.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_api_rejects_skip_without_node_skip_handler(monkeypatch):
    from backend.routes import lifecycle
    from backend.models.api_requests import RetryNodeRequest
    from fastapi import HTTPException

    state = AnalysisState(user_id="u", status=TaskStatus.FAILED)
    monkeypatch.setattr(lifecycle, "require_session_ownership", AsyncMock(return_value=state))
    with pytest.raises(HTTPException, match="") as exc:
        await lifecycle.retry_node("s1", SimpleNamespace(state=SimpleNamespace(user_id="u")),
            RetryNodeRequest(node_name="planning_dialogue", skip=True))
    assert exc.value.status_code == 400
    assert not (state.sandbox_state or {}).get("node_retry")
