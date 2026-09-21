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
async def test_resume_from_pause_at_waiting_input_does_not_auto_signal_continue(monkeypatch):
    from backend.routes import lifecycle

    state = AnalysisState(user_id="u", initial_instruction="x")
    state.status = TaskStatus.PAUSED
    state.pre_pause_status = TaskStatus.WAITING_INPUT
    state.user_chosen_direction = None

    request = SimpleNamespace(state=SimpleNamespace(user_id="u"))
    workflow = MagicMock()
    workflow.signal_resume = MagicMock()
    workflow.signal_continue = MagicMock()
    workflow.signal_annotation_confirmed = MagicMock()

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
    assert state.status == TaskStatus.WAITING_INPUT
    assert state.pre_pause_status is None
    assert not state.user_chosen_direction


@pytest.mark.asyncio
async def test_resume_from_pause_at_waiting_annotation_does_not_auto_confirm(monkeypatch):
    from backend.routes import lifecycle

    state = AnalysisState(user_id="u", initial_instruction="x")
    state.status = TaskStatus.PAUSED
    state.pre_pause_status = TaskStatus.WAITING_ANNOTATION

    request = SimpleNamespace(state=SimpleNamespace(user_id="u"))
    workflow = MagicMock()
    workflow.signal_resume = MagicMock()
    workflow.signal_continue = MagicMock()
    workflow.signal_annotation_confirmed = MagicMock()

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
    assert state.status == TaskStatus.WAITING_ANNOTATION
    assert state.pre_pause_status is None


@pytest.mark.asyncio
async def test_resume_from_pause_at_waiting_plan_review_does_not_auto_skip(monkeypatch):
    from backend.routes import lifecycle

    state = AnalysisState(user_id="u", initial_instruction="x")
    state.status = TaskStatus.PAUSED
    state.pre_pause_status = TaskStatus.WAITING_PLAN_REVIEW

    request = SimpleNamespace(state=SimpleNamespace(user_id="u"))
    workflow = MagicMock()
    workflow.signal_resume = MagicMock()
    workflow.signal_plan_review = MagicMock()
    workflow.signal_continue = MagicMock()
    workflow.signal_annotation_confirmed = MagicMock()

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
    workflow.signal_plan_review.assert_not_called()
    workflow.signal_continue.assert_not_called()
    workflow.signal_annotation_confirmed.assert_not_called()
    assert state.status == TaskStatus.WAITING_PLAN_REVIEW


@pytest.mark.asyncio
async def test_resume_from_pause_at_running_goes_to_running(monkeypatch):
    """Regression: PAUSED with no pre_pause_status still resumes to RUNNING."""
    from backend.routes import lifecycle

    state = AnalysisState(user_id="u", initial_instruction="x")
    state.status = TaskStatus.PAUSED
    state.pre_pause_status = None

    request = SimpleNamespace(state=SimpleNamespace(user_id="u"))
    workflow = MagicMock()
    workflow.signal_resume = MagicMock()
    workflow.signal_continue = MagicMock()
    workflow.signal_annotation_confirmed = MagicMock()

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
    assert state.status == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_resume_while_running_is_rejected_without_mutating_state(monkeypatch):
    """A stale duplicate /resume must not touch runtime or workflow state."""
    from fastapi import HTTPException

    from backend.routes import lifecycle

    state = AnalysisState(user_id="u", initial_instruction="x")
    state.status = TaskStatus.RUNNING
    state.current_round = 3
    state.max_rounds = 7
    state.original_max_rounds = 11
    state.runtime_restore_status = "stale"
    state.runtime_restore_warning = "keep this warning"
    state.skip_required_snapshot_phase = "round_start"
    before = state.model_dump(mode="json")

    request = SimpleNamespace(state=SimpleNamespace(user_id="u"))
    require = AsyncMock(return_value=state)
    monkeypatch.setattr(lifecycle, "require_session_ownership", require)
    active_workflow = object()
    workflow_map = {"s1": active_workflow}
    monkeypatch.setattr(lifecycle, "workflows", workflow_map)

    # These are the mutators/probes that the old implementation reached before
    # discovering that the analysis was already running. None may be touched by
    # a stale request; status validation is intentionally the first operation.
    recovery = MagicMock(side_effect=AssertionError("recovery check called"))
    update_cache = AsyncMock(side_effect=AssertionError("cache update called"))
    save = AsyncMock(side_effect=AssertionError("session save called"))
    monkeypatch.setattr(lifecycle, "assert_recovery_clear", recovery)
    monkeypatch.setattr(lifecycle, "_update_session_cache", update_cache)
    monkeypatch.setattr(lifecycle, "async_save_session", save)

    with pytest.raises(HTTPException) as exc_info:
        await lifecycle.resume_analysis("s1", request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Analysis already running"
    require.assert_awaited_once_with(session_id="s1", user_id="u")
    recovery.assert_not_called()
    update_cache.assert_not_awaited()
    save.assert_not_awaited()
    assert lifecycle.workflows is workflow_map
    assert lifecycle.workflows["s1"] is active_workflow
    assert state.model_dump(mode="json") == before


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
@pytest.mark.parametrize("skip", [False, True])
async def test_retry_api_persists_target_for_resume(monkeypatch, skip):
    from backend.routes import lifecycle
    from backend.models.api_requests import RetryNodeRequest

    state = AnalysisState(user_id="u", status=TaskStatus.FAILED)
    monkeypatch.setattr(lifecycle, "require_session_ownership", AsyncMock(return_value=state))
    save = AsyncMock()
    monkeypatch.setattr(lifecycle, "async_save_session", save)
    request = SimpleNamespace(state=SimpleNamespace(user_id="u"))
    result = await lifecycle.retry_node("s1", request,
        RetryNodeRequest(node_name="literature_research", skip=skip))
    assert result.success
    intent = dict(state.sandbox_state["node_retry"])
    assert intent["id"]
    assert intent["node_name"] == "literature_research"
    assert intent["skip"] is skip
    save.assert_awaited_once_with("s1", state, force=True)

    monkeypatch.setattr(lifecycle, "workflows", {})
    monkeypatch.setattr(lifecycle, "_update_session_cache", AsyncMock())
    resumed = await lifecycle.resume_analysis("s1", request)
    assert resumed.success
    assert resumed.data["needs_reconnect"] is True
    assert state.sandbox_state["node_retry"] == intent
    assert save.await_args.kwargs["force"] is True


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
