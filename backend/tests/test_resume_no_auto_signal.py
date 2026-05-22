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
