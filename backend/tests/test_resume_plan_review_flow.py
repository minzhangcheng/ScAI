"""Lifecycle resume must not auto-skip plan review."""
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from backend.models.schemas import AnalysisState, TaskStatus


@pytest.mark.asyncio
async def test_resume_waiting_plan_review_does_not_signal_skip(monkeypatch):
    from backend.routes import lifecycle

    state = AnalysisState(user_id="u", initial_instruction="x")
    state.status = TaskStatus.WAITING_PLAN_REVIEW

    request = SimpleNamespace(state=SimpleNamespace(user_id="u"))
    workflow = MagicMock()
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
    assert "plan review" in resp.message.lower()
    workflow.signal_plan_review.assert_not_called()
    assert state.status == TaskStatus.WAITING_PLAN_REVIEW