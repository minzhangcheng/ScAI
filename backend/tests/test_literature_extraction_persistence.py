"""Literature extraction checkpoints survive pause and respect workflow ownership."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from backend.config import ensure_session_dirs
from backend.models.schemas import AnalysisState, TaskStatus
from backend.services import session_store
from backend.state import sessions
from backend.workflow.agent_nodes.literature_node import LiteratureNodeMixin


@pytest.mark.asyncio
async def test_pause_cannot_overwrite_durable_extraction_checkpoint(monkeypatch):
    from backend.routes import lifecycle

    sid = "extraction-cache-regression"
    ensure_session_dirs(sid)
    cached = AnalysisState(id=sid, user_id="u", status=TaskStatus.RUNNING)
    await session_store.update_session_cache(sid, cached)
    # LangGraph reconstructs a separate AnalysisState at the next node.
    node_state = cached.model_copy(deep=True)
    record = {"identity": "current-input", "phase": "pending", "report_ids": ["report-1"],
              "result": {"findings": ["retained research"], "reports": ["long report " * 200]}}
    node_state.sandbox_state["literature_extraction_resume_v1"] = record
    host = LiteratureNodeMixin()
    host.session_id = sid

    async def publish(state, *, force=False):
        await session_store.update_session_cache(sid, state)
        await session_store.async_save_session(sid, state, force=force)
        return True

    host.state_callback = AsyncMock(side_effect=publish)
    await host._save_literature_extraction_state(node_state)
    host.state_callback.assert_awaited_once_with(node_state, force=True)
    assert session_store.load_session_from_disk(sid).sandbox_state["literature_extraction_resume_v1"] == record

    # A normal Pause arrives while the extraction request is still running.
    async def ownership(**kwargs):
        return sessions[sid]

    async def pause(session_id, analysis_state):
        analysis_state.status = TaskStatus.PAUSED
        return {"success": True}

    monkeypatch.setattr(lifecycle, "require_session_ownership", ownership)
    monkeypatch.setattr(lifecycle, "workflows", {sid: SimpleNamespace(pause=pause)})
    monkeypatch.setattr(lifecycle, "_update_session_cache", session_store.update_session_cache)
    monkeypatch.setattr(lifecycle, "async_save_session", session_store.async_save_session)
    result = await lifecycle.pause_analysis(sid, SimpleNamespace(state=SimpleNamespace(user_id="u")))
    assert result.success
    reloaded = session_store.load_session_from_disk(sid)
    assert reloaded.sandbox_state.get("literature_extraction_resume_v1") == record


@pytest.mark.asyncio
@pytest.mark.parametrize("callback_error", [False, True])
async def test_callback_failure_does_not_fall_back_to_unowned_save(monkeypatch, callback_error):
    direct_save = AsyncMock(return_value=True)
    monkeypatch.setattr(session_store, "async_save_session", direct_save)
    state = AnalysisState(id="extraction-callback", user_id="u")
    host = LiteratureNodeMixin()
    host.session_id = state.id
    host.state_callback = AsyncMock(
        side_effect=OSError("disk unavailable") if callback_error else None,
        return_value=False,
    )

    await host._save_literature_extraction_state(state)

    host.state_callback.assert_awaited_once_with(state, force=True)
    direct_save.assert_not_awaited()
