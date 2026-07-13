"""POST /snapshot-failure/skip route glue."""
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import HTTPException

from backend.models.schemas import AnalysisState, TaskStatus
import backend.routes.lifecycle as lifecycle


def _gated_state(retry_context):
    st = AnalysisState(id="sess-skip-route", name="T", user_id="u1",
                       initial_instruction="analyze")
    st.status = TaskStatus.WAITING_INPUT
    st.wait_input_reason = "snapshot_failure"
    st.snapshot_failure = {
        "code": "required_runtime_unpersistable",
        "message": "loom_con not found",
        "allowed_actions": ["retry", "skip", "manage", "terminate"],
        "retry_context": retry_context,
    }
    return st


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(lifecycle, "_update_session_cache", AsyncMock())
    monkeypatch.setattr(lifecycle, "async_save_session", AsyncMock())


def _request():
    return SimpleNamespace(state=SimpleNamespace(user_id="u1"))


@pytest.mark.asyncio
async def test_skip_workflow_boundary_clears_gate_and_arms_marker(monkeypatch, patched):
    state = _gated_state({
        "producer": "workflow_boundary",
        "resume_policy": "graph_checkpoint",
        "snapshot_request": {"phase": "pre_annotation"},
        "previous_status": "running",
    })
    monkeypatch.setattr(lifecycle, "require_session_ownership", AsyncMock(return_value=state))

    resp = await lifecycle.snapshot_failure_skip("sess-skip-route", _request())

    assert resp.success is True
    assert state.snapshot_failure is None
    assert state.status == TaskStatus.RUNNING
    assert state.skip_required_snapshot_phase == "pre_annotation"


@pytest.mark.asyncio
async def test_skip_pre_restore_returns_409(monkeypatch, patched):
    state = _gated_state({
        "producer": "pre_restore",
        "resume_policy": "restore_transaction",
        "snapshot_request": {"phase": "pre_restore"},
        "previous_status": "running",
    })
    monkeypatch.setattr(lifecycle, "require_session_ownership", AsyncMock(return_value=state))

    with pytest.raises(HTTPException) as exc:
        await lifecycle.snapshot_failure_skip("sess-skip-route", _request())
    assert exc.value.status_code == 409
    # Gate is left intact.
    assert state.snapshot_failure is not None


@pytest.mark.asyncio
async def test_skip_requires_active_gate(monkeypatch, patched):
    state = AnalysisState(id="sess-skip-route", name="T", user_id="u1",
                          initial_instruction="analyze")
    state.status = TaskStatus.RUNNING
    monkeypatch.setattr(lifecycle, "require_session_ownership", AsyncMock(return_value=state))

    with pytest.raises(HTTPException) as exc:
        await lifecycle.snapshot_failure_skip("sess-skip-route", _request())
    assert exc.value.status_code == 409
