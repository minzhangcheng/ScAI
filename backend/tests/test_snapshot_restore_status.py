import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.models.schemas import AnalysisState, TaskStatus


@pytest.mark.asyncio
async def test_per_session_lock_prevents_stale_overwrite():
    """The per-session lock serializes restore vs stale writes correctly.

    Reproduces the production race: a stale state_callback / CancelledError
    handler tries to write PAUSED concurrently with restore writing WAITING_INPUT.
    The lock must ensure restore is always the final writer.
    """
    from backend.state import sessions, workflows, get_session_lock

    session_id = "test-lock-serial"
    sess_lock = get_session_lock(session_id)

    old_state = AnalysisState(
        session_id=session_id, user_id="test", status=TaskStatus.RUNNING, current_round=3
    )
    sessions[session_id] = old_state
    mock_wf = object()
    workflows[session_id] = mock_wf

    restored = AnalysisState(
        session_id=session_id, user_id="test", status=TaskStatus.WAITING_INPUT, current_round=3
    )

    write_order = []

    async def stale_write():
        async with sess_lock:
            if workflows.get(session_id) is not mock_wf:
                return
            sessions[session_id] = AnalysisState(
                session_id=session_id, user_id="test", status=TaskStatus.PAUSED
            )
            write_order.append("stale")

    async def restore_write():
        async with sess_lock:
            workflows.pop(session_id, None)
            sessions[session_id] = restored
            write_order.append("restore")

    await asyncio.gather(stale_write(), restore_write())

    assert sessions[session_id].status == TaskStatus.WAITING_INPUT, (
        f"Expected WAITING_INPUT but got {sessions[session_id].status}. "
        f"Write order: {write_order}"
    )
    assert len(write_order) >= 1, "At least restore should have written"
    assert write_order[-1] == "restore", (
        f"Restore must be the last write, but order was {write_order}"
    )

    sessions.pop(session_id, None)
    workflows.pop(session_id, None)


@pytest.mark.asyncio
async def test_guard_skips_when_workflow_popped():
    """state_callback / cancel handler must skip entirely after workflows.pop.

    Tests the exact guard pattern used in production (修改 2/3):
      if workflows.get(session_id) is not workflow: return
    """
    from backend.state import sessions, workflows, get_session_lock

    session_id = "test-guard-skip"
    sess_lock = get_session_lock(session_id)
    mock_wf = object()
    workflows[session_id] = mock_wf

    async with sess_lock:
        workflows.pop(session_id, None)

    did_write = False
    async with sess_lock:
        if workflows.get(session_id) is not mock_wf:
            pass
        else:
            sessions[session_id] = AnalysisState(
                session_id=session_id, user_id="test", status=TaskStatus.PAUSED
            )
            did_write = True

    assert not did_write, (
        "Guard must skip write after workflows.pop. "
        "workflows.get returned: %s" % workflows.get(session_id)
    )

    sessions.pop(session_id, None)
    workflows.pop(session_id, None)


@pytest.mark.asyncio
async def test_guard_allows_write_when_workflow_active():
    """Guard must allow write when workflow is still registered.

    This is the normal (non-restore) path: state_callback saves progress
    during active workflow execution.
    """
    from backend.state import sessions, workflows, get_session_lock

    session_id = "test-guard-allow"
    sess_lock = get_session_lock(session_id)
    mock_wf = object()
    workflows[session_id] = mock_wf

    did_write = False
    async with sess_lock:
        if workflows.get(session_id) is not mock_wf:
            pass
        else:
            sessions[session_id] = AnalysisState(
                session_id=session_id, user_id="test", status=TaskStatus.PAUSED
            )
            did_write = True

    assert did_write, "Guard must allow write when workflow is still active"

    sessions.pop(session_id, None)
    workflows.pop(session_id, None)


@pytest.mark.asyncio
async def test_resume_respects_waiting_input_status():
    """resume() must not override WAITING_INPUT to RUNNING."""
    from backend.workflow import SingleCellAnalysisWorkflow

    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "test-resume-wi"
    workflow.compiled_graph = MagicMock()
    workflow.compiled_graph.aget_state = AsyncMock(return_value=None)
    workflow._current_analysis_state = None
    workflow._send_message = AsyncMock()

    analysis_state = AnalysisState(
        session_id="test-resume-wi",
        user_id="test",
        status=TaskStatus.WAITING_INPUT,
        current_round=2
    )

    result = await workflow.resume("thread-1", analysis_state)

    assert result is analysis_state
    assert result.status == TaskStatus.WAITING_INPUT


@pytest.mark.asyncio
async def test_resume_respects_waiting_annotation_status():
    """resume() must not override WAITING_ANNOTATION to RUNNING."""
    from backend.workflow import SingleCellAnalysisWorkflow

    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "test-resume-wa"
    workflow.compiled_graph = MagicMock()
    workflow.compiled_graph.aget_state = AsyncMock(return_value=None)
    workflow._current_analysis_state = None
    workflow._send_message = AsyncMock()

    analysis_state = AnalysisState(
        session_id="test-resume-wa",
        user_id="test",
        status=TaskStatus.WAITING_ANNOTATION,
        current_round=2
    )

    result = await workflow.resume("thread-1", analysis_state)

    assert result is analysis_state
    assert result.status == TaskStatus.WAITING_ANNOTATION


@pytest.mark.asyncio
async def test_resume_starts_workflow_for_paused_status():
    """resume() must still start workflow for PAUSED status.

    This is the normal path used by the /resume lifecycle endpoint,
    not by snapshot restore. After restore, resume() should never
    receive WAITING_INPUT — it is caught by HITL-hold.
    """
    from backend.workflow import SingleCellAnalysisWorkflow

    mock_graph = AsyncMock()
    mock_graph.aget_state = AsyncMock(return_value=None)

    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "test-resume-paused"
    workflow.compiled_graph = None
    workflow._current_analysis_state = None
    workflow._send_message = AsyncMock()
    workflow._set_status = MagicMock()
    workflow._make_initial_workflow_state = MagicMock(return_value={"test": "state"})

    with patch.object(SingleCellAnalysisWorkflow, 'initialize', new_callable=AsyncMock):
        await workflow.initialize()
        workflow.compiled_graph = mock_graph

    analysis_state = AnalysisState(
        session_id="test-resume-paused",
        user_id="test",
        status=TaskStatus.PAUSED,
        current_round=1
    )

    with patch.object(workflow, 'initialize', new_callable=AsyncMock):
        workflow._state = analysis_state
        mock_graph.aget_state = AsyncMock(return_value=None)
        mock_graph.ainvoke = AsyncMock(return_value={"analysis_state": analysis_state})

        result = await workflow.resume("thread-1", analysis_state)

    assert result is analysis_state