from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import asyncio
import pytest

from backend.workflow.core import SingleCellAnalysisWorkflow
from backend.utils.sandbox_parts.kernel import KernelMixin
from backend.workflow.helpers import WorkflowSupersededError


def _workflow_for_close():
    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "close-test"
    workflow._close_lock = asyncio.Lock()
    workflow._close_complete = False
    workflow._close_status = {}
    workflow._close_task = None
    workflow._close_retry_task = None
    workflow._discard_runtime_on_close = False
    workflow._close_discard_event = asyncio.Event()
    workflow._restore_superseded = False
    workflow.sandbox = SimpleNamespace(stop=AsyncMock())
    workflow._db_conn = SimpleNamespace(close=AsyncMock())
    workflow._compact_checkpoints = AsyncMock()
    workflow.llm_manager = SimpleNamespace(close=AsyncMock())
    workflow.literature_agent = SimpleNamespace(close_runtime_clients=AsyncMock())
    workflow._settings_provider = SimpleNamespace(cleanup=AsyncMock())
    return workflow


def _install_close_gate_state(workflow, status):
    """Publish a live AnalysisState this workflow owns.

    Named for the close gate it used to host; with the gate gone it is still
    how a close test gets a registered session to arbitrate ownership against.
    """
    from backend.models.schemas import AnalysisState
    from backend.state import sessions, workflows

    state = AnalysisState(id=workflow.session_id, user_id="u1", status=status)
    sessions[workflow.session_id] = state
    workflows[workflow.session_id] = workflow
    workflow._current_analysis_state = state
    return state


def _remove_close_gate_state(session_id):
    from backend.state import sessions, workflows

    sessions.pop(session_id, None)
    workflows.pop(session_id, None)


@pytest.mark.asyncio
async def test_workflow_close_retries_only_failed_resources_until_complete():
    workflow = _workflow_for_close()
    db_conn = workflow._db_conn
    workflow.sandbox.stop.side_effect = [RuntimeError("sandbox close failed"), None]
    db_conn.close.side_effect = [RuntimeError("db close failed"), None]
    workflow.llm_manager.close.side_effect = [RuntimeError("manager failed"), None]
    workflow._settings_provider.cleanup.side_effect = [RuntimeError("provider failed"), None]

    await workflow.close()

    assert workflow._close_complete is False
    assert workflow._db_conn is db_conn
    workflow.sandbox.stop.assert_awaited_once_with(save=True)
    db_conn.close.assert_awaited_once()
    workflow.llm_manager.close.assert_awaited_once()
    workflow._settings_provider.cleanup.assert_awaited_once()
    workflow._compact_checkpoints.assert_awaited_once()
    workflow.literature_agent.close_runtime_clients.assert_awaited_once()

    await workflow.close()

    assert workflow._close_complete is True
    assert workflow._db_conn is None
    assert workflow.sandbox.stop.await_count == 2
    assert db_conn.close.await_count == 2
    assert workflow.llm_manager.close.await_count == 2
    assert workflow._settings_provider.cleanup.await_count == 2
    # Resources that succeeded on the first pass are never closed twice.
    workflow._compact_checkpoints.assert_awaited_once()
    workflow.literature_agent.close_runtime_clients.assert_awaited_once()

    await workflow.close()
    assert workflow.sandbox.stop.await_count == 2
    assert db_conn.close.await_count == 2
    assert workflow.llm_manager.close.await_count == 2
    assert workflow._settings_provider.cleanup.await_count == 2


@pytest.mark.asyncio
async def test_workflow_close_finishes_other_resources_before_reraising_cancellation():
    workflow = _workflow_for_close()
    workflow.sandbox.stop.side_effect = [asyncio.CancelledError(), None]

    with pytest.raises(asyncio.CancelledError):
        await workflow.close()

    workflow._compact_checkpoints.assert_awaited_once()
    workflow.llm_manager.close.assert_awaited_once()
    workflow.literature_agent.close_runtime_clients.assert_awaited_once()
    workflow._settings_provider.cleanup.assert_awaited_once()
    assert workflow._close_complete is False

    await workflow.close()
    assert workflow._close_complete is True
    assert workflow.sandbox.stop.await_count == 2
    workflow.llm_manager.close.assert_awaited_once()
    workflow._compact_checkpoints.assert_awaited_once()
    workflow.literature_agent.close_runtime_clients.assert_awaited_once()
    workflow._settings_provider.cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_incomplete_registered_close_retries_and_releases_registry():
    from backend.models.schemas import TaskStatus
    from backend.state import workflows

    workflow = _workflow_for_close()
    state = _install_close_gate_state(workflow, TaskStatus.COMPLETED)
    workflow.sandbox.stop.side_effect = [RuntimeError("shutdown failed"), None]
    try:
        assert await workflow.close() is False
        retry_task = workflow._close_retry_task
        assert retry_task is not None
        await asyncio.wait_for(retry_task, timeout=2)
        assert workflow._close_complete is True
        assert workflows.get(workflow.session_id) is not workflow
    finally:
        _remove_close_gate_state(workflow)

    assert workflow.sandbox.stop.await_count == 2


@pytest.mark.asyncio
async def test_stale_workflow_retries_generic_close_without_touching_replacement():
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.state import sessions, workflows

    workflow = _workflow_for_close()
    workflow.session_id = "stale-close-retry"
    old_state = AnalysisState(
        id=workflow.session_id,
        user_id="u1",
        status=TaskStatus.COMPLETED,
    )
    replacement_state = AnalysisState(
        id=workflow.session_id,
        user_id="u1",
        status=TaskStatus.PAUSED,
    )
    replacement_workflow = object()
    workflow._current_analysis_state = old_state
    workflow.sandbox.stop.side_effect = [RuntimeError("shutdown failed"), None]
    sessions[workflow.session_id] = replacement_state
    workflows[workflow.session_id] = replacement_workflow
    try:
        assert await workflow.close() is False
        retry_task = workflow._close_retry_task
        assert retry_task is not None
        await asyncio.wait_for(retry_task, timeout=2)
        assert workflow._close_complete is True
        assert sessions[workflow.session_id] is replacement_state
        assert workflows[workflow.session_id] is replacement_workflow
    finally:
        if workflows.get(workflow.session_id) is replacement_workflow:
            workflows.pop(workflow.session_id, None)
        if sessions.get(workflow.session_id) is replacement_state:
            sessions.pop(workflow.session_id, None)

    assert [call.kwargs for call in workflow.sandbox.stop.await_args_list] == [
        {"save": True},
        {"save": False},
    ]
    assert workflow._restore_superseded is True


@pytest.mark.asyncio
async def test_teardown_false_retains_owner_tasks_and_checkpoint(monkeypatch):
    from backend.routes import _helpers
    from backend.state import _monitor_tasks, _run_tasks, workflows

    session_id = "teardown-incomplete"
    workflow = SimpleNamespace(close=AsyncMock(return_value=False))
    release = asyncio.Event()
    run_task = asyncio.create_task(release.wait())
    monitor_task = asyncio.create_task(release.wait())
    workflows[session_id] = workflow
    _run_tasks[session_id] = run_task
    _monitor_tasks[session_id] = monitor_task

    manager = MagicMock()
    monkeypatch.setattr("backend.utils.snapshot.SnapshotManager", manager)
    try:
        assert (
            await _helpers.teardown_running_workflow(
                session_id, discard_runtime=True,
            )
            is False
        )
        assert workflows.get(session_id) is workflow
        assert _run_tasks.get(session_id) is run_task
        assert _monitor_tasks.get(session_id) is monitor_task
        assert not run_task.done()
        assert not monitor_task.done()
        manager.assert_not_called()
    finally:
        release.set()
        await asyncio.gather(run_task, monitor_task)
        workflows.pop(session_id, None)
        _run_tasks.pop(session_id, None)
        _monitor_tasks.pop(session_id, None)


@pytest.mark.asyncio
async def test_teardown_timeout_retains_live_run_owner_and_checkpoint(monkeypatch):
    from backend.routes import _helpers
    from backend.state import _monitor_tasks, _run_tasks, workflows

    session_id = "teardown-run-timeout"
    release = asyncio.Event()

    async def ignore_first_cancellation():
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    run_task = asyncio.create_task(ignore_first_cancellation())
    workflows.pop(session_id, None)
    _run_tasks[session_id] = run_task
    manager = MagicMock()
    monkeypatch.setattr("backend.utils.snapshot.SnapshotManager", manager)

    async def immediate_timeout(awaitable, *, timeout):
        assert timeout == 5.0
        awaitable.cancel()
        raise asyncio.TimeoutError

    monkeypatch.setattr(_helpers.asyncio, "wait_for", immediate_timeout)
    try:
        assert await _helpers.teardown_running_workflow(session_id) is False
        assert _run_tasks.get(session_id) is run_task
        assert not run_task.done()
        manager.assert_not_called()
    finally:
        release.set()
        await asyncio.gather(run_task, return_exceptions=True)
        _run_tasks.pop(session_id, None)
        _monitor_tasks.pop(session_id, None)
