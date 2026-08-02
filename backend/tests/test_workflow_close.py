from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import asyncio
import pytest

from backend.workflow.core import SingleCellAnalysisWorkflow
from backend.services.runtime_persistence_gate import (
    RuntimePersistenceShutdownBlocked,
)
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
    workflow._runtime_persistence_event = asyncio.Event()
    workflow._runtime_persistence_gate_active = False
    workflow._runtime_persistence_action = None
    workflow._runtime_persistence_action_pending = False
    workflow._runtime_persistence_gate_token = ""
    workflow._runtime_persistence_gate_generation = 0
    workflow._restore_superseded = False
    workflow.sandbox = SimpleNamespace(stop=AsyncMock())
    workflow._db_conn = SimpleNamespace(close=AsyncMock())
    workflow._compact_checkpoints = AsyncMock()
    workflow.llm_manager = SimpleNamespace(close=AsyncMock())
    workflow.literature_agent = SimpleNamespace(close_runtime_clients=AsyncMock())
    workflow._settings_provider = SimpleNamespace(cleanup=AsyncMock())
    return workflow


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
@pytest.mark.parametrize("failure", [False, RuntimeError("serialization failed")])
async def test_kernel_stop_preserves_live_kernel_when_final_save_fails(failure):
    sandbox = SimpleNamespace(
        session_id="preserve-live-kernel",
        await_pending_saves=AsyncMock(),
        save_session=AsyncMock(
            return_value=failure if failure is False else True,
            side_effect=failure if isinstance(failure, Exception) else None,
        ),
        notebook_state=SimpleNamespace(save=MagicMock()),
        kc=SimpleNamespace(stop_channels=MagicMock()),
        km=SimpleNamespace(shutdown_kernel=MagicMock()),
        _initialized=True,
    )
    original_kc = sandbox.kc
    original_km = sandbox.km

    with pytest.raises(RuntimePersistenceShutdownBlocked):
        await KernelMixin.stop(sandbox, save=True)

    assert sandbox.kc is original_kc
    assert sandbox.km is original_km
    assert sandbox._initialized is True
    sandbox.kc.stop_channels.assert_not_called()
    sandbox.km.shutdown_kernel.assert_not_called()


def _install_close_gate_state(workflow, status):
    from backend.models.schemas import AnalysisState
    from backend.state import sessions, workflows

    state = AnalysisState(
        session_id=workflow.session_id,
        user_id="u1",
        name="close persistence",
        status=status,
    )
    workflow._current_analysis_state = state
    workflow.state_callback = AsyncMock()
    workflow._send_state_sync = AsyncMock()
    sessions[workflow.session_id] = state
    workflows[workflow.session_id] = workflow
    return state


def _remove_close_gate_state(workflow):
    from backend.state import sessions, workflows

    if workflows.get(workflow.session_id) is workflow:
        workflows.pop(workflow.session_id, None)
    sessions.pop(workflow.session_id, None)


@pytest.mark.asyncio
async def test_workflow_close_retry_commits_before_kernel_shutdown():
    from backend.models.schemas import TaskStatus

    workflow = _workflow_for_close()
    state = _install_close_gate_state(workflow, TaskStatus.COMPLETED)
    workflow.sandbox.stop.side_effect = [
        RuntimePersistenceShutdownBlocked("save failed"),
        None,
    ]
    workflow._wait_for_runtime_persistence_intervention = AsyncMock(
        return_value="retry"
    )
    try:
        assert await workflow.close() is True
    finally:
        _remove_close_gate_state(workflow)

    assert [call.kwargs for call in workflow.sandbox.stop.await_args_list] == [
        {"save": True},
        {"save": True},
    ]
    assert state.status is TaskStatus.COMPLETED
    workflow._wait_for_runtime_persistence_intervention.assert_awaited_once_with(
        state,
        successful_status=TaskStatus.COMPLETED,
        persist_transitions=True,
        rotate_generation=False,
    )


@pytest.mark.asyncio
async def test_workflow_close_terminate_explicitly_discards_unsaved_kernel():
    from backend.models.schemas import TaskStatus

    workflow = _workflow_for_close()
    state = _install_close_gate_state(workflow, TaskStatus.COMPLETED)
    workflow.sandbox.stop.side_effect = [
        RuntimePersistenceShutdownBlocked("save failed"),
        None,
    ]
    workflow._wait_for_runtime_persistence_intervention = AsyncMock(
        return_value="terminate"
    )
    try:
        assert await workflow.close() is True
    finally:
        _remove_close_gate_state(workflow)

    assert [call.kwargs for call in workflow.sandbox.stop.await_args_list] == [
        {"save": True},
        {"save": False},
    ]
    assert state.status is TaskStatus.STOPPED
    assert state.termination_reason == "runtime_persistence_failure"
    workflow.state_callback.assert_awaited()


@pytest.mark.asyncio
async def test_workflow_close_manage_explicitly_discards_unsaved_kernel():
    from backend.models.schemas import TaskStatus

    workflow = _workflow_for_close()
    state = _install_close_gate_state(workflow, TaskStatus.COMPLETED)
    workflow.sandbox.stop.side_effect = [
        RuntimePersistenceShutdownBlocked("save failed"),
        None,
    ]
    workflow._wait_for_runtime_persistence_intervention = AsyncMock(
        return_value="recompute"
    )
    try:
        assert await workflow.close() is True
    finally:
        _remove_close_gate_state(workflow)

    assert [call.kwargs for call in workflow.sandbox.stop.await_args_list] == [
        {"save": True},
        {"save": False},
    ]
    assert state.status is TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_close_gate_discard_flag_wins_race_with_gate_initialization():
    from backend.models.schemas import AnalysisState, TaskStatus

    workflow = _workflow_for_close()
    state = AnalysisState(
        id="close-discard-race",
        user_id="u1",
        status=TaskStatus.COMPLETED,
    )
    workflow.session_id = state.id
    workflow._discard_runtime_on_close = False
    workflow._runtime_persistence_event = asyncio.Event()
    workflow._runtime_persistence_gate_active = False
    workflow._runtime_persistence_action = None
    workflow._runtime_persistence_action_pending = False

    def _sync(target):
        workflow._discard_runtime_on_close = True
        payload = {"status": "failed", "current_revision": 2}
        target.runtime_persistence = dict(payload)
        return payload

    workflow.sandbox.sync_runtime_persistence_state = MagicMock(
        side_effect=_sync
    )

    assert (
        await workflow._wait_for_runtime_persistence_intervention(state)
        == "discard"
    )
    assert state.snapshot_failure is None
    assert state.wait_input_reason is None
    assert workflow._runtime_persistence_gate_active is False


@pytest.mark.asyncio
async def test_discard_interrupts_close_gate_state_publish_without_deadlock():
    from backend.models.schemas import TaskStatus

    workflow = _workflow_for_close()
    state = _install_close_gate_state(workflow, TaskStatus.COMPLETED)
    workflow._runtime_persistence_event = asyncio.Event()
    workflow._runtime_persistence_gate_active = False
    workflow._runtime_persistence_action = None
    workflow._runtime_persistence_action_pending = False
    workflow._discard_runtime_on_close = False
    workflow._close_discard_event = asyncio.Event()
    workflow.sandbox.stop.side_effect = [
        RuntimePersistenceShutdownBlocked("save failed"),
        None,
    ]
    workflow.sandbox.sync_runtime_persistence_state = MagicMock(
        side_effect=lambda target: {
            "status": "failed",
            "current_revision": 2,
        }
    )
    publish_started = asyncio.Event()
    publish_cancelled = asyncio.Event()

    async def _blocked_state_callback(_state):
        publish_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            publish_cancelled.set()
            raise

    workflow.state_callback = _blocked_state_callback
    workflow._send_state_sync = AsyncMock()
    try:
        first_close = asyncio.create_task(workflow.close())
        await asyncio.wait_for(publish_started.wait(), timeout=1)
        discard_close = asyncio.create_task(
            workflow.close(save_runtime=False)
        )
        assert await asyncio.wait_for(discard_close, timeout=1) is True
        assert await asyncio.wait_for(first_close, timeout=1) is True
    finally:
        _remove_close_gate_state(workflow)

    assert publish_cancelled.is_set()
    assert workflow._close_complete is True
    assert [call.kwargs for call in workflow.sandbox.stop.await_args_list] == [
        {"save": True},
        {"save": False},
    ]


@pytest.mark.asyncio
async def test_close_safe_point_supersession_finishes_with_no_save_shutdown():
    from backend.models.schemas import TaskStatus

    workflow = _workflow_for_close()
    state = _install_close_gate_state(workflow, TaskStatus.COMPLETED)
    workflow.sandbox.stop.side_effect = [
        RuntimePersistenceShutdownBlocked("save failed"),
        None,
    ]
    workflow._wait_for_runtime_persistence_intervention = AsyncMock(
        side_effect=WorkflowSupersededError("restored generation installed")
    )
    try:
        assert await workflow.close() is True
    finally:
        _remove_close_gate_state(workflow)

    assert [call.kwargs for call in workflow.sandbox.stop.await_args_list] == [
        {"save": True},
        {"save": False},
    ]
    assert workflow._restore_superseded is True


@pytest.mark.asyncio
async def test_cancelled_close_gate_retries_without_orphaning_live_kernel():
    from backend.models.schemas import TaskStatus

    workflow = _workflow_for_close()
    state = _install_close_gate_state(workflow, TaskStatus.COMPLETED)
    workflow.sandbox.stop.side_effect = [
        RuntimePersistenceShutdownBlocked("save failed"),
        None,
    ]
    workflow._wait_for_runtime_persistence_intervention = AsyncMock(
        side_effect=asyncio.CancelledError()
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await workflow.close()
        retry_task = workflow._close_retry_task
        assert retry_task is not None
        await asyncio.wait_for(retry_task, timeout=2)
        assert workflow._close_complete is True
    finally:
        _remove_close_gate_state(workflow)

    assert [call.kwargs for call in workflow.sandbox.stop.await_args_list] == [
        {"save": True},
        {"save": True},
    ]
    workflow._compact_checkpoints.assert_awaited_once()
    workflow.llm_manager.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_real_close_gate_defers_owned_task_cancellation_until_gate_clears():
    from backend.models.schemas import TaskStatus
    from backend.services.runtime_persistence_gate import (
        runtime_persistence_gate_identity,
    )
    from backend.state import workflows

    workflow = _workflow_for_close()
    state = _install_close_gate_state(workflow, TaskStatus.COMPLETED)
    workflow.sandbox.stop.side_effect = [
        RuntimePersistenceShutdownBlocked("save failed"),
        None,
    ]
    workflow.sandbox.save_session = AsyncMock(return_value=True)
    persistence_states = iter([
        {"status": "failed", "current_revision": 2},
        {"status": "committed", "committed_revision": 3},
    ])

    def _sync_runtime(target):
        payload = next(persistence_states)
        target.runtime_persistence = dict(payload)
        return payload

    workflow.sandbox.sync_runtime_persistence_state = MagicMock(
        side_effect=_sync_runtime
    )
    persisted = []

    async def _persist(current):
        persisted.append({
            "status": current.status,
            "snapshot_failure": current.snapshot_failure,
            "wait_input_reason": current.wait_input_reason,
            "runtime_persistence": dict(current.runtime_persistence or {}),
        })
        return True

    workflow.state_callback = _persist
    caller = asyncio.create_task(workflow.close())
    try:
        for _ in range(100):
            if workflow._runtime_persistence_gate_active and state.snapshot_failure:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("real final-save gate was not armed")

        owned = workflow._close_task
        assert owned is not None
        owned.cancel()
        token, generation = runtime_persistence_gate_identity(state)
        assert workflow.signal_runtime_persistence_action(
            "retry", gate_token=token, generation=generation,
        ).accepted

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(caller, timeout=2)

        assert workflow._close_complete is True
        assert workflows.get(workflow.session_id) is not workflow
        assert state.snapshot_failure is None
        assert state.wait_input_reason is None
        assert state.status is TaskStatus.COMPLETED
        assert state.runtime_persistence["status"] == "committed"
        assert persisted[-1]["snapshot_failure"] is None
        assert persisted[-1]["status"] is TaskStatus.COMPLETED
        assert [call.kwargs for call in workflow.sandbox.stop.await_args_list] == [
            {"save": True},
            {"save": True},
        ]
    finally:
        _remove_close_gate_state(workflow)


@pytest.mark.asyncio
async def test_close_keeps_kernel_until_cleared_gate_is_durable():
    from backend.models.schemas import TaskStatus
    from backend.services.runtime_persistence_gate import (
        runtime_persistence_gate_identity,
    )
    from backend.state import workflows

    workflow = _workflow_for_close()
    state = _install_close_gate_state(workflow, TaskStatus.COMPLETED)
    workflow.sandbox.stop.side_effect = [
        RuntimePersistenceShutdownBlocked("save failed"),
        None,
    ]
    workflow.sandbox.save_session = AsyncMock(return_value=True)
    persistence_states = iter([
        {"status": "failed", "current_revision": 2},
        {"status": "committed", "committed_revision": 3},
    ])

    def _sync_runtime(target):
        payload = next(persistence_states)
        target.runtime_persistence = dict(payload)
        return payload

    workflow.sandbox.sync_runtime_persistence_state = MagicMock(
        side_effect=_sync_runtime
    )
    first_clear_failed = asyncio.Event()
    retry_started = asyncio.Event()
    allow_retry = asyncio.Event()
    durable = {}
    clear_attempts = 0

    async def _persist(current):
        nonlocal clear_attempts
        snapshot = {
            "status": current.status,
            "snapshot_failure": current.snapshot_failure,
            "wait_input_reason": current.wait_input_reason,
        }
        if current.snapshot_failure is None:
            clear_attempts += 1
            if clear_attempts == 1:
                first_clear_failed.set()
                raise OSError("metadata write failed")
            retry_started.set()
            await allow_retry.wait()
        durable.clear()
        durable.update(snapshot)
        return True

    workflow.state_callback = _persist
    caller = asyncio.create_task(workflow.close())
    try:
        for _ in range(100):
            if workflow._runtime_persistence_gate_active and state.snapshot_failure:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("real final-save gate was not armed")
        token, generation = runtime_persistence_gate_identity(state)
        assert workflow.signal_runtime_persistence_action(
            "retry", gate_token=token, generation=generation,
        ).accepted

        await asyncio.wait_for(first_clear_failed.wait(), timeout=1)
        await asyncio.wait_for(retry_started.wait(), timeout=1)
        assert workflow.sandbox.stop.await_count == 1
        assert workflows.get(workflow.session_id) is workflow
        assert durable["snapshot_failure"] is not None
        assert not caller.done()

        allow_retry.set()
        assert await asyncio.wait_for(caller, timeout=2) is True
        assert workflow._close_complete is True
        assert workflow.sandbox.stop.await_count == 2
        assert durable["snapshot_failure"] is None
        assert durable["wait_input_reason"] is None
        assert durable["status"] is TaskStatus.COMPLETED
    finally:
        allow_retry.set()
        _remove_close_gate_state(workflow)


@pytest.mark.asyncio
async def test_tombstoned_report_close_never_republishes_stale_workflow():
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.state import sessions, workflows

    workflow = _workflow_for_close()
    workflow.session_id = "tombstoned-report-close"
    state = AnalysisState(
        id=workflow.session_id,
        user_id="u1",
        status=TaskStatus.COMPLETED,
    )
    workflow._current_analysis_state = state
    workflow.sandbox.stop.side_effect = [
        RuntimePersistenceShutdownBlocked("save failed"),
        None,
    ]
    sessions.pop(workflow.session_id, None)
    workflows.pop(workflow.session_id, None)

    assert await workflow.close() is True

    assert workflow.session_id not in workflows
    assert [call.kwargs for call in workflow.sandbox.stop.await_args_list] == [
        {"save": True},
        {"save": False},
    ]


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
async def test_report_only_close_temporarily_registers_its_live_failure_gate():
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.state import sessions, workflows

    workflow = _workflow_for_close()
    workflow.session_id = "report-close-persistence"
    state = AnalysisState(
        session_id=workflow.session_id,
        user_id="u1",
        name="report close",
        status=TaskStatus.COMPLETED,
    )
    workflow._current_analysis_state = state
    workflow.state_callback = AsyncMock()
    workflow._send_state_sync = AsyncMock()
    sessions[workflow.session_id] = state
    workflows.pop(workflow.session_id, None)
    workflow.sandbox.stop.side_effect = [
        RuntimePersistenceShutdownBlocked("save failed"),
        None,
    ]

    async def _observe_registration(*_args, **_kwargs):
        assert workflows.get(workflow.session_id) is workflow
        return "retry"

    workflow._wait_for_runtime_persistence_intervention = AsyncMock(
        side_effect=_observe_registration
    )
    try:
        assert await workflow.close() is True
        assert workflow.session_id not in workflows
    finally:
        if workflows.get(workflow.session_id) is workflow:
            workflows.pop(workflow.session_id, None)
        sessions.pop(workflow.session_id, None)
