from types import SimpleNamespace

import pytest

from backend.models.schemas import AnalysisState, TaskStatus
from backend.state import (
    _run_tasks,
    detailed_report_tasks,
    initializing_sessions,
    initializing_session_owners,
    report_workflows,
    sessions,
    workflows,
)


class _FakeWorkflow:
    instances = []

    def __init__(self, config, session_id, message_callback=None, state_callback=None):
        self.config = config
        self.session_id = session_id
        self.message_callback = message_callback
        self.state_callback = state_callback
        self.run_calls = []
        self.resume_calls = []
        self.close_calls = 0
        _FakeWorkflow.instances.append(self)

    async def run(self, initial_state, thread_id):
        self.run_calls.append((initial_state, thread_id))
        initial_state.status = TaskStatus.COMPLETED
        return initial_state

    async def resume(self, thread_id, analysis_state):
        self.resume_calls.append((thread_id, analysis_state))
        analysis_state.status = TaskStatus.COMPLETED
        return analysis_state

    async def close(self):
        self.close_calls += 1


@pytest.fixture(autouse=True)
def _clear_workflow_globals(monkeypatch):
    from backend.services import workflow_runner
    from backend.services.workflow_runner import _report_tasks

    monkeypatch.setattr(
        workflow_runner, "assert_generation_writable", lambda *_args: None,
    )

    sessions.clear()
    workflows.clear()
    report_workflows.clear()
    initializing_sessions.clear()
    initializing_session_owners.clear()
    _run_tasks.clear()
    _report_tasks.clear()
    detailed_report_tasks.clear()
    _FakeWorkflow.instances.clear()
    yield
    sessions.clear()
    workflows.clear()
    report_workflows.clear()
    initializing_sessions.clear()
    initializing_session_owners.clear()
    _run_tasks.clear()
    _report_tasks.clear()
    detailed_report_tasks.clear()
    _FakeWorkflow.instances.clear()


@pytest.mark.asyncio
async def test_headless_runner_calls_run_with_state_and_thread_id(monkeypatch):
    from backend.services import workflow_runner
    import backend.workflow as workflow_module

    state = AnalysisState(id="s", user_id="u", status=TaskStatus.RUNNING, current_round=0)
    sessions["s"] = state

    async def _fake_resolve_config_for_user(user_id: str):
        return SimpleNamespace(max_rounds=3)

    async def _fake_save_session(session_id: str, state: AnalysisState):
        return None

    async def _fake_update_session_cache(session_id: str, state: AnalysisState):
        sessions[session_id] = state

    monkeypatch.setattr(workflow_module, "SingleCellAnalysisWorkflow", _FakeWorkflow)
    monkeypatch.setattr(workflow_runner, "resolve_config_for_user", _fake_resolve_config_for_user)
    monkeypatch.setattr(workflow_runner, "async_save_session", _fake_save_session)
    monkeypatch.setattr(workflow_runner, "_update_session_cache", _fake_update_session_cache)

    await workflow_runner._headless_workflow_runner("s")

    workflow = _FakeWorkflow.instances[0]
    assert workflow.run_calls == [(state, "s")]
    assert workflow.resume_calls == []
    assert workflow.close_calls == 1
    assert workflows == {}


@pytest.mark.asyncio
async def test_headless_runner_resumes_existing_round_state(monkeypatch):
    from backend.services import workflow_runner
    import backend.workflow as workflow_module

    state = AnalysisState(id="s", user_id="u", status=TaskStatus.RUNNING, current_round=3)
    sessions["s"] = state

    async def _fake_resolve_config_for_user(user_id: str):
        return SimpleNamespace(max_rounds=4)

    async def _fake_save_session(session_id: str, state: AnalysisState):
        return None

    async def _fake_update_session_cache(session_id: str, state: AnalysisState):
        sessions[session_id] = state

    monkeypatch.setattr(workflow_module, "SingleCellAnalysisWorkflow", _FakeWorkflow)
    monkeypatch.setattr(workflow_runner, "resolve_config_for_user", _fake_resolve_config_for_user)
    monkeypatch.setattr(workflow_runner, "async_save_session", _fake_save_session)
    monkeypatch.setattr(workflow_runner, "_update_session_cache", _fake_update_session_cache)

    await workflow_runner._headless_workflow_runner("s")

    workflow = _FakeWorkflow.instances[0]
    assert workflow.run_calls == []
    assert workflow.resume_calls == [("s", state)]
    assert workflow.close_calls == 1
    assert workflows == {}


@pytest.mark.asyncio
async def test_headless_start_refuses_parallel_report_owner():
    from backend.services import workflow_runner

    report_workflows["s"] = object()
    await workflow_runner.start_workflow_if_needed("s")

    assert "s" not in initializing_sessions
    assert "s" not in _run_tasks


@pytest.mark.asyncio
async def test_old_report_done_callback_cannot_remove_replacement(monkeypatch):
    import asyncio

    from backend.services import workflow_runner

    release_old = asyncio.Event()
    release_new = asyncio.Event()

    async def _old_report(_session_id):
        await release_old.wait()

    monkeypatch.setattr(workflow_runner, "_run_report_generation", _old_report)
    sessions["s"] = AnalysisState(id="s", user_id="u")
    await workflow_runner.start_report_generation("s")
    old_task = workflow_runner._report_tasks["s"]
    replacement = asyncio.create_task(release_new.wait())
    workflow_runner._report_tasks["s"] = replacement
    try:
        release_old.set()
        await old_task
        await asyncio.sleep(0)
        assert workflow_runner._report_tasks.get("s") is replacement
    finally:
        release_new.set()
        await replacement
        if workflow_runner._report_tasks.get("s") is replacement:
            workflow_runner._report_tasks.pop("s", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_kind", ["workflow", "run_task"])
async def test_standalone_report_refuses_live_analysis_owner(owner_kind):
    import asyncio

    from backend.services import workflow_runner

    sessions["s"] = AnalysisState(id="s", user_id="u")
    release = asyncio.Event()
    task = None
    if owner_kind == "workflow":
        workflows["s"] = object()
    else:
        task = asyncio.create_task(release.wait())
        _run_tasks["s"] = task
    try:
        with pytest.raises(workflow_runner.SessionWorkflowConflictError):
            await workflow_runner.start_report_generation("s")
        assert "s" not in workflow_runner._report_tasks
        assert "s" not in report_workflows
    finally:
        workflows.pop("s", None)
        release.set()
        if task is not None:
            await task
            _run_tasks.pop("s", None)


@pytest.mark.asyncio
async def test_detailed_report_owner_blocks_both_workflow_entrypoints():
    import asyncio

    from backend.services import workflow_runner

    sessions["s"] = AnalysisState(id="s", user_id="u")
    release = asyncio.Event()
    detailed = asyncio.create_task(release.wait())
    detailed_report_tasks["s"] = detailed
    try:
        with pytest.raises(workflow_runner.SessionWorkflowConflictError):
            await workflow_runner.start_report_generation("s")
        await workflow_runner.start_workflow_if_needed("s")
        assert "s" not in initializing_sessions
        assert "s" not in _run_tasks
        assert "s" not in workflow_runner._report_tasks
    finally:
        release.set()
        await detailed
        detailed_report_tasks.pop("s", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("safe_point_state", ["pending", "acquired"])
async def test_snapshot_safe_point_blocks_both_workflow_entrypoints(
    safe_point_state,
):
    from backend.services import workflow_runner
    from backend.services.snapshot_v2.safe_point import (
        get_snapshot_safe_point_manager,
    )

    state = AnalysisState(id="s", user_id="u")
    sessions["s"] = state
    safe_point = get_snapshot_safe_point_manager()
    owner_token = f"test:{safe_point_state}"
    if safe_point_state == "pending":
        assert await safe_point.request_safe_point("s", owner_token)
    else:
        assert await safe_point.acquire(
            "s", {"status": "capture"}, owner_token=owner_token,
        )

    prepared = False

    async def _prepare():
        nonlocal prepared
        prepared = True

    try:
        with pytest.raises(workflow_runner.SessionWorkflowConflictError):
            await workflow_runner.start_report_generation("s")
        with pytest.raises(workflow_runner.SessionWorkflowConflictError):
            await workflow_runner.start_workflow_if_needed(
                "s", expected_state=state, prepare=_prepare,
            )
        assert prepared is False
        assert "s" not in initializing_sessions
        assert "s" not in _run_tasks
        assert "s" not in workflow_runner._report_tasks
        assert "s" not in report_workflows
        assert "s" not in workflows
    finally:
        if safe_point_state == "pending":
            await safe_point.cancel_safe_point_request("s", owner_token)
        else:
            await safe_point.release("s", owner_token)


@pytest.mark.asyncio
async def test_standalone_report_reservation_blocks_headless_start(monkeypatch):
    import asyncio

    from backend.services import workflow_runner

    state = AnalysisState(id="s", user_id="u")
    sessions["s"] = state
    prepare_started = asyncio.Event()
    release_prepare = asyncio.Event()
    report_started = asyncio.Event()
    release_report = asyncio.Event()

    async def _prepare():
        prepare_started.set()
        await release_prepare.wait()

    async def _report(_session_id):
        report_started.set()
        await release_report.wait()

    monkeypatch.setattr(workflow_runner, "_run_report_generation", _report)
    starter = asyncio.create_task(
        workflow_runner.start_report_generation(
            "s", expected_state=state, prepare=_prepare,
        )
    )
    try:
        await prepare_started.wait()
        await workflow_runner.start_workflow_if_needed("s")
        assert "s" not in initializing_sessions
        assert "s" not in _run_tasks

        release_prepare.set()
        assert await starter is True
        await report_started.wait()
        assert "s" in workflow_runner._report_tasks
    finally:
        release_prepare.set()
        release_report.set()
        await asyncio.gather(starter, return_exceptions=True)
        task = workflow_runner._report_tasks.get("s")
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_request_cancellation_during_prepare_keeps_registered_owner(
    monkeypatch,
):
    import asyncio

    from backend.services import workflow_runner

    state = AnalysisState(id="s", user_id="u")
    sessions["s"] = state
    prepare_started = asyncio.Event()
    release_prepare = asyncio.Event()
    prepare_finished = asyncio.Event()
    report_started = asyncio.Event()
    release_report = asyncio.Event()

    async def _prepare():
        prepare_started.set()
        await release_prepare.wait()
        prepare_finished.set()

    async def _report(_session_id):
        report_started.set()
        await release_report.wait()

    monkeypatch.setattr(workflow_runner, "_run_report_generation", _report)
    request_owner = asyncio.create_task(
        workflow_runner.start_report_generation(
            "s", expected_state=state, prepare=_prepare,
        )
    )
    try:
        await prepare_started.wait()
        registered = workflow_runner._report_tasks["s"]
        request_owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request_owner

        assert workflow_runner._report_tasks.get("s") is registered
        assert not registered.done()
        release_prepare.set()
        await prepare_finished.wait()
        await report_started.wait()
        assert not registered.done()
    finally:
        release_prepare.set()
        release_report.set()
        await asyncio.gather(request_owner, return_exceptions=True)
        registered = workflow_runner._report_tasks.get("s")
        if registered is not None:
            await asyncio.gather(registered, return_exceptions=True)


@pytest.mark.asyncio
async def test_old_headless_finally_cannot_clear_replacement_initializing_owner(
    monkeypatch,
):
    import asyncio

    from backend.services import workflow_runner
    from backend.state import (
        claim_initializing_session,
        initializing_session_owners,
    )

    build_started = asyncio.Event()
    release_build = asyncio.Event()

    async def _delayed_empty_build(_session_id):
        build_started.set()
        await release_build.wait()
        return None, None

    monkeypatch.setattr(
        workflow_runner, "_build_workflow_for_session", _delayed_empty_build,
    )
    old_token = claim_initializing_session("s")
    old_runner = asyncio.create_task(
        workflow_runner._headless_workflow_runner(
            "s", initializing_token=old_token,
        )
    )
    try:
        await build_started.wait()
        new_token = claim_initializing_session("s")
        release_build.set()
        await old_runner

        assert "s" in initializing_sessions
        assert initializing_session_owners.get("s") is new_token
    finally:
        release_build.set()
        await asyncio.gather(old_runner, return_exceptions=True)


@pytest.mark.asyncio
async def test_websocket_create_gate_observes_report_task_before_owner_registers():
    import asyncio

    from backend.routes.websockets import _session_workflow_activity
    from backend.services import workflow_runner

    release = asyncio.Event()
    report_task = asyncio.create_task(release.wait())
    workflow_runner._report_tasks["s"] = report_task
    try:
        assert _session_workflow_activity("s") == (
            False,
            True,
            True,
            False,
        )
        assert "s" not in initializing_sessions
    finally:
        release.set()
        await report_task
        if workflow_runner._report_tasks.get("s") is report_task:
            workflow_runner._report_tasks.pop("s", None)
