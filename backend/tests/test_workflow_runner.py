import asyncio
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

    async def _noop_persist(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        workflow_runner, "assert_generation_writable", lambda *_args: None,
    )
    monkeypatch.setattr(
        workflow_runner,
        "_async_save_session_under_lifecycle",
        _noop_persist,
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
async def test_headless_runner_runs_when_it_holds_the_reservation(monkeypatch):
    from backend.services import workflow_runner
    from backend.state import claim_initializing_session
    import backend.workflow as workflow_module

    state = AnalysisState(id="reserved", user_id="u", status=TaskStatus.RUNNING)
    sessions[state.id] = state

    async def _resolve(_user_id):
        return SimpleNamespace(max_rounds=3)

    async def _save(_session_id, _state):
        return None

    async def _update(session_id, current):
        sessions[session_id] = current

    monkeypatch.setattr(workflow_module, "SingleCellAnalysisWorkflow", _FakeWorkflow)
    monkeypatch.setattr(workflow_runner, "resolve_config_for_user", _resolve)
    monkeypatch.setattr(
        workflow_runner, "_async_save_session_under_lifecycle", _save,
    )
    monkeypatch.setattr(workflow_runner, "_update_session_cache", _update)

    token = claim_initializing_session(state.id)
    _run_tasks[state.id] = asyncio.current_task()
    await workflow_runner._headless_workflow_runner(
        state.id, initializing_token=token,
    )

    workflow = _FakeWorkflow.instances[0]
    assert workflow.run_calls == [(state, state.id)]
    assert state.status is TaskStatus.COMPLETED
    assert workflow.close_calls == 1
    assert workflows == {}
    assert state.id not in _run_tasks
    assert state.id not in initializing_sessions


@pytest.mark.asyncio
async def test_headless_runner_registers_before_first_state_sync(monkeypatch):
    from backend.services import workflow_runner
    import backend.workflow as workflow_module

    session_id = "headless-registration"
    state = AnalysisState(id=session_id, user_id="u", status=TaskStatus.RUNNING)
    observed = {}
    saved = []

    class _FirstNodeWorkflow:
        def __init__(self, config, session_id, message_callback=None, state_callback=None):
            self.state_callback = state_callback

        async def run(self, initial_state, thread_id):
            observed["registered"] = workflows.get(thread_id) is self
            observed["written"] = await self.state_callback(
                AnalysisState(
                    id=thread_id,
                    user_id="u",
                    status=TaskStatus.RUNNING,
                    current_round=1,
                )
            )
            return initial_state

        async def close(self):
            return None

    async def _resolve(_user_id):
        return SimpleNamespace(max_rounds=3)

    async def _save(_session_id, current):
        saved.append(current)

    async def _update(session, current):
        sessions[session] = current

    monkeypatch.setattr(workflow_module, "SingleCellAnalysisWorkflow", _FirstNodeWorkflow)
    monkeypatch.setattr(workflow_runner, "resolve_config_for_user", _resolve)
    monkeypatch.setattr(
        workflow_runner, "_async_save_session_under_lifecycle", _save,
    )
    monkeypatch.setattr(workflow_runner, "_update_session_cache", _update)
    sessions[session_id] = state

    await workflow_runner._headless_workflow_runner(session_id)

    assert observed == {"registered": True, "written": True}
    assert saved and saved[0].current_round == 1


@pytest.mark.asyncio
async def test_headless_state_callback_accepts_graph_state_swap(monkeypatch):
    from backend.services import workflow_runner
    import backend.workflow as workflow_module

    session_id = "headless-graph-swap"
    original = AnalysisState(id=session_id, user_id="u", status=TaskStatus.RUNNING)
    replacement = AnalysisState(
        id=session_id, user_id="u", status=TaskStatus.WAITING_INPUT,
    )
    saved = []

    class _GraphSwapWorkflow:
        def __init__(self, config, session_id, message_callback=None, state_callback=None):
            self.state_callback = state_callback

        async def run(self, initial_state, thread_id):
            graph_state = AnalysisState(
                id=thread_id,
                user_id="u",
                status=TaskStatus.RUNNING,
                current_round=4,
            )
            assert graph_state is not initial_state
            assert await self.state_callback(graph_state) is True

            # Exercise the commit-point recheck: a replacement that publishes
            # while history is being carried must not be overwritten by the
            # stale graph owner.
            from backend.routes import websockets as websocket_routes

            real_preserve = websocket_routes._preserve_realtime_stream

            def _replace_during_carry(previous, incoming):
                real_preserve(previous, incoming)
                workflows[thread_id] = object()
                sessions[thread_id] = replacement

            monkeypatch.setattr(
                websocket_routes,
                "_preserve_realtime_stream",
                _replace_during_carry,
            )
            try:
                assert await self.state_callback(
                    AnalysisState(
                        id=thread_id,
                        user_id="u",
                        status=TaskStatus.RUNNING,
                        current_round=5,
                    )
                ) is False
            finally:
                monkeypatch.setattr(
                    websocket_routes,
                    "_preserve_realtime_stream",
                    real_preserve,
                )
            return graph_state

        async def close(self):
            return None

    async def _resolve(_user_id):
        return SimpleNamespace(max_rounds=5)

    async def _save(_session_id, current):
        saved.append(current)

    async def _update(session, current):
        sessions[session] = current

    monkeypatch.setattr(workflow_module, "SingleCellAnalysisWorkflow", _GraphSwapWorkflow)
    monkeypatch.setattr(workflow_runner, "resolve_config_for_user", _resolve)
    monkeypatch.setattr(
        workflow_runner, "_async_save_session_under_lifecycle", _save,
    )
    monkeypatch.setattr(workflow_runner, "_update_session_cache", _update)
    sessions[session_id] = original

    await workflow_runner._headless_workflow_runner(session_id)

    assert sessions[session_id] is replacement
    assert len(saved) == 1
    assert saved[0].current_round == 4


@pytest.mark.asyncio
async def test_headless_state_callback_rejects_superseded_workflow(monkeypatch):
    from backend.services import workflow_runner
    import backend.workflow as workflow_module

    session_id = "headless-superseded"
    original = AnalysisState(id=session_id, user_id="u", status=TaskStatus.RUNNING)
    restored = AnalysisState(
        id=session_id, user_id="u", status=TaskStatus.WAITING_INPUT,
    )
    saved = []

    class _SupersededWorkflow:
        def __init__(self, config, session_id, message_callback=None, state_callback=None):
            self.state_callback = state_callback

        async def run(self, initial_state, thread_id):
            workflows[thread_id] = object()
            sessions[thread_id] = restored
            assert await self.state_callback(initial_state) is False
            raise RuntimeError("superseded workflow failure")

        async def close(self):
            return None

    async def _resolve(_user_id):
        return SimpleNamespace(max_rounds=3)

    async def _save(_session_id, current):
        saved.append(current)

    async def _update(session, current):
        sessions[session] = current

    monkeypatch.setattr(workflow_module, "SingleCellAnalysisWorkflow", _SupersededWorkflow)
    monkeypatch.setattr(workflow_runner, "resolve_config_for_user", _resolve)
    monkeypatch.setattr(workflow_runner, "async_save_session", _save)
    monkeypatch.setattr(workflow_runner, "_update_session_cache", _update)
    sessions[session_id] = original

    await workflow_runner._headless_workflow_runner(session_id)

    assert sessions[session_id] is restored
    assert restored.status is TaskStatus.WAITING_INPUT
    assert saved == []


async def _capture_headless_callbacks(
    monkeypatch, state, *, owner_registry=None,
):
    from backend.services import workflow_runner
    import backend.workflow as workflow_module

    captured = {}

    class _CapturingWorkflow:
        def __init__(self, config, session_id, message_callback=None, state_callback=None):
            captured["message_callback"] = message_callback
            captured["state_callback"] = state_callback

    async def _resolve(_user_id):
        return SimpleNamespace(max_rounds=3)

    async def _save(_session_id, _state):
        return None

    async def _update(session_id, current):
        sessions[session_id] = current

    monkeypatch.setattr(workflow_module, "SingleCellAnalysisWorkflow", _CapturingWorkflow)
    monkeypatch.setattr(workflow_runner, "resolve_config_for_user", _resolve)
    monkeypatch.setattr(workflow_runner, "async_save_session", _save)
    monkeypatch.setattr(workflow_runner, "_update_session_cache", _update)
    sessions[state.id] = state
    workflow, built = await workflow_runner._build_workflow_for_session(
        state.id, owner_registry=owner_registry,
    )
    assert built is state
    captured["workflow"] = workflow
    if owner_registry is not None:
        owner_registry[state.id] = workflow
    return captured


@pytest.mark.asyncio
async def test_headless_message_callback_delivers_live_and_persists_sanitized(monkeypatch):
    from backend.constants import STRIPPED_FIGURE_PLACEHOLDER
    from backend.models.schemas import WebSocketMessage
    from backend.services import workflow_runner
    import backend.workflow as workflow_module

    state = AnalysisState(id="headless-payload", user_id="owner")
    saved = []
    live = []

    class Workflow:
        def __init__(self, config, session_id, message_callback=None, state_callback=None):
            self.message_callback = message_callback
            self.state_callback = state_callback

    async def resolve(_user_id):
        return SimpleNamespace(max_rounds=3)

    async def save(_session_id, current):
        saved.append(current)

    monkeypatch.setattr(workflow_module, "SingleCellAnalysisWorkflow", Workflow)
    monkeypatch.setattr(workflow_runner, "resolve_config_for_user", resolve)
    monkeypatch.setattr(workflow_runner, "_async_save_session_under_lifecycle", save)
    monkeypatch.setattr(workflow_runner, "adopt_persisted_generation", lambda _state: None)
    from backend.utils import websocket_delivery

    async def deliver(_session_id, payload):
        live.append(payload)
        return True

    monkeypatch.setattr(websocket_delivery, "send_analysis_json", deliver)

    sessions[state.id] = state
    workflow, _ = await workflow_runner._build_workflow_for_session(
        state.id, owner_registry=workflows,
    )
    workflows[state.id] = workflow
    message = WebSocketMessage(
        type="figure",
        content={"image": "iVBORw0KGgo" + ("A" * 2048)},
    )
    assert await workflow.message_callback(message) is True
    assert state.realtime_messages[0].content["image"] == STRIPPED_FIGURE_PLACEHOLDER
    assert message.content["image"].startswith("iVBORw0KGgo")
    assert live and live[0]["content"]["image"].startswith("iVBORw0KGgo")

    updated = AnalysisState(id=state.id, user_id=state.user_id)
    assert await workflow.state_callback(updated) is True
    assert saved and saved[-1] is updated
    assert saved[-1].realtime_messages[0].content["image"] == STRIPPED_FIGURE_PLACEHOLDER


@pytest.mark.asyncio
async def test_headless_message_callback_is_async_bounded_and_binary_free(monkeypatch):
    from backend.constants import STRIPPED_FIGURE_PLACEHOLDER
    from backend.models.schemas import WebSocketMessage

    state = AnalysisState(id="headless-messages", user_id="u")
    captured = await _capture_headless_callbacks(
        monkeypatch, state, owner_registry=workflows,
    )
    callback = captured["message_callback"]
    assert asyncio.iscoroutinefunction(callback)

    for index in range(1001):
        await callback(
            WebSocketMessage(
                type="status",
                content={"index": index, "blob": b"binary"},
            )
        )

    assert len(state.realtime_messages) == 1000
    assert state.realtime_messages[0].content["index"] == 1
    assert all(
        message.content["blob"] == STRIPPED_FIGURE_PLACEHOLDER
        for message in state.realtime_messages
    )

    before = len(state.realtime_messages)
    sessions.pop(state.id)
    assert await callback(
        WebSocketMessage(type="status", content="missing")
    ) is False
    assert len(state.realtime_messages) == before

    sessions[state.id] = state
    workflows[state.id] = object()
    assert await callback(
        WebSocketMessage(type="status", content="superseded")
    ) is False
    assert len(state.realtime_messages) == before


@pytest.mark.asyncio
async def test_headless_state_swap_carries_and_persists_live_history(monkeypatch):
    from backend.models.schemas import WebSocketMessage
    from backend.services import workflow_runner
    import backend.workflow as workflow_module

    session_id = "headless-history"
    original = AnalysisState(id=session_id, user_id="u", status=TaskStatus.RUNNING)
    saved = []

    class _HistoryWorkflow:
        def __init__(self, config, session_id, message_callback=None, state_callback=None):
            self.message_callback = message_callback
            self.state_callback = state_callback

        async def run(self, initial_state, thread_id):
            await self.message_callback(
                WebSocketMessage(type="status", content={"text": "live", "raw": b"x"})
            )
            graph_state = AnalysisState(
                id=thread_id,
                user_id="u",
                status=TaskStatus.RUNNING,
                current_round=2,
            )
            assert await self.state_callback(graph_state) is True
            return graph_state

        async def close(self):
            return None

    async def _resolve(_user_id):
        return SimpleNamespace(max_rounds=3)

    async def _save(_session_id, current):
        saved.append(current)

    async def _update(session, current):
        sessions[session] = current

    monkeypatch.setattr(workflow_module, "SingleCellAnalysisWorkflow", _HistoryWorkflow)
    monkeypatch.setattr(workflow_runner, "resolve_config_for_user", _resolve)
    monkeypatch.setattr(
        workflow_runner, "_async_save_session_under_lifecycle", _save,
    )
    monkeypatch.setattr(workflow_runner, "_update_session_cache", _update)
    sessions[session_id] = original

    await workflow_runner._headless_workflow_runner(session_id)

    assert sessions[session_id] is not original
    assert sessions[session_id].current_round == 2
    assert [m.content["text"] for m in sessions[session_id].realtime_messages] == ["live"]
    assert sessions[session_id].realtime_messages[0].content["raw"] == "[image]"
    assert saved and saved[-1] is sessions[session_id]


@pytest.mark.asyncio
async def test_report_callback_retains_object_identity_guard(monkeypatch):
    state = AnalysisState(
        id="report-identity", user_id="u", status=TaskStatus.GENERATING_REPORT,
    )
    captured = await _capture_headless_callbacks(monkeypatch, state)
    from_graph = AnalysisState(
        id=state.id, user_id="u", status=TaskStatus.WAITING_INPUT,
    )
    sessions[state.id] = from_graph

    assert await captured["state_callback"](state) is False
    assert sessions[state.id] is from_graph


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

    async def _delayed_empty_build(_session_id, **_kwargs):
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
