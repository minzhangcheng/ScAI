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

    async def _save(_session_id, current, force=False):
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

    async def save(_session_id, current, force=False):
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
