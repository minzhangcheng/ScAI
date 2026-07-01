from types import SimpleNamespace

import pytest

from backend.models.schemas import AnalysisState, TaskStatus
from backend.state import initializing_sessions, sessions, workflows


class _FakeWorkflow:
    instances = []

    def __init__(self, config, session_id, message_callback=None, state_callback=None):
        self.config = config
        self.session_id = session_id
        self.message_callback = message_callback
        self.state_callback = state_callback
        self.run_calls = []
        self.resume_calls = []
        _FakeWorkflow.instances.append(self)

    async def run(self, initial_state, thread_id):
        self.run_calls.append((initial_state, thread_id))
        initial_state.status = TaskStatus.COMPLETED
        return initial_state

    async def resume(self, thread_id, analysis_state):
        self.resume_calls.append((thread_id, analysis_state))
        analysis_state.status = TaskStatus.COMPLETED
        return analysis_state


@pytest.fixture(autouse=True)
def _clear_workflow_globals():
    sessions.clear()
    workflows.clear()
    initializing_sessions.clear()
    _FakeWorkflow.instances.clear()
    yield
    sessions.clear()
    workflows.clear()
    initializing_sessions.clear()
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
    assert workflows == {}
