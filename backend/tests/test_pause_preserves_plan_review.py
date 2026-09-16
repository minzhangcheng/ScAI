import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.config._config import SystemConfig
from backend.tests.split_facade import setattr_facade
from backend.workflow.core import SingleCellAnalysisWorkflow


class _FakeLLMManager:
    def __init__(self, *args, **kwargs):
        self.openrouter = MagicMock()
        self.openrouter.on_retry = None


class _FakeSandbox:
    def __init__(self, *args, **kwargs):
        return None


class _FakeCheckpointer:
    @classmethod
    def from_conn_string(cls, *args, **kwargs):
        return cls()

    async def setup(self):
        return None


@pytest.mark.asyncio
async def test_pause_keeps_plan_review_action(monkeypatch, tmp_path):
    setattr_facade(monkeypatch, "backend.workflow.core", "LLMManager", _FakeLLMManager)
    setattr_facade(monkeypatch, "backend.workflow.core", "LocalJupyterSandbox", _FakeSandbox)
    setattr_facade(monkeypatch, "backend.workflow.core", "AsyncSqliteSaver", _FakeCheckpointer)
    setattr_facade(monkeypatch, "backend.workflow.core", "get_session_states", lambda session_id: tmp_path / "states")
    setattr_facade(monkeypatch, "backend.workflow.core", "get_session_workspace", lambda session_id: tmp_path / "workspace")
    monkeypatch.setattr(SingleCellAnalysisWorkflow, "ensure_agent_ready", lambda self, agent_name: None)
    monkeypatch.setattr(SingleCellAnalysisWorkflow, "_build_graph", lambda self: None)

    from backend.models.schemas import AnalysisState, TaskStatus

    workflow = SingleCellAnalysisWorkflow(config=SystemConfig(), session_id="test-pause-session")

    state = AnalysisState(
        id="test-pause-session",
        name="Test Pause Session",
        user_id="user-1",
        initial_instruction="test",
    )
    workflow._state = state
    workflow._db_conn = MagicMock()
    workflow._db_conn.close = AsyncMock()

    # Mock the compiled_graph.aget_state to return a minimal state
    fake_state = MagicMock()
    fake_state.values = {}
    mock_graph = MagicMock()
    mock_graph.aget_state = AsyncMock(return_value=fake_state)
    workflow.compiled_graph = mock_graph

    await workflow.initialize()

    workflow._plan_review_action = {"action": "approve"}
    workflow.sandbox = None
    state.status = TaskStatus.RUNNING
    result = await workflow.pause(
        thread_id="test-thread", analysis_state=state,
    )

    assert workflow._plan_review_action == {"action": "approve"}
    assert result["success"] is True
    assert result["status"] == TaskStatus.PAUSED.value
    assert state.status is TaskStatus.PAUSED


@pytest.mark.asyncio
async def test_pause_waits_for_active_bioinformatics_turn_before_saving(monkeypatch):
    from backend.agents.bioinformatics import executor_loop
    from backend.models.schemas import AnalysisState, TaskStatus

    state = AnalysisState(
        id="test-pause-boundary",
        name="Test Pause Boundary",
        user_id="user-1",
        initial_instruction="test",
        status=TaskStatus.RUNNING,
    )
    pause_event = asyncio.Event()
    pause_event.set()
    turn_lock = asyncio.Lock()
    agent = SimpleNamespace(
        _pause_event=pause_event,
        _pause_turn_lock=turn_lock,
    )
    turn_started = asyncio.Event()
    allow_turn_to_finish = asyncio.Event()
    revision = {"current": 1}
    status_seen_at_turn_end = []
    saved_revisions = []

    async def fake_start_turn(context):
        turn_started.set()
        return SimpleNamespace(
            response=SimpleNamespace(content="active turn"),
            exit=None,
        )

    async def fake_process_response(context, response, agent_reasoning):
        await allow_turn_to_finish.wait()
        revision["current"] = 2
        status_seen_at_turn_end.append(state.status)
        context.completed = True
        return object()

    async def fake_finish_turn(context, effects):
        return executor_loop.LoopExit.CONTINUE

    async def fake_finalize_run(context):
        return {}

    monkeypatch.setattr(executor_loop, "start_turn", fake_start_turn)
    monkeypatch.setattr(executor_loop, "process_response", fake_process_response)
    monkeypatch.setattr(executor_loop, "finish_turn", fake_finish_turn)
    monkeypatch.setattr(executor_loop, "finalize_run", fake_finalize_run)

    context = SimpleNamespace(
        agent=agent,
        step_count=0,
        max_steps=1,
        completed=False,
    )

    async def save_session():
        saved_revisions.append(revision["current"])
        return True

    sandbox = SimpleNamespace(
        save_session=save_session,
        sync_runtime_persistence_state=lambda analysis_state: {
            "status": "current",
            "current_revision": revision["current"],
            "saved_revision": revision["current"],
        },
        notebook_state=SimpleNamespace(save=MagicMock()),
    )
    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = state.id
    workflow._pause_event = pause_event
    workflow._continue_event = asyncio.Event()
    workflow._annotation_event = asyncio.Event()
    workflow._plan_review_event = asyncio.Event()
    workflow._species_event = asyncio.Event()
    workflow._species_user_response = None
    workflow.bioinformatics_agent = agent
    workflow.sandbox = sandbox
    workflow.compiled_graph = SimpleNamespace(
        aget_state=AsyncMock(return_value=SimpleNamespace(values={}))
    )

    turn_task = asyncio.create_task(executor_loop.run_turn_loop(context))
    await turn_started.wait()
    pause_task = asyncio.create_task(
        workflow.pause(state.id, analysis_state=state)
    )
    await asyncio.sleep(0)
    assert not pause_task.done()
    assert saved_revisions == []
    allow_turn_to_finish.set()
    await asyncio.gather(turn_task, pause_task)

    assert status_seen_at_turn_end == [TaskStatus.RUNNING]
    assert saved_revisions == [2]
    assert state.status is TaskStatus.PAUSED


@pytest.mark.asyncio
async def test_retry_gate_drains_an_active_bioinformatics_phase_before_pause():
    from backend.workflow.core_parts.initialization import (
        _wait_for_retry_pause_gate,
    )

    pause_event = asyncio.Event()
    pause_event.clear()
    agent = SimpleNamespace(
        _pause_turn_active=True,
        _pause_gate_deferred=False,
    )
    workflow = SimpleNamespace(
        _pause_event=pause_event,
        bioinformatics_agent=agent,
    )

    # Pause is waiting for the active turn lock, so a retry inside that turn
    # must be allowed to drain instead of waiting back on Resume.
    await asyncio.wait_for(_wait_for_retry_pause_gate(workflow), timeout=0.1)

    # Outside an owned Bioinformatics phase, the original retry gate remains.
    agent._pause_turn_active = False
    retry_gate = asyncio.create_task(_wait_for_retry_pause_gate(workflow))
    await asyncio.sleep(0)
    assert not retry_gate.done()
    pause_event.set()
    await retry_gate


@pytest.mark.asyncio
async def test_pause_waits_for_bioinformatics_tail_before_saving(monkeypatch):
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.workflow.agent_nodes import bioinformatics_node

    state = AnalysisState(
        id="test-pause-tail",
        name="Test Pause Tail",
        user_id="user-1",
        initial_instruction="test",
        status=TaskStatus.RUNNING,
    )
    pause_event = asyncio.Event()
    pause_event.set()
    tail_started = asyncio.Event()
    allow_tail_to_finish = asyncio.Event()
    revision = {"current": 1}
    saved_revisions = []
    status_seen_at_tail_end = []
    agent = SimpleNamespace(
        _pause_event=pause_event,
        _pause_turn_lock=asyncio.Lock(),
        _pause_turn_active=False,
        _pause_gate_deferred=False,
    )

    async def fake_start(workflow, graph_state):
        return {"_continue": object()}

    async def fake_prepare(workflow, graph_state, continuation):
        return {"analysis_state": state}

    async def fake_execute(workflow, context):
        return {}

    async def fake_collect(workflow, context, result):
        assert agent._pause_gate_deferred is True
        tail_started.set()
        await allow_tail_to_finish.wait()
        revision["current"] = 2

    async def fake_finalize(workflow, context, result):
        status_seen_at_tail_end.append(state.status)
        return {}

    monkeypatch.setattr(bioinformatics_node, "start_bioinformatics_analysis", fake_start)
    monkeypatch.setattr(bioinformatics_node, "prepare_bioinformatics_instruction", fake_prepare)
    monkeypatch.setattr(bioinformatics_node, "execute_bioinformatics_analysis", fake_execute)
    monkeypatch.setattr(bioinformatics_node, "collect_bioinformatics_results", fake_collect)
    monkeypatch.setattr(bioinformatics_node, "finalize_bioinformatics_analysis", fake_finalize)

    async def save_session():
        saved_revisions.append(revision["current"])
        return True

    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = state.id
    workflow._pause_event = pause_event
    workflow._pause_bioinformatics_tail_lock = asyncio.Lock()
    workflow._continue_event = asyncio.Event()
    workflow._annotation_event = asyncio.Event()
    workflow._plan_review_event = asyncio.Event()
    workflow._species_event = asyncio.Event()
    workflow._species_user_response = None
    workflow.bioinformatics_agent = agent
    workflow.sandbox = SimpleNamespace(
        save_session=save_session,
        sync_runtime_persistence_state=lambda analysis_state: {
            "status": "current",
            "current_revision": revision["current"],
            "saved_revision": revision["current"],
        },
        notebook_state=SimpleNamespace(save=MagicMock()),
    )
    workflow.compiled_graph = SimpleNamespace(
        aget_state=AsyncMock(return_value=SimpleNamespace(values={}))
    )

    tail_task = asyncio.create_task(
        bioinformatics_node.run_bioinformatics_analysis(
            workflow, {"analysis_state": state},
        )
    )
    await tail_started.wait()
    pause_task = asyncio.create_task(
        workflow.pause(state.id, analysis_state=state)
    )
    await asyncio.sleep(0)

    assert not pause_task.done()
    assert saved_revisions == []
    allow_tail_to_finish.set()
    await asyncio.gather(tail_task, pause_task)

    assert status_seen_at_tail_end == [TaskStatus.RUNNING]
    assert saved_revisions == [2]
    assert state.status is TaskStatus.PAUSED
    assert agent._pause_gate_deferred is False
