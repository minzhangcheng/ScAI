import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.workflow.core import SingleCellAnalysisWorkflow


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
