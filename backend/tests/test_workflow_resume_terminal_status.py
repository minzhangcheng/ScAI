from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.models.schemas import AnalysisState, TaskStatus
from backend.workflow import SingleCellAnalysisWorkflow


@pytest.mark.asyncio
async def test_resume_at_graph_end_preserves_failed_status():
    workflow = object.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "failed-session"
    workflow._set_status = MagicMock()

    failed = AnalysisState(
        id="failed-session",
        user_id="user",
        status=TaskStatus.FAILED,
    )
    workflow.compiled_graph = MagicMock()
    workflow.compiled_graph.aget_state = AsyncMock(return_value=SimpleNamespace(
        values={"analysis_state": failed},
        next=(),
    ))

    result = await workflow.resume("failed-session")

    assert result is failed
    assert result.status == TaskStatus.FAILED
    workflow._set_status.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("skip", [False, True])
async def test_explicit_retry_rearms_ended_checkpoint_without_resetting_results(monkeypatch, skip):
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from backend.workflow.state import WorkflowState
    from backend.workflow import core

    monkeypatch.setattr(core, "is_live_runtime_continuation_safe", lambda *a: True)
    workflow = object.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "retry-checkpoint"
    workflow._set_status = MagicMock()
    seen = []

    async def execute(state):
        seen.append(state)
        result = state["analysis_state"].model_copy(deep=True)
        result.status = TaskStatus.COMPLETED
        return {"analysis_state": result}

    graph = StateGraph(WorkflowState)
    graph.add_node("restore_entry", lambda state: {})
    graph.add_node("literature_research", execute)
    graph.set_entry_point("restore_entry")
    graph.add_conditional_edges("restore_entry", workflow._route_from_restore_entry,
        {"hold": END, "route_literature_research": "literature_research"})
    graph.add_edge("literature_research", END)
    workflow.compiled_graph = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": workflow.session_id}}
    failed = AnalysisState(id=workflow.session_id, user_id="user", current_round=2,
                           status=TaskStatus.FAILED)
    await workflow.compiled_graph.ainvoke({
        "analysis_state": failed, "current_round": 2,
        "failed_node": "literature_research", "should_continue": False,
        "pi_instructions": {"retained": "plan"},
        "bioinformatics_result": {"retained": "execution"},
        "user_plan_feedback": "Keep the approved revision",
        "_restore_entry_plan": {"intent": "idle"},
    }, config)
    retry = failed.model_copy(deep=True)
    retry.sandbox_state = {"node_retry": {
        "id": "request-1", "node_name": "literature_research", "skip": skip}}
    result = await workflow.resume(workflow.session_id, retry)
    assert result.status == TaskStatus.COMPLETED
    assert len(seen) == 1
    assert seen[0]["current_round"] == 2
    assert seen[0]["pi_instructions"] == {"retained": "plan"}
    assert seen[0]["bioinformatics_result"] == {"retained": "execution"}
    assert seen[0]["user_plan_feedback"] == "Keep the approved revision"
    assert seen[0]["skip_nodes"] == (["literature_research"] if skip else [])
    assert "node_retry" not in result.sandbox_state
    # A stale session.json intent must not replay the already committed action.
    again = await workflow.resume(workflow.session_id, retry)
    assert again.status == TaskStatus.COMPLETED
    assert len(seen) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_skip", [False, True])
async def test_previous_retry_skip_list_does_not_override_next_round(monkeypatch, legacy_skip):
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from backend.workflow import core
    from backend.workflow.state import WorkflowState
    from backend.tests.test_report_retry_flow import _workflow, _state, _park

    monkeypatch.setattr(core, "is_live_runtime_continuation_safe", lambda *a: True)
    workflow = _workflow()
    state = _state()
    await _park(workflow, state, failed_node="generate_final_report", skip_nodes=["literature_research"])
    state.sandbox_state["node_retry"] = {"id": "retry-1", "node_name": "generate_final_report", "skip": False}
    state = await workflow.resume(workflow.session_id, state)
    assert not state.sandbox_state.get("skip_nodes")
    if legacy_skip:
        state.sandbox_state["skip_nodes"] = ["literature_research"]
    seen = []

    async def next_round(graph_state):
        graph_state["analysis_state"].current_round += 1
        return {**graph_state, "skip_nodes": []}

    async def literature(graph_state):
        seen.append(graph_state["skip_nodes"])
        return graph_state

    graph = StateGraph(WorkflowState)
    graph.add_node("initialize_round", next_round)
    graph.add_node("literature_research", literature)
    graph.set_entry_point("initialize_round")
    graph.add_edge("initialize_round", "literature_research")
    graph.add_edge("literature_research", END)
    workflow.compiled_graph = graph.compile(checkpointer=MemorySaver(), interrupt_before=["literature_research"])
    config = {"configurable": {"thread_id": workflow.session_id}}
    await workflow.compiled_graph.ainvoke({"analysis_state": state}, config)
    checkpoint = await workflow.compiled_graph.aget_state(config)
    await workflow.resume(workflow.session_id, checkpoint.values["analysis_state"])
    assert seen == [[]]
