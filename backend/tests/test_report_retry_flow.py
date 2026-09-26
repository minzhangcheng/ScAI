"""Report-only recovery must share the durable graph and honor user termination."""
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend.models.schemas import AnalysisState, RoundResult, TaskStatus
from backend.workflow.helpers import WorkflowTerminatedError
from backend.workflow.state import WorkflowState


def _make_workflow():
    from backend.workflow import SingleCellAnalysisWorkflow
    wf = object.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "test-session"
    wf._send_message = AsyncMock()
    wf._send_state_sync = AsyncMock()

    def _set_status(state, status):
        state.status = status

    wf._set_status = staticmethod(_set_status).__get__(wf, SingleCellAnalysisWorkflow)
    wf._settings_provider = MagicMock()
    wf._settings_provider.get_vision_setting = MagicMock(return_value=True)
    wf.painter_agent = None
    wf.bioinformatics_agent = None
    wf.supervisor_agent = None
    wf.llm_manager = MagicMock()
    wf.sandbox = None
    return wf


def _report_graph(workflow):
    graph = StateGraph(WorkflowState)
    graph.add_node("restore_entry", lambda state: {})
    names = ["generate_final_report", "generate_graphic_abstract", "generate_detailed_report"]

    def call_node(name):
        async def invoke(state):
            return await getattr(workflow, f"_{name}")(state)
        return invoke

    for name in names:
        graph.add_node(name, call_node(name))
    graph.set_entry_point("restore_entry")
    graph.add_conditional_edges("restore_entry", workflow._route_from_restore_entry,
        {"hold": END, **{f"route_{name}": name for name in names}})
    for left, right in zip(names, names[1:]):
        graph.add_edge(left, right)
    graph.add_edge(names[-1], END)
    return graph.compile(checkpointer=MemorySaver())


def _workflow():
    workflow = _make_workflow()
    workflow.initialize = AsyncMock()
    for name in ("generate_final_report", "generate_graphic_abstract", "generate_detailed_report"):
        setattr(workflow, f"_{name}", AsyncMock(side_effect=lambda state: state))
    workflow.compiled_graph = _report_graph(workflow)
    return workflow


async def _park(workflow, state, **extra):
    await workflow.compiled_graph.ainvoke({
        "analysis_state": state, "current_round": state.current_round,
        "_restore_entry_plan": {"intent": "idle"}, **extra,
    }, {"configurable": {"thread_id": workflow.session_id}})


def _state():
    return AnalysisState(id="test-session", user_id="test", current_round=1,
        status=TaskStatus.RUNNING, round_results=[RoundResult(round_number=1)])


@pytest.mark.asyncio
async def test_report_only_failure_retries_only_failed_report(monkeypatch):
    from backend.workflow import core
    monkeypatch.setattr(core, "is_live_runtime_continuation_safe", lambda *a: True)
    workflow = _workflow()
    state = _state()
    await _park(workflow, state, bioinformatics_result={"findings": ["retained result"]})

    async def detail(graph_state):
        result = graph_state["analysis_state"]
        if workflow._generate_detailed_report.await_count == 1:
            result.status = TaskStatus.FAILED
            result.sandbox_state["failed_node"] = "generate_detailed_report"
            return {**graph_state, "failed_node": "generate_detailed_report"}
        assert graph_state["bioinformatics_result"] == {"findings": ["retained result"]}
        result.status = TaskStatus.COMPLETED
        return {**graph_state, "failed_node": None}

    workflow._generate_detailed_report.side_effect = detail
    failed = await workflow.generate_report(state, workflow.session_id)
    assert failed.status == TaskStatus.FAILED
    checkpoint = await workflow.compiled_graph.aget_state({"configurable": {"thread_id": workflow.session_id}})
    assert checkpoint.values["failed_node"] == "generate_detailed_report"
    failed.sandbox_state["node_retry"] = {"id": "retry-1", "node_name": "generate_detailed_report", "skip": False}
    completed = await workflow.resume(workflow.session_id, failed)
    assert completed.status == TaskStatus.COMPLETED
    assert "node_retry" not in completed.sandbox_state
    workflow._generate_final_report.assert_awaited_once()
    workflow._generate_graphic_abstract.assert_awaited_once()
    assert workflow._generate_detailed_report.await_count == 2


@pytest.mark.asyncio
async def test_resumed_report_terminate_keeps_completed_summary(monkeypatch):
    from backend.workflow import core
    monkeypatch.setattr(core, "is_live_runtime_continuation_safe", lambda *a: True)
    workflow = _workflow()
    state = _state()
    await _park(workflow, state, failed_node="generate_final_report")

    async def summary(graph_state):
        graph_state["analysis_state"].final_summary = "Summary produced by this retry"
        return graph_state

    workflow._generate_final_report.side_effect = summary
    workflow._generate_graphic_abstract.side_effect = WorkflowTerminatedError("User chose Terminate")
    state.sandbox_state["node_retry"] = {"id": "retry-1", "node_name": "generate_final_report", "skip": False}
    result = await workflow.resume(workflow.session_id, state)
    assert result.status == TaskStatus.STOPPED
    assert result.final_summary == "Summary produced by this retry"
    workflow._generate_detailed_report.assert_not_awaited()
