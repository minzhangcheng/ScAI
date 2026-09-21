"""Report-only recovery must share the durable graph and honor user termination."""
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend.models.schemas import AnalysisState, RoundResult, TaskStatus
from backend.tests.test_reporting_shortcircuit import _make_workflow
from backend.workflow.helpers import WorkflowTerminatedError
from backend.workflow.state import WorkflowState


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
@pytest.mark.parametrize("in_synthesis", [False, True])
async def test_report_only_terminate_stops_all_following_phases(in_synthesis):
    workflow = _workflow()
    state = _state()
    if in_synthesis:
        state.round_results = []
        workflow._pi_synthesis = AsyncMock(side_effect=WorkflowTerminatedError("User chose Terminate"))
    else:
        workflow._generate_final_report.side_effect = WorkflowTerminatedError("User chose Terminate")
    await _park(workflow, state)
    result = await workflow.generate_report(state, workflow.session_id)
    assert result.status == TaskStatus.STOPPED
    assert result.wait_input_reason is None
    workflow._generate_graphic_abstract.assert_not_awaited()
    workflow._generate_detailed_report.assert_not_awaited()
    if in_synthesis:
        workflow._generate_final_report.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("has_checkpoint", [False, True])
async def test_legacy_report_failure_marker_resumes_exact_report(monkeypatch, has_checkpoint):
    from backend.workflow import core
    monkeypatch.setattr(core, "is_live_runtime_continuation_safe", lambda *a: True)
    workflow = _workflow()
    state = _state()
    if has_checkpoint:
        await _park(workflow, state)
    state.status = TaskStatus.FAILED
    state.sandbox_state = {"failed_node": "generate_detailed_report", "node_retry": {
        "id": "legacy-retry", "node_name": "generate_detailed_report", "skip": False}}
    await workflow.resume(workflow.session_id, state)
    workflow._generate_detailed_report.assert_awaited_once()
    workflow._generate_final_report.assert_not_awaited()
    workflow._generate_graphic_abstract.assert_not_awaited()


@pytest.mark.asyncio
async def test_report_retry_without_matching_failure_marker_is_rejected(monkeypatch):
    from backend.workflow import core
    monkeypatch.setattr(core, "is_live_runtime_continuation_safe", lambda *a: True)
    workflow = _workflow()
    state = _state()
    await _park(workflow, state)
    state.sandbox_state = {"node_retry": {
        "id": "unmatched-retry", "node_name": "generate_detailed_report", "skip": False}}
    with pytest.raises(ValueError, match="does not match"):
        await workflow.resume(workflow.session_id, state)
    workflow._generate_detailed_report.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminate", [False, True])
async def test_later_report_failure_preserves_completed_summary(terminate):
    workflow = _workflow()
    state = _state()
    await _park(workflow, state)

    async def summary(graph_state):
        graph_state["analysis_state"].final_summary = "Already generated summary"
        return graph_state

    workflow._generate_final_report.side_effect = summary
    workflow._generate_graphic_abstract.side_effect = (
        WorkflowTerminatedError("User chose Terminate") if terminate else RuntimeError("renderer unavailable")
    )
    result = await workflow.generate_report(state, workflow.session_id)
    assert result.status == (TaskStatus.STOPPED if terminate else TaskStatus.FAILED)
    assert result.final_summary == "Already generated summary"
    workflow._generate_detailed_report.assert_not_awaited()


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
