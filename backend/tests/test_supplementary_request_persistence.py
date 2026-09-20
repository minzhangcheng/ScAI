"""A2: durable request id survives route -> facade -> queue."""
from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.constants import LlmFailureAction
from backend.models.schemas import AnalysisState, TaskStatus
from backend.utils.llm_client import UpstreamExhaustedError
from backend.workflow.core import SingleCellAnalysisWorkflow
from backend.workflow.core_parts import initialization, supplementary
from backend.workflow.helpers import WorkflowTerminatedError


def test_facade_forwards_request_id_to_queue():
    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow._supplementary_queue = deque()
    workflow._annotation_event = asyncio.Event()
    workflow._continue_event = asyncio.Event()

    SingleCellAnalysisWorkflow.signal_supplementary_analysis(
        workflow, "supp_abc123", "annotation", request_id="supp_abc123",
    )

    assert list(workflow._supplementary_queue) == [
        ("supp_abc123", "annotation")
    ]


def test_route_durable_record_shape_keeps_scope_and_request_id():
    # The route writes this shape before signalling; this guards the fields the
    # executor resolves later.
    record = {
        "request_id": "supp_abc123",
        "request_text": "run a supplementary task",
        "target": "round",
        "round_number": 2,
        "request_scope_id": "supp_scope_xyz",
        "status": "queued",
        "adata_source": None,
        "source_selection_signature": None,
        "adata_source_preparation_id": None,
        "created_at": "2026-09-14T00:00:00",
        "updated_at": "2026-09-14T00:00:00",
        "error_summary": None,
    }
    assert record["request_scope_id"].startswith("supp_scope_")
    assert record["request_id"].startswith("supp_")


@pytest.mark.asyncio
async def test_durable_request_rebuilds_once_at_matching_gate():
    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow._supplementary_queue = deque()
    workflow._annotation_event = asyncio.Event()
    workflow._continue_event = asyncio.Event()
    state = AnalysisState(
        id="supp-rebuild",
        user_id="user",
        status=TaskStatus.WAITING_ANNOTATION,
        sandbox_state={
            "supplementary_requests_v1": {
                "supp_abc123": {
                    "request_id": "supp_abc123",
                    "request_text": "run a supplementary task",
                    "target": "annotation",
                    "round_number": 1,
                    "request_scope_id": "supp_scope_xyz",
                    "status": "queued",
                    "created_at": "2026-09-14T00:00:00",
                }
            }
        },
    )
    workflow._state = state

    await initialization._rebuild_supplementary_queue(workflow)
    await initialization._rebuild_supplementary_queue(workflow)

    assert list(workflow._supplementary_queue) == [
        ("supp_abc123", "annotation")
    ]
    assert workflow._annotation_event.is_set()


def _supplementary_workflow(state, plan, literature_process):
    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = str(state.id)
    workflow.state_callback = None
    workflow._send_message = AsyncMock()
    workflow._wait_for_llm_intervention = AsyncMock()
    workflow._figure_review_callback = AsyncMock()
    workflow._current_graph_state = {
        "analysis_state": state,
        "bioinformatics_result": None,
        "literature_result": None,
    }
    workflow._build_and_save_detailed_segment = AsyncMock(
        return_value=SimpleNamespace(id="supp-segment", supplement_owner_key=None),
    )
    workflow.pi_agent = MagicMock()
    workflow.pi_agent.plan_supplementary_analysis = AsyncMock(return_value=plan)
    workflow.pi_agent.plan_literature_search = AsyncMock()
    workflow.pi_agent.review_literature_search = AsyncMock()
    workflow.bioinformatics_agent = SimpleNamespace(
        sandbox=SimpleNamespace(),
        run_autonomous_analysis=AsyncMock(return_value={
            "status": "completed",
            "findings": ["completed bioinformatics work"],
            "code_history": [],
            "analysis_log": [],
            "settlement_attempt_execution_ids": [],
            "executor_settlement_scope_id": "supp-scope",
            "resolved_adata_source_context": {},
        }),
    )
    workflow.literature_agent = SimpleNamespace(process=literature_process)
    return workflow


def _queued_state():
    return AnalysisState(
        id="supp-hitl", user_id="user", current_round=1,
        sandbox_state={
            "supplementary_requests_v1": {
                "supp-request": {
                    "request_id": "supp-request",
                    "request_text": "run supplementary evidence",
                    "target": "round",
                    "round_number": 1,
                    "request_scope_id": "supp-scope",
                    "status": "queued",
                    "created_at": "2026-09-20T00:00:00",
                    "updated_at": "2026-09-20T00:00:00",
                },
            },
        },
    )


async def _capture_and_settle(*_args, before_state_save=None, **_kwargs):
    outcome = SimpleNamespace(
        available=True, disclosure=None, error_summary=None, record=None,
    )
    if before_state_save is not None:
        before_state_save(outcome)
    return outcome


def _upstream_error():
    return UpstreamExhaustedError(
        "supplementary upstream exhausted",
        agent_name="literature", model="test", llm_server="test",
    )


@pytest.mark.asyncio
async def test_literature_retry_is_local_and_request_settles_without_rerunning_bioinfo(
    monkeypatch,
):
    state = _queued_state()
    literature_process = AsyncMock(side_effect=[
        _upstream_error(), {"report": "retrieved supplementary evidence"},
    ])
    workflow = _supplementary_workflow(
        state,
        {
            "bioinformatics_tasks": ["completed once"],
            "literature_topics": ["retry this literature call"],
        },
        literature_process,
    )
    workflow._wait_for_llm_intervention.return_value = LlmFailureAction.RETRY
    monkeypatch.setattr(
        "backend.services.session_store.async_save_session", AsyncMock(),
    )
    monkeypatch.setattr(
        "backend.workflow.adata_checkpoint_hooks.capture_and_persist_boundary",
        _capture_and_settle,
    )

    await supplementary._run_supplementary_analysis(
        workflow, state, "supp-request", "round",
    )

    assert workflow.bioinformatics_agent.run_autonomous_analysis.await_count == 1
    assert literature_process.await_count == 2
    workflow._wait_for_llm_intervention.assert_awaited_once()
    assert state.sandbox_state["supplementary_requests_v1"]["supp-request"][
        "status"
    ] == "settled"


@pytest.mark.asyncio
async def test_literature_skip_preserves_completed_work_and_settles_technical_failure(
    monkeypatch,
):
    state = _queued_state()
    literature_process = AsyncMock(side_effect=_upstream_error())
    workflow = _supplementary_workflow(
        state,
        {
            "bioinformatics_tasks": ["completed once"],
            "literature_topics": ["skip failed literature call"],
        },
        literature_process,
    )
    workflow._wait_for_llm_intervention.return_value = LlmFailureAction.SKIP
    monkeypatch.setattr(
        "backend.services.session_store.async_save_session", AsyncMock(),
    )
    monkeypatch.setattr(
        "backend.workflow.adata_checkpoint_hooks.capture_and_persist_boundary",
        _capture_and_settle,
    )

    await supplementary._run_supplementary_analysis(
        workflow, state, "supp-request", "round",
    )

    assert workflow.bioinformatics_agent.run_autonomous_analysis.await_count == 1
    assert literature_process.await_count == 1
    assert workflow._current_graph_state["bioinformatics_result"]["findings"] == [
        "[Supplementary] completed bioinformatics work"
    ]
    durable = state.sandbox_state["supplementary_requests_v1"]["supp-request"]
    assert durable["status"] == "failed"
    assert durable["error_summary"] == (
        "Supplementary literature was skipped after LLM failure"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    {"source_failure": {"mode": "pubmed_llm", "reason": "ValueError: invalid XML"}},
    {"source_processing_error": "RuntimeError: extraction failed"},
    {"search_meta": {"evaluation_failed": True}},
    {},
])
async def test_returned_literature_failure_is_settled_and_preserved_without_reexecution(
    monkeypatch, failure,
):
    state = _queued_state()
    literature_process = AsyncMock(return_value={
        "status": "partial", "report": "retained evidence", **failure,
    })
    workflow = _supplementary_workflow(
        state,
        {"bioinformatics_tasks": ["completed once"], "literature_topics": ["topic"]},
        literature_process,
    )
    monkeypatch.setattr("backend.services.session_store.async_save_session", AsyncMock())
    monkeypatch.setattr(
        "backend.workflow.adata_checkpoint_hooks.capture_and_persist_boundary",
        _capture_and_settle,
    )

    await supplementary._run_supplementary_analysis(workflow, state, "supp-request", "round")

    workflow.bioinformatics_agent.run_autonomous_analysis.assert_awaited_once()
    literature_process.assert_awaited_once()
    workflow._wait_for_llm_intervention.assert_not_awaited()
    result = workflow._current_graph_state["literature_result"]
    assert any("retained evidence" in finding for finding in result["findings"])
    durable = state.sandbox_state["supplementary_requests_v1"]["supp-request"]
    assert durable["status"] == ("failed" if failure else "settled")
    if failure:
        assert result["status"] == "partial"
        for key, value in failure.items():
            assert result["technical_failures"][0][key] == value
            assert durable["literature_technical_failures"][0][key] == value
    else:
        assert "literature_technical_failures" not in durable


@pytest.mark.asyncio
async def test_pi_planning_retry_is_local_and_request_settles(monkeypatch):
    state = _queued_state()
    workflow = _supplementary_workflow(state, {}, AsyncMock())
    workflow.pi_agent.plan_supplementary_analysis = AsyncMock(side_effect=[
        _upstream_error(), {},
    ])
    workflow._wait_for_llm_intervention.return_value = LlmFailureAction.RETRY
    monkeypatch.setattr(
        "backend.services.session_store.async_save_session", AsyncMock(),
    )
    monkeypatch.setattr(
        "backend.workflow.adata_checkpoint_hooks.capture_and_persist_boundary",
        _capture_and_settle,
    )

    await supplementary._run_supplementary_analysis(
        workflow, state, "supp-request", "round",
    )

    assert workflow.pi_agent.plan_supplementary_analysis.await_count == 2
    assert state.sandbox_state["supplementary_requests_v1"]["supp-request"][
        "status"
    ] == "settled"


@pytest.mark.asyncio
async def test_pi_planning_terminate_is_persisted_before_propagation(monkeypatch):
    state = _queued_state()
    workflow = _supplementary_workflow(state, {}, AsyncMock())
    workflow.pi_agent.plan_supplementary_analysis = AsyncMock(
        side_effect=_upstream_error(),
    )
    workflow._wait_for_llm_intervention.return_value = LlmFailureAction.TERMINATE
    monkeypatch.setattr(
        "backend.services.session_store.async_save_session", AsyncMock(),
    )

    with pytest.raises(WorkflowTerminatedError):
        await supplementary._run_supplementary_analysis(
            workflow, state, "supp-request", "round",
        )

    durable = state.sandbox_state["supplementary_requests_v1"]["supp-request"]
    assert durable["status"] == "interrupted"
    assert "terminated" in durable["error_summary"].lower()
