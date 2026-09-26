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
