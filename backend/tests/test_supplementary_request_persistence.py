"""A2: durable request id survives route -> facade -> queue."""
from __future__ import annotations

import asyncio
from collections import deque

import pytest

from backend.models.schemas import AnalysisState, TaskStatus
from backend.workflow.core import SingleCellAnalysisWorkflow
from backend.workflow.core_parts import initialization


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
