"""Lifecycle routes: run, pause, resume, stop, retry-node, failed-node, resume-bioinformatics.

Extracted from backend/main.py to decouple lifecycle route handlers.
"""

import json

import logging

from datetime import datetime

from typing import Any, Dict, Literal

from fastapi import APIRouter, HTTPException, Request, Depends

from pydantic import BaseModel

from backend.config import get_session_states

from backend.constants import WaitInputReason, SEMANTIC_WAIT_GATES

from backend.models.api_requests import LlmFailureActionRequest, RetryNodeRequest, StopAnalysisRequest

from backend.models.schemas import (
    APIResponse, TaskStatus,
)

from backend.services.session_store import (
    async_save_session,
    assert_recovery_clear,
    ensure_session_loaded,
    pause_and_restart_ws as _pause_and_restart_ws_impl,
)

from backend.utils.auth import require_session_ownership

from backend.utils.runtime_apply import is_forward_resume_blocked

from backend.services.snapshot_v2.checkpoint_paths import checkpoint_sqlite_path

from backend.services.snapshot_v2.errors import (
    SNAPSHOT_FAILURE_ACTIONS,
    SnapshotTransactionError,
    is_snapshot_failure_blocked,
)

from backend.services.snapshot_v2.feature_gate import require_snapshot_v2_enabled

from backend.utils.files import validate_session_id

from backend.state import (
    _run_tasks,
    state_lock as _state_lock,
    workflows,
    get_session_lock,
    get_snapshot_action_lock,
)

from backend.services.session_store import update_session_cache as _update_session_cache

from backend.routes._helpers import teardown_running_workflow

from backend.utils.websocket_delivery import send_analysis_json

from backend.utils.websocket_delivery import close_analysis_websocket

logger = logging.getLogger(__name__)

WAITING_HITL_STATUSES = frozenset({
    TaskStatus.WAITING_INPUT,
    TaskStatus.WAITING_ANNOTATION,
    TaskStatus.WAITING_PLAN_REVIEW,
    TaskStatus.WAITING_SPECIES_INPUT,
})

router = APIRouter(prefix="/api/session", tags=["lifecycle"])

import sys
from .lifecycle_parts import interventions as lifecycle_interventions
from .lifecycle_parts import run_resume as lifecycle_run_resume
from .lifecycle_parts import runtime_persistence as lifecycle_runtime_persistence
from .lifecycle_parts import shared as lifecycle_shared
from .lifecycle_parts import snapshot_failures as lifecycle_snapshot_failures

async def _resume_workflow(session_id, state, signal_fn=None):
    lifecycle_shared.bind_facade(sys.modules[__name__])
    return await lifecycle_shared._resume_workflow(session_id, state, signal_fn)

def resolve_hitl_skip_target(state):
    lifecycle_shared.bind_facade(sys.modules[__name__])
    return lifecycle_shared.resolve_hitl_skip_target(state)

async def _pause_and_restart_ws(session_id: str, state):
    lifecycle_shared.bind_facade(sys.modules[__name__])
    return await lifecycle_shared._pause_and_restart_ws(session_id, state)

async def _close_hitl_socket_for_redrive(session_id: str, reason: str) -> None:
    lifecycle_shared.bind_facade(sys.modules[__name__])
    return await lifecycle_shared._close_hitl_socket_for_redrive(session_id, reason)

@router.post('/{session_id}/run')
async def run_analysis(session_id: str, request: Request):
    lifecycle_run_resume.bind_facade(sys.modules[__name__])
    return await lifecycle_run_resume.run_analysis(session_id, request)

@router.post('/{session_id}/pause')
async def pause_analysis(session_id: str, request: Request):
    lifecycle_run_resume.bind_facade(sys.modules[__name__])
    return await lifecycle_run_resume.pause_analysis(session_id, request)

async def _pause_analysis_locked(session_id: str, user_id: str):
    lifecycle_run_resume.bind_facade(sys.modules[__name__])
    return await lifecycle_run_resume._pause_analysis_locked(session_id, user_id)

@router.post('/{session_id}/resume')
async def resume_analysis(session_id: str, request: Request):
    lifecycle_run_resume.bind_facade(sys.modules[__name__])
    return await lifecycle_run_resume.resume_analysis(session_id, request)

async def _resume_analysis_locked(session_id: str, user_id: str):
    lifecycle_run_resume.bind_facade(sys.modules[__name__])
    return await lifecycle_run_resume._resume_analysis_locked(session_id, user_id)

@router.post('/{session_id}/stop')
async def stop_analysis(session_id: str, request: Request, req: StopAnalysisRequest):
    lifecycle_run_resume.bind_facade(sys.modules[__name__])
    return await lifecycle_run_resume.stop_analysis(session_id, request, req)

async def _stop_analysis_locked(session_id: str, user_id: str, req: StopAnalysisRequest):
    lifecycle_run_resume.bind_facade(sys.modules[__name__])
    return await lifecycle_run_resume._stop_analysis_locked(session_id, user_id, req)

@router.post('/{session_id}/llm-failure-action')
async def llm_failure_action(session_id: str, request: Request, req: LlmFailureActionRequest):
    lifecycle_interventions.bind_facade(sys.modules[__name__])
    return await lifecycle_interventions.llm_failure_action(session_id, request, req)

async def _llm_failure_action_locked(session_id: str, user_id: str, req: LlmFailureActionRequest):
    lifecycle_interventions.bind_facade(sys.modules[__name__])
    return await lifecycle_interventions._llm_failure_action_locked(session_id, user_id, req)

def _raise_runtime_persistence_action_409(result) -> None:
    lifecycle_runtime_persistence.bind_facade(sys.modules[__name__])
    return lifecycle_runtime_persistence._raise_runtime_persistence_action_409(result)

async def _runtime_persistence_action_locked(session_id: str, user_id: str, action: str, *, gate_token: str | None=None, generation: int | None=None):
    lifecycle_runtime_persistence.bind_facade(sys.modules[__name__])
    return await lifecycle_runtime_persistence._runtime_persistence_action_locked(session_id, user_id, action, gate_token=gate_token, generation=generation)

async def _runtime_persistence_action(session_id: str, request: Request, action: str):
    lifecycle_runtime_persistence.bind_facade(sys.modules[__name__])
    return await lifecycle_runtime_persistence._runtime_persistence_action(session_id, request, action)

@router.post('/{session_id}/runtime-persistence/retry')
async def runtime_persistence_retry(session_id: str, request: Request):
    lifecycle_runtime_persistence.bind_facade(sys.modules[__name__])
    return await lifecycle_runtime_persistence.runtime_persistence_retry(session_id, request)

@router.post('/{session_id}/runtime-persistence/recompute')
async def runtime_persistence_recompute(session_id: str, request: Request):
    lifecycle_runtime_persistence.bind_facade(sys.modules[__name__])
    return await lifecycle_runtime_persistence.runtime_persistence_recompute(session_id, request)

@router.post('/{session_id}/runtime-persistence/terminate')
async def runtime_persistence_terminate(session_id: str, request: Request):
    lifecycle_runtime_persistence.bind_facade(sys.modules[__name__])
    return await lifecycle_runtime_persistence.runtime_persistence_terminate(session_id, request)

@router.post('/{session_id}/retry-node')
async def retry_node(session_id: str, request: Request, body: RetryNodeRequest):
    lifecycle_interventions.bind_facade(sys.modules[__name__])
    return await lifecycle_interventions.retry_node(session_id, request, body)

@router.get('/{session_id}/failed-node')
async def get_failed_node(session_id: str, request: Request):
    lifecycle_interventions.bind_facade(sys.modules[__name__])
    return await lifecycle_interventions.get_failed_node(session_id, request)

@router.post('/{session_id}/resume-bioinformatics')
async def resume_bioinformatics(session_id: str, request: Request):
    lifecycle_interventions.bind_facade(sys.modules[__name__])
    return await lifecycle_interventions.resume_bioinformatics(session_id, request)

@router.post('/{session_id}/snapshot-failure/retry', dependencies=[Depends(require_snapshot_v2_enabled)])
async def snapshot_failure_retry(session_id: str, request: Request):
    lifecycle_snapshot_failures.bind_facade(sys.modules[__name__])
    return await lifecycle_snapshot_failures.snapshot_failure_retry(session_id, request)

async def _snapshot_failure_retry_locked(session_id: str, user_id: str):
    lifecycle_snapshot_failures.bind_facade(sys.modules[__name__])
    return await lifecycle_snapshot_failures._snapshot_failure_retry_locked(session_id, user_id)

@router.post('/{session_id}/snapshot-failure/proceed', dependencies=[Depends(require_snapshot_v2_enabled)])
async def snapshot_failure_proceed(session_id: str, request: Request):
    lifecycle_snapshot_failures.bind_facade(sys.modules[__name__])
    return await lifecycle_snapshot_failures.snapshot_failure_proceed(session_id, request)

async def _snapshot_failure_proceed_locked(session_id: str, user_id: str):
    lifecycle_snapshot_failures.bind_facade(sys.modules[__name__])
    return await lifecycle_snapshot_failures._snapshot_failure_proceed_locked(session_id, user_id)

@router.post('/{session_id}/snapshot-failure/skip', dependencies=[Depends(require_snapshot_v2_enabled)])
async def snapshot_failure_skip(session_id: str, request: Request):
    lifecycle_snapshot_failures.bind_facade(sys.modules[__name__])
    return await lifecycle_snapshot_failures.snapshot_failure_skip(session_id, request)

async def _snapshot_failure_skip_locked(session_id: str, user_id: str):
    lifecycle_snapshot_failures.bind_facade(sys.modules[__name__])
    return await lifecycle_snapshot_failures._snapshot_failure_skip_locked(session_id, user_id)

@router.post('/{session_id}/snapshot-failure/terminate', dependencies=[Depends(require_snapshot_v2_enabled)])
async def snapshot_failure_terminate(session_id: str, request: Request):
    lifecycle_snapshot_failures.bind_facade(sys.modules[__name__])
    return await lifecycle_snapshot_failures.snapshot_failure_terminate(session_id, request)

async def _snapshot_failure_terminate_locked(session_id: str, user_id: str):
    lifecycle_snapshot_failures.bind_facade(sys.modules[__name__])
    return await lifecycle_snapshot_failures._snapshot_failure_terminate_locked(session_id, user_id)
