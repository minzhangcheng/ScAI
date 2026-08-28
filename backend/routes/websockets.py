"""WebSocket endpoints for real-time updates and report generation."""

import asyncio

import logging

from datetime import datetime

from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import WS_RECEIVE_TIMEOUT

from backend.models.schemas import WebSocketMessage, AnalysisState, TaskStatus, AnalysisPhase

from backend.workflow.direction_coverage import active_batch_incomplete

from backend.state import (
    workflows, report_workflows, _monitor_tasks,
    _run_tasks, state_lock, initializing_sessions,
    claim_initializing_session, release_initializing_session,
    detailed_reports,
    detailed_report_tasks,
    sessions as _sessions,
    get_session_lock,
)

from backend.utils.auth import (
    WS_SESSION_GONE_CODE,
    WS_SESSION_NOT_FOUND_CODE,
    ws_authenticate_and_load,
)

from backend.utils.asyncio_tasks import consume_current_cancellation_request

from backend.utils.websocket_delivery import (
    WebSocketConnection,
    append_durable_realtime_message,
    register_analysis_websocket,
    register_report_websocket,
    send_analysis_json,
    send_connection_json,
    unregister_websocket,
    serialize_realtime_message,
)

from .serialization import serialize_round_results

from backend.utils.config_resolver import resolve_config_for_user

from backend.services.session_store import (
    update_session_cache as _update_session_cache,
    async_save_session as _async_save_session,
)

from backend.services.session_lifecycle import (
    SessionGenerationError,
    SessionGoneError,
    assert_generation_writable,
    get_lifecycle_lock,
    snapshot_operation_active,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websockets"])

import sys
from .websocket_parts import analysis_connection as websocket_analysis_connection
from .websocket_parts import ownership as websocket_ownership
from .websocket_parts import state_sync as websocket_state_sync

def _session_workflow_activity(session_id: str) -> tuple[bool, bool, bool, bool]:
    websocket_state_sync.bind_facade(sys.modules[__name__])
    return websocket_state_sync._session_workflow_activity(session_id)

def _needs_report_only(state, effective_max_rounds: int, in_integration_loop: bool) -> bool:
    websocket_state_sync.bind_facade(sys.modules[__name__])
    return websocket_state_sync._needs_report_only(state, effective_max_rounds, in_integration_loop)

def _in_integration_loop(state) -> bool:
    websocket_state_sync.bind_facade(sys.modules[__name__])
    return websocket_state_sync._in_integration_loop(state)

def should_resume_workflow(state) -> bool:
    websocket_state_sync.bind_facade(sys.modules[__name__])
    return websocket_state_sync.should_resume_workflow(state)

async def _send_restore_state_sync(websocket, state, session_id: str) -> None:
    websocket_state_sync.bind_facade(sys.modules[__name__])
    return await websocket_state_sync._send_restore_state_sync(websocket, state, session_id)

async def _resend_wait_input_state_sync(websocket, state, session_id: str) -> None:
    websocket_state_sync.bind_facade(sys.modules[__name__])
    return await websocket_state_sync._resend_wait_input_state_sync(websocket, state, session_id)

def _preserve_realtime_stream(prev, new) -> None:
    websocket_state_sync.bind_facade(sys.modules[__name__])
    return websocket_state_sync._preserve_realtime_stream(prev, new)

def _serialize_realtime_messages(messages):
    websocket_state_sync.bind_facade(sys.modules[__name__])
    return websocket_state_sync._serialize_realtime_messages(messages)

def _pop_if_owned(registry: dict, session_id: str, owned: Any) -> bool:
    websocket_ownership.bind_facade(sys.modules[__name__])
    return websocket_ownership._pop_if_owned(registry, session_id, owned)

async def _publish_stop_summary_task(session_id: str, state: AnalysisState, workflow: Any, prior_task: asyncio.Task | None) -> asyncio.Task:
    websocket_ownership.bind_facade(sys.modules[__name__])
    return await websocket_ownership._publish_stop_summary_task(session_id, state, workflow, prior_task)

async def _cleanup_owned_workflow_handler(session_id: str, *, workflow: Any, run_task: asyncio.Task | None, monitor_task: asyncio.Task | None, connection_ref: WebSocketConnection, initializing_token: object | None=None) -> None:
    websocket_ownership.bind_facade(sys.modules[__name__])
    return await websocket_ownership._cleanup_owned_workflow_handler(session_id, workflow=workflow, run_task=run_task, monitor_task=monitor_task, connection_ref=connection_ref, initializing_token=initializing_token)

@router.websocket('/ws/{session_id}')
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    websocket_analysis_connection.bind_facade(sys.modules[__name__])
    return await websocket_analysis_connection.websocket_endpoint(websocket, session_id)

@router.websocket('/ws/report/{session_id}')
async def report_websocket(websocket: WebSocket, session_id: str):
    websocket_analysis_connection.bind_facade(sys.modules[__name__])
    return await websocket_analysis_connection.report_websocket(websocket, session_id)
