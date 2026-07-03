"""WebSocket endpoints for real-time updates and report generation."""
import asyncio
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import WS_RECEIVE_TIMEOUT
from backend.models.schemas import WebSocketMessage, AnalysisState, TaskStatus
from backend.state import (
    websockets, workflows, _monitor_tasks,
    _run_tasks, state_lock, initializing_sessions,
    report_websockets, report_ws_lock, detailed_reports,
    sessions as _sessions,
    get_session_lock,
)
from backend.utils.auth import ws_authenticate_and_load
from backend.utils.config_resolver import resolve_config_for_user
from backend.services.session_store import (
    update_session_cache as _update_session_cache,
    async_save_session as _async_save_session,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websockets"])


def _in_integration_loop(state) -> bool:
    from ..models.schemas import AnalysisPhase
    from ..workflow.direction_coverage import direction_coverage_status_for_batch
    # User explicitly chose to write the final report — not in the loop.
    if getattr(state, "final_report_requested", False):
        return False
    batch = state.active_direction_batch or []
    batch_incomplete = bool(batch) and not direction_coverage_status_for_batch(state, batch).complete
    return bool(
        state.pending_integration
        or state.analysis_phase == AnalysisPhase.INTEGRATION
        or batch_incomplete
        or state.integration_gate_candidates
    )


async def _resend_wait_input_state_sync(websocket, state, session_id: str) -> None:
    """Re-emit state_sync for pending HITL gates after a reconnect.

    The LlmTimeoutDialog and disabled-agent dialog are driven exclusively by
    state_sync messages (useApi.ts). On page reload, the workflow may already
    be parked in _wait_for_llm_intervention with llm_failure_info set, but the
    reconnect path only replays message_history — which does not feed the
    state_sync handler. Without this resend the dialog never reopens.
    """
    reason = getattr(state, "wait_input_reason", None)
    _HITL_RESEND_REASONS = (
        "llm_failure",
        "agent_disabled",
        "marker_db_missing",
        "species_input",
        "gene_species_mismatch",
    )
    if reason not in _HITL_RESEND_REASONS:
        return
    payload = {
        "change_type": "llm_failure" if reason == "marker_db_missing" else reason,
        "version": getattr(state, "state_version", 0),
        "status": state.status.value if hasattr(state.status, "value") else state.status,
        "current_round": getattr(state, "current_round", 0),
        "analysis_phase": getattr(state, "analysis_phase", None),
        "timestamp": datetime.now().isoformat(),
        "unresolved_errors": len(state.get_unresolved_errors()) if hasattr(state, "get_unresolved_errors") else 0,
        "wait_input_reason": reason,
        "llm_failure_info": getattr(state, "llm_failure_info", None),
        "agent_disabled_info": getattr(state, "agent_disabled_info", None),
        "species_input_info": getattr(state, "species_input_info", None),
    }
    try:
        await websocket.send_json({
            "type": "state_sync",
            "agent": None,
            "content": payload,
            "timestamp": datetime.now().isoformat(),
        })
        logger.info(f"[{session_id}] Re-sent state_sync for pending {reason} on reconnect")
    except Exception:
        logger.warning(f"[{session_id}] Failed to resend wait-input state_sync", exc_info=True)


def _serialize_realtime_messages(messages):
    """Serialize a list of WebSocketMessage to JSON-safe dicts."""
    msg_data_list = []
    for msg in messages:
        msg_data_list.append({
            "type": msg.type,
            "agent": msg.agent.value if msg.agent else None,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat() if msg.timestamp else datetime.now().isoformat(),
            "id": msg.id
        })
    return msg_data_list


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket for real-time updates during analysis"""
    result = await ws_authenticate_and_load(websocket, session_id)
    if result is None:
        return
    user_id, state = result

    config = await resolve_config_for_user(user_id)

    if config is None:
        await websocket.send_json({"error": "No API key configured. Please configure your LLM API key in Settings before starting analysis."})
        await websocket.close()
        return

    logger.info(f"WebSocket connected for session {session_id}, state={state.status.value}")

    # P1-8: acquire session-scoped lock BEFORE the create-or-observe gate so
    # two concurrent connections cannot both decide to create a workflow
    # (TOCTOU). Lock order: session_lock first, then state_lock.
    _sess_lock = get_session_lock(session_id)
    async with _sess_lock:
        # Check if workflow is already running BEFORE updating websocket reference
        # Use lock to prevent race condition between check and set
        async with state_lock:
            workflow_running = session_id in workflows
            workflow_initializing = session_id in initializing_sessions
            if not workflow_running and not workflow_initializing:
                # Only set websocket reference if we're the main connection (not an observer)
                websockets[session_id] = websocket
                # Mark as initializing to prevent concurrent workflow creation
                initializing_sessions.add(session_id)

    if workflow_running or workflow_initializing:
        logger.info(f"Workflow already running for session {session_id}, reconnecting WebSocket...")

        # is_restoring is True ONLY when initializing_sessions has the session
        # but workflows does NOT. This happens during snapshot restore, which
        # adds to initializing_sessions before popping workflows. In this case
        # there is no running workflow to observe — just wait for restore to
        # finish. Don't send stale status/history to the frontend.
        is_restoring = workflow_initializing and not workflow_running

        # Cancel old monitor task to prevent zombie background tasks
        old_monitor = _monitor_tasks.pop(session_id, None)
        if old_monitor and not old_monitor.done():
            old_monitor.cancel()
            logger.info(f"Cancelled old monitor task for session {session_id}")

        # Update websocket reference so message_callback sends to new connection
        async with state_lock:
            websockets[session_id] = websocket

        if not is_restoring:
            # Only send status/history if a workflow is actually running
            try:
                await websocket.send_json({
                    "type": "status",
                    "agent": None,
                    "content": f"Analysis in progress (Round {state.current_round})...",
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                logger.exception("Failed to send status to observer")

            # Send message history from state (same approach as completed/HITL restore)
            if state.realtime_messages:
                msg_data_list = _serialize_realtime_messages(state.realtime_messages)
                logger.info(f"Sending {len(msg_data_list)} messages from history to observer")
                try:
                    await websocket.send_json({
                        "type": "message_history",
                        "content": {
                            "messages": msg_data_list,
                            "count": len(msg_data_list)
                        },
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Failed to send history to observer: {e}")

            # Re-emit pending HITL state_sync so the frontend can reopen the
            # LlmTimeoutDialog / disabled-agent dialog after a page reload.
            # message_history alone doesn't drive the state_sync handler, so
            # llmFailureInfo would otherwise stay null until the next event.
            await _resend_wait_input_state_sync(websocket, state, session_id)

        # Wait for workflow to complete AND restore to finish
        try:
            while session_id in workflows or session_id in initializing_sessions:
                # Keep connection alive by waiting for messages
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=WS_RECEIVE_TIMEOUT)
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive through proxies
                    try:
                        await websocket.send_json({
                            "type": "ping",
                            "timestamp": datetime.now().isoformat()
                        })
                    except WebSocketDisconnect:
                        logger.debug("Observer WebSocket disconnected during ping")
                        break
                    except Exception:
                        logger.warning("Observer WebSocket ping failed, closing", exc_info=True)
                        break
                except WebSocketDisconnect:
                    logger.debug("Observer WebSocket disconnected normally")
                    break
                except Exception:
                    logger.warning("Observer WebSocket receive failed, closing", exc_info=True)
                    break
        except WebSocketDisconnect:
            logger.info(f"WebSocket observer disconnected for session {session_id}")
        # Don't delete websockets[session_id] - we never owned it
        return

    # Check if session is already completed - don't restart workflow
    if state.status == TaskStatus.COMPLETED:
        logger.info(f"Session {session_id} already completed, not restarting workflow")
        # Don't overwrite global websockets[session_id] -- multiple clients may
        # connect to a completed session simultaneously (read-only). Each handler
        # uses its own local `websocket` variable instead.
        # Remove from initializing set since we're not starting a workflow
        async with state_lock:
            initializing_sessions.discard(session_id)

        try:
            # Send completed message with full session data for UI restoration
            await websocket.send_json({
                "type": "completed",
                "agent": None,
                "content": {
                    "status": "completed",
                    "message": "Analysis already completed. View results in the Reports tab.",
                    "final_summary": state.final_summary,
                    "round_results": [
                        {
                            "round": r.round_number,
                            "summary": r.pi_summary,
                            "pathology_findings": r.pathology_findings,
                            "bioinformatics_findings": r.bioinformatics_findings,
                            "biochemistry_findings": r.biochemistry_findings,
                            "literature_findings": r.literature_findings,
                        }
                        for r in state.round_results
                    ],
                },
                "timestamp": datetime.now().isoformat()
            })

            # Send message_history for UI message stream restoration
            if state.realtime_messages:
                msg_data_list = _serialize_realtime_messages(state.realtime_messages)
                await websocket.send_json({
                    "type": "message_history",
                    "content": {
                        "messages": msg_data_list,
                        "count": len(msg_data_list)
                    },
                    "timestamp": datetime.now().isoformat()
                })
                logger.info(f"[{session_id}] Sent {len(msg_data_list)} messages from history to reconnecting client")
        except Exception as e:
            logger.exception(f"[{session_id}] Failed to send completed status")

        # Keep connection open for potential client requests
        try:
            while True:
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=WS_RECEIVE_TIMEOUT)
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for completed session {session_id}")
        return

    # HITL-hold: after snapshot restore, hold connection without starting workflow.
    # The frontend will show the HITL dialog from data loaded via the /load endpoint.
    # When user confirms/chooses, the confirm/choose endpoint closes this WebSocket
    # with code 4000, triggering auto-reconnect which creates a new workflow.
    if state.status in (TaskStatus.WAITING_ANNOTATION, TaskStatus.WAITING_INPUT):
        logger.info(f"[{session_id}] HITL-hold: status={state.status.value}, waiting for user action")
        async with state_lock:
            initializing_sessions.discard(session_id)
        try:
            # Send message history (includes synthesized HITL messages from restore)
            if state.realtime_messages:
                msg_data_list = _serialize_realtime_messages(state.realtime_messages)
                await websocket.send_json({
                    "type": "message_history",
                    "content": {
                        "messages": msg_data_list,
                        "count": len(msg_data_list)
                    },
                    "timestamp": datetime.now().isoformat()
                })
            await _resend_wait_input_state_sync(websocket, state, session_id)
            # Hold connection until user action triggers close
            while True:
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=WS_RECEIVE_TIMEOUT)
                except asyncio.TimeoutError:
                    try:
                        await websocket.send_json({
                            "type": "ping",
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception:
                        break
        except Exception:
            logger.debug("WebSocket receive loop ended unexpectedly", exc_info=True)
        finally:
            async with state_lock:
                # Only clean up if we're still the active websocket
                if websockets.get(session_id) is websocket:
                    del websockets[session_id]
        return

    # Create message callback that uses the global websockets dict
    # This way, if the websocket reconnects, messages go to the new connection
    # Track WebSocket closed state with lock for thread-safe access
    ws_closed = False
    active_ws = websocket
    ws_lock = asyncio.Lock()

    async def message_callback(msg: WebSocketMessage):
        nonlocal ws_closed, active_ws

        # Save message to state.realtime_messages for session restoration
        # Strip base64 from figure messages before persisting to avoid memory bloat
        if msg.type == "figure" and isinstance(msg.content, str) and len(msg.content) > 1000:
            from copy import copy
            stripped_msg = copy(msg)
            stripped_msg.content = "[image]"
            state.realtime_messages.append(stripped_msg)
        else:
            state.realtime_messages.append(msg)
        # Cap realtime_messages to prevent unbounded memory growth
        if len(state.realtime_messages) > 1000:
            state.realtime_messages = state.realtime_messages[-1000:]

        msg_data = {
            "type": msg.type,
            "agent": msg.agent.value if msg.agent else None,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat() if msg.timestamp else datetime.now().isoformat(),
            "id": msg.id  # Include message ID for deduplication on client
        }

        # Hold ws_lock for the entire check-and-send sequence to prevent concurrent
        # coroutines from all reading ws_closed=False and then all trying to send
        # to an already-closed connection.
        async with ws_lock:
            current_ws = websockets.get(session_id)
            if ws_closed:
                # Check if a new websocket has connected (reset ws_closed if so)
                if current_ws and current_ws != active_ws:
                    # New connection detected, reset closed state
                    ws_closed = False
                    active_ws = current_ws
                else:
                    return  # Still closed, skip
            else:
                if current_ws and current_ws != active_ws:
                    active_ws = current_ws

            if not current_ws:
                logger.warning(f"[{session_id}] No WebSocket connection, message buffered: {msg.type}")
                return
            try:
                await current_ws.send_json(msg_data)
            except Exception as e:
                # Mark as closed to prevent further attempts
                ws_closed = True
                logger.warning(f"[{session_id}] WebSocket closed during send, marking as closed: {e}", exc_info=True)

    # Helper to safely send JSON
    async def safe_send(data: dict):
        nonlocal ws_closed, active_ws
        # Hold ws_lock for the entire check-and-send sequence (same pattern as message_callback)
        async with ws_lock:
            current_ws = websockets.get(session_id)
            if ws_closed:
                # Check if a new websocket has connected
                if current_ws and current_ws != active_ws:
                    ws_closed = False
                    active_ws = current_ws
                else:
                    return False
            else:
                if current_ws and current_ws != active_ws:
                    active_ws = current_ws

            if not current_ws:
                return False
            try:
                await current_ws.send_json(data)
                return True
            except Exception:
                logger.warning("WebSocket send_json failed, marking closed", exc_info=True)
                ws_closed = True
                return False

    # Send initial connection confirmation message
    try:
        await websocket.send_json({
            "type": "status",
            "agent": None,
            "content": "WebSocket connected, starting analysis...",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.exception(f"[{session_id}] Failed to send initial message")

    # State callback to save after each round
    # (_sess_lock acquired above for create-or-observe gate; reuse same handle)

    async def _write_state_if_current(state: AnalysisState, *, set_paused=False, set_failed=False) -> bool:
        """Write state to cache and disk only if this workflow is still current.
        
        Returns True if state was written, False if workflow was superseded.
        If set_paused/set_failed is True, applies the status change inside the lock
        before writing — ensuring no write happens if superseded.
        """
        async with _sess_lock:
            if workflows.get(session_id) is not workflow:
                logger.info(f"[{session_id}] Workflow superseded, skipping stale write")
                return False
            if set_paused and state.status == TaskStatus.RUNNING:
                state.status = TaskStatus.PAUSED
            if set_failed:
                state.status = TaskStatus.FAILED
            await _update_session_cache(session_id, state)
            await _async_save_session(session_id, state)
            return True

    async def state_callback(state: AnalysisState):
        if not await _write_state_if_current(state):
            return

    # Import workflow here to avoid circular imports
    from backend.workflow import SingleCellAnalysisWorkflow

    # Create workflow with session_id for state persistence.
    # If construction fails we are NOT yet inside the run try/finally (below),
    # so initializing_sessions would leak — clean it up here, otherwise later
    # observer connections spin forever in `while session_id in initializing_sessions`.
    try:
        workflow = SingleCellAnalysisWorkflow(
            config=config,
            session_id=session_id,
            message_callback=message_callback,
            state_callback=state_callback
        )
        # Add workflow to tracking dict (protected by lock to prevent race conditions)
        async with state_lock:
            workflows[session_id] = workflow
    except Exception:
        logger.exception(f"[{session_id}] Failed to create workflow")
        async with state_lock:
            initializing_sessions.discard(session_id)
            if websockets.get(session_id) is websocket:
                del websockets[session_id]
        try:
            await websocket.send_json({
                "type": "error",
                "agent": None,
                "content": "Failed to initialize analysis workflow.",
                "timestamp": datetime.now().isoformat()
            })
        except Exception:
            pass
        return

    # Background task to monitor WebSocket for client disconnect
    async def monitor_websocket():
        """Listen for WebSocket disconnect while workflow is running, with heartbeat"""
        nonlocal ws_closed
        try:
            while True:
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=WS_RECEIVE_TIMEOUT)
                except asyncio.TimeoutError:
                    # Send heartbeat ping to keep connection alive through proxies
                    async with ws_lock:
                        if ws_closed:
                            break
                        current_ws = websockets.get(session_id)
                        if current_ws:
                            try:
                                await current_ws.send_json({
                                    "type": "ping",
                                    "timestamp": datetime.now().isoformat()
                                })
                            except Exception:
                                logger.warning(f"[{session_id}] WebSocket ping failed, marking closed", exc_info=True)
                                ws_closed = True
                                break
        except WebSocketDisconnect:
            logger.info(f"[{session_id}] Client disconnected during workflow execution")
            async with ws_lock:
                ws_closed = True
        except asyncio.CancelledError:
            async with ws_lock:
                ws_closed = True
        except Exception as e:
            logger.exception(f"[{session_id}] Unexpected WebSocket monitor error")
            async with ws_lock:
                ws_closed = True

    monitor_task = asyncio.create_task(monitor_websocket())
    _monitor_tasks[session_id] = monitor_task

    # Detect sessions that completed analysis but were interrupted during report generation
    effective_max_rounds = state.max_rounds if state.max_rounds > 0 else config.max_rounds
    in_integration_loop = _in_integration_loop(state)
    finalize_requested = getattr(state, "final_report_requested", False)
    needs_report_only = finalize_requested or (
        state.status in (TaskStatus.RUNNING, TaskStatus.WAITING_INPUT,
                         TaskStatus.FAILED, TaskStatus.GENERATING_REPORT)
        and state.current_round >= effective_max_rounds
        and len(state.round_results) >= effective_max_rounds
        and not in_integration_loop
    )

    try:
        # Run workflow (or report-only if analysis was already complete)
        if needs_report_only:
            logger.info(f"Session {session_id} needs report generation only "
                        f"(round {state.current_round}/{effective_max_rounds}, "
                        f"{len(state.round_results)} round results)")
            run_coro = workflow.generate_report(state, session_id)
        else:
            # Check if we should resume or start fresh
            if state.current_round > 0 or state.status in (
                TaskStatus.PAUSED, TaskStatus.WAITING_INPUT, TaskStatus.WAITING_ANNOTATION
            ):
                logger.info(f"Resuming workflow for session {session_id}")
                run_coro = workflow.resume(session_id, state)
            else:
                logger.info(f"Starting workflow for session {session_id}")
                run_coro = workflow.run(state, session_id)

        # Wrap in asyncio.Task for cancellation support (stop endpoint)
        run_task = asyncio.create_task(run_coro)
        _run_tasks[session_id] = run_task
        try:
            final_state = await run_task
        except asyncio.CancelledError:
            logger.info(f"[{session_id}] Workflow task was cancelled")
            current_state = _sessions.get(session_id, state)
            if current_state.status == TaskStatus.GENERATING_REPORT:
                # Stop-and-summarize (not hit during restore; only normal stop)
                try:
                    logger.info(f"[{session_id}] Starting report generation after stop-and-summarize")
                    report_state = await workflow.generate_report(current_state, session_id)
                    await _update_session_cache(session_id, report_state)
                    await _async_save_session(session_id, report_state)
                    await safe_send({
                        "type": "completed",
                        "content": {
                            "status": report_state.status.value,
                            "rounds": len(report_state.round_results),
                            "final_summary": report_state.final_summary
                        }
                    })
                    final_state = report_state
                except Exception as e:
                    logger.exception(f"[{session_id}] Report generation after stop failed")
                    current_state.status = TaskStatus.FAILED
                    await _update_session_cache(session_id, current_state)
                    await _async_save_session(session_id, current_state)
                    final_state = current_state
            else:
                if await _write_state_if_current(current_state, set_paused=True):
                    final_state = current_state
                else:
                    final_state = _sessions.get(session_id, state)
        finally:
            _run_tasks.pop(session_id, None)

        # Second write after inner finally — guarded by per-session lock
        if await _write_state_if_current(final_state):
            logger.info(f"Workflow completed for session {session_id}")

        # Send completion message (safely) - only if actually completed
        if final_state.status == TaskStatus.COMPLETED:
            await safe_send({
                "type": "completed",
                "content": {
                    "status": final_state.status.value,
                    "rounds": len(final_state.round_results),
                    "final_summary": final_state.final_summary
                }
            })

    except Exception as e:
        # Log the full error traceback
        logger.exception(f"Workflow error for session {session_id}")

        # Try to send error, but don't fail if WebSocket is closed
        await safe_send({
            "type": "error",
            "content": str(e)
        })
        current_state = _sessions.get(session_id, state)
        await _write_state_if_current(current_state, set_failed=True)

    finally:
        # Clean up task tracking
        _run_tasks.pop(session_id, None)

        # Cancel the WebSocket monitor task
        monitor_task.cancel()
        _monitor_tasks.pop(session_id, None)
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        # Clean up with lock protection
        logger.info(f"Cleaning up session {session_id}")
        try:
            await workflow.close()
        except Exception as e:
            logger.exception("Error closing workflow")
        async with state_lock:
            if session_id in workflows:
                del workflows[session_id]
            if websockets.get(session_id) is websocket:
                del websockets[session_id]
            # Remove from initializing set
            initializing_sessions.discard(session_id)


@router.websocket("/ws/report/{session_id}")
async def report_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for report generation progress — authenticated via scai.jwt.v1 subprotocol."""
    result = await ws_authenticate_and_load(websocket, session_id)
    if result is None:
        return

    async with report_ws_lock:
        report_websockets[session_id] = websocket

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with report_ws_lock:
            if report_websockets.get(session_id) is websocket:
                del report_websockets[session_id]
