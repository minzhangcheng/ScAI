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


def _session_workflow_activity(
    session_id: str,
) -> tuple[bool, bool, bool, bool]:
    """Classify graph and report-only owners for create-or-observe routing."""
    from backend.services.workflow_runner import _report_tasks

    report_task = _report_tasks.get(session_id)
    report_task_running = bool(
        report_task is not None and not report_task.done()
    )
    report_workflow_running = session_id in report_workflows
    detailed_report_running = bool(
        (detailed_task := detailed_report_tasks.get(session_id)) is not None
        and not detailed_task.done()
    )
    analysis_task_running = bool(
        (analysis_task := _run_tasks.get(session_id)) is not None
        and not analysis_task.done()
    )
    workflow_running = bool(
        session_id in workflows
        or report_workflow_running
        or analysis_task_running
    )
    workflow_initializing = bool(
        session_id in initializing_sessions
        or report_task_running
        or detailed_report_running
    )
    return (
        workflow_running,
        workflow_initializing,
        report_task_running,
        report_workflow_running,
    )

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websockets"])


def _needs_report_only(state, effective_max_rounds: int, in_integration_loop: bool) -> bool:
    """Whether this attach should generate the report instead of running the graph.

    Scoped to a session that completed its analysis and was interrupted during
    report generation — not to a user-initiated run. An armed
    ``restore_start_intent`` is exactly that ask: ``POST /run`` sets it as the
    "re-run from your data" the blocked-resume message promises, and import arms it
    for a checkpoint-less session. Answering it with a report fabricated a final
    report out of the imported rounds, cleared the runtime warning on the way in,
    and left ``runtime_recompute_pending`` unconsumed — only the bioinformatics
    node clears it — so the finished session's one remaining trace of an empty
    kernel was that flag.

    Suppressing report-only is sufficient on its own: ``_should_continue`` tests
    ``current_round >= max_rounds`` only *after* a round body runs, so the
    preserved round re-executes and then routes to the report exactly once.
    Bumping ``max_rounds`` instead would run an extra round, because
    ``preserve_round_on_initialize`` keeps ``current_round`` where it is.

    ``final_report_requested`` keeps precedence: that is the explicit
    "write the report" intent, and it is already exempt from the integration-loop
    guard.
    """
    if getattr(state, "final_report_requested", False):
        return True
    if getattr(state, "restore_start_intent", None) == "explicit_start":
        return False
    return (
        state.status in (TaskStatus.RUNNING, TaskStatus.WAITING_INPUT,
                         TaskStatus.FAILED, TaskStatus.GENERATING_REPORT)
        and state.current_round >= effective_max_rounds
        and len(state.round_results) >= effective_max_rounds
        and not in_integration_loop
    )


def _in_integration_loop(state) -> bool:
    # User explicitly chose to write the final report — not in the loop.
    if getattr(state, "final_report_requested", False):
        return False
    # Cheap boolean checks first; only fall back to the O(N) batch scan last.
    if getattr(state, "pending_integration", False):
        return True
    if state.analysis_phase == AnalysisPhase.INTEGRATION:
        return True
    if getattr(state, "integration_gate_candidates", None):
        return True
    return active_batch_incomplete(state)


def should_resume_workflow(state) -> bool:
    """Whether this connection continues an existing run or starts a fresh one.

    Named rather than inlined because it is the point where an imported session
    is committed to a path: ``resume()`` refuses to invent a fresh
    ``initialize_round`` run and moves only on an explicit restore intent, while
    ``run()`` always starts one. A session that has executed at least one round
    must take the former even after a runtime-less import, and one that never
    got past round 0 must take the latter, or it holds without executing.
    """
    return state.current_round > 0 or state.status in (
        TaskStatus.PAUSED, TaskStatus.WAITING_INPUT, TaskStatus.WAITING_ANNOTATION
    )


async def _send_restore_state_sync(websocket, state, session_id: str) -> None:
    """Emit a state_sync after a restored session connects.

    Communicates the restore kind and branch identity to the frontend so it
    can render the appropriate banner/dialog (graph-node restore, terminal
    restore, idle restore, etc.).
    """
    kind = getattr(state, "restore_resume_kind", None)
    if kind is None:
        return
    branch_id = getattr(state, "restore_branch_id", None) or ""
    payload = {
        "change_type": f"restore_{kind}",
        "version": getattr(state, "state_version", 0),
        "status": state.status.value if hasattr(state.status, "value") else state.status,
        "current_round": getattr(state, "current_round", 0),
        "analysis_phase": getattr(state, "analysis_phase", None),
        "timestamp": datetime.now().isoformat(),
        "restore_resume_kind": kind,
        "restore_resume_node": getattr(state, "restore_resume_node", None),
        "restore_resume_gate": getattr(state, "restore_resume_gate", None),
        "restore_branch_id": branch_id,
        "restore_branch_label": getattr(state, "restore_branch_label", None) or "",
    }
    try:
        message = {
            "type": "state_sync",
            "agent": None,
            "content": payload,
            "timestamp": datetime.now().isoformat(),
        }
        if isinstance(websocket, WebSocketConnection):
            sent = await send_connection_json(websocket, message)
        else:  # Backward-compatible test/helper caller.
            await websocket.send_json(message)
            sent = True
        if sent:
            logger.info(
                f"[{session_id}] Sent restore state_sync for kind={kind} "
                f"branch_id={branch_id}"
            )
    except Exception:
        logger.warning(f"[{session_id}] Failed to send restore state_sync", exc_info=True)


async def _resend_wait_input_state_sync(websocket, state, session_id: str) -> None:
    """Re-emit state_sync for pending HITL gates after a reconnect.

    The LlmTimeoutDialog and disabled-agent dialog are driven exclusively by
    state_sync messages (useApi.ts). On page reload, the workflow may already
    be parked in _wait_for_llm_intervention with llm_failure_info set, but the
    reconnect path only replays message_history — which does not feed the
    state_sync handler. Without this resend the dialog never reopens.

    Also re-emits snapshot_failure so the failure HITL gate survives reconnect.
    """
    from backend.services.snapshot_v2.failure_gate import (
        upgrade_legacy_snapshot_failure_gate,
    )

    upgraded_legacy_gate = upgrade_legacy_snapshot_failure_gate(state)
    if upgraded_legacy_gate:
        logger.info(
            "[%s] Upgraded legacy snapshot-failure gate on reconnect",
            session_id,
        )

    reason = getattr(state, "wait_input_reason", None)
    snapshot_failure = getattr(state, "snapshot_failure", None)
    _HITL_RESEND_REASONS = (
        "llm_failure",
        "agent_disabled",
        "marker_db_missing",
        "species_input",
        "gene_species_mismatch",
        "snapshot_failure",
    )
    if reason not in _HITL_RESEND_REASONS and not snapshot_failure:
        return
    # Prefer snapshot_failure payload when present even if reason drifted.
    effective_reason = reason
    if snapshot_failure and reason != "snapshot_failure":
        effective_reason = "snapshot_failure"
    change_type = (
        "snapshot_failure"
        if effective_reason == "snapshot_failure" or snapshot_failure
        else ("llm_failure" if effective_reason == "marker_db_missing" else effective_reason)
    )
    payload = {
        "change_type": change_type,
        "version": getattr(state, "state_version", 0),
        "status": state.status.value if hasattr(state.status, "value") else state.status,
        "current_round": getattr(state, "current_round", 0),
        "analysis_phase": getattr(state, "analysis_phase", None),
        "timestamp": datetime.now().isoformat(),
        "unresolved_errors": len(state.get_unresolved_errors()) if hasattr(state, "get_unresolved_errors") else 0,
        "wait_input_reason": effective_reason or ("snapshot_failure" if snapshot_failure else reason),
        "llm_failure_info": getattr(state, "llm_failure_info", None),
        "agent_disabled_info": getattr(state, "agent_disabled_info", None),
        "species_input_info": getattr(state, "species_input_info", None),
        "snapshot_failure": snapshot_failure,
        "snapshot_transaction": getattr(state, "snapshot_transaction", None),
    }
    try:
        message = {
            "type": "state_sync",
            "agent": None,
            "content": payload,
            "timestamp": datetime.now().isoformat(),
        }
        if isinstance(websocket, WebSocketConnection):
            sent = await send_connection_json(websocket, message)
        else:  # Backward-compatible test/helper caller.
            await websocket.send_json(message)
            sent = True
        if sent:
            logger.info(f"[{session_id}] Re-sent state_sync for pending {effective_reason or change_type} on reconnect")
    except Exception:
        logger.warning(f"[{session_id}] Failed to resend wait-input state_sync", exc_info=True)


def _preserve_realtime_stream(prev, new) -> None:
    """Carry the fuller realtime-message stream from the outgoing cached state
    object onto the incoming one before it becomes canonical.

    The running workflow replaces sessions[session_id] with fresh AnalysisState
    objects (LangGraph checkpoint round-trips). Only the WebSocket
    message_callback ever appends to realtime_messages, and it appends to the
    *currently cached* object, so the cached (prev) object always holds the
    fuller stream while the incoming graph object carries a stale/short copy.
    Move the longer stream forward so /load and reconnect keep serving the full
    history instead of the handful of messages that predate the first swap.
    """
    if (
        prev is not None
        and prev is not new
        and len(prev.realtime_messages) > len(new.realtime_messages)
    ):
        new.realtime_messages = prev.realtime_messages


def _serialize_realtime_messages(messages):
    """Serialize a list of WebSocketMessage to JSON-safe dicts."""
    return [serialize_realtime_message(msg, strip_binary=True) for msg in messages]


def _pop_if_owned(registry: dict, session_id: str, owned: Any) -> bool:
    """Remove a runtime handle only when this handler still owns the slot."""
    if owned is not None and registry.get(session_id) is owned:
        registry.pop(session_id, None)
        return True
    return False


async def _publish_stop_summary_task(
    session_id: str,
    state: AnalysisState,
    workflow: Any,
    prior_task: asyncio.Task | None,
) -> asyncio.Task:
    """Publish cancellation-triggered report work before it can execute.

    User stop-and-summarize legitimately cancels the graph task and starts a
    report. Session deletion uses the same cancellation signal, but first
    tombstones the generation and detaches every owner. Revalidating all three
    identities under the lifecycle barrier distinguishes those cases, while
    republishing the report task makes a later deletion able to cancel and
    drain it.
    """
    async with get_lifecycle_lock(session_id):
        assert_generation_writable(session_id, state)
        async with state_lock:
            if _sessions.get(session_id) is not state:
                raise SessionGenerationError(
                    "session generation changed before stop-summary publication"
                )
            if workflows.get(session_id) is not workflow:
                raise SessionGenerationError(
                    "workflow owner changed before stop-summary publication"
                )
            existing = _run_tasks.get(session_id)
            if (
                existing is not None
                and existing is not prior_task
                and not existing.done()
            ):
                raise SessionGenerationError(
                    "another analysis task owns stop-summary publication"
                )
            report_task = asyncio.create_task(
                workflow.generate_report(state, session_id),
                name=f"analysis-stop-summary:{session_id}",
            )
            _run_tasks[session_id] = report_task
            return report_task


async def _cleanup_owned_workflow_handler(
    session_id: str,
    *,
    workflow: Any,
    run_task: asyncio.Task | None,
    monitor_task: asyncio.Task | None,
    connection_ref: WebSocketConnection,
    initializing_token: object | None = None,
) -> None:
    """Retire one handler without touching a replacement generation.

    Snapshot restore and a delayed WebSocket ``finally`` can overlap.  Every
    process-wide registry therefore uses compare-and-remove semantics.  The old
    objects are still closed/cancelled so they do not leak, but a newer object
    occupying the same session slot is never cancelled, closed, or removed.
    """
    _pop_if_owned(_run_tasks, session_id, run_task)

    cancellation: asyncio.CancelledError | None = None
    close_complete = False
    logger.info("Cleaning up session %s", session_id)
    try:
        close_result = await workflow.close()
        # Compatibility for workflow stand-ins whose close() predates the
        # explicit boolean contract: only literal False means retained.
        close_complete = close_result is not False
    except asyncio.CancelledError as exc:
        # close() owns and shields its cleanup task. Preserve the registry while
        # that task keeps the persistence gate/live kernel reachable.
        cancellation = exc
        consume_current_cancellation_request()
        close_complete = bool(getattr(workflow, "_close_complete", False))
    except Exception:
        logger.exception("Error closing workflow")
        close_complete = bool(getattr(workflow, "_close_complete", False))

    # Keep the delivery generation registered until close() has either saved the
    # runtime or established its HITL gate. Otherwise the current client receives
    # COMPLETED but misses the corrective WAITING_INPUT state_sync.
    if monitor_task is not None:
        if not monitor_task.done():
            monitor_task.cancel()
        _pop_if_owned(_monitor_tasks, session_id, monitor_task)
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

    async with state_lock:
        if close_complete and workflows.get(session_id) is workflow:
            workflows.pop(session_id, None)
            if initializing_token is not None:
                release_initializing_session(session_id, initializing_token)
        elif not close_complete and workflows.get(session_id) is workflow:
            logger.warning(
                "[%s] Workflow cleanup is incomplete; retaining registry owner",
                session_id,
            )

    await unregister_websocket(connection_ref)
    if cancellation is not None:
        raise cancellation


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

    # The lifecycle lock is both the create-or-observe mutex and the deletion
    # barrier. If registration wins, deletion observes and retires the socket
    # plus initializing marker; if a tombstone wins, the captured generation
    # cannot publish either surface afterwards.
    _sess_lock = get_session_lock(session_id)

    initializing_token: object | None = None
    snapshot_operation_running = False

    async def _discard_captured_initializing() -> None:
        """Clear only this generation's create-or-observe marker."""
        try:
            async with get_lifecycle_lock(session_id):
                assert_generation_writable(session_id, state)
                async with state_lock:
                    if _sessions.get(session_id) is state:
                        release_initializing_session(
                            session_id, initializing_token,
                        )
        except (SessionGoneError, SessionGenerationError):
            # Deletion/replacement owns every marker at this point. A stale
            # handler must not erase the next generation's initialization.
            return

    claimed_initializing = False
    try:
        async with get_lifecycle_lock(session_id):
            assert_generation_writable(session_id, state)
            snapshot_operation_running = snapshot_operation_active(session_id)
            async with state_lock:
                from backend.services.workflow_runner import _report_tasks

                (
                    workflow_running,
                    workflow_initializing,
                    report_task_running,
                    report_workflow_running,
                ) = _session_workflow_activity(session_id)
                if (
                    not workflow_running
                    and not workflow_initializing
                    and not snapshot_operation_running
                ):
                    initializing_token = claim_initializing_session(session_id)
                    claimed_initializing = True
            connection_ref = await register_analysis_websocket(
                session_id, websocket,
            )
    except SessionGoneError:
        if claimed_initializing:
            await _discard_captured_initializing()
        await websocket.close(
            code=WS_SESSION_GONE_CODE,
            reason="SESSION_GONE",
        )
        return
    except SessionGenerationError:
        if claimed_initializing:
            await _discard_captured_initializing()
        await websocket.close(
            code=WS_SESSION_NOT_FOUND_CODE,
            reason="SESSION_NOT_FOUND",
        )
        return
    except BaseException:
        if claimed_initializing:
            await _discard_captured_initializing()
        raise

    if (
        snapshot_operation_running
        and not workflow_running
        and not workflow_initializing
    ):
        await send_connection_json(connection_ref, {
            "type": "status",
            "agent": None,
            "content": "Snapshot capture or restore is active; analysis start deferred.",
            "timestamp": datetime.now().isoformat(),
        })
        await unregister_websocket(connection_ref)
        await websocket.close(code=1013)
        return

    if workflow_running or workflow_initializing:
        logger.info(f"Workflow already running for session {session_id}, reconnecting WebSocket...")

        # is_restoring is True ONLY when initializing_sessions has the session
        # but workflows does NOT. This happens during snapshot restore, which
        # adds to initializing_sessions before popping workflows. In this case
        # there is no running workflow to observe — just wait for restore to
        # finish. Don't send stale status/history to the frontend.
        is_restoring = bool(
            session_id in initializing_sessions
            and session_id not in workflows
            and not report_task_running
            and not report_workflow_running
        )

        if not is_restoring:
            # Only send status/history if a workflow is actually running
            try:
                await send_connection_json(connection_ref, {
                    "type": "status",
                    "agent": None,
                    "content": f"Analysis in progress (Round {state.current_round})...",
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                logger.exception("Failed to send status to observer")

            # Re-emit pending HITL state_sync so the frontend can reopen the
            # LlmTimeoutDialog / disabled-agent dialog after a page reload.
            await _resend_wait_input_state_sync(connection_ref, state, session_id)

        # Wait for workflow to complete AND restore to finish
        try:
            while (
                session_id in workflows
                or session_id in report_workflows
                or session_id in initializing_sessions
                or (
                    (analysis_task := _run_tasks.get(session_id)) is not None
                    and not analysis_task.done()
                )
                or (
                    (report_task := _report_tasks.get(session_id)) is not None
                    and not report_task.done()
                )
                or (
                    (detailed_task := detailed_report_tasks.get(session_id))
                    is not None
                    and not detailed_task.done()
                )
            ):
                # Keep connection alive by waiting for messages
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=WS_RECEIVE_TIMEOUT)
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive through proxies
                    try:
                        await send_connection_json(connection_ref, {
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
        finally:
            await unregister_websocket(connection_ref)
        return

    # Check if session is already completed - don't restart workflow
    if state.status == TaskStatus.COMPLETED:
        logger.info(f"Session {session_id} already completed, not restarting workflow")
        # Remove from initializing set since we're not starting a workflow
        await _discard_captured_initializing()

        try:
            # Send completed message with full session data for UI restoration
            await send_connection_json(connection_ref, {
                "type": "completed",
                "agent": None,
                "content": {
                    "status": "completed",
                    "message": "Analysis already completed. View results in the Reports tab.",
                    "final_summary": state.final_summary,
                    "round_results": serialize_round_results(state.round_results),
                },
                "timestamp": datetime.now().isoformat()
            })

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
        finally:
            await unregister_websocket(connection_ref)
        return

    # HITL-hold: after snapshot restore, hold connection without starting workflow.
    # The frontend will show the HITL dialog from data loaded via the /load endpoint.
    # When user confirms/chooses, the confirm/choose endpoint closes this WebSocket
    # with code 4000, triggering auto-reconnect which creates a new workflow.
    if state.status in (TaskStatus.WAITING_ANNOTATION, TaskStatus.WAITING_INPUT):
        logger.info(f"[{session_id}] HITL-hold: status={state.status.value}, waiting for user action")
        await _discard_captured_initializing()
        try:
            await _resend_wait_input_state_sync(connection_ref, state, session_id)
            # Hold connection until user action triggers close
            while True:
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=WS_RECEIVE_TIMEOUT)
                except asyncio.TimeoutError:
                    try:
                        await send_connection_json(connection_ref, {
                            "type": "ping",
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception:
                        break
        except Exception:
            logger.debug("WebSocket receive loop ended unexpectedly", exc_info=True)
        finally:
            await unregister_websocket(connection_ref)
        return

    # Persist before realtime delivery so reconnect history remains authoritative.
    async def message_callback(msg: WebSocketMessage):
        # Save message to realtime_messages for session restoration.
        # IMPORTANT: append to the *currently cached* session object, not the
        # `state` captured at handler entry. The running workflow replaces
        # sessions[session_id] with fresh AnalysisState objects (LangGraph
        # checkpoint round-trips surfaced via state_callback), so appending to
        # the stale closure would strand the live stream on an orphaned object
        # that /load and reconnect never serve — the reason realtime messages
        # vanished on refresh. _write_state_if_current carries this stream
        # forward across each swap so nothing is lost.
        target = _sessions.get(session_id) or state
        append_durable_realtime_message(target, msg)
        msg_data = serialize_realtime_message(msg, strip_binary=False)

        if not await send_analysis_json(session_id, msg_data):
            logger.debug(
                "[%s] No live WebSocket generation; message remains buffered: %s",
                session_id,
                msg.type,
            )

    # Helper to safely send JSON
    async def safe_send(data: dict):
        return await send_analysis_json(session_id, data)

    # Send initial connection confirmation message
    try:
        await send_connection_json(connection_ref, {
            "type": "status",
            "agent": None,
            "content": "WebSocket connected, starting analysis...",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.exception(f"[{session_id}] Failed to send initial message")

    # Notify frontend about restored snapshot state (if any).
    # Must happen before the workflow is started so the UI can show
    # a restore banner / dialog before analysis begins.
    await _send_restore_state_sync(connection_ref, state, session_id)

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
            # Preserve the live realtime-message stream across state-object swaps
            # (see message_callback and _preserve_realtime_stream).
            _preserve_realtime_stream(_sessions.get(session_id), state)
            await _update_session_cache(session_id, state)
            await _async_save_session(session_id, state)
            return True

    async def state_callback(state: AnalysisState):
        return await _write_state_if_current(state)

    # Import workflow here to avoid circular imports
    from backend.workflow import SingleCellAnalysisWorkflow
    from backend.services.session_store import assert_recovery_clear
    from backend.services.snapshot_v2.errors import SnapshotTransactionError

    try:
        assert_recovery_clear(state, session_id=session_id)
    except SnapshotTransactionError as exc:
        logger.error("[%s] recovery blocked; refusing workflow construction: %s", session_id, exc)
        await _discard_captured_initializing()
        await send_connection_json(connection_ref, {
                "type": "error",
                "agent": None,
                "content": str(exc),
                "timestamp": datetime.now().isoformat()
        })
        await unregister_websocket(connection_ref)
        return

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
        # Publish the producer under the same lifecycle barrier as deletion.
        # A force-delete that won after socket registration closes that socket,
        # removes the initializing marker, and leaves a tombstone here; this
        # second gate prevents the detached handler from resurrecting a workflow.
        async with get_lifecycle_lock(session_id):
            assert_generation_writable(session_id, state)
            if snapshot_operation_active(session_id):
                raise SessionGenerationError(
                    "snapshot operation started before workflow publication"
                )
            async with state_lock:
                if _sessions.get(session_id) is not state:
                    raise SessionGenerationError(
                        "session state changed before workflow publication"
                    )
                workflows[session_id] = workflow
                release_initializing_session(session_id, initializing_token)
    except Exception:
        logger.exception(f"[{session_id}] Failed to create workflow")
        if "workflow" in locals():
            workflow._restore_superseded = True
            try:
                await workflow.close(save_runtime=False)
            except Exception:
                logger.exception(
                    "[%s] Failed to close unpublished workflow", session_id,
                )
        await _discard_captured_initializing()
        await send_connection_json(connection_ref, {
                "type": "error",
                "agent": None,
                "content": "Failed to initialize analysis workflow.",
                "timestamp": datetime.now().isoformat()
        })
        await unregister_websocket(connection_ref)
        return

    # Background task to monitor WebSocket for client disconnect
    async def monitor_websocket():
        """Listen for WebSocket disconnect while workflow is running, with heartbeat"""
        try:
            while True:
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=WS_RECEIVE_TIMEOUT)
                except asyncio.TimeoutError:
                    if not await send_connection_json(connection_ref, {
                        "type": "ping",
                        "timestamp": datetime.now().isoformat(),
                    }):
                        break
        except WebSocketDisconnect:
            logger.debug(f"[{session_id}] Client disconnected during workflow execution")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(
                "[%s] WebSocket monitor ended type=%s",
                session_id,
                type(exc).__name__,
            )
        finally:
            _pop_if_owned(_monitor_tasks, session_id, asyncio.current_task())
            await unregister_websocket(connection_ref)

    monitor_task = asyncio.create_task(monitor_websocket())
    _monitor_tasks[session_id] = monitor_task
    run_task: asyncio.Task | None = None

    # Detect sessions that completed analysis but were interrupted during report generation
    effective_max_rounds = state.max_rounds if state.max_rounds > 0 else config.max_rounds
    in_integration_loop = _in_integration_loop(state)
    needs_report_only = _needs_report_only(
        state, effective_max_rounds, in_integration_loop
    )

    try:
        # Creating the task and publishing its handle is a second producer
        # linearization point. Once this lock is released, force-delete either
        # sees the task and cancels it or has already tombstoned the generation
        # and this block refuses to start it.
        async with get_lifecycle_lock(session_id):
            assert_generation_writable(session_id, state)
            if workflows.get(session_id) is not workflow:
                raise SessionGenerationError(
                    "workflow ownership changed before task publication"
                )
            if needs_report_only:
                logger.info(
                    f"Session {session_id} needs report generation only "
                    f"(round {state.current_round}/{effective_max_rounds}, "
                    f"{len(state.round_results)} round results)"
                )
                run_coro = workflow.generate_report(state, session_id)
            else:
                if should_resume_workflow(state):
                    logger.info(f"Resuming workflow for session {session_id}")
                    run_coro = workflow.resume(session_id, state)
                else:
                    logger.info(f"Starting workflow for session {session_id}")
                    run_coro = workflow.run(state, session_id)
            run_task = asyncio.create_task(run_coro)
            _run_tasks[session_id] = run_task
        try:
            final_state = await run_task
        except asyncio.CancelledError:
            logger.info(f"[{session_id}] Workflow task was cancelled")
            current_state = _sessions.get(session_id, state)
            if current_state.status == TaskStatus.GENERATING_REPORT:
                # Stop-and-summarize (not hit during restore; only normal stop)
                report_task: asyncio.Task | None = None
                try:
                    logger.info(f"[{session_id}] Starting report generation after stop-and-summarize")
                    report_task = await _publish_stop_summary_task(
                        session_id,
                        current_state,
                        workflow,
                        run_task,
                    )
                    report_state = await report_task
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
                except (SessionGoneError, SessionGenerationError):
                    logger.info(
                        "[%s] Generation retired; skipping cancelled-run "
                        "report fallback",
                        session_id,
                    )
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.exception(f"[{session_id}] Report generation after stop failed")
                    current_state.status = TaskStatus.FAILED
                    await _update_session_cache(session_id, current_state)
                    await _async_save_session(session_id, current_state)
                    final_state = current_state
                finally:
                    _pop_if_owned(_run_tasks, session_id, report_task)
            else:
                if await _write_state_if_current(current_state, set_paused=True):
                    final_state = current_state
                else:
                    final_state = _sessions.get(session_id, state)
        finally:
            _pop_if_owned(_run_tasks, session_id, run_task)

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
        await _cleanup_owned_workflow_handler(
            session_id,
            workflow=workflow,
            run_task=run_task,
            monitor_task=monitor_task,
            connection_ref=connection_ref,
            initializing_token=initializing_token,
        )


@router.websocket("/ws/report/{session_id}")
async def report_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for report generation progress — authenticated via scai.jwt.v1 subprotocol."""
    result = await ws_authenticate_and_load(websocket, session_id)
    if result is None:
        return

    _user_id, state = result
    try:
        async with get_lifecycle_lock(session_id):
            assert_generation_writable(session_id, state)
            connection_ref = await register_report_websocket(
                session_id, websocket,
            )
    except SessionGoneError:
        await websocket.close(
            code=WS_SESSION_GONE_CODE,
            reason="SESSION_GONE",
        )
        return
    except SessionGenerationError:
        await websocket.close(
            code=WS_SESSION_NOT_FOUND_CODE,
            reason="SESSION_NOT_FOUND",
        )
        return

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await unregister_websocket(connection_ref)
