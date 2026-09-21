"""WebSocket analysis connection owners."""
from __future__ import annotations

_OWNED_NAMES = frozenset({'report_websocket', 'websocket_endpoint'})

bindings = None


def bind_facade(facade) -> None:
    global bindings
    bindings = facade
    globals().update({name: value for name, value in vars(facade).items() if not name.startswith("__") and name not in _OWNED_NAMES})


async def _handle_existing_or_held_connection(
    websocket, session_id, state, connection_ref, *,
    snapshot_operation_running, workflow_running, workflow_initializing,
    report_task_running, report_workflow_running, report_tasks,
    _discard_captured_initializing,
) -> bool:
    if (
        snapshot_operation_running
        and not workflow_running
        and not workflow_initializing
    ):
        await bindings.send_connection_json(connection_ref, {
            "type": "status",
            "agent": None,
            "content": "Snapshot capture or restore is active; analysis start deferred.",
            "timestamp": datetime.now().isoformat(),
        })
        await bindings.unregister_websocket(connection_ref)
        await websocket.close(code=1013)
        return True

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
                await bindings.send_connection_json(connection_ref, {
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
                    (report_task := report_tasks.get(session_id)) is not None
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
                        await bindings.send_connection_json(connection_ref, {
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
            await bindings.unregister_websocket(connection_ref)
        return True

    # Check if session is already completed - don't restart workflow
    if state.status == TaskStatus.COMPLETED:
        logger.info(f"Session {session_id} already completed, not restarting workflow")
        # Remove from initializing set since we're not starting a workflow
        await _discard_captured_initializing()

        try:
            # Send completed message with full session data for UI restoration
            await bindings.send_connection_json(connection_ref, {
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
            await bindings.unregister_websocket(connection_ref)
        return True

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
                        await bindings.send_connection_json(connection_ref, {
                            "type": "ping",
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception:
                        break
        except Exception:
            logger.debug("WebSocket receive loop ended unexpectedly", exc_info=True)
        finally:
            await bindings.unregister_websocket(connection_ref)
        return True
    return False


async def _run_owned_analysis_connection(
    websocket, session_id, state, config, connection_ref, *,
    _sess_lock, initializing_token, _discard_captured_initializing,
):
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
        await bindings.send_connection_json(connection_ref, {
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

    async def _write_state_if_current(
        state: AnalysisState, *, set_paused=False, set_failed=False, force=False,
    ) -> bool:
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
            bindings._preserve_realtime_stream(_sessions.get(session_id), state)
            await bindings._update_session_cache(session_id, state)
            if force:
                await bindings._async_save_session(session_id, state, force=True)
            else:
                await bindings._async_save_session(session_id, state)
            return True

    async def state_callback(state: AnalysisState, *, force: bool = False):
        return await _write_state_if_current(state, force=force)

    # Import workflow here to avoid circular imports
    from backend.workflow import SingleCellAnalysisWorkflow
    from backend.services.session_store import assert_recovery_clear
    from backend.services.legacy_state_sanitizer import RecoveryBlockedError

    try:
        assert_recovery_clear(state, session_id=session_id)
    except RecoveryBlockedError as exc:
        logger.error("[%s] recovery blocked; refusing workflow construction: %s", session_id, exc)
        await _discard_captured_initializing()
        await bindings.send_connection_json(connection_ref, {
                "type": "error",
                "agent": None,
                "content": str(exc),
                "timestamp": datetime.now().isoformat()
        })
        await bindings.unregister_websocket(connection_ref)
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
            bindings.assert_generation_writable(session_id, state)
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
        await bindings.send_connection_json(connection_ref, {
                "type": "error",
                "agent": None,
                "content": "Failed to initialize analysis workflow.",
                "timestamp": datetime.now().isoformat()
        })
        await bindings.unregister_websocket(connection_ref)
        return

    # Background task to monitor WebSocket for client disconnect
    async def monitor_websocket():
        """Listen for WebSocket disconnect while workflow is running, with heartbeat"""
        try:
            while True:
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=WS_RECEIVE_TIMEOUT)
                except asyncio.TimeoutError:
                    if not await bindings.send_connection_json(connection_ref, {
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
            await bindings.unregister_websocket(connection_ref)

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
            bindings.assert_generation_writable(session_id, state)
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
                    await bindings._update_session_cache(session_id, report_state)
                    await bindings._async_save_session(session_id, report_state)
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
                    await bindings._update_session_cache(session_id, current_state)
                    await bindings._async_save_session(session_id, current_state)
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


async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket for real-time updates during analysis"""
    result = await bindings.ws_authenticate_and_load(websocket, session_id)
    if result is None:
        return
    user_id, state = result

    config = await bindings.resolve_config_for_user(user_id)

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
                bindings.assert_generation_writable(session_id, state)
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
            bindings.assert_generation_writable(session_id, state)
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
                ):
                    initializing_token = claim_initializing_session(session_id)
                    claimed_initializing = True
            connection_ref = await bindings.register_analysis_websocket(
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

    handled = await _handle_existing_or_held_connection(
        websocket, session_id, state, connection_ref,
        snapshot_operation_running=snapshot_operation_running,
        workflow_running=workflow_running,
        workflow_initializing=workflow_initializing,
        report_task_running=report_task_running,
        report_workflow_running=report_workflow_running,
        report_tasks=_report_tasks,
        _discard_captured_initializing=_discard_captured_initializing,
    )
    if handled:
        return
    return await _run_owned_analysis_connection(
        websocket, session_id, state, config, connection_ref,
        _sess_lock=_sess_lock,
        initializing_token=initializing_token,
        _discard_captured_initializing=_discard_captured_initializing,
    )

async def report_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for report generation progress — authenticated via scai.jwt.v1 subprotocol."""
    result = await bindings.ws_authenticate_and_load(websocket, session_id)
    if result is None:
        return

    _user_id, state = result
    try:
        async with get_lifecycle_lock(session_id):
            bindings.assert_generation_writable(session_id, state)
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
        await bindings.unregister_websocket(connection_ref)
