"""WebSocket ownership owners."""
from __future__ import annotations

_OWNED_NAMES = frozenset({'_publish_stop_summary_task', '_cleanup_owned_workflow_handler', '_pop_if_owned'})

bindings = None


def bind_facade(facade) -> None:
    global bindings
    bindings = facade
    globals().update({name: value for name, value in vars(facade).items() if not name.startswith("__") and name not in _OWNED_NAMES})


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
        bindings.assert_generation_writable(session_id, state)
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

    await bindings.unregister_websocket(connection_ref)
    if cancellation is not None:
        raise cancellation
