"""Workflow runner service: starts and manages headless analysis workflows.

Extracted from backend/main.py to decouple lifecycle route handlers from
workflow management logic.
"""
import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from backend.models.schemas import TaskStatus
from backend.workflow.helpers import WorkflowSupersededError
from backend.state import (
    _run_tasks,
    detailed_report_tasks,
    report_workflows, sessions, workflows,
    session_access_times,
    state_lock as _state_lock,
    initializing_sessions as _initializing_sessions,
    claim_initializing_session,
    release_initializing_session,
)
from backend.services.session_store import (
    _async_save_session_under_lifecycle,
    async_save_session,
    update_session_cache as _update_session_cache,
)
from backend.services.session_lifecycle import (
    assert_generation_writable,
    adopt_persisted_generation,
    get_lifecycle_lock,
)
from backend.utils.config_resolver import resolve_config_for_user

logger = logging.getLogger(__name__)

# In-flight report generation tasks: session_id -> asyncio.Task
_report_tasks: dict[str, asyncio.Task] = {}


class SessionWorkflowConflictError(RuntimeError):
    """A second workflow owner was requested for one session generation."""


def _graph_owner_running(session_id: str) -> bool:
    run_task = _run_tasks.get(session_id)
    return bool(
        session_id in workflows
        or session_id in _initializing_sessions
        or (run_task is not None and not run_task.done())
        or _task_running(detailed_report_tasks, session_id)
    )


def _task_running(registry: dict[str, asyncio.Task], session_id: str) -> bool:
    task = registry.get(session_id)
    return bool(task is not None and not task.done())


def session_has_any_workflow_owner(session_id: str) -> bool:
    """Return whether analysis or either report producer owns the session."""
    return bool(
        session_id in workflows
        or session_id in report_workflows
        or session_id in _initializing_sessions
        or _task_running(_run_tasks, session_id)
        or _task_running(_report_tasks, session_id)
        or _task_running(detailed_report_tasks, session_id)
    )


async def _build_workflow_for_session(
    session_id: str,
    *,
    fail_on_missing_config: bool = True,
    owner_registry: dict | None = None,
):
    """Load state + resolve config + construct the workflow. Returns (workflow, state) or (None, state/None).

    When ``fail_on_missing_config`` is False (capture-only rebuild), missing
    config returns ``(None, state)`` without mutating status to FAILED or
    persisting a failed session. Headless runners keep the default True.
    """
    from backend.workflow import SingleCellAnalysisWorkflow

    state = sessions.get(session_id)
    if state is None:
        return None, None

    user_id = getattr(state, "user_id", "") if state else ""
    cfg = await resolve_config_for_user(user_id)
    if cfg is None:
        logger.error("No global config available for session %s; aborting workflow", session_id)
        if fail_on_missing_config:
            state.status = TaskStatus.FAILED
            state.llm_failure_info = {
                "agent": "system",
                "error": "No API key configured. Please configure your LLM API key in Settings.",
            }
            await async_save_session(session_id, state)
            return None, None
        return None, state

    # Filled after construction so state_callback can identify the workflow
    # that owns graph writes without capturing a stale state object.
    published: dict[str, object] = {}

    def _owner_is_current(owner_registry, owner, current) -> bool:
        if current is None:
            return False
        if owner_registry is not None:
            return owner is not None and owner_registry.get(session_id) is owner
        # Report-only generation never round-trips through the graph, so it
        # retains the state-object identity guard.
        return current is state

    async def message_callback(msg):
        """Append only to a live canonical state owned by this workflow."""
        from backend.utils.websocket_delivery import (
            append_durable_realtime_message,
            send_analysis_json,
            serialize_realtime_message,
        )

        async with get_lifecycle_lock(session_id):
            async with _state_lock:
                current = sessions.get(session_id)
                owner = published.get("workflow")
                if not _owner_is_current(owner_registry, owner, current):
                    return False
                append_durable_realtime_message(current, msg)
        # Headless workflows have no socket callback in the workflow object,
        # but an already-connected analysis client still owns this channel.
        # Keep the live copy (including images) separate from the durable copy.
        await send_analysis_json(
            session_id,
            serialize_realtime_message(msg, strip_binary=False),
        )
        return True

    async def state_callback(updated_state, *, force: bool = False):
        """Atomically carry history, publish the graph state, and persist it."""
        from backend.routes.websockets import _preserve_realtime_stream

        # update_session_cache() and async_save_session() each acquire this
        # lifecycle lock themselves. Keep the complete callback transaction on
        # the same boundary, update the cache directly under state_lock, and
        # use the under-lifecycle persistence primitive to avoid re-entry.
        async with get_lifecycle_lock(session_id):
            async with _state_lock:
                current = sessions.get(session_id)
                owner = published.get("workflow")
                if not _owner_is_current(owner_registry, owner, current):
                    return False
                _preserve_realtime_stream(current, updated_state)
                # History carry is synchronous, but a restore/replacement
                # may have published while the incoming graph object was being
                # prepared. Recheck before changing the canonical cache so a
                # stale owner cannot overwrite that replacement.
                if not _owner_is_current(
                    owner_registry, owner, sessions.get(session_id)
                ):
                    return False
                adopt_persisted_generation(updated_state)
                sessions[session_id] = updated_state
                session_access_times[session_id] = time.time()

                # Reconfirm at the canonical commit point while the state lock
                # still excludes ordinary restore/cache replacement writers.
                if not _owner_is_current(
                    owner_registry, owner, sessions.get(session_id)
                ) or sessions.get(session_id) is not updated_state:
                    return False

            # The lifecycle lock excludes deletion/restore generation changes;
            # no state_lock is held while persistence snapshots the model.
            await _async_save_session_under_lifecycle(
                session_id, updated_state, force=force,
            )
        return True

    workflow = SingleCellAnalysisWorkflow(
        config=cfg,
        session_id=session_id,
        message_callback=message_callback,
        state_callback=state_callback,
    )
    published["workflow"] = workflow

    return workflow, state


async def start_workflow_if_needed(
    session_id: str,
    *,
    expected_state=None,
    prepare: Callable[[], Awaitable[None]] | None = None,
) -> bool:
    """Reserve and start one headless workflow for the current generation."""
    from backend.state import _run_tasks

    loop = asyncio.get_running_loop()
    preparation_done = loop.create_future() if prepare is not None else None
    initializing_token: object | None = None

    async def _reserved_runner() -> None:
        try:
            if prepare is not None:
                await prepare()
        except asyncio.CancelledError:
            if preparation_done is not None and not preparation_done.done():
                preparation_done.cancel()
            raise
        except BaseException as exc:
            if preparation_done is not None and not preparation_done.done():
                preparation_done.set_exception(exc)
            raise
        else:
            if preparation_done is not None and not preparation_done.done():
                preparation_done.set_result(None)
        await _headless_workflow_runner(
            session_id,
            initializing_token=initializing_token,
        )

    async with get_lifecycle_lock(session_id):
        async with _state_lock:
            current_state = sessions.get(session_id)
            if current_state is None or (
                expected_state is not None and current_state is not expected_state
            ):
                return False
            assert_generation_writable(session_id, current_state)
            report_task = _report_tasks.get(session_id)
            report_running = bool(
                session_id in report_workflows
                or (report_task is not None and not report_task.done())
                or _task_running(detailed_report_tasks, session_id)
            )
            if (
                session_id not in workflows
                and session_id not in _initializing_sessions
                and not _task_running(_run_tasks, session_id)
                and not report_running
            ):
                initializing_token = claim_initializing_session(session_id)
                # Register the handle in `_run_tasks`. That dict is the
                # process-wide proof that something is executing for this
                # session, including the build/publication gap.
                task = asyncio.create_task(
                    _reserved_runner(),
                    name=f"headless-workflow:{session_id}",
                )
                _run_tasks[session_id] = task

                def _retire_reserved_runner(done: asyncio.Task) -> None:
                    if _run_tasks.get(session_id) is done:
                        _run_tasks.pop(session_id, None)
                    if initializing_token is not None:
                        release_initializing_session(
                            session_id, initializing_token,
                        )
                    if preparation_done is not None and not preparation_done.done():
                        if done.cancelled():
                            preparation_done.cancel()
                        else:
                            exc = done.exception()
                            if exc is None:
                                preparation_done.set_result(None)
                            else:
                                preparation_done.set_exception(exc)
                    try:
                        done.result()
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        logger.exception(
                            "Reserved headless workflow failed for %s", session_id,
                        )

                task.add_done_callback(_retire_reserved_runner)
            else:
                return False

    if preparation_done is None:
        return True
    try:
        await asyncio.shield(preparation_done)
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None and current.cancelling() > 0:
            # Request cancellation must not cancel the independently registered
            # workflow owner halfway through its state preparation.
            raise
        await asyncio.gather(task, return_exceptions=True)
        raise
    except BaseException:
        await asyncio.gather(task, return_exceptions=True)
        raise
    return True


async def start_report_generation(
    session_id: str,
    *,
    expected_state=None,
    prepare: Callable[[], Awaitable[None]] | None = None,
) -> bool:
    """Reserve the sole workflow slot for one current lifecycle generation."""
    loop = asyncio.get_running_loop()
    preparation_done = loop.create_future()

    async def _reserved_runner() -> None:
        try:
            if prepare is not None:
                await prepare()
        except asyncio.CancelledError:
            if not preparation_done.done():
                preparation_done.cancel()
            raise
        except BaseException as exc:
            if not preparation_done.done():
                preparation_done.set_exception(exc)
            raise
        else:
            if not preparation_done.done():
                preparation_done.set_result(None)
        await _run_report_generation(session_id)

    # Deletion takes the same lifecycle -> state lock order. Publishing the
    # task under both prevents a post-tombstone producer that deletion could no
    # longer discover, cancel, and await.
    async with get_lifecycle_lock(session_id):
        async with _state_lock:
            current_state = sessions.get(session_id)
            if current_state is None or (
                expected_state is not None and current_state is not expected_state
            ):
                raise SessionWorkflowConflictError(
                    f"session generation changed before report start: {session_id}"
                )
            assert_generation_writable(session_id, current_state)
            if _graph_owner_running(session_id):
                raise SessionWorkflowConflictError(
                    f"analysis workflow is already active for {session_id}"
                )
            if session_id in report_workflows:
                return False
            if session_id in _report_tasks and not _report_tasks[session_id].done():
                return False
            task = asyncio.create_task(
                _reserved_runner(), name=f"standalone-report:{session_id}",
            )
            _report_tasks[session_id] = task

            def _remove_own_report_task(done: asyncio.Task) -> None:
                # A deleted/restored generation may have installed a replacement
                # under the same ID before this callback gets event-loop time.
                if _report_tasks.get(session_id) is done:
                    _report_tasks.pop(session_id, None)
                try:
                    done.result()
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception(
                        "Standalone report task failed for %s", session_id,
                    )

            task.add_done_callback(_remove_own_report_task)
    try:
        await asyncio.shield(preparation_done)
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None and current.cancelling() > 0:
            # The producer is independently registered and now owns any state
            # transition performed by prepare(). A disconnected HTTP caller
            # must not strand RUNNING state by cancelling that owner halfway.
            raise
        # The preparation itself was cancelled (for example by deletion).
        await asyncio.gather(task, return_exceptions=True)
        if _report_tasks.get(session_id) is task:
            _report_tasks.pop(session_id, None)
        raise
    except BaseException:
        # Preparation failed inside the owned task; drain and retire it before
        # returning the same failure to the caller.
        await asyncio.gather(task, return_exceptions=True)
        if _report_tasks.get(session_id) is task:
            _report_tasks.pop(session_id, None)
        raise
    return True


async def _run_report_generation(session_id: str):
    """Run only the report generation phase, bypassing analysis rounds."""
    workflow, state = await _build_workflow_for_session(session_id)
    if workflow is None:
        return
    stale_before_registration = False
    async with _state_lock:
        if sessions.get(session_id) is not state or _graph_owner_running(session_id):
            stale_before_registration = True
        else:
            report_workflows[session_id] = workflow
    if stale_before_registration:
        workflow._restore_superseded = True
        await workflow.close(save_runtime=False)
        return
    try:
        await workflow.generate_report(state, session_id)
    except WorkflowSupersededError:
        logger.info(
            "Standalone report workflow superseded by restore for %s",
            session_id,
        )
    except Exception:
        logger.exception("Report generation failed for %s", session_id)
    finally:
        # `generate_report` calls `initialize()`, which unconditionally opens an
        # aiosqlite WAL connection to the checkpoint database plus the LLM
        # manager, the settings provider and the eager agents' clients. Only
        # `close()` compacts and closes them. Leaving them open is not merely a
        # delayed reclaim: this workflow is never registered in
        # `backend.state.workflows`, so `_retire_superseded_workflow` — which
        # exists to close `_db_conn` before a restore's `apply()` renames the
        # checkpoint file into place — cannot reach it, and the orphaned
        # connection can checkpoint stale WAL frames into the restored file.
        # Mirrors `_headless_workflow_runner`'s finally.
        close_complete = False
        try:
            from backend.services.session_lifecycle import is_session_tombstoned

            close_result = await workflow.close(
                save_runtime=not is_session_tombstoned(session_id)
            )
            close_complete = close_result is not False
            if close_result is False:
                logger.warning(
                    "Report workflow cleanup remains incomplete for %s; "
                    "retaining its report-workflow owner",
                    session_id,
                )
        except Exception:
            logger.exception(
                "Report generation cleanup failed for %s", session_id
            )
        finally:
            if close_complete:
                async with _state_lock:
                    if report_workflows.get(session_id) is workflow:
                        report_workflows.pop(session_id, None)


async def _headless_workflow_runner(
    session_id: str,
    *,
    initializing_token: object | None = None,
):
    """Run the analysis workflow without a WebSocket connection (for API usage)."""
    # An import inside this function makes `_run_tasks` local for the whole
    # function. Bind it before the reservation check; otherwise a reserved
    # runner fails before reaching its first graph node.
    from backend.state import _run_tasks

    workflow = None
    try:
        workflow, state = await _build_workflow_for_session(
            session_id,
            owner_registry=workflows,
        )
        if workflow is None:
            return

        async with get_lifecycle_lock(session_id):
            assert_generation_writable(session_id, state)
            async with _state_lock:
                if (
                    sessions.get(session_id) is not state
                    or (
                        initializing_token is not None
                        and _run_tasks.get(session_id) is not asyncio.current_task()
                    )
                ):
                    raise WorkflowSupersededError(
                        f"headless workflow owner changed for {session_id}"
                    )
                workflows[session_id] = workflow
                if initializing_token is not None:
                    release_initializing_session(session_id, initializing_token)

        # One rule for resume-vs-fresh-start, shared with the WebSocket path so
        # the two cannot commit an imported session to different paths. Imported
        # locally: the helper lives in the routes layer, which must not be pulled
        # in while this service module is still being imported.
        from backend.routes.websockets import should_resume_workflow

        if should_resume_workflow(state):
            await workflow.resume(session_id, state)
        else:
            await workflow.run(state, session_id)
        logger.info(f"Headless workflow completed for session {session_id}")

    except WorkflowSupersededError:
        # A snapshot restore replaced the state this run was built on. The
        # restore already persisted the authoritative state; writing FAILED
        # here would overwrite exactly what it installed.
        logger.info(
            "Headless workflow superseded by a snapshot restore for session %s",
            session_id,
        )
    except Exception:
        logger.exception(f"Headless workflow failed for session {session_id}")
        # A replaced/deleted workflow must not resurrect its old state as
        # FAILED. Use the same lifecycle + workflow identity boundary as the
        # normal graph callback, then persist without re-entering either lock.
        async with get_lifecycle_lock(session_id):
            failed_state = None
            async with _state_lock:
                state = sessions.get(session_id)
                if (
                    state is None
                    or workflow is None
                    or workflows.get(session_id) is not workflow
                ):
                    logger.info(
                        "[%s] Failed workflow no longer owns the session; "
                        "skipping stale FAILED write",
                        session_id,
                    )
                else:
                    state.status = TaskStatus.FAILED
                    failed_state = state
            if failed_state is not None:
                await _async_save_session_under_lifecycle(session_id, failed_state)
    finally:
        close_complete = True
        if workflow is not None:
            try:
                close_result = await workflow.close()
                close_complete = close_result is not False
                if not close_complete:
                    logger.warning(
                        "Headless workflow cleanup remains incomplete for %s; "
                        "retaining its workflow owner",
                        session_id,
                    )
            except Exception:
                close_complete = bool(
                    getattr(workflow, "_close_complete", False)
                )
                logger.exception(
                    "Headless workflow cleanup failed for session %s", session_id
                )
        async with _state_lock:
            if initializing_token is not None:
                release_initializing_session(session_id, initializing_token)
            if close_complete and workflows.get(session_id) is workflow:
                workflows.pop(session_id, None)
            # Only clear our own handle: a later run may already have registered
            # a new one under the same session id.
            if _run_tasks.get(session_id) is asyncio.current_task():
                _run_tasks.pop(session_id, None)
