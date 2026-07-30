"""Workflow runner service: starts and manages headless analysis workflows.

Extracted from backend/main.py to decouple lifecycle route handlers from
workflow management logic.
"""
import asyncio
import logging
from backend.models.schemas import TaskStatus
from backend.workflow.helpers import WorkflowSupersededError
from backend.state import (
    sessions, workflows,
    state_lock as _state_lock,
    initializing_sessions as _initializing_sessions,
)
from backend.services.session_store import async_save_session, update_session_cache as _update_session_cache
from backend.utils.config_resolver import resolve_config_for_user

logger = logging.getLogger(__name__)

# In-flight report generation tasks: session_id -> asyncio.Task
_report_tasks: dict[str, asyncio.Task] = {}


async def _build_workflow_for_session(
    session_id: str,
    *,
    fail_on_missing_config: bool = True,
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

    def message_callback(msg):
        pass

    async def state_callback(updated_state):
        # Is-current guard, the headless counterpart of the WebSocket runner's
        # `_write_state_if_current`. That one keys on workflow identity, which
        # this runner cannot use: the report-only workflow is never registered
        # in `workflows`, so it has no anchor there — and for the same reason
        # `_retire_superseded_workflow` returns early on it and never sets the
        # superseded flag it would otherwise observe. A restore that commits
        # mid-run installs a NEW AnalysisState object in `sessions`, so object
        # identity is the signal that survives: without this check the run
        # finished on the state it started with and wrote it straight back over
        # the cache entry and session.json the restore had just installed,
        # silently undoing the restore.
        #
        # Checked under `state_lock` and released before writing —
        # `_update_session_cache` takes the same (non-reentrant) lock.
        async with _state_lock:
            current = sessions.get(session_id)
            superseded = current is not None and current is not updated_state
        if superseded:
            logger.info(
                "[%s] Session state was replaced (restore); skipping stale write",
                session_id,
            )
            return
        await _update_session_cache(session_id, updated_state)
        await async_save_session(session_id, updated_state)

    workflow = SingleCellAnalysisWorkflow(
        config=cfg,
        session_id=session_id,
        message_callback=message_callback,
        state_callback=state_callback,
    )

    return workflow, state


async def start_workflow_if_needed(session_id: str) -> None:
    """Start a headless workflow runner if not already running. Public API for v1 routes."""
    from backend.state import _run_tasks

    async with _state_lock:
        if session_id not in workflows and session_id not in _initializing_sessions:
            _initializing_sessions.add(session_id)
            # Register the handle in `_run_tasks`. That dict is the process-wide
            # proof that something is executing for this session, and until now
            # the WebSocket runner was its only writer — so a headless run was
            # invisible to every consumer. Most consequentially
            # `RestoreJobManager._acknowledge_if_quiescent` reads it to decide
            # whether a session is idle: a headless run blocked at a HITL gate
            # looked quiescent, and the restore acknowledged its own safe point
            # on behalf of a workflow that was very much alive and holding a
            # live kernel — capturing pre_restore with `kernel_present: False`
            # and tearing the checkpointer down under a running graph.
            _run_tasks[session_id] = asyncio.create_task(
                _headless_workflow_runner(session_id)
            )


async def start_report_generation(session_id: str) -> None:
    """Launch report generation as a tracked background task (not fire-and-forget)."""
    async with _state_lock:
        if session_id in _report_tasks and not _report_tasks[session_id].done():
            return
        task = asyncio.create_task(_run_report_generation(session_id))
        _report_tasks[session_id] = task
        task.add_done_callback(lambda t: _report_tasks.pop(session_id, None))


async def _run_report_generation(session_id: str):
    """Run only the report generation phase, bypassing analysis rounds."""
    workflow, state = await _build_workflow_for_session(session_id)
    if workflow is None:
        return
    try:
        await workflow.generate_report(state, session_id)
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
        try:
            await workflow.close()
        except Exception:
            logger.exception(
                "Report generation cleanup failed for %s", session_id
            )


async def _headless_workflow_runner(session_id: str):
    """Run the analysis workflow without a WebSocket connection (for API usage)."""
    workflow = None
    try:
        workflow, state = await _build_workflow_for_session(session_id)
        if workflow is None:
            return

        async with _state_lock:
            workflows[session_id] = workflow
            _initializing_sessions.discard(session_id)

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
        # This run IS the superseded one, and it has now unwound — clear the
        # session-scoped flag so the next run does not inherit it. Every other
        # handler of this exception does the same; only this one did not.
        try:
            from backend.services.snapshot_v2.safe_point import (
                get_snapshot_safe_point_manager,
            )

            get_snapshot_safe_point_manager().clear_superseded(session_id)
        except Exception:
            logger.exception(
                "Failed to clear the superseded flag for session %s", session_id,
            )
    except Exception:
        logger.exception(f"Headless workflow failed for session {session_id}")
        state = sessions.get(session_id)
        if state:
            state.status = TaskStatus.FAILED
            await _update_session_cache(session_id, state)
            await async_save_session(session_id, state)
    finally:
        if workflow is not None:
            try:
                await workflow.close()
            except Exception:
                logger.exception(
                    "Headless workflow cleanup failed for session %s", session_id
                )
        from backend.state import _run_tasks

        async with _state_lock:
            _initializing_sessions.discard(session_id)
            workflows.pop(session_id, None)
            # Only clear our own handle: a later run may already have registered
            # a new one under the same session id.
            if _run_tasks.get(session_id) is asyncio.current_task():
                _run_tasks.pop(session_id, None)
