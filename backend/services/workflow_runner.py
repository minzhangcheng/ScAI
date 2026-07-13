"""Workflow runner service: starts and manages headless analysis workflows.

Extracted from backend/main.py to decouple lifecycle route handlers from
workflow management logic.
"""
import asyncio
import logging
from backend.models.schemas import TaskStatus
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
    async with _state_lock:
        if session_id not in workflows and session_id not in _initializing_sessions:
            _initializing_sessions.add(session_id)
            asyncio.create_task(_headless_workflow_runner(session_id))


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


async def _headless_workflow_runner(session_id: str):
    """Run the analysis workflow without a WebSocket connection (for API usage)."""
    try:
        workflow, state = await _build_workflow_for_session(session_id)
        if workflow is None:
            return

        async with _state_lock:
            workflows[session_id] = workflow
            _initializing_sessions.discard(session_id)

        if state.current_round > 0 or state.status in (
            TaskStatus.PAUSED,
            TaskStatus.WAITING_INPUT,
            TaskStatus.WAITING_ANNOTATION,
        ):
            await workflow.resume(session_id, state)
        else:
            await workflow.run(state, session_id)
        logger.info(f"Headless workflow completed for session {session_id}")

    except Exception:
        logger.exception(f"Headless workflow failed for session {session_id}")
        state = sessions.get(session_id)
        if state:
            state.status = TaskStatus.FAILED
            await _update_session_cache(session_id, state)
            await async_save_session(session_id, state)
    finally:
        async with _state_lock:
            _initializing_sessions.discard(session_id)
            workflows.pop(session_id, None)
