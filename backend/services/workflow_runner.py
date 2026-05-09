"""Workflow runner service: starts and manages headless analysis workflows.

Extracted from backend/main.py to decouple lifecycle route handlers from
workflow management logic.
"""
import asyncio
import logging
from typing import Optional

from backend.models.schemas import AnalysisState, TaskStatus
from backend.state import (
    sessions, workflows,
    state_lock as _state_lock,
    initializing_sessions as _initializing_sessions,
)
from backend.services.session_store import async_save_session, update_session_cache as _update_session_cache
from backend.utils.config_resolver import resolve_config_for_user

logger = logging.getLogger(__name__)


async def start_workflow_if_needed(session_id: str) -> None:
    """Start a headless workflow runner if not already running. Public API for v1 routes."""
    async with _state_lock:
        if session_id not in workflows and session_id not in _initializing_sessions:
            _initializing_sessions.add(session_id)
            asyncio.create_task(_headless_workflow_runner(session_id))


async def _headless_workflow_runner(session_id: str):
    """Run the analysis workflow without a WebSocket connection (for API usage)."""
    from backend.workflow import SingleCellAnalysisWorkflow

    try:
        state = sessions.get(session_id)
        if state is None:
            return

        user_id = getattr(state, "user_id", "") if state else ""
        cfg = await resolve_config_for_user(user_id)
        if cfg is None:
            logger.error("No global config available for session %s; aborting workflow", session_id)
            state.status = TaskStatus.FAILED
            state.error = "No API key configured. Please configure your LLM API key in Settings."
            await async_save_session(session_id, state)
            return

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

        async with _state_lock:
            workflows[session_id] = workflow
            _initializing_sessions.discard(session_id)

        await workflow.run()
        logger.info(f"Headless workflow completed for session {session_id}")

    except Exception as e:
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
