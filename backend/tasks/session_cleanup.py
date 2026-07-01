"""Periodic in-memory cache eviction for idle sessions + stale import jobs."""
import asyncio
import logging
import time
from typing import Any, Dict

from backend.config import (
    CLEANUP_INTERVAL, IMPORT_JOB_TTL, MAX_CACHED_SESSIONS, SESSION_CACHE_TTL,
)
from backend.services.session_store import async_save_session
from backend.state import (
    sessions, workflows, initializing_sessions,
    state_lock, session_locks, session_access_times, last_save_times,
    get_session_lock,
    report_websockets, detailed_reports,
)

logger = logging.getLogger(__name__)


async def _collect_sessions_to_evict(current_time: float) -> list[str]:
    """Pick idle sessions + overflow sessions. Holds state_lock once."""
    to_evict: list[str] = []
    async with state_lock:
        for session_id, last_access in list(session_access_times.items()):
            if session_id in workflows or session_id in initializing_sessions:
                continue
            if current_time - last_access > SESSION_CACHE_TTL:
                to_evict.append(session_id)

        if len(sessions) > MAX_CACHED_SESSIONS:
            sorted_sessions = sorted(
                [(sid, session_access_times.get(sid, 0))
                 for sid in sessions.keys()
                 if sid not in workflows and sid not in initializing_sessions],
                key=lambda x: x[1],
            )
            evict_count = len(sessions) - MAX_CACHED_SESSIONS
            for sid, _ in sorted_sessions[:evict_count]:
                if sid not in to_evict:
                    to_evict.append(sid)
    return to_evict


async def _evict_single_session(
    session_id: str,
    chat_task_lock: asyncio.Lock,
    chat_sandboxes: Dict[str, Any],
) -> None:
    """Save + remove one session. Save happens outside all locks."""
    # get_session_lock mutates session_locks dict, but in single-threaded
    # asyncio this is atomic between awaits — no state_lock needed here.
    session_lock = get_session_lock(session_id)

    async with session_lock:
        async with state_lock:
            if session_id in initializing_sessions or session_id in workflows:
                return
            state_to_save = sessions.get(session_id)

        if state_to_save:
            try:
                await async_save_session(session_id, state_to_save, force=True)
                logger.info(f"Saved state for evicted session {session_id}")
            except Exception:
                logger.exception(f"Failed to save state for evicted session {session_id}")

        async with state_lock:
            if session_id in initializing_sessions or session_id in workflows:
                return
            sessions.pop(session_id, None)
            session_access_times.pop(session_id, None)
            last_save_times.pop(session_id, None)
            session_locks.pop(session_id, None)
            report_websockets.pop(session_id, None)
            # Clean up report caches and pending flags
            from backend.routes.reports import clear_detailed_report_cache, clear_generation_pending
            clear_detailed_report_cache(session_id)
            clear_generation_pending(session_id)

    async with chat_task_lock:
        chat_sandbox = chat_sandboxes.pop(session_id, None)
    if chat_sandbox:
        try:
            await chat_sandbox.stop()
            logger.info(f"Stopped chat sandbox for evicted session {session_id}")
        except Exception:
            logger.exception(f"Failed to stop chat sandbox for session {session_id}")


async def _cleanup_stale_import_jobs(
    current_time: float,
    import_jobs: Dict[str, dict],
    import_jobs_lock: asyncio.Lock,
) -> None:
    async with import_jobs_lock:
        stale = [
            jid for jid, job in import_jobs.items()
            if job.get("status") in ("completed", "failed", "cancelled")
            and current_time - job.get("created_at", 0) > IMPORT_JOB_TTL
        ]
        for jid in stale:
            del import_jobs[jid]
    if stale:
        logger.info(f"Cleaned up {len(stale)} stale import jobs")


async def cleanup_stale_sessions(
    chat_task_lock: asyncio.Lock,
    chat_sandboxes: Dict[str, Any],
    import_jobs: Dict[str, dict],
    import_jobs_lock: asyncio.Lock,
) -> None:
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL)
            current_time = time.time()

            sessions_to_evict = await _collect_sessions_to_evict(current_time)

            for session_id in sessions_to_evict:
                await _evict_single_session(session_id, chat_task_lock, chat_sandboxes)

            if sessions_to_evict:
                logger.info(f"Evicted {len(sessions_to_evict)} stale sessions from memory")

            await _cleanup_stale_import_jobs(current_time, import_jobs, import_jobs_lock)

        except Exception:
            logger.exception("Error in session cleanup task")