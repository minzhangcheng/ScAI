"""Periodic in-memory cache eviction for idle sessions + stale import jobs."""
import asyncio
import logging
import time
from typing import Any, Dict

from pathlib import Path

from backend.config import (
    CLEANUP_INTERVAL, EXPORT_TEMP_PREFIXES, IMPORT_JOB_TTL, MAX_CACHED_SESSIONS,
    PREPARED_EXPORT_TTL, SESSION_CACHE_TTL, UPLOAD_TEMP_DIR,
)
from backend.services.session_store import _async_save_session_under_lifecycle
from backend.services.export_lifecycle import (
    EXPORT_OWNERSHIP_MARKER_SUFFIX,
    cleanup_export_record,
    export_ownership_marker_matches,
    unlink_export_archive_and_marker,
)
from backend.services.session_lifecycle import get_lifecycle_lock
from backend.utils.asyncio_tasks import run_in_owned_thread
from backend.state import (
    sessions, workflows, report_workflows, initializing_sessions,
    state_lock, session_access_times, last_save_times,
    get_session_lock,
    report_websockets, detailed_reports, has_live_session_task,
    prepared_exports as global_prepared_exports,
    prepared_exports_lock as global_prepared_exports_lock,
)

logger = logging.getLogger(__name__)


async def _collect_sessions_to_evict(current_time: float) -> list[str]:
    """Pick idle sessions + overflow sessions. Holds state_lock once."""
    to_evict: list[str] = []
    async with state_lock:
        for session_id, last_access in list(session_access_times.items()):
            if (
                session_id in workflows
                or session_id in report_workflows
                or session_id in initializing_sessions
                or has_live_session_task(session_id)
            ):
                continue
            if current_time - last_access > SESSION_CACHE_TTL:
                to_evict.append(session_id)

        if len(sessions) > MAX_CACHED_SESSIONS:
            sorted_sessions = sorted(
                [(sid, session_access_times.get(sid, 0))
                 for sid in sessions.keys()
                 if sid not in workflows
                 and sid not in report_workflows
                 and sid not in initializing_sessions
                 and not has_live_session_task(sid)],
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
    """Save + remove one cache entry as one lifecycle operation."""
    # get_session_lock mutates session_locks dict, but in single-threaded
    # asyncio this is atomic between awaits — no state_lock needed here.
    session_lock = get_session_lock(session_id)

    async with session_lock:
        # ensure_session_loaded begins with this same lock. It can therefore
        # observe either the cached generation before eviction or the complete
        # disk write after eviction, never the save/pop gap between them.
        async with get_lifecycle_lock(session_id):
            async with state_lock:
                if (
                    session_id in initializing_sessions
                    or session_id in workflows
                    or session_id in report_workflows
                    or has_live_session_task(session_id)
                ):
                    return
                state_to_save = sessions.get(session_id)

            if state_to_save:
                try:
                    await _async_save_session_under_lifecycle(
                        session_id, state_to_save, force=True
                    )
                    logger.info(f"Saved state for evicted session {session_id}")
                except Exception:
                    logger.exception(f"Failed to save state for evicted session {session_id}")

            # stop(save=True) now preserves a live kernel when its final save
            # fails. Do this before evicting the cache entry and remove the
            # registry handle only after shutdown is proven, so the next sweep
            # can retry instead of orphaning the kernel.
            async with chat_task_lock:
                chat_sandbox = chat_sandboxes.get(session_id)
            if chat_sandbox:
                try:
                    await chat_sandbox.stop()
                    logger.info(
                        "Stopped chat sandbox for evicted session %s", session_id
                    )
                except Exception:
                    logger.exception(
                        "Failed to stop chat sandbox for session %s; "
                        "deferring cache eviction",
                        session_id,
                    )
                    return
                async with chat_task_lock:
                    if chat_sandboxes.get(session_id) is chat_sandbox:
                        chat_sandboxes.pop(session_id, None)

            async with state_lock:
                if (
                    session_id in initializing_sessions
                    or session_id in workflows
                    or session_id in report_workflows
                    or has_live_session_task(session_id)
                ):
                    return
                # Do not remove a replacement installed by another owner of
                # the cache; lifecycle serialization makes this mostly
                # defensive, while identity documents the intended generation.
                if sessions.get(session_id) is state_to_save:
                    sessions.pop(session_id, None)
                    session_access_times.pop(session_id, None)
                    last_save_times.pop(session_id, None)
                report_websockets.pop(session_id, None)
                # Keep the per-session mutation lock: eviction is not deletion,
                # and dropping it lets a concurrent route create a second lock.
                from backend.routes.reports import clear_detailed_report_cache, clear_generation_pending
                clear_detailed_report_cache(session_id)
                clear_generation_pending(session_id)


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


async def _cleanup_stale_prepared_exports(
    current_time: float,
    prepared_exports: Dict[str, dict],
    prepared_exports_lock: asyncio.Lock,
) -> None:
    """Delete export archives that were built but never downloaded.

    A prepared export is a full-size copy of the session on disk. The only
    unlink hangs off a successful download, so closing the "packaging complete"
    dialog -- or restarting the process -- used to strand the bytes forever,
    with no way left to reach them once the in-memory token was gone.
    """
    async with prepared_exports_lock:
        stale = [
            (token, info)
            for token, info in prepared_exports.items()
            if current_time - float(info.get("created_at", 0) or 0) > PREPARED_EXPORT_TTL
            and str(info.get("phase") or "ready")
            not in ("building", "claimed", "streaming")
            and not (
                isinstance(info.get("task"), asyncio.Task)
                and not info["task"].done()
            )
        ]
        for _token, info in stale:
            info["phase"] = "cleaning"

    removed = 0
    for token, info in stale:
        if await cleanup_export_record(
            info,
            registry=prepared_exports,
            registry_lock=prepared_exports_lock,
        ):
            removed += 1
    if removed:
        logger.info("Cleaned up %d stale prepared exports", removed)


def _unlink_orphaned_export_archives(
    max_age_s: float | None,
    protected_paths: frozenset[str] = frozenset(),
) -> int:
    """Delete export archives on disk that no live token can reach any more.

    ``_cleanup_stale_prepared_exports`` walks the in-memory ``prepared_exports``
    dict, so it can only reclaim what this process still remembers. A restart
    forgets every token and strands the files — full-size copies of a session —
    with nothing left that can ever name them. This is the disk-side half.
    """
    removed = 0
    try:
        entries = list(UPLOAD_TEMP_DIR.iterdir())
    except OSError:
        return 0
    now = time.time()
    for marker in entries:
        if not marker.name.endswith(EXPORT_OWNERSHIP_MARKER_SUFFIX):
            continue
        archive_name = marker.name[:-len(EXPORT_OWNERSHIP_MARKER_SUFFIX)]
        if not archive_name.startswith(EXPORT_TEMP_PREFIXES):
            continue
        path = marker.with_name(archive_name)
        if str(path) in protected_paths:
            continue
        try:
            if not export_ownership_marker_matches(path):
                continue
            if max_age_s is not None and now - path.stat().st_mtime <= max_age_s:
                continue
            unlink_export_archive_and_marker(
                path,
                require_marker=True,
                remove_marker=True,
            )
            removed += 1
        except OSError as exc:
            logger.warning("Failed to delete orphaned export archive %s: %s", path, exc)
    if removed:
        logger.info("Cleaned up %d orphaned export archives", removed)
    return removed


async def sweep_orphaned_export_archives(max_age_s: float | None = None) -> int:
    async with global_prepared_exports_lock:
        protected = frozenset(
            str(info.get("path"))
            for info in global_prepared_exports.values()
            if info.get("path")
        )
    return await run_in_owned_thread(
        _unlink_orphaned_export_archives,
        max_age_s,
        protected,
    )


async def cleanup_stale_sessions(
    chat_task_lock: asyncio.Lock,
    chat_sandboxes: Dict[str, Any],
    import_jobs: Dict[str, dict],
    import_jobs_lock: asyncio.Lock,
    prepared_exports: Dict[str, dict] | None = None,
    prepared_exports_lock: asyncio.Lock | None = None,
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
            if prepared_exports is not None and prepared_exports_lock is not None:
                await _cleanup_stale_prepared_exports(
                    current_time, prepared_exports, prepared_exports_lock
                )
            # In-memory sweep first, so the dict never outlives the file it names.
            await sweep_orphaned_export_archives(PREPARED_EXPORT_TTL)
            from backend.services.session_deletion import (
                sweep_tombstoned_generations,
            )

            await sweep_tombstoned_generations()

        except Exception:
            logger.exception("Error in session cleanup task")
