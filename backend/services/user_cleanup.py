"""User data cleanup service: cascade delete sessions and files for a user."""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

from backend.services.session_lifecycle import (
    list_lifecycle_records,
    request_import_generation_cancellation,
)
from backend.services.session_store import list_saved_sessions, load_session_from_disk
from backend.state import (
    import_jobs,
    import_jobs_lock,
    import_generation_tasks,
    sessions,
    state_lock,
    users_pending_deletion,
)
from backend.utils.auth import get_db


logger = logging.getLogger(__name__)

_TERMINAL_IMPORT_STATUSES = frozenset({"completed", "failed", "cancelled"})


class UserCleanupPendingError(RuntimeError):
    """Account access is revoked, but some owned data still needs cleanup."""

    def __init__(
        self,
        user_id: str,
        *,
        pending_sessions: list[str],
        pending_import_jobs: list[str],
        failures: dict[str, str],
    ) -> None:
        self.user_id = user_id
        self.pending_sessions = sorted(set(pending_sessions))
        self.pending_import_jobs = sorted(set(pending_import_jobs))
        self.failures = dict(sorted(failures.items()))
        super().__init__(
            f"user {user_id} cleanup remains pending: "
            f"{len(self.pending_sessions)} session(s), "
            f"{len(self.pending_import_jobs)} import job(s), "
            f"{len(self.failures)} failure(s)"
        )

    def as_detail(self) -> dict[str, Any]:
        """Return a stable, retry-oriented API error payload."""
        return {
            "code": "user_cleanup_pending",
            "message": (
                "Account access was revoked, but data cleanup is still in "
                "progress. Retry deletion after active producers stop."
            ),
            "user_id": self.user_id,
            "pending_sessions": self.pending_sessions,
            "pending_import_jobs": self.pending_import_jobs,
            "failures": self.failures,
            "retryable": True,
        }


async def _cancel_user_import_jobs(user_id: str) -> list[str]:
    """Request cooperative cancellation for every non-terminal user import."""
    cancelled: list[str] = []
    async with import_jobs_lock:
        for job_id, job in import_jobs.items():
            if str(job.get("user_id") or "") != user_id:
                continue
            if str(job.get("status") or "") in _TERMINAL_IMPORT_STATUSES:
                continue
            job["cancel_requested"] = True
            cancelled.append(job_id)
    return cancelled


async def _pending_user_import_jobs(user_id: str) -> list[str]:
    async with import_jobs_lock:
        return sorted(
            job_id
            for job_id, job in import_jobs.items()
            if str(job.get("user_id") or "") == user_id
            and str(job.get("status") or "") not in _TERMINAL_IMPORT_STATUSES
        )


async def _live_import_generations(user_id: str) -> tuple[set[str], bool]:
    """Return live imported ids and whether a live job is not bound yet."""
    live_ids = {
        sid
        for sid, task in import_generation_tasks.items()
        if isinstance(task, asyncio.Task) and not task.done()
    }
    unbound_job = False
    async with import_jobs_lock:
        for job in import_jobs.values():
            if str(job.get("user_id") or "") != user_id:
                continue
            if str(job.get("status") or "") in _TERMINAL_IMPORT_STATUSES:
                continue
            session_id = str(job.get("session_id") or "")
            if session_id:
                live_ids.add(session_id)
            else:
                # Allocation and publication of session_id are consecutive in
                # the extraction thread, but account cleanup must remain safe
                # even if it observes that tiny interval.
                unbound_job = True
    return live_ids, unbound_job


def _owned_lifecycle_records(user_id: str) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("session_id") or ""): record
        for record in list_lifecycle_records()
        if record.get("session_id")
        and str(record.get("owner_id") or "") == user_id
    }


def _count_cleanup_stats(totals: dict[str, int], stats: dict[str, int]) -> None:
    totals["deleted_sessions"] += int(stats.get("deleted_sessions", 0) or 0)
    totals["deleted_state_files"] += int(
        stats.get("deleted_state_files", 0) or 0
    )


async def delete_user_data(user_id: str) -> dict:
    """Revoke an account and delete every generation it owns.

    Deletion is deliberately retryable. The account is made non-active and
    its API keys are revoked before producer cleanup starts. All independently
    owned sessions are then attempted even when one producer fails to stop.
    The user row is removed only after lifecycle records prove that every
    generation is tombstoned and physically clean and every import job is
    terminal.
    """
    # This synchronous publication closes the race with import generation
    # claims in the same process: either the import registered first and is
    # found below, or its route observes this marker and refuses the claim.
    users_pending_deletion.add(user_id)

    totals = {"deleted_sessions": 0, "deleted_state_files": 0}
    failures: dict[str, str] = {}

    try:
        db = await get_db()
        status_changed = await db.set_user_status(user_id, "pending")
    except BaseException:
        # No durable revocation was established, so do not leave a process-local
        # marker that would permanently reject imports for a nonexistent or
        # still-active account.
        users_pending_deletion.discard(user_id)
        raise
    # This is the account-level revocation boundary. Existing JWTs subsequently
    # fail require_active_user, while deleting API keys closes that credential
    # path immediately. Retrying deletion keeps the account non-active.
    if not status_changed:
        users_pending_deletion.discard(user_id)
        raise ValueError(f"user {user_id} does not exist")
    await db.delete_api_keys_by_user(user_id)

    await _cancel_user_import_jobs(user_id)

    # Capture lifecycle inventory directly. list_saved_sessions intentionally
    # hides importing and deleted records and therefore cannot be the source of
    # truth for account erasure.
    lifecycle_records = await asyncio.to_thread(_owned_lifecycle_records, user_id)
    initially_importing = {
        sid
        for sid, record in lifecycle_records.items()
        if record.get("status") == "importing"
    }

    live_import_ids, has_unbound_import_job = await _live_import_generations(user_id)

    # A durable cancellation marker makes the eventual commit fail closed but
    # intentionally leaves the record non-sweepable while its producer writes.
    # If the marker predates this request and no producer remains, this is a
    # crash/restart leftover and can be retired under the cleanup lease now.
    from backend.services.session_deletion import cleanup_failed_import_generation

    for sid in sorted(initially_importing):
        try:
            record = lifecycle_records[sid]
            if (
                record.get("import_cancel_requested") is True
                and sid not in live_import_ids
                and not has_unbound_import_job
            ):
                cleaned = await asyncio.to_thread(
                    cleanup_failed_import_generation, sid
                )
                if not cleaned:
                    failures[f"session:{sid}"] = (
                        "cancelled import changed before cleanup"
                    )
                continue
            generation = await asyncio.to_thread(
                request_import_generation_cancellation,
                sid,
                user_id,
            )
            if generation is None:
                failures[f"session:{sid}"] = (
                    "import generation changed before cancellation was recorded"
                )
        except Exception as exc:
            failures[f"session:{sid}"] = str(exc)

    async with state_lock:
        owned_states = {
            sid: state
            for sid, state in sessions.items()
            if getattr(state, "user_id", None) == user_id
        }

    saved = await asyncio.to_thread(list_saved_sessions)
    for sess in saved:
        if sess.get("user_id") != user_id:
            continue
        sid = str(sess.get("id") or "")
        if sid and sid not in owned_states:
            loaded = await asyncio.to_thread(load_session_from_disk, sid)
            if loaded is not None and loaded.user_id == user_id:
                owned_states[sid] = loaded

    # A lifecycle record can outlive a damaged/missing session.json. Use the
    # durable owner/generation data so such a session is still tombstoned.
    for sid, record in lifecycle_records.items():
        if record.get("status") != "active" or sid in owned_states:
            continue
        owned_states[sid] = SimpleNamespace(
            id=sid,
            user_id=user_id,
            lifecycle_generation=int(record.get("generation") or 0),
        )

    from backend.services.session_deletion import (
        cleanup_tombstoned_generation,
        delete_session_generation,
    )

    # Do not fail fast: each generation has its own tombstone/producer boundary,
    # so later sessions can still be safely retired when an earlier one stalls.
    for sid, state in sorted(owned_states.items()):
        if sid in initially_importing:
            continue
        try:
            stats = await delete_session_generation(
                sid,
                state,
                user_id,
                force_stop=True,
            )
            _count_cleanup_stats(totals, stats)
        except Exception as exc:
            failures[f"session:{sid}"] = str(exc)

    # Actively retry tombstones that predated this account deletion. Freshly
    # abandoned imports are excluded because their producer still owns the
    # destination until it observes cancellation and exits.
    for sid, record in sorted(lifecycle_records.items()):
        if sid in initially_importing:
            continue
        if record.get("status") != "deleted" or record.get("cleanup_complete") is True:
            continue
        try:
            stats = await cleanup_tombstoned_generation(sid, owner_id=user_id)
            _count_cleanup_stats(totals, stats)
            if int(stats.get("pending_producers", 0) or 0):
                failures[f"session:{sid}"] = "session producers are still stopping"
        except Exception as exc:
            failures[f"session:{sid}"] = str(exc)

    current_records = await asyncio.to_thread(_owned_lifecycle_records, user_id)
    pending_sessions = [
        sid
        for sid, record in current_records.items()
        if record.get("status") != "deleted"
        or record.get("cleanup_complete") is not True
    ]
    pending_import_jobs = await _pending_user_import_jobs(user_id)

    if failures or pending_sessions or pending_import_jobs:
        logger.warning(
            "User %s cleanup pending: sessions=%s imports=%s failures=%s",
            user_id,
            pending_sessions,
            pending_import_jobs,
            failures,
        )
        raise UserCleanupPendingError(
            user_id,
            pending_sessions=pending_sessions,
            pending_import_jobs=pending_import_jobs,
            failures=failures,
        )

    await db.delete_user_llm_config(user_id)
    await db.delete_user(user_id)
    users_pending_deletion.discard(user_id)

    logger.info(
        "User %s deleted: %s sessions, %s state files",
        user_id,
        totals["deleted_sessions"],
        totals["deleted_state_files"],
    )
    return {"user_id": user_id, **totals}
