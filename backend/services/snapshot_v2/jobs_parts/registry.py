"""Restore-job registry, journal persistence, and protection lookup."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.snapshot_v2.errors import SnapshotTransactionError
from backend.services.snapshot_v2.models import SnapshotTransactionRecord

logger = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_transaction_id() -> str:
    """Job status journal id (distinct from restore_apply mutation journal)."""
    return f"restore-job-{uuid.uuid4().hex[:16]}"


def restore_apply_transaction_id(job_tx_id: str) -> str:
    """Map restore_job id to restore_apply id so both can coexist on disk."""
    if job_tx_id.startswith("restore-job-"):
        return "restore-apply-" + job_tx_id[len("restore-job-"):]
    if job_tx_id.startswith("restore-apply-"):
        return job_tx_id
    return job_tx_id


def get_restore_job(transaction_id: str):
    """Return the process-wide handle for *transaction_id*, or ``None``."""
    from backend.state import snapshot_restore_jobs

    return snapshot_restore_jobs.get(transaction_id)


def load_transaction(transaction_id: str, *, paths: Any = None, bindings=None):
    """Load the latest transaction record from the shared handle or journal."""
    handle = bindings.get_restore_job(transaction_id)
    if handle is not None:
        return handle.record
    if paths is None:
        return None
    from backend.services.snapshot_v2.errors import SnapshotIntegrityError
    from backend.services.snapshot_v2.journal_store import JournalStore
    from backend.services.snapshot_v2.recovery import RecoveryFailure

    store = JournalStore(paths)
    try:
        env = store.try_read(transaction_id)
    except RecoveryFailure:
        journals = getattr(paths, "journals", None)
        if journals is None:
            return None
        journal_path = Path(journals) / f"{transaction_id}.json"
        if not journal_path.is_file():
            return None
        try:
            return SnapshotTransactionRecord.model_validate_json(
                journal_path.read_text()
            )
        except Exception:
            return None
    except SnapshotIntegrityError:
        return None
    if env is None:
        return None
    if env.kind == "restore_job":
        payload = env.payload or {}
        raw = payload.get("record")
        if isinstance(raw, dict):
            try:
                return SnapshotTransactionRecord.model_validate(raw)
            except Exception:
                pass
        try:
            return SnapshotTransactionRecord(
                transaction_id=env.transaction_id,
                session_id=env.session_id,
                snapshot_id=str(payload.get("snapshot_id") or ""),
                status=env.status,
                recovery_mode=str(payload.get("recovery_mode") or "rollback"),
                failure=payload.get("failure"),
            )
        except Exception:
            logger.debug(
                "[snapshot-v2] load_transaction: failed to parse restore_job %s",
                transaction_id,
                exc_info=True,
            )
    return None


async def cancel_restore_job(
    transaction_id: str,
    *,
    session_id: str | None = None,
    bindings=None,
) -> None:
    """Cancel a restore job via the process-wide registry."""
    handle = bindings.get_restore_job(transaction_id)
    if handle is None:
        raise SnapshotTransactionError(
            "TRANSACTION_NOT_FOUND",
            f"Restore transaction not found: {transaction_id}",
        )
    record = handle.record
    if session_id is not None and record.session_id != session_id:
        raise SnapshotTransactionError(
            "TRANSACTION_NOT_FOUND",
            f"Restore transaction not found for session: {transaction_id}",
        )
    if record.status in ("committed", "failed", "timed_out", "cancelled"):
        logger.debug(
            "[snapshot-v2] cancel_restore_job: %s already terminal (%s)",
            transaction_id,
            record.status,
        )
        return
    if record.status == "applying":
        handle.cancel_event.set()
        return
    handle.record = record.model_copy(update={"status": "cancelled"})
    bindings._persist_handle(handle)
    owner_token = handle.owner_token or f"restore:{record.transaction_id}"
    await bindings.get_snapshot_safe_point_manager().cancel_safe_point_request(
        record.session_id,
        owner_token,
    )
    handle.cancel_event.set()
    if handle.task is not None and not handle.task.done():
        handle.task.cancel()


def persist_handle(handle) -> bool:
    """Write the current restore-job record before any WebSocket event."""
    from backend.services.session_lifecycle import (
        SessionGenerationError,
        SessionGoneError,
        assert_session_generation_active,
    )

    try:
        assert_session_generation_active(
            handle.record.session_id,
            expected_generation=handle.lifecycle_generation,
        )
    except (SessionGoneError, SessionGenerationError):
        logger.info(
            "[snapshot-v2] suppress stale restore job persist %s for "
            "session generation %s",
            handle.record.transaction_id,
            handle.lifecycle_generation,
        )
        return False
    if handle.paths is None or getattr(handle.paths, "journals", None) is None:
        return True
    from backend.services.snapshot_v2.errors import SnapshotIntegrityError
    from backend.services.snapshot_v2.journal_store import JournalStore, make_envelope
    from backend.services.snapshot_v2.recovery import RecoveryFailure

    store = JournalStore(handle.paths)
    tx_id = handle.record.transaction_id
    try:
        existing = store.try_read(tx_id)
        if existing is not None and existing.kind == "restore_apply":
            logger.warning(
                "[snapshot-v2] refuse restore_job persist over restore_apply %s",
                tx_id,
            )
            return False
    except (SnapshotIntegrityError, RecoveryFailure):
        logger.warning(
            "[snapshot-v2] refuse restore_job persist over unreadable journal %s",
            tx_id,
        )
        return False
    record = handle.record
    store.write(make_envelope(
        kind="restore_job",
        transaction_id=tx_id,
        session_id=record.session_id,
        status=record.status,
        payload={
            "snapshot_id": record.snapshot_id,
            "recovery_mode": record.recovery_mode,
            "failure": record.failure,
            "lifecycle_generation": handle.lifecycle_generation,
            "record": record.model_dump(mode="json"),
        },
    ))
    return True


def collect_active_restore_protected_ids(
    session_id: str,
    *,
    paths: Any = None,
) -> dict[str, set[str]]:
    """Return active restore target and pre-restore protections."""
    out: dict[str, set[str]] = {"restore_target": set(), "pre_restore": set()}
    terminal = {"committed", "failed", "cancelled", "timed_out"}
    try:
        from backend.state import snapshot_restore_jobs
        for handle in list(snapshot_restore_jobs.values()):
            record = getattr(handle, "record", None)
            if record is None or getattr(record, "session_id", None) != session_id:
                continue
            if getattr(record, "status", None) in terminal:
                continue
            snapshot_id = getattr(record, "snapshot_id", None)
            if snapshot_id:
                out["restore_target"].add(str(snapshot_id))
    except Exception:
        pass
    if paths is None:
        return out
    from backend.services.snapshot_v2.journal_store import JournalStore

    store = JournalStore(paths)
    journals = getattr(paths, "journals", None)
    if journals is None:
        return out
    journal_dir = Path(journals)
    if not journal_dir.exists():
        return out
    for entry in journal_dir.iterdir():
        if not entry.is_file() or entry.suffix != ".json":
            continue
        try:
            env = store.read(entry.stem)
        except Exception:
            continue
        if env.session_id not in (None, "", session_id):
            continue
        status = str(env.status or "").lower()
        payload = env.payload or {}
        if env.kind == "restore_apply":
            if status in ("cancelled", "timed_out"):
                continue
            if status == "committed" and not payload.get("pre_restore_ref"):
                continue
            target = payload.get("target_snapshot_id")
            if target:
                out["restore_target"].add(str(target))
            pre_restore = payload.get("pre_restore_ref")
            if pre_restore:
                out["pre_restore"].add(str(pre_restore))
        elif env.kind == "restore_job" and status not in terminal:
            target = payload.get("snapshot_id")
            if target:
                out["restore_target"].add(str(target))
    return out
