"""Kind-dispatched recovery envelope handlers."""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from backend.services.snapshot_v2.checkpoint_paths import (
    SQLITE_COMPANION_SUFFIXES,
    checkpoint_sqlite_path,
    sqlite_companion_paths,
)
from backend.services.snapshot_v2.errors import SnapshotIntegrityError
from backend.services.snapshot_v2.object_store import (
    ManifestStore,
    ObjectStore,
    reclaim_stale_staging_temp,
)
from backend.services.snapshot_v2.runtime_names import runtime_h5ad_path
from backend.services.snapshot_v2.workspace import WorkspaceScope

if TYPE_CHECKING:
    from backend.services.snapshot_v2.gc import GCResult

logger = logging.getLogger(__name__)

from .primitives import *

def _recover_restore_envelope(
    envelope,
    *,
    paths,
    session_id: str,
    restorer: "_RecoveryRestorer",
    recovered_txns: list,
) -> bool:
    """Recover a restore_apply envelope (Task 1 state machine preserved).

    Returns True when the live ``session.json`` was rewritten (rollback or
    forward-complete), so the caller can evict the now-stale in-memory cache.
    """
    from backend.services.snapshot_v2.journal_store import (
        JournalStore,
        envelope_to_restore_dict,
    )
    from backend.services.snapshot_v2.restore import (
        RestoreJournal,
        migrate_verified_restore_v0,
        read_verified_restore_journal,
    )

    journal = envelope_to_restore_dict(envelope)
    tx_id = envelope.transaction_id
    status = journal.get("status", "")
    recovery_mode = journal.get("recovery_mode", "rollback")
    restore_journal_version = int(
        journal.get("restore_journal_version")
        if journal.get("restore_journal_version") is not None
        else 0
    )
    jp = _journal_path(paths, tx_id)

    if status == "COMMITTED":
        verified = read_verified_restore_journal(paths, tx_id)
        if restore_journal_version == 0:
            proof = restorer.verify_target_complete(journal)
            migrate_verified_restore_v0(paths, verified, proof)
            verified = read_verified_restore_journal(paths, tx_id)
            journal = verified.to_dict()
            restore_journal_version = int(verified.restore_journal_version)
        elif not (
            journal.get("verified_live_root_hash")
            or verified.verified_live_root_hash
        ):
            raise RecoveryFailure(
                "COMMITTED_WITHOUT_VERIFICATION",
                f"transaction {tx_id} lacks live verification proof",
            )
        restorer.verify_target_complete(journal)
        if journal.get("runtime_restore_status") == "complete":
            from backend.utils.runtime_apply import (
                commit_runtime_persistence_metadata,
            )

            commit_runtime_persistence_metadata(
                session_id,
                source="snapshot_apply",
            )
        restorer.remove_staging(journal)
        JournalStore(paths).remove(tx_id)
        recovered_txns.append(tx_id)
        logger.info(
            "Recovery [%s]: cleanup for committed restore_apply %s",
            session_id, tx_id,
        )
        return False

    if status in ("APPLYING", "STAGED"):
        if recovery_mode == "forward_only":
            ok = restorer.forward_complete(journal)
            if not ok:
                raise RecoveryFailure(
                    "FORWARD_COMPLETE_FAILED",
                    f"forward_complete returned failure for {tx_id}",
                )
            # Live contract gate: never drop APPLYING journal without verify.
            restorer.verify_target_complete(journal)
            if journal.get("runtime_restore_status") == "complete":
                from backend.utils.runtime_apply import (
                    commit_runtime_persistence_metadata,
                )

                commit_runtime_persistence_metadata(
                    session_id,
                    source="snapshot_apply",
                )
            restorer.remove_staging(journal)
            JournalStore(paths).remove(tx_id)
            recovered_txns.append(tx_id)
            logger.info(
                "Recovery [%s]: forward-completed restore_apply %s",
                session_id, tx_id,
            )
        else:
            ok = restorer.rollback(journal)
            if not ok:
                raise RecoveryFailure(
                    "ROLLBACK_FAILED",
                    f"rollback returned failure for {tx_id}",
                )
            restorer.remove_staging(journal)
            JournalStore(paths).remove(tx_id)
            recovered_txns.append(tx_id)
            logger.info(
                "Recovery [%s]: rolled back restore_apply %s",
                session_id, tx_id,
            )
        return True

    if status == "STAGING":
        restorer.remove_staging(journal)
        JournalStore(paths).remove(tx_id)
        recovered_txns.append(tx_id)
        logger.info(
            "Recovery [%s]: removed partial staging for restore_apply %s",
            session_id, tx_id,
        )
        return False

    logger.warning(
        "Recovery [%s]: restore_apply %s unknown status %r — left in place",
        session_id, tx_id, status,
    )
    return False


def _recover_restore_job_envelope(
    envelope,
    *,
    paths,
    session_id: str,
    recovered_txns: list,
) -> None:
    """Non-mutating job status recovery: retire terminal and abandoned jobs.

    A non-terminal envelope is only meaningful while the job that owns it is
    alive in this process. ``_run_job`` is its sole writer, so once the handle
    is gone from ``snapshot_restore_jobs`` — always the case after a restart —
    nothing will ever drive the record to a terminal status again. Leaving it
    was permanent: ``compute_protected_snapshot_ids`` treats any non-terminal
    ``restore_job`` envelope as a live ``restore_target`` protection, so the
    snapshot being restored became undeletable and un-reclaimable forever;
    status polls reported ``waiting_safe_point`` indefinitely; and
    ``cancel_restore_job`` answered 404, so no API could clear it either.
    """
    from backend.services.snapshot_v2.journal_store import JournalStore, make_envelope
    from backend.services.snapshot_v2.jobs import get_restore_job

    st = str(envelope.status or "").lower()
    terminal = {"committed", "failed", "cancelled", "timed_out"}
    if st in terminal:
        JournalStore(paths).remove(envelope.transaction_id)
        recovered_txns.append(envelope.transaction_id)
        logger.info(
            "Recovery [%s]: cleaned terminal restore_job %s (%s)",
            session_id, envelope.transaction_id, st,
        )
        return

    if get_restore_job(envelope.transaction_id) is not None:
        # Live job in this process — its own runner owns the record.
        logger.info(
            "Recovery [%s]: restore_job %s status=%s is live, left in place",
            session_id, envelope.transaction_id, st,
        )
        return

    payload = dict(envelope.payload or {})
    payload["failure"] = {
        "code": "RESTORE_JOB_ABANDONED",
        "message": (
            "The process running this restore job did not survive; the job "
            "status is reported as failed. The session itself is repaired by "
            "restore_apply recovery."
        ),
    }
    record = payload.get("record")
    if isinstance(record, dict):
        record = dict(record)
        record["status"] = "failed"
        record["failure"] = payload["failure"]
        payload["record"] = record
    JournalStore(paths).write(
        make_envelope(
            kind="restore_job",
            transaction_id=envelope.transaction_id,
            session_id=envelope.session_id,
            status="failed",
            payload=payload,
        )
    )
    recovered_txns.append(envelope.transaction_id)
    logger.warning(
        "Recovery [%s]: restore_job %s was abandoned at status=%s; marked failed",
        session_id, envelope.transaction_id, st,
    )


def _recover_delete_envelope(
    envelope,
    *,
    paths,
    session_id: str,
    recovered_txns: list,
) -> None:
    from backend.services.snapshot_v2.deletion import recover_delete_transaction

    journal = dict(envelope.payload or {})
    journal["transaction_id"] = envelope.transaction_id
    journal["session_id"] = envelope.session_id
    journal["status"] = envelope.status
    recover_delete_transaction(journal, paths=paths, session_id=session_id)
    recovered_txns.append(envelope.transaction_id)
    logger.info(
        "Recovery [%s]: recovered delete transaction %s",
        session_id, envelope.transaction_id,
    )


def _recover_import_envelope(
    envelope,
    *,
    paths,
    session_id: str,
    recovered_txns: list,
    live_txns: list | None = None,
) -> None:
    """Import recovery: STAGING cleans import staging; STAGED reconciles;
    COMMITTED cleans up; FAILED remains visible (no restore masquerade).

    An import that is still running in this process is not crash debris and is
    deferred to itself; *live_txns* collects those transaction ids for the
    caller.
    """
    from backend.services.snapshot_v2.journal_store import (
        JOURNAL_OWNER_ID,
        JournalStore,
    )
    from backend.services.snapshot_v2.object_store import ManifestStore

    tx_id = envelope.transaction_id
    status = str(envelope.status or "").upper()
    store = JournalStore(paths)
    staging_root = Path(str(paths.staging))

    # An in-flight journal stamped with this process's owner id belongs to an
    # import still running in a worker thread, not to a crashed one. Recovery
    # runs on the event loop for every request, so it can land in the middle of
    # the import: at STAGING it would delete the journal and let the sweep below
    # take the objects Pass 2 has already written, at STAGED it would also delete
    # the manifests the import just published and its batch record. Neither is
    # recoverable - the import continues from memory and commits manifests whose
    # objects are gone. Defer instead; the import finishes its own transaction,
    # and a process that actually crashed loses this tag with its pid.
    owner = str((envelope.payload or {}).get("owner") or "")
    if status in ("STAGING", "STAGED") and owner == JOURNAL_OWNER_ID:
        if live_txns is not None:
            live_txns.append(tx_id)
        logger.info(
            "Recovery [%s]: import %s is live in this process (%s) - deferred",
            session_id, tx_id, status,
        )
        return

    if status == "STAGING":
        # Remove only import staging dirs prefixed with this txn
        if staging_root.exists():
            for entry in list(staging_root.iterdir()):
                if entry.name.startswith(f"{tx_id}") or entry.name.startswith(
                    f"{tx_id}_"
                ):
                    if entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=True)
                    else:
                        try:
                            entry.unlink()
                        except OSError:
                            pass
        store.remove(tx_id)
        recovered_txns.append(tx_id)
        logger.info(
            "Recovery [%s]: cleaned STAGING import journal %s",
            session_id, tx_id,
        )
        return

    if status == "STAGED":
        # Collect snapshot IDs from the envelope payload so we can clean
        # up any pending-import markers left over from the interrupted
        # batch.  The batch was never committed, so these manifests are
        # orphaned and must be removed.
        payload = envelope.payload or {}
        pending_ids = set(
            str(sid) for sid in payload.get("protected_snapshot_ids") or []
            if sid
        )

        # Remove the manifests directly if they have pending-import
        # markers (the batch was never published).
        if pending_ids:
            from backend.services.snapshot_v2.deletion import DeletionService
            from backend.services.snapshot_v2.object_store import ObjectStore

            del_paths = paths
            del_service = DeletionService(
                session_id, del_paths,
                manifest_store=ManifestStore(del_paths),
                object_store=ObjectStore(del_paths),
            )
            try:
                del_service.delete_pending_import_manifests(list(pending_ids))
            except Exception as exc:
                logger.warning(
                    "Recovery [%s]: cleanup of pending-import manifests "
                    "failed for %s: %s",
                    session_id, tx_id, exc,
                )

        # Also remove any partial batch record
        try:
            from backend.services.snapshot_v2.batch_store import ImportBatchStore
            batch_store = ImportBatchStore(paths.root)
            batch_store.remove(tx_id)
        except Exception as exc:
            logger.warning(
                "Recovery [%s]: cleanup of partial batch record "
                "failed for %s: %s",
                session_id, tx_id, exc,
            )

        try:
            reconcile_result = ManifestStore(paths).reconcile()
        except Exception as exc:
            logger.error(
                "Recovery [%s]: import STAGED reconcile failed for %s: %s",
                session_id, tx_id, exc,
            )
            raise RecoveryFailure(
                "IMPORT_RECONCILE_FAILED",
                f"Import STAGED reconcile failed for {tx_id}: {exc}",
                details={"transaction_id": tx_id, "session_id": session_id},
            ) from exc
        if reconcile_result.corrupt_ids:
            logger.error(
                "Recovery [%s]: import STAGED reconcile found corrupt "
                "manifests for %s: %s",
                session_id, tx_id, reconcile_result.corrupt_ids,
            )
            raise RecoveryFailure(
                "IMPORT_RECONCILE_CORRUPT",
                f"Import STAGED reconcile found corrupt manifests: "
                f"{reconcile_result.corrupt_ids}",
                details={
                    "transaction_id": tx_id,
                    "session_id": session_id,
                    "corrupt_ids": list(reconcile_result.corrupt_ids),
                },
            )
        try:
            from backend.services.snapshot_v2.gc import SnapshotGarbageCollector
            SnapshotGarbageCollector(session_id, paths).run()
        except Exception as exc:
            logger.warning(
                "Recovery [%s]: import STAGED GC failed for %s: %s",
                session_id, tx_id, exc,
            )
        store.remove(tx_id)
        recovered_txns.append(tx_id)
        logger.info(
            "Recovery [%s]: reconciled STAGED import journal %s",
            session_id, tx_id,
        )
        return

    if status == "COMMITTED":
        store.remove(tx_id)
        recovered_txns.append(tx_id)
        logger.info(
            "Recovery [%s]: cleaned COMMITTED import journal %s",
            session_id, tx_id,
        )
        return

    if status == "FAILED":
        logger.warning(
            "Recovery [%s]: FAILED import journal %s left in place",
            session_id, tx_id,
        )
        return

    logger.warning(
        "Recovery [%s]: import journal %s unknown status %r — left in place",
        session_id, tx_id, status,
    )


def _recover_export_protection_envelope(
    envelope,
    *,
    paths,
    session_id: str,
    recovered_txns: list,
) -> None:
    """Export protection is non-mutating; reap it once its writer is gone.

    The finally in export() is the journal's only remover, and it runs
    in the process that wrote it — so a crash during the archive build (minutes
    of work over tens of GB, which is where a crash is most likely) leaves an
    ACTIVE journal behind with nothing left to retire it.  Leaving it was
    permanent, not merely untidy: compute_protected_snapshot_ids folds every
    export_protection envelope into the protected map, so each snapshot the
    payload names answered 409 PROTECTED to DELETE and its bytes stayed
    unreclaimable by GC for the life of the session, with no API able to clear
    the journal.  "They only block deletes until an operator removes them" had
    no operator-facing removal path.

    Neither status nor age can tell crash debris from a live export — the
    journal reads ACTIVE for both, and because the archive is built in a worker
    thread while recovery runs on the event loop, recovery can legitimately
    land in the middle of one.  The owner stamp can: a journal carrying another
    process's id, or none at all (only writers predating the stamp produce
    that), provably has no live export handle behind it.  This is the same
    argument applied to abandoned restore_job envelopes above.
    """
    from backend.services.snapshot_v2.journal_store import (
        JOURNAL_OWNER_ID,
        JournalStore,
    )

    owner = str((envelope.payload or {}).get("owner") or "")
    if owner == JOURNAL_OWNER_ID:
        logger.info(
            "Recovery [%s]: export_protection %s is owned by this process, "
            "left in place",
            session_id, envelope.transaction_id,
        )
        return

    JournalStore(paths).remove(envelope.transaction_id)
    recovered_txns.append(envelope.transaction_id)
    logger.info(
        "Recovery [%s]: reaped orphaned export_protection %s (owner=%s)",
        session_id, envelope.transaction_id, owner or "<unstamped>",
    )


def _gc_blocked_by_active_safe_point(session_id: str) -> bool:
    """True when a capture or restore currently owns *session_id*'s safe point.

    Fail-open (False) when the manager cannot be reached: recovery must still
    reclaim on a process where the safe-point machinery is unavailable.
    """
    try:
        from backend.services.snapshot_v2.safe_point import (
            get_snapshot_safe_point_manager,
        )

        if not get_snapshot_safe_point_manager().is_paused(session_id):
            return False
    except Exception:
        return False
    logger.info(
        "Recovery [%s]: skipping GC - a capture/restore owns the safe point",
        session_id,
    )
    return True


def _report_partial_session_state_rewrite(restorer, result) -> None:
    """Carry a durable session.json rewrite out of a *failed* recovery pass.

    ``_recover_restore_envelope`` reports the rewrite through its return value,
    which only exists when the whole pass completed — but ``rollback()`` streams
    the pre_restore session.json over the live file with the notebook and
    checkpoint steps still ahead of it. A failure in between left the flag off,
    so the caller kept the cached AnalysisState the interrupted restore had
    installed and wrote that target state straight back over the file the
    rollback had just restored, half-undoing it. The write is what the cache has
    to be invalidated against, not the success of the steps after it.
    """
    if getattr(restorer, "session_json_rewritten", False):
        result.session_state_rewritten = True

__all__ = [name for name in globals() if not name.startswith("__")]
