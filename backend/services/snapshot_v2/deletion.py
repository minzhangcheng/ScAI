"""Protected leaf and branch deletion for Snapshot V2.

Journal-based crash recovery: every deletion progresses through
STAGED -> APPLYING -> COMMITTED journal states so an interrupted
operation can be recovered via ``recover_delete_transaction()``.

Safety invariants:
- Delete journals survive every non-COMMITTED failure (no ``finally``
  that always removes them).
- GC uses ``SnapshotGarbageCollector`` which verifies every surviving
  manifest before sweep; corrupt survivors abort without object removal.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from backend.services.snapshot_v2.errors import (
    SnapshotDeleteConflict,
    SnapshotIntegrityError,
    SnapshotStorageError,
)
from backend.services.snapshot_v2.gc import SnapshotGarbageCollector
from backend.services.snapshot_v2.models import (
    DeleteResult,
    SnapshotManifestV2,
)
from backend.services.snapshot_v2.object_store import ManifestStore, ObjectStore

logger = logging.getLogger(__name__)


def _new_delete_tx_id() -> str:
    import uuid
    return f"del-{uuid.uuid4().hex[:16]}"


def _journal_path(paths, transaction_id: str) -> Path:
    return Path(str(paths.journals)) / f"{transaction_id}.json"


def _delete_payload_from_record(record: dict) -> dict:
    """Extract delete-specific fields for envelope payload."""
    keys = (
        "delete_set",
        "surviving_manifest_ids",
        "previous_lineage_head",
        "new_lineage_head",
        "error",
    )
    return {k: record[k] for k in keys if k in record}


def _record_from_delete_envelope(env) -> dict:
    """Flatten delete envelope into the dict recover_delete_transaction expects."""
    d = dict(env.payload or {})
    d["transaction_id"] = env.transaction_id
    d["session_id"] = env.session_id
    d["status"] = env.status
    return d


def _read_journal(paths, transaction_id: str) -> dict | None:
    """Read verified delete journal envelope; return flat record dict."""
    from backend.services.snapshot_v2.journal_store import JournalStore

    jp = _journal_path(paths, transaction_id)
    if not jp.exists():
        return None
    store = JournalStore(paths)
    env = store.read(transaction_id)
    if env.kind != "delete":
        # Not a delete journal — treat as absent for delete recovery path
        return None
    return _record_from_delete_envelope(env)


def read_delete_journal(paths, transaction_id: str | None = None):
    """Load a delete journal as :class:`DeleteJournalRecord`.

    Unwraps the schema-v1 kind envelope (``schema_version`` / ``kind`` /
    ``payload`` / ``hmac``) into the flat record shape tests and recovery
    expect. When *transaction_id* is omitted, requires exactly one
    ``del-*.json`` journal under ``paths.journals``.
    """
    from backend.services.snapshot_v2.models import DeleteJournalRecord

    if transaction_id is None:
        journals = sorted(Path(str(paths.journals)).glob("del-*.json"))
        if len(journals) != 1:
            raise FileNotFoundError(
                f"expected one delete journal, found {len(journals)}"
            )
        transaction_id = journals[0].stem

    flat = _read_journal(paths, transaction_id)
    if flat is None:
        raise FileNotFoundError(
            f"delete journal not found or not kind=delete: {transaction_id}"
        )
    return DeleteJournalRecord.model_validate(flat)


def _write_journal(paths, transaction_id: str, record: dict) -> None:
    """Write a signed delete envelope."""
    from backend.services.snapshot_v2.journal_store import JournalStore, make_envelope

    env = make_envelope(
        kind="delete",
        transaction_id=transaction_id,
        session_id=str(record.get("session_id") or ""),
        status=str(record.get("status") or "STAGED"),
        payload=_delete_payload_from_record(record),
    )
    JournalStore(paths).write(env)


def _remove_journal(paths, transaction_id: str) -> None:
    from backend.services.snapshot_v2.journal_store import JournalStore

    JournalStore(paths).remove(transaction_id)


def _remove_visibility(paths, delete_set: list[str]) -> None:
    """Remove committed manifests and integrity signatures for *delete_set*."""
    for sid in delete_set:
        m_path = Path(str(paths.committed)) / f"{sid}.json"
        if m_path.exists():
            m_path.unlink()
        s_path = Path(str(paths.integrity)) / f"{sid}.sig"
        if s_path.exists():
            s_path.unlink()


def recover_delete_transaction(
    journal: dict[str, Any],
    *,
    paths,
    session_id: str,
    manifest_store: ManifestStore | None = None,
    object_store: ObjectStore | None = None,
) -> DeleteResult:
    """Idempotently complete or abandon a delete transaction from its journal.

    * ``STAGED`` — no manifests were touched; remove journal only.
    * ``APPLYING`` — ensure visibility removal, run fail-closed GC, then
      mark COMMITTED and remove journal.
    * ``COMMITTED`` — cleanup journal only.
    * ``FAILED`` — leave journal for operator (no mutations).

    Corrupt surviving manifests cause ``SnapshotIntegrityError`` and leave
    the journal in place (fail closed — no object sweep).
    """
    tx_id = str(journal.get("transaction_id") or "")
    status = journal.get("status", "")
    delete_set = list(journal.get("delete_set") or [])
    surviving_ids = list(journal.get("surviving_manifest_ids") or [])

    if status == "STAGED":
        if tx_id:
            _remove_journal(paths, tx_id)
        logger.info(
            "Recovered delete [%s]: cleaned up STAGED journal %s",
            session_id, tx_id,
        )
        return DeleteResult(
            deleted_ids=[],
            freed_bytes=0,
            surviving_manifests=surviving_ids,
        )

    if status == "COMMITTED":
        if tx_id:
            _remove_journal(paths, tx_id)
        logger.info(
            "Recovered delete [%s]: cleaned up COMMITTED journal %s",
            session_id, tx_id,
        )
        return DeleteResult(
            deleted_ids=delete_set,
            freed_bytes=0,
            surviving_manifests=surviving_ids,
        )

    if status == "FAILED":
        logger.warning(
            "Recovered delete [%s]: FAILED journal %s left in place",
            session_id, tx_id,
        )
        return DeleteResult(
            deleted_ids=[],
            freed_bytes=0,
            surviving_manifests=surviving_ids,
        )

    if status != "APPLYING":
        logger.warning(
            "Recovered delete [%s]: unknown status %r for journal %s",
            session_id, status, tx_id,
        )
        return DeleteResult(
            deleted_ids=[],
            freed_bytes=0,
            surviving_manifests=surviving_ids,
        )

    # APPLYING: complete visibility removal + fail-closed GC
    _remove_visibility(paths, delete_set)

    gc = SnapshotGarbageCollector(
        session_id,
        paths,
        manifest_store=manifest_store,
        object_store=object_store,
    )
    # Verify + mark + sweep; raises SnapshotIntegrityError on bad survivor
    gc_result = gc.run()
    freed_bytes = gc_result.removed_bytes

    record = dict(journal)
    record["status"] = "COMMITTED"
    if tx_id:
        _write_journal(paths, tx_id, record)
        _remove_journal(paths, tx_id)

    logger.info(
        "Recovered delete [%s]: completed APPLYING journal %s "
        "(%d manifests, GC removed %d objects / %d bytes)",
        session_id, tx_id, len(delete_set),
        gc_result.removed_count, freed_bytes,
    )
    return DeleteResult(
        deleted_ids=delete_set,
        freed_bytes=freed_bytes,
        surviving_manifests=surviving_ids,
    )


class DeletionService:
    """Transactional deletion for V2 snapshot manifests.

    Usage::

        service = DeletionService(session_id, paths)
        result = service.delete_leaf("snap-abc")
        # or
        result = service.delete_branch("snap-abc", descendant_ids=[...])
    """

    def __init__(
        self,
        session_id: str,
        paths,
        manifest_store: ManifestStore | None = None,
        object_store: ObjectStore | None = None,
        gc: SnapshotGarbageCollector | None = None,
    ) -> None:
        self.session_id = session_id
        self.paths = paths
        self.manifest_store = manifest_store or ManifestStore(paths)
        self.object_store = object_store or ObjectStore(paths)
        self.gc = gc or SnapshotGarbageCollector(
            session_id,
            paths,
            manifest_store=self.manifest_store,
            object_store=self.object_store,
        )

    # ── Public API ─────────────────────────────────────────────────────

    def delete_leaf(
        self,
        snapshot_id: str,
        *,
        protected_ids: set[str] | None = None,
    ) -> DeleteResult:
        """Delete a single leaf snapshot.

        Raises ``SnapshotDeleteConflict`` (→ HTTP 409) when the snapshot
        has children, is protected, or is referenced by an active restore/
        export/import operation.
        """
        return self._delete_impl(
            snapshot_id=snapshot_id,
            extra_ids=None,
            protected_ids=protected_ids,
        )

    def delete_branch(
        self,
        snapshot_id: str,
        descendant_ids: list[str],
        *,
        protected_ids: set[str] | None = None,
    ) -> DeleteResult:
        """Delete a snapshot and all its descendants.

        *descendant_ids* must be the complete set of transitive descendants
        reachable from *snapshot_id*.
        """
        return self._delete_impl(
            snapshot_id=snapshot_id,
            extra_ids=descendant_ids,
            protected_ids=protected_ids,
        )

    def recover_delete(self, transaction_id: str) -> DeleteResult | None:
        """Recover from a crash during deletion.

        Returns ``DeleteResult`` when a journal was processed, or ``None``
        when the journal is absent.
        """
        journal = _read_journal(self.paths, transaction_id)
        if journal is None:
            return None
        return recover_delete_transaction(
            journal,
            paths=self.paths,
            session_id=self.session_id,
            manifest_store=self.manifest_store,
            object_store=self.object_store,
        )

    def _delete_impl(
        self,
        snapshot_id: str,
        extra_ids: list[str] | None,
        protected_ids: set[str] | None,
    ) -> DeleteResult:
        tx_id = _new_delete_tx_id()

        all_manifests = self._load_all_committed()
        manifest_map = {m.snapshot_id: m for m in all_manifests}

        target = manifest_map.get(snapshot_id)
        if target is None:
            raise SnapshotStorageError(
                "SNAPSHOT_NOT_FOUND",
                f"Snapshot {snapshot_id} not found",
            )

        delete_set = [snapshot_id]
        if extra_ids:
            delete_set.extend(extra_ids)
        delete_set_set = set(delete_set)

        # Protect non-leaf snapshots for leaf-only delete
        if not extra_ids:
            for other in all_manifests:
                if other.snapshot_id != snapshot_id and other.parent_id == snapshot_id:
                    raise SnapshotDeleteConflict(
                        "HAS_CHILDREN",
                        f"Snapshot {snapshot_id} has children and is not a leaf",
                    )

        # Collect active export protection records
        export_protected: set[str] = set()
        try:
            from backend.services.snapshot_v2.exporter import (
                load_active_export_protected_ids,
            )
            export_protected = load_active_export_protected_ids(self.paths)
        except Exception:
            pass

        all_protected = set(protected_ids or {})
        all_protected.update(export_protected)

        if all_protected:
            for sid in delete_set_set:
                if sid in all_protected:
                    raise SnapshotDeleteConflict(
                        "PROTECTED",
                        f"Snapshot {sid} is protected and cannot be deleted",
                        details={"reason": "protected", "snapshot_id": sid},
                    )

        surviving_ids = [
            m.snapshot_id for m in all_manifests
            if m.snapshot_id not in delete_set_set
        ]

        parent_id = target.parent_id
        new_head = parent_id if parent_id and parent_id in surviving_ids else None

        record = {
            "transaction_id": tx_id,
            "session_id": self.session_id,
            "delete_set": delete_set,
            "surviving_manifest_ids": surviving_ids,
            "previous_lineage_head": snapshot_id,
            "new_lineage_head": new_head,
            "status": "STAGED",
        }
        _write_journal(self.paths, tx_id, record)

        freed_bytes = 0
        try:
            record["status"] = "APPLYING"
            _write_journal(self.paths, tx_id, record)

            _remove_visibility(self.paths, delete_set)

            # Fail-closed mark-and-sweep via shared GC
            gc_result = self.gc.run()
            freed_bytes = gc_result.removed_bytes

            record["status"] = "COMMITTED"
            _write_journal(self.paths, tx_id, record)
        except Exception:
            logger.exception(
                "[%s] delete transaction %s failed, leaving journal for recovery",
                self.session_id, tx_id,
            )
            raise

        # Journal removed only after durable COMMITTED
        _remove_journal(self.paths, tx_id)

        return DeleteResult(
            deleted_ids=delete_set,
            freed_bytes=freed_bytes,
            surviving_manifests=surviving_ids,
        )

    def _load_all_committed(self) -> list[SnapshotManifestV2]:
        return self.manifest_store.verify_all_committed()
