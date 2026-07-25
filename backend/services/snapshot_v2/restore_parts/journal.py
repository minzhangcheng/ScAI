# Restore journal: envelope types, HMAC signing, and verified read/write.
# Moved verbatim out of backend/services/snapshot_v2/restore.py.
"""Crash-recoverable multi-artifact restore coordinator for Snapshot V2.

Provides ``RestoreCoordinator`` with four-phase lifecycle::

    stage(snapshot_id, ...) -> StagedRestore
    apply(staged, paths, ...) -> AppliedRestore
    commit(transaction_id, paths) -> None
    cleanup(transaction_id, paths) -> None

Every transition is journaled under ``journals/<transaction_id>.json`` with
HMAC integrity, so ``recover_snapshot_transactions`` in ``recovery.py`` can
complete or roll back an interrupted transaction.
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from backend.services.snapshot_v2.errors import SnapshotIntegrityError

logger = logging.getLogger(__name__)


# -- Result types ------------------------------------------------------------


@dataclass
class StagedRestore:
    transaction_id: str
    session_id: str
    snapshot_id: str
    recovery_mode: str
    pre_restore_ref: str
    branch_id: str
    continuation_kind: str
    manifest: dict
    staged_workspace: str
    staged_runtime_pkl: Optional[str] = None
    staged_runtime_h5ads: dict = field(default_factory=dict)
    staged_notebook: Optional[str] = None
    staged_analysis_state: Optional[str] = None
    staged_checkpoint: Optional[str] = None

@dataclass
class AppliedRestore:
    transaction_id: str
    steps_applied: list
    workspace_files_count: int = 0
    runtime_applied: bool = False
    notebook_applied: bool = False
    state_applied: bool = False
    checkpoint_applied: bool = False
    runtime_restore_status: str = "complete"
    runtime_warning: Optional[str] = None

@dataclass
class RestoreJournal:
    """Validated restore journal (HMAC stripped after verification).

    ``restore_journal_version`` is distinct from the generic envelope
    ``schema_version`` introduced later. Absence of the field means legacy
    restore journal version 0.
    """

    transaction_id: str
    target_snapshot_id: str
    session_id: str
    recovery_mode: str
    status: str
    pre_restore_ref: str = ""
    branch_id: str = "main"
    continuation_kind: str = "idle"
    target_root_hash: str = ""
    applied_step_ledger: list = field(default_factory=list)
    restore_journal_version: Literal[0, 1] = 0
    verified_live_root_hash: str = ""
    component_hashes: dict = field(default_factory=dict)
    verified_at: str = ""
    raw: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        """Return a plain dict suitable for re-signing (no hmac)."""
        d = {
            "transaction_id": self.transaction_id,
            "target_snapshot_id": self.target_snapshot_id,
            "session_id": self.session_id,
            "recovery_mode": self.recovery_mode,
            "status": self.status,
            "pre_restore_ref": self.pre_restore_ref,
            "branch_id": self.branch_id,
            "continuation_kind": self.continuation_kind,
            "applied_step_ledger": list(self.applied_step_ledger),
            "restore_journal_version": int(self.restore_journal_version),
        }
        if self.target_root_hash:
            d["target_root_hash"] = self.target_root_hash
        if self.verified_live_root_hash:
            d["verified_live_root_hash"] = self.verified_live_root_hash
        if self.component_hashes:
            d["component_hashes"] = dict(self.component_hashes)
        if self.verified_at:
            d["verified_at"] = self.verified_at
        for k, v in self.raw.items():
            if k not in d and k != "hmac":
                d[k] = v
        return d

def _new_transaction_id() -> str:
    return f"restore-{uuid.uuid4().hex[:16]}"

def _journal_path(paths, transaction_id: str):
    return Path(str(paths.journals)) / f"{transaction_id}.json"

def _journal_hmac(data: bytes) -> str:
    """Legacy flat-body HMAC (Task 1 restore shape)."""
    from backend.utils.crypto import sign_hmac_sha256
    return sign_hmac_sha256(data)

def _canonicalize_journal_payload(payload: dict) -> bytes:
    """Canonical JSON of flat legacy journal body with hmac removed."""
    body = {k: v for k, v in payload.items() if k != "hmac"}
    return json.dumps(body, sort_keys=True, ensure_ascii=False).encode()

def _restore_dict_from_envelope_payload(
    transaction_id: str,
    session_id: str,
    status: str,
    payload: dict,
) -> dict:
    """Merge envelope identity + payload into flat restore journal dict."""
    body = dict(payload or {})
    body["transaction_id"] = transaction_id
    body["session_id"] = session_id
    body["status"] = status
    return body

def _restore_journal_from_body(body: dict, transaction_id: str) -> RestoreJournal:
    """Parse a flat restore body (legacy or envelope-flattened) into RestoreJournal."""
    required = (
        "transaction_id",
        "target_snapshot_id",
        "session_id",
        "recovery_mode",
        "status",
    )
    missing = [k for k in required if k not in body]
    if missing:
        raise SnapshotIntegrityError(
            "JOURNAL_HMAC",
            f"Restore journal missing fields {missing}: {transaction_id}",
            details={"transaction_id": transaction_id, "missing": missing},
        )

    if "kind" in body and body.get("kind") not in (
        None,
        "",
        "restore_apply",
        "restore",
    ):
        raise SnapshotIntegrityError(
            "JOURNAL_HMAC",
            f"Restore journal has conflicting kind {body.get('kind')!r}: "
            f"{transaction_id}",
            details={"transaction_id": transaction_id, "kind": body.get("kind")},
        )

    raw_version = body.get("restore_journal_version")
    if raw_version is None:
        version: Literal[0, 1] = 0
    else:
        try:
            version_int = int(raw_version)
        except (TypeError, ValueError) as exc:
            raise SnapshotIntegrityError(
                "JOURNAL_HMAC",
                f"Restore journal invalid restore_journal_version: {transaction_id}",
                details={"transaction_id": transaction_id},
            ) from exc
        if version_int not in (0, 1):
            raise SnapshotIntegrityError(
                "JOURNAL_HMAC",
                f"Restore journal unsupported restore_journal_version={version_int}: "
                f"{transaction_id}",
                details={"transaction_id": transaction_id},
            )
        version = version_int  # type: ignore[assignment]

    return RestoreJournal(
        transaction_id=str(body["transaction_id"]),
        target_snapshot_id=str(body["target_snapshot_id"]),
        session_id=str(body["session_id"]),
        recovery_mode=str(body["recovery_mode"]),
        status=str(body["status"]),
        pre_restore_ref=str(body.get("pre_restore_ref") or ""),
        branch_id=str(body.get("branch_id") or "main"),
        continuation_kind=str(body.get("continuation_kind") or "idle"),
        target_root_hash=str(body.get("target_root_hash") or ""),
        applied_step_ledger=list(body.get("applied_step_ledger") or []),
        restore_journal_version=version,
        verified_live_root_hash=str(body.get("verified_live_root_hash") or ""),
        component_hashes=dict(body.get("component_hashes") or {}),
        verified_at=str(body.get("verified_at") or ""),
        raw=body,
    )

def _verify_flat_legacy_restore_hmac(payload: dict, transaction_id: str) -> dict:
    """Verify Task-1 flat restore journal HMAC; return body without hmac."""
    stored_hmac = payload.get("hmac")
    if not stored_hmac or not isinstance(stored_hmac, str):
        raise SnapshotIntegrityError(
            "JOURNAL_HMAC",
            f"Restore journal missing hmac: {transaction_id}",
            details={"transaction_id": transaction_id},
        )
    from backend.utils.crypto import verify_hmac_sha256
    canonical = _canonicalize_journal_payload(payload)
    if not verify_hmac_sha256(canonical, stored_hmac):
        raise SnapshotIntegrityError(
            "JOURNAL_HMAC",
            f"Restore journal HMAC mismatch: {transaction_id}",
            details={"transaction_id": transaction_id},
        )
    return {k: v for k, v in payload.items() if k != "hmac"}

def read_verified_restore_journal(paths, transaction_id: str) -> RestoreJournal:
    """Load a restore journal and verify its HMAC integrity.

    Accepts schema-v1 ``restore_apply`` envelopes (via JournalStore) and
    exact Task-1 flat HMAC restore journals (versions 0 and 1).

    Raises ``SnapshotIntegrityError`` with code ``JOURNAL_HMAC`` on missing
    signature, parse failure, or signature mismatch. Callers must treat this
    as a recovery-required, mutation-blocking condition.
    """
    jp = _journal_path(paths, transaction_id)
    if not jp.exists():
        raise SnapshotIntegrityError(
            "JOURNAL_HMAC",
            f"Restore journal missing: {transaction_id}",
            details={"transaction_id": transaction_id, "path": str(jp)},
        )
    try:
        raw_text = jp.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotIntegrityError(
            "JOURNAL_HMAC",
            f"Restore journal unreadable/corrupt: {transaction_id}: {exc}",
            details={"transaction_id": transaction_id, "path": str(jp)},
        ) from exc

    if not isinstance(payload, dict):
        raise SnapshotIntegrityError(
            "JOURNAL_HMAC",
            f"Restore journal is not an object: {transaction_id}",
            details={"transaction_id": transaction_id},
        )

    # Schema-v1 envelope (kind restore_apply)
    if payload.get("schema_version") == 1 and payload.get("kind") == "restore_apply":
        from backend.services.snapshot_v2.journal_store import JournalStore

        env = JournalStore(paths).read(transaction_id)
        if env.kind != "restore_apply":
            raise SnapshotIntegrityError(
                "JOURNAL_HMAC",
                f"Expected restore_apply envelope, got {env.kind!r}: {transaction_id}",
                details={"transaction_id": transaction_id, "kind": env.kind},
            )
        body = _restore_dict_from_envelope_payload(
            env.transaction_id, env.session_id, env.status, env.payload
        )
        return _restore_journal_from_body(body, transaction_id)

    # Flat legacy restore (Task 1)
    body = _verify_flat_legacy_restore_hmac(payload, transaction_id)
    return _restore_journal_from_body(body, transaction_id)

def _read_journal(paths, transaction_id: str) -> Optional[dict]:
    """Verified journal read. Returns None if absent; raises on tamper."""
    jp = _journal_path(paths, transaction_id)
    if not jp.exists():
        return None
    journal = read_verified_restore_journal(paths, transaction_id)
    return journal.to_dict()

def _write_journal_hmac_signed(paths, transaction_id: str, journal: dict) -> None:
    """Atomically write a restore_apply envelope (or flat v0 for test helper).

    Production writers always use schema-v1 ``restore_apply`` envelopes via
    JournalStore. The flat HMAC form is retained only for
    ``write_committed_restore_v0_for_test`` (explicit Task-1 migration fixtures).

    On-disk JSON mirrors restore fields at both top-level (compat with tests
    and readers that inspect flat keys) and inside ``payload`` (canonical
    envelope contract). HMAC is over the full envelope body excluding hmac.
    """
    # If caller requests explicit flat legacy (no envelope), keep Task-1 shape.
    if journal.get("_force_flat_legacy_v0"):
        body = {k: v for k, v in journal.items() if k not in ("hmac", "_force_flat_legacy_v0")}
        jp = _journal_path(paths, transaction_id)
        jp.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
        hmac_val = _journal_hmac(data)
        signed = json.loads(data)
        signed["hmac"] = hmac_val
        tmp = jp.with_suffix(jp.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(signed, sort_keys=True, ensure_ascii=False))
        fd = os.open(str(tmp), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(str(tmp), str(jp))
        dfd = os.open(str(jp.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        return

    from backend.services.snapshot_v2.journal_store import (
        JournalStore,
        make_envelope,
        sign_envelope_body,
    )
    from backend.utils.files import atomic_write_bytes as _atomic_write_bytes

    body = {k: v for k, v in journal.items() if k != "hmac"}
    status = str(body.get("status") or "")
    session_id = str(body.get("session_id") or "")
    payload = {
        k: v
        for k, v in body.items()
        if k
        not in (
            "transaction_id",
            "session_id",
            "status",
            "schema_version",
            "kind",
            "payload",
            "hmac",
        )
    }
    env = make_envelope(
        kind="restore_apply",
        transaction_id=transaction_id,
        session_id=session_id,
        status=status,
        payload=payload,
    )
    # Write envelope with restore fields also mirrored at top-level for
    # backward-compatible journal inspection (status, restore_journal_version, …).
    # Signing includes mirrored fields so tamper detection covers them.
    disk: dict = env.model_dump(mode="json")
    disk.pop("hmac", None)
    for k, v in payload.items():
        if k not in disk:
            disk[k] = v
    # Ensure identity fields present
    disk["transaction_id"] = transaction_id
    disk["session_id"] = session_id
    disk["status"] = status
    disk["kind"] = "restore_apply"
    disk["schema_version"] = 1
    disk["payload"] = payload
    sig = sign_envelope_body(disk)
    disk["hmac"] = sig
    text = json.dumps(disk, sort_keys=True, ensure_ascii=False)
    path = _journal_path(paths, transaction_id)
    _atomic_write_bytes(path, text.encode())

def migrate_verified_restore_v0(
    paths,
    journal: RestoreJournal | dict,
    proof: dict,
) -> RestoreJournal:
    """Atomically rewrite a verified v0 COMMITTED journal as signed v1 envelope.

    Never accepts unsigned input — *journal* must already be HMAC-verified.
    Writes a schema-v1 ``restore_apply`` envelope with restore_journal_version=1.
    """
    if isinstance(journal, RestoreJournal):
        body = journal.to_dict()
        tx_id = journal.transaction_id
    else:
        body = dict(journal)
        tx_id = str(body["transaction_id"])
    body["status"] = "COMMITTED"
    body["restore_journal_version"] = 1
    body["verified_live_root_hash"] = str(proof.get("verified_live_root_hash") or "")
    body["component_hashes"] = dict(proof.get("component_hashes") or {})
    body["verified_at"] = str(proof.get("verified_at") or "")
    _write_journal_hmac_signed(paths, tx_id, body)
    return read_verified_restore_journal(paths, tx_id)
