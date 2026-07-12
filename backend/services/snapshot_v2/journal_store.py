"""Signed discriminated journal store for Snapshot V2.

All journal kinds (restore_apply, restore_job, delete, import,
export_protection) share one envelope schema and host HMAC.

Legacy HMAC-valid restore journals (Task 1 exact shape, restore journal
version 0 or 1, no schema_version envelope) are accepted on read and
projected into a ``restore_apply`` envelope. Unsigned or ambiguous JSON
raises ``RecoveryFailure("LEGACY_UNSIGNED_JOURNAL")`` and is never mutated.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator

from backend.services.snapshot_v2.errors import SnapshotIntegrityError
from backend.services.snapshot_v2.models import JournalKind, SnapshotJournalEnvelope

logger = logging.getLogger(__name__)


def _legacy_unsigned(transaction_id: str, path: Path, message: str | None = None):
    from backend.services.snapshot_v2.recovery import RecoveryFailure

    raise RecoveryFailure(
        "LEGACY_UNSIGNED_JOURNAL",
        message
        or (
            f"Unsigned or ambiguous journal {transaction_id} cannot be recovered "
            "automatically"
        ),
        details={"transaction_id": transaction_id, "path": str(path)},
    )

_ENVELOPE_TOP_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "transaction_id",
        "session_id",
        "status",
        "payload",
        "hmac",
    }
)

_RESTORE_REQUIRED = (
    "transaction_id",
    "target_snapshot_id",
    "session_id",
    "recovery_mode",
    "status",
)


def _secret_bytes() -> bytes:
    from backend.utils.crypto import get_integrity_secret

    secret = get_integrity_secret()
    if isinstance(secret, bytes):
        return secret
    return secret.encode()


def canonicalize_envelope_body(body: dict[str, Any]) -> bytes:
    """Canonical JSON of envelope body with hmac removed (sorted, compact)."""
    payload = {k: v for k, v in body.items() if k != "hmac"}
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def sign_envelope_body(body: dict[str, Any]) -> str:
    return _hmac.new(
        _secret_bytes(), canonicalize_envelope_body(body), hashlib.sha256
    ).hexdigest()


def _journal_path(paths, transaction_id: str) -> Path:
    return Path(str(paths.journals)) / f"{transaction_id}.json"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_bytes(data)
    fd = os.open(str(tmp), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(str(tmp), str(path))
    dfd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _is_schema_v1_envelope(payload: dict[str, Any]) -> bool:
    if payload.get("schema_version") == 1 and "kind" in payload:
        return True
    # Explicit envelope keys without schema_version still count if kind present
    # and payload field is a dict (discriminated shape).
    if (
        payload.get("kind") in (
            "restore_apply",
            "restore_job",
            "delete",
            "import",
            "export_protection",
        )
        and isinstance(payload.get("payload"), dict)
        and "schema_version" in payload
    ):
        return True
    return False


def _normalize_envelope_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip mirrored restore fields so pydantic envelope validation succeeds.

    Production restore_apply journals may mirror payload keys at top-level
    for inspection compatibility; those extras are not model fields.
    """
    if not isinstance(payload.get("payload"), dict):
        # Ensure payload exists for kinds that only used top-level fields
        payload = dict(payload)
        payload.setdefault("payload", {})
    # Keep only known envelope keys for validation; extras ignored via dump.
    return {
        "schema_version": payload.get("schema_version", 1),
        "kind": payload.get("kind"),
        "transaction_id": payload.get("transaction_id"),
        "session_id": payload.get("session_id"),
        "status": payload.get("status"),
        "payload": payload.get("payload") or {},
        "hmac": payload.get("hmac", ""),
    }


def _looks_like_legacy_restore(payload: dict[str, Any]) -> bool:
    """Exact Task-1 restore shape (no envelope schema_version / nested payload)."""
    if payload.get("schema_version") == 1:
        return False
    if isinstance(payload.get("payload"), dict) and payload.get("kind") in (
        "restore_apply",
        "restore_job",
        "delete",
        "import",
        "export_protection",
    ):
        return False
    for key in _RESTORE_REQUIRED:
        if key not in payload:
            return False
    kind = payload.get("kind")
    if kind not in (None, "", "restore_apply", "restore"):
        return False
    return True


def _restore_journal_to_envelope(body: dict[str, Any]) -> SnapshotJournalEnvelope:
    """Project verified legacy restore body into restore_apply envelope."""
    payload = {
        k: v
        for k, v in body.items()
        if k
        not in (
            "hmac",
            "schema_version",
            "kind",
            "transaction_id",
            "session_id",
            "status",
            "payload",
        )
    }
    # Keep restore-specific identity fields in payload
    for key in (
        "target_snapshot_id",
        "recovery_mode",
        "pre_restore_ref",
        "branch_id",
        "continuation_kind",
        "target_root_hash",
        "applied_step_ledger",
        "restore_journal_version",
        "verified_live_root_hash",
        "component_hashes",
        "verified_at",
        "runtime_choice",
        "object_refs",
    ):
        if key in body and key not in payload:
            payload[key] = body[key]
    return SnapshotJournalEnvelope(
        schema_version=1,
        kind="restore_apply",
        transaction_id=str(body["transaction_id"]),
        session_id=str(body["session_id"]),
        status=str(body["status"]),
        payload=payload,
        hmac="",
    )


def envelope_to_restore_dict(envelope: SnapshotJournalEnvelope) -> dict[str, Any]:
    """Flatten restore_apply envelope into the dict shape restore recovery expects."""
    if envelope.kind != "restore_apply":
        raise ValueError(f"expected restore_apply, got {envelope.kind}")
    d = dict(envelope.payload or {})
    d["transaction_id"] = envelope.transaction_id
    d["session_id"] = envelope.session_id
    d["status"] = envelope.status
    d.setdefault("target_snapshot_id", d.get("target_snapshot_id", ""))
    d.setdefault("recovery_mode", d.get("recovery_mode", "rollback"))
    d.setdefault("pre_restore_ref", d.get("pre_restore_ref", ""))
    d.setdefault("applied_step_ledger", d.get("applied_step_ledger") or [])
    return d


def make_envelope(
    *,
    kind: JournalKind,
    transaction_id: str,
    session_id: str,
    status: str,
    payload: dict[str, Any] | None = None,
) -> SnapshotJournalEnvelope:
    return SnapshotJournalEnvelope(
        schema_version=1,
        kind=kind,
        transaction_id=transaction_id,
        session_id=session_id,
        status=status,
        payload=dict(payload or {}),
        hmac="",
    )


class JournalStore:
    """Atomic HMAC-signed journal I/O for all Snapshot V2 journal kinds."""

    def __init__(self, paths) -> None:
        self.paths = paths

    def write(self, envelope: SnapshotJournalEnvelope) -> SnapshotJournalEnvelope:
        """Sign and atomically write *envelope*. Returns the signed envelope."""
        body = envelope.model_dump(mode="json")
        body.pop("hmac", None)
        sig = sign_envelope_body(body)
        body["hmac"] = sig
        # Prefer human-readable sorted JSON on disk (stable keys)
        text = json.dumps(body, sort_keys=True, ensure_ascii=False)
        # Re-sign over compact canonical form — ensure hmac matches canonicalize
        # of the same body dict (already signed above via sign_envelope_body).
        path = _journal_path(self.paths, envelope.transaction_id)
        _atomic_write_bytes(path, text.encode())
        return SnapshotJournalEnvelope.model_validate(body)

    def read(self, transaction_id: str) -> SnapshotJournalEnvelope:
        """Load and verify a journal; project legacy restore into envelope.

        Raises:
            SnapshotIntegrityError: missing file, HMAC mismatch on envelope
                or on exact HMAC-valid restore legacy shape.
            RecoveryFailure(LEGACY_UNSIGNED_JOURNAL): unsigned/ambiguous JSON.
        """
        path = _journal_path(self.paths, transaction_id)
        if not path.exists():
            raise SnapshotIntegrityError(
                "JOURNAL_HMAC",
                f"Journal missing: {transaction_id}",
                details={"transaction_id": transaction_id, "path": str(path)},
            )
        try:
            raw_text = path.read_text(encoding="utf-8")
            payload = json.loads(raw_text)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SnapshotIntegrityError(
                "JOURNAL_HMAC",
                f"Journal unreadable/corrupt: {transaction_id}: {exc}",
                details={"transaction_id": transaction_id, "path": str(path)},
            ) from exc

        if not isinstance(payload, dict):
            raise SnapshotIntegrityError(
                "JOURNAL_HMAC",
                f"Journal is not an object: {transaction_id}",
                details={"transaction_id": transaction_id},
            )

        if _is_schema_v1_envelope(payload):
            return self._verify_envelope(payload, transaction_id)

        # Legacy path: only exact HMAC-valid restore shape
        if _looks_like_legacy_restore(payload):
            return self._read_legacy_restore(payload, transaction_id, path)

        # Anything else: unsigned/ambiguous — do not mutate
        _legacy_unsigned(transaction_id, path)

    def _verify_envelope(
        self, payload: dict[str, Any], transaction_id: str
    ) -> SnapshotJournalEnvelope:
        stored_hmac = payload.get("hmac")
        if not stored_hmac or not isinstance(stored_hmac, str):
            raise SnapshotIntegrityError(
                "JOURNAL_HMAC",
                f"Journal missing hmac: {transaction_id}",
                details={"transaction_id": transaction_id},
            )
        # Sign over the full on-disk body (including mirrored top-level fields).
        expected = sign_envelope_body(payload)
        if not _hmac.compare_digest(expected, stored_hmac):
            raise SnapshotIntegrityError(
                "JOURNAL_HMAC",
                f"Journal HMAC mismatch: {transaction_id}",
                details={"transaction_id": transaction_id},
            )
        # Prefer nested payload; if empty, promote mirrored restore fields.
        nested = dict(payload.get("payload") or {})
        if not nested:
            for k, v in payload.items():
                if k not in _ENVELOPE_TOP_KEYS and k != "hmac":
                    nested[k] = v
        else:
            # Merge any mirrored keys missing from nested payload
            for k, v in payload.items():
                if k not in _ENVELOPE_TOP_KEYS and k != "hmac" and k not in nested:
                    nested[k] = v
        try:
            env = SnapshotJournalEnvelope.model_validate(
                {
                    "schema_version": payload.get("schema_version", 1),
                    "kind": payload.get("kind"),
                    "transaction_id": payload.get("transaction_id"),
                    "session_id": payload.get("session_id"),
                    "status": payload.get("status"),
                    "payload": nested,
                    "hmac": stored_hmac,
                }
            )
        except Exception as exc:
            raise SnapshotIntegrityError(
                "JOURNAL_HMAC",
                f"Journal envelope invalid: {transaction_id}: {exc}",
                details={"transaction_id": transaction_id},
            ) from exc
        return env

    def _read_legacy_restore(
        self,
        payload: dict[str, Any],
        transaction_id: str,
        path: Path,
    ) -> SnapshotJournalEnvelope:
        """Verify Task-1 restore HMAC and project to restore_apply envelope.

        Only the exact HMAC-valid restore shape is migrated. Missing or
        invalid HMAC on a flat legacy file is ``LEGACY_UNSIGNED_JOURNAL``
        (not envelope JOURNAL_HMAC) so operators can inspect without
        recovery mutations.
        """
        from backend.services.snapshot_v2.restore import (
            read_verified_restore_journal,
        )

        stored_hmac = payload.get("hmac")
        if not stored_hmac or not isinstance(stored_hmac, str):
            _legacy_unsigned(
                transaction_id,
                path,
                f"LEGACY_UNSIGNED_JOURNAL: unsigned legacy restore journal "
                f"{transaction_id}",
            )

        try:
            verified = read_verified_restore_journal(self.paths, transaction_id)
        except SnapshotIntegrityError as exc:
            # HMAC mismatch / corrupt flat body → treat as legacy unsigned
            # (do not confuse with schema-v1 envelope JOURNAL_HMAC).
            _legacy_unsigned(
                transaction_id,
                path,
                f"LEGACY_UNSIGNED_JOURNAL: invalid legacy restore journal "
                f"{transaction_id}: {exc}",
            )
        except Exception as exc:
            _legacy_unsigned(
                transaction_id,
                path,
                f"Legacy journal {transaction_id} failed restore verification: {exc}",
            )
        return _restore_journal_to_envelope(verified.to_dict())

    def iter_verified(self, session_id: str) -> Iterator[SnapshotJournalEnvelope]:
        """Yield verified envelopes for *session_id* (sorted by filename).

        Integrity / legacy failures propagate to the caller (fail closed).
        """
        journals_dir = Path(str(self.paths.journals))
        if not journals_dir.exists():
            return
        for jp in sorted(journals_dir.iterdir()):
            if not jp.is_file() or jp.suffix != ".json":
                continue
            env = self.read(jp.stem)
            if env.session_id not in (None, "", session_id):
                continue
            yield env

    def remove(self, transaction_id: str) -> None:
        path = _journal_path(self.paths, transaction_id)
        if path.exists():
            path.unlink()

    def try_read(self, transaction_id: str) -> SnapshotJournalEnvelope | None:
        """Return envelope or None if missing; re-raise integrity/legacy errors."""
        path = _journal_path(self.paths, transaction_id)
        if not path.exists():
            return None
        return self.read(transaction_id)
