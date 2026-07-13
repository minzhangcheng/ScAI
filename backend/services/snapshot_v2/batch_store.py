"""Signed import-batch commit records and membership validation for Snapshot V2.

Each ImportBatchRecord represents an atomic import transaction: member
manifests are installed under ``committed/{snapshot_id}.json`` but remain
invisible to ``ManifestStore.list_*`` / ``load()`` / ``verify_all_committed()``
until the signed batch record verifies that exact set of snapshot IDs and
root hashes.  A batch record is the single visibility gate.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import Literal

from backend.services.snapshot_v2.errors import SnapshotIntegrityError
from backend.utils.crypto import sign_hmac_sha256, verify_hmac_sha256
from backend.utils.files import atomic_write_bytes, atomic_write_text, atomic_write_json

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_batch_document_bytes(record: ImportBatchRecord) -> bytes:
    """Canonical JSON of a batch record (sorted, compact, no hmac)."""
    payload = record.model_dump(mode="json")
    payload.pop("hmac", None)
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


class ImportBatchRecord(BaseModel):
    """Signed record that makes an import batch visible.

    The ``manifest_roots`` dict maps snapshot_id → root_hash for every
    member snapshot.  The batch is only authoritative when its ``hmac``
    field is verified against the canonical body.
    """

    schema_version: Literal[1] = 1
    batch_id: str
    session_id: str
    manifest_roots: dict[str, str]
    created_at: str = ""
    hmac: str = ""

    def model_dump(self, *args, **kwargs):
        # Ensure hmac is excluded by default from serialized output
        # unless explicitly included.
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)


class ImportBatchStore:
    """Manages signed batch records under ``<v2>/batches/``.

    The on-disk layout is::

        batches/
            {batch_id}.json   # signed ImportBatchRecord
            {batch_id}.sig    # HMAC signature (written first for atomicity)
    """

    def __init__(self, v2_root: Path):
        self._batches_dir = v2_root / "batches"

    def _record_path(self, batch_id: str) -> Path:
        return self._batches_dir / f"{batch_id}.json"

    def _signature_path(self, batch_id: str) -> Path:
        return self._batches_dir / f"{batch_id}.sig"

    def stage(self, record: ImportBatchRecord) -> ImportBatchRecord:
        """Write an incomplete batch record (no hmac) to disk.

        The record is not visible until ``publish()`` finalises it.
        """
        self._batches_dir.mkdir(parents=True, exist_ok=True)
        if not record.created_at:
            record.created_at = _now_iso()
        body = _canonical_batch_document_bytes(record)
        path = self._record_path(record.batch_id)
        atomic_write_bytes(path, body)
        return record

    def publish(self, record: ImportBatchRecord) -> ImportBatchRecord:
        """Atomically sign and commit a batch record.

        Writes the signature first, then the signed record body.
        Returns the verified record loaded from disk.
        """
        self._batches_dir.mkdir(parents=True, exist_ok=True)
        if not record.created_at:
            record.created_at = _now_iso()
        body = _canonical_batch_document_bytes(record)
        signature = sign_hmac_sha256(body)
        # Write signature first (atomic)
        atomic_write_text(self._signature_path(record.batch_id), signature)
        # Write the record (atomic)
        atomic_write_bytes(self._record_path(record.batch_id), body)
        return self.load(record.batch_id)

    def load(self, batch_id: str) -> ImportBatchRecord | None:
        """Load and verify a batch record.

        Returns None if the record does not exist.  Raises
        SnapshotIntegrityError on HMAC mismatch or parse failure.
        """
        path = self._record_path(batch_id)
        sig_path = self._signature_path(batch_id)
        if not path.exists():
            return None
        if not sig_path.exists():
            raise SnapshotIntegrityError(
                "BATCH_HMAC",
                f"Batch record {batch_id} has no signature file",
            )
        body = path.read_bytes()
        sig = sig_path.read_text().strip()
        if not verify_hmac_sha256(body, sig):
            raise SnapshotIntegrityError(
                "BATCH_HMAC",
                f"Batch record HMAC mismatch: {batch_id}",
            )
        data = json.loads(body)
        return ImportBatchRecord(**data)

    def remove(self, batch_id: str) -> bool:
        """Remove batch record and signature files unconditionally.

        Returns True if files existed and were removed, False if neither
        file was present.
        """
        rp = self._record_path(batch_id)
        sp = self._signature_path(batch_id)
        existed = rp.exists() or sp.exists()
        rp.unlink(missing_ok=True)
        sp.unlink(missing_ok=True)
        return existed

    def remove_if_empty(self, batch_id: str) -> bool:
        """Remove batch files if the record has no member manifests.

        Returns True if removed, False if not empty or not found.
        """
        record = self.load(batch_id)
        if record is None:
            return False
        if record.manifest_roots:
            return False
        self._record_path(batch_id).unlink(missing_ok=True)
        self._signature_path(batch_id).unlink(missing_ok=True)
        return True

    def list_batch_ids(self) -> list[str]:
        """Return all batch IDs from the batches directory."""
        if not self._batches_dir.exists():
            return []
        return sorted(
            p.stem for p in self._batches_dir.iterdir()
            if p.suffix == ".json" and p.is_file()
        )
