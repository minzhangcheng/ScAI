"""Immutable figure transaction types and constants."""
from __future__ import annotations

import uuid
from typing import Any

TRANSACTION_SCHEMA_VERSION = 2

TRANSACTION_ROOT = ".transactions"

JOURNAL_FILENAME = "journal.json"

COMMIT_MARKER = "COMMITTED"

ABANDON_MARKER = "ABANDONED"

FAILURE_MARKER = "FAILED"

ROLLBACK_DURABLE_MARKER = "ROLLBACK_DURABLE"

_PROCESS_TOKEN = uuid.uuid4().hex

_LockOwner = tuple[str, str, str, tuple[str, ...]]

_HELD_LOCKS: dict[str, tuple[_LockOwner, Any]] = {}

_MAX_JOURNAL_BYTES = 4 * 1024 * 1024

_MAX_GC_ENTRIES = 100

_STALE_ENTRY_SECONDS = 24 * 60 * 60

_QUARANTINE_RETENTION_SECONDS = 7 * 24 * 60 * 60

_RUNTIME_JOURNAL_FIELDS = frozenset({
    "attempt_dir",
    "staged_jpg",
    "staged_pdf",
    "final_jpg",
    "final_pdf",
    "backup_jpg",
    "backup_pdf",
    "archive_jpg",
    "archive_pdf",
    "journal_path",
    "lock_keys",
})

_TERMINAL_MARKERS = frozenset({COMMIT_MARKER, ABANDON_MARKER, FAILURE_MARKER})

_TRANSACTION_RECORD_FIELDS = frozenset({
    "png_path",
    "pdf_path",
    "render_sha256",
    "preview_sha256",
    "pdf_sha256",
    "producer_status",
    "is_published_head",
    "supersedes_id",
    "revision_of_id",
})
