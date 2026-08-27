"""Crash-recoverable publication-figure file and record transactions.

A publication render is a pair of files plus the ``PublicationFigure`` record
that binds review metadata to those exact bytes.  Replacing the JPG first and
the PDF second is not a transaction: a process death between either rename and
the state write can leave a new image attached to an old record.  This module
keeps a durable journal beside hidden render attempts, fsyncs every boundary,
and either forward-commits a record that reached durable session state or rolls
the pair back and leaves an explicit abandoned audit record.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import shutil
import stat
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from backend.utils.crypto import sign_hmac_sha256
from .figure_transaction_parts import (
    filesystem as _filesystem_owner,
    journal as _journal_owner,
    locks as _locks_owner,
)
from .figure_transaction_parts.types import (
    TRANSACTION_SCHEMA_VERSION,
    TRANSACTION_ROOT,
    JOURNAL_FILENAME,
    COMMIT_MARKER,
    ABANDON_MARKER,
    FAILURE_MARKER,
    ROLLBACK_DURABLE_MARKER,
    _PROCESS_TOKEN,
    _LockOwner,
    _HELD_LOCKS,
    _MAX_JOURNAL_BYTES,
    _MAX_GC_ENTRIES,
    _STALE_ENTRY_SECONDS,
    _QUARANTINE_RETENTION_SECONDS,
    _RUNTIME_JOURNAL_FIELDS,
    _TERMINAL_MARKERS,
    _TRANSACTION_RECORD_FIELDS,
)

logger = logging.getLogger(__name__)


# These values are useful to the live caller but are deliberately not persisted.
# Every one is derived from the authenticated identity fields and trusted roots;
# accepting a second path representation from disk would make the journal a path
# traversal capability.


def _value(item: Any, key: str, default: Any = None) -> Any:
    return _filesystem_owner._value(item, key, default, bindings=sys.modules[__name__])


def _set(item: Any, key: str, value: Any) -> None:
    return _filesystem_owner._set(item, key, value, bindings=sys.modules[__name__])


def _safe_component(value: Any, fallback: str) -> str:
    return _filesystem_owner._safe_component(value, fallback, bindings=sys.modules[__name__])


def _archive_record_component(record_id: Any) -> str:
    """Map an arbitrary record identity to one collision-resistant path part."""
    return _filesystem_owner._archive_record_component(record_id, bindings=sys.modules[__name__])


def _absolute(path: Path) -> Path:
    """Return a lexical absolute path without following its final component."""
    return _filesystem_owner._absolute(path, bindings=sys.modules[__name__])


def _directory_flags() -> int:
    return _filesystem_owner._directory_flags(bindings=sys.modules[__name__])


def _open_directory_nofollow(path: Path) -> int:
    """Open one real directory, rejecting a file or symlink final component."""
    return _filesystem_owner._open_directory_nofollow(path, bindings=sys.modules[__name__])


def _open_child_directory_nofollow(
    parent_fd: int,
    name: str,
    *,
    create: bool = False,
) -> int:
    """Open one direct child directory relative to an already trusted FD."""
    return _filesystem_owner._open_child_directory_nofollow(parent_fd, name, create=create, bindings=sys.modules[__name__])


def _require_directory_entry_binding(
    parent_fd: int,
    name: str,
    child_fd: int,
    *,
    label: str,
) -> None:
    """Require a live parent entry to still name the opened directory inode."""
    return _filesystem_owner._require_directory_entry_binding(parent_fd, name, child_fd, label=label, bindings=sys.modules[__name__])


def _open_transaction_root(
    figures_dir: Path,
    *,
    create: bool = False,
) -> tuple[int, int]:
    """Open ``figures/.transactions`` without following either component."""
    return _filesystem_owner._open_transaction_root(figures_dir, create=create, bindings=sys.modules[__name__])


def _open_transaction_directory(transaction_dir: Path) -> tuple[int, int, int]:
    """Open one direct transaction directory through trusted directory FDs."""
    return _filesystem_owner._open_transaction_directory(transaction_dir, bindings=sys.modules[__name__])


def _open_attempt_directory(
    attempt_dir: Path,
    *,
    create: bool = False,
) -> tuple[int, int, int]:
    """Open ``figures/.attempts/<attempt>`` without following components."""
    return _filesystem_owner._open_attempt_directory(attempt_dir, create=create, bindings=sys.modules[__name__])


def _open_revision_directory(
    revision_dir: Path,
    *,
    create: bool = False,
) -> tuple[int, int, int, int]:
    """Open ``figures/.revisions/<record>/<attempt>`` component by component."""
    return _filesystem_owner._open_revision_directory(revision_dir, create=create, bindings=sys.modules[__name__])


def _require_transaction_directory_binding(
    figures_fd: int,
    root_fd: int,
    transaction_fd: int,
    attempt_id: str,
) -> None:
    return _filesystem_owner._require_transaction_directory_binding(figures_fd, root_fd, transaction_fd, attempt_id, bindings=sys.modules[__name__])


def _require_attempt_directory_binding(
    figures_fd: int,
    attempts_fd: int,
    attempt_fd: int,
    attempt_id: str,
) -> None:
    return _filesystem_owner._require_attempt_directory_binding(figures_fd, attempts_fd, attempt_fd, attempt_id, bindings=sys.modules[__name__])


def _open_known_file_parent(
    path: Path,
    *,
    create: bool = False,
) -> tuple[int, list[int]]:
    """Open the trusted parent for public and hidden figure transaction files."""
    return _filesystem_owner._open_known_file_parent(path, create=create, bindings=sys.modules[__name__])


def _close_parent_chain(parent_fd: int, ancestors: Iterable[int]) -> None:
    return _filesystem_owner._close_parent_chain(parent_fd, ancestors, bindings=sys.modules[__name__])


def _digest_regular_file_at(parent_fd: int, name: str) -> str:
    """Hash one stable regular direct child using one no-follow descriptor."""
    return _filesystem_owner._digest_regular_file_at(parent_fd, name, bindings=sys.modules[__name__])


def _digest_open_regular_file(descriptor: int, *, label: str) -> str:
    """Hash an already opened regular file and restore its seek position."""
    return _filesystem_owner._digest_open_regular_file(descriptor, label=label, bindings=sys.modules[__name__])


def _open_frozen_staged_file(
    attempt_fd: int,
    name: str,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[int, os.stat_result]:
    return _filesystem_owner._open_frozen_staged_file(attempt_fd, name, expected_sha256, label=label, bindings=sys.modules[__name__])


def _open_stable_staged_file(
    attempt_fd: int,
    name: str,
    *,
    label: str,
) -> tuple[int, os.stat_result, str]:
    """Open and hash one pending staged file without trusting its path again."""
    return _filesystem_owner._open_stable_staged_file(attempt_fd, name, label=label, bindings=sys.modules[__name__])


def _require_regular_entry_binding(
    parent_fd: int,
    name: str,
    descriptor: int,
    *,
    label: str,
) -> None:
    """Require a direct filename to still name the opened regular-file inode."""
    return _filesystem_owner._require_regular_entry_binding(parent_fd, name, descriptor, label=label, bindings=sys.modules[__name__])


def _regular_child_exists(parent_fd: int, name: str, *, label: str) -> bool:
    return _filesystem_owner._regular_child_exists(parent_fd, name, label=label, bindings=sys.modules[__name__])


def _replace_staged_entry(
    attempt_fd: int,
    figures_fd: int,
    source_name: str,
    destination_name: str,
) -> None:
    return _filesystem_owner._replace_staged_entry(attempt_fd, figures_fd, source_name, destination_name, bindings=sys.modules[__name__])


def _verify_published_entry(
    figures_fd: int,
    name: str,
    staged_fd: int,
    staged_metadata: os.stat_result,
    expected_sha256: str,
    *,
    label: str,
) -> None:
    return _filesystem_owner._verify_published_entry(figures_fd, name, staged_fd, staged_metadata, expected_sha256, label=label, bindings=sys.modules[__name__])


def _read_regular_file_at(
    parent_fd: int,
    name: str,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    """Read and validate one regular file entirely through one no-follow FD."""
    return _filesystem_owner._read_regular_file_at(parent_fd, name, max_bytes=max_bytes, label=label, bindings=sys.modules[__name__])


def _read_transaction_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    return _filesystem_owner._read_transaction_file(path, max_bytes=max_bytes, label=label, bindings=sys.modules[__name__])


def _fsync_file(path: Path) -> None:
    return _filesystem_owner._fsync_file(path, bindings=sys.modules[__name__])


def _fsync_dir(path: Path) -> None:
    return _filesystem_owner._fsync_dir(path, bindings=sys.modules[__name__])


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    return _filesystem_owner._atomic_json(path, payload, bindings=sys.modules[__name__])


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return _journal_owner._canonical_json(payload, bindings=sys.modules[__name__])


def _journal_body(payload: Mapping[str, Any]) -> bytes:
    return _journal_owner._journal_body(payload, bindings=sys.modules[__name__])


def _marker_body(payload: Mapping[str, Any]) -> bytes:
    return _journal_owner._marker_body(payload, bindings=sys.modules[__name__])


def _write_marker(path: Path) -> None:
    """Write an authenticated transaction marker bound to journal identity."""
    return _journal_owner._write_marker(path, bindings=sys.modules[__name__])


def _authenticated_marker_exists(
    transaction_dir: Path,
    marker: str,
    journal: Mapping[str, Any],
) -> bool:
    return _journal_owner._authenticated_marker_exists(transaction_dir, marker, journal, bindings=sys.modules[__name__])


def _unlink_transaction_regular_file_if_present(
    transaction_dir: Path,
    name: str,
) -> None:
    """Unlink one direct transaction child only through a verified directory FD."""
    return _journal_owner._unlink_transaction_regular_file_if_present(transaction_dir, name, bindings=sys.modules[__name__])


def _sha256(path: Path) -> str | None:
    return _journal_owner._sha256(path, bindings=sys.modules[__name__])


def _snapshot_record(record: Any, *, include_preview: bool = False) -> dict[str, Any]:
    return _journal_owner._snapshot_record(record, include_preview=include_preview, bindings=sys.modules[__name__])


def _find_record(records: Iterable[Any], record_id: str) -> Any | None:
    return _journal_owner._find_record(records, record_id, bindings=sys.modules[__name__])


def _journal_paths(figures_dir: Path, attempt_id: str) -> tuple[Path, Path]:
    return _journal_owner._journal_paths(figures_dir, attempt_id, bindings=sys.modules[__name__])


def _publish_lock_keys(
    figures_dir: Path,
    final_jpg: Path,
    published_record: Any,
) -> list[str]:
    return _locks_owner._publish_lock_keys(figures_dir, final_jpg, published_record, bindings=sys.modules[__name__])


def _attempt_lock_key(figures_dir: Path, attempt_id: str) -> str:
    return _locks_owner._attempt_lock_key(figures_dir, attempt_id, bindings=sys.modules[__name__])


def _transaction_creation_is_live(figures_dir: Path, attempt_id: str) -> bool:
    """Probe the attempt-scoped flock before classifying a pre-journal directory."""
    return _locks_owner._transaction_creation_is_live(figures_dir, attempt_id, bindings=sys.modules[__name__])


def _require_within(path: Path, root: Path, *, label: str) -> Path:
    return _journal_owner._require_within(path, root, label=label, bindings=sys.modules[__name__])


def _validate_regular_child(parent_fd: int, name: str, *, label: str) -> None:
    return _journal_owner._validate_regular_child(parent_fd, name, label=label, bindings=sys.modules[__name__])


def _validate_optional_regular_child(
    parent_fd: int,
    name: str,
    *,
    label: str,
) -> None:
    return _journal_owner._validate_optional_regular_child(parent_fd, name, label=label, bindings=sys.modules[__name__])


def _validate_registration_paths(
    *,
    attempt_dir: Path,
    staged_jpg: Path,
    staged_pdf: Path,
    final_jpg: Path,
    final_pdf: Path,
    published_record: Any,
) -> tuple[Path, str, str]:
    return _journal_owner._validate_registration_paths(attempt_dir=attempt_dir, staged_jpg=staged_jpg, staged_pdf=staged_pdf, final_jpg=final_jpg, final_pdf=final_pdf, published_record=published_record, bindings=sys.modules[__name__])


def _hydrate_verified_journal(
    payload: dict[str, Any],
    journal_path: Path,
) -> dict[str, Any]:
    return _journal_owner._hydrate_verified_journal(payload, journal_path, bindings=sys.modules[__name__])


def _read_verified_journal(journal_path: Path) -> dict[str, Any]:
    return _journal_owner._read_verified_journal(journal_path, bindings=sys.modules[__name__])


def _quarantine_transaction_dir(transaction_dir: Path) -> bool:
    """Isolate one untrusted root entry without following any journal path."""
    return _journal_owner._quarantine_transaction_dir(transaction_dir, bindings=sys.modules[__name__])


def _lock_file(figures_dir: Path, key: str) -> Path:
    return _locks_owner._lock_file(figures_dir, key, bindings=sys.modules[__name__])


def _lock_owner(
    figures_dir: Path,
    session_id: str,
    attempt_id: str,
    keys: Iterable[str],
) -> _LockOwner:
    return _locks_owner._lock_owner(figures_dir, session_id, attempt_id, keys, bindings=sys.modules[__name__])


def _acquire_publish_locks(
    figures_dir: Path,
    attempt_id: str,
    keys: Iterable[str],
    *,
    session_id: str = "",
) -> str | None:
    return _locks_owner._acquire_publish_locks(figures_dir, attempt_id, keys, session_id=session_id, bindings=sys.modules[__name__])


def _release_publish_locks(
    attempt_id: str,
    *,
    keys: Iterable[str] | None = None,
    figures_dir: Path | None = None,
    session_id: str | None = None,
) -> None:
    return _locks_owner._release_publish_locks(attempt_id, keys=keys, figures_dir=figures_dir, session_id=session_id, bindings=sys.modules[__name__])


def acquire_figure_publish_locks(
    *,
    figures_dir: Path,
    final_jpg: Path,
    published_record: Any,
) -> str | None:
    """Acquire path and semantic-lineage locks before choosing the old head."""
    return _locks_owner.acquire_figure_publish_locks(figures_dir=figures_dir, final_jpg=final_jpg, published_record=published_record, bindings=sys.modules[__name__])


def _persist_journal(journal_path: Path, journal: dict[str, Any]) -> None:
    return _journal_owner._persist_journal(journal_path, journal, bindings=sys.modules[__name__])


def _complete_step(
    journal_path: Path,
    journal: dict[str, Any],
    step: str,
    *,
    status: str | None = None,
) -> None:
    return _journal_owner._complete_step(journal_path, journal, step, status=status, bindings=sys.modules[__name__])


def _copy_durable(source: Path, destination: Path) -> None:
    """Copy a regular file atomically without following hidden path entries."""
    return _filesystem_owner._copy_durable(source, destination, bindings=sys.modules[__name__])


def register_pending_figure_attempt(
    *,
    attempt_dir: Path,
    staged_jpg: Path,
    staged_pdf: Path,
    final_jpg: Path,
    final_pdf: Path,
    published_record: Any,
    session_id: str = "",
) -> dict[str, Any]:
    """Persist the producer record while owning both publication locks."""
    figures_dir, attempt_id, figure_name = _validate_registration_paths(
        attempt_dir=attempt_dir,
        staged_jpg=staged_jpg,
        staged_pdf=staged_pdf,
        final_jpg=final_jpg,
        final_pdf=final_pdf,
        published_record=published_record,
    )
    if not str(session_id or ""):
        raise ValueError("publication transaction requires a session identity")
    transaction_dir, journal_path = _journal_paths(figures_dir, attempt_id)
    lock_keys = _publish_lock_keys(figures_dir, Path(final_jpg), published_record)
    lock_error = _acquire_publish_locks(
        figures_dir,
        attempt_id,
        lock_keys,
        session_id=str(session_id),
    )
    if lock_error:
        raise RuntimeError(lock_error)

    created_transaction_dir = False
    try:
        figures_fd, root_fd = _open_transaction_root(figures_dir, create=True)
        try:
            os.mkdir(attempt_id, mode=0o700, dir_fd=root_fd)
            created_transaction_dir = True
            transaction_fd = _open_child_directory_nofollow(
                root_fd, attempt_id, create=False
            )
            os.close(transaction_fd)
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
            os.close(figures_fd)
        _fsync_file(Path(staged_jpg))
        _fsync_file(Path(staged_pdf))
        _fsync_dir(Path(attempt_dir))
        new_render_sha256 = _sha256(Path(staged_jpg))
        new_pdf_sha256 = _sha256(Path(staged_pdf))
        if not new_render_sha256 or not new_pdf_sha256:
            raise ValueError("publication staged pair is not stable and regular")
        journal = {
            "schema_version": TRANSACTION_SCHEMA_VERSION,
            "session_id": str(session_id),
            "attempt_id": attempt_id,
            "figure_name": figure_name,
            "owner_pid": os.getpid(),
            "owner_process_token": _PROCESS_TOKEN,
            "status": "REVIEW_PENDING",
            "completed_steps": [],
            "had_final_jpg": Path(final_jpg).exists(),
            "had_final_pdf": Path(final_pdf).exists(),
            "new_record_id": str(_value(published_record, "id", "")),
            "new_record": _snapshot_record(published_record),
            "old_record_id": "",
            "old_record_before": {},
            "new_render_sha256": new_render_sha256,
            "new_pdf_sha256": new_pdf_sha256,
        }
        journal["journal_hmac"] = sign_hmac_sha256(_journal_body(journal))
        runtime = _hydrate_verified_journal(journal, journal_path)
        _persist_journal(journal_path, runtime)
        return _read_verified_journal(journal_path)
    except BaseException:
        _release_publish_locks(
            attempt_id,
            keys=lock_keys,
            figures_dir=figures_dir,
            session_id=str(session_id),
        )
        if created_transaction_dir:
            _remove_transaction_dir(transaction_dir)
        raise


def refreeze_pending_figure_attempt(
    pending_transaction: Mapping[str, Any],
    *,
    attempt_dir: Path,
    staged_jpg: Path,
    staged_pdf: Path,
    final_jpg: Path,
    final_pdf: Path,
    published_record: Any,
) -> dict[str, Any]:
    """Authenticate and freeze the final reviewed bytes of a pending attempt.

    Revisions replace the JPG/PDF inside the already-journaled attempt.  Only
    the process that owns the original authenticated transaction and all of its
    locks may advance the signed digests to those final candidate bytes.
    """
    figures_dir, attempt_id, figure_name = _validate_registration_paths(
        attempt_dir=attempt_dir,
        staged_jpg=staged_jpg,
        staged_pdf=staged_pdf,
        final_jpg=final_jpg,
        final_pdf=final_pdf,
        published_record=published_record,
    )
    _transaction_dir, journal_path = _journal_paths(figures_dir, attempt_id)
    if _absolute(Path(str(pending_transaction.get("journal_path") or ""))) != journal_path:
        raise ValueError("pending publication transaction journal identity changed")
    journal = _read_verified_journal(journal_path)
    for field, expected in {
        "session_id": journal.get("session_id"),
        "attempt_id": attempt_id,
        "figure_name": figure_name,
        "new_record_id": str(_value(published_record, "id", "") or ""),
        "journal_hmac": journal.get("journal_hmac"),
    }.items():
        if str(pending_transaction.get(field) or "") != str(expected or ""):
            raise ValueError(
                f"pending publication transaction {field} identity changed"
            )
    if (
        str(journal.get("attempt_id") or "") != attempt_id
        or str(journal.get("figure_name") or "") != figure_name
        or str(journal.get("new_record_id") or "")
        != str(_value(published_record, "id", "") or "")
        or str(journal.get("status") or "") != "REVIEW_PENDING"
        or list(journal.get("completed_steps") or [])
        or str(journal.get("old_record_id") or "")
    ):
        raise ValueError("publication transaction is not an untouched pending attempt")
    if (
        int(journal.get("owner_pid") or -1) != os.getpid()
        or str(journal.get("owner_process_token") or "") != _PROCESS_TOKEN
    ):
        raise RuntimeError("publication transaction is not owned by this producer")
    if any(
        _authenticated_marker_exists(journal_path.parent, marker, journal)
        for marker in _TERMINAL_MARKERS
    ):
        raise RuntimeError("publication transaction is already terminal")

    lock_keys = _publish_lock_keys(figures_dir, Path(final_jpg), published_record)
    if list(journal.get("lock_keys") or []) != lock_keys:
        raise ValueError("publication transaction lock identity changed")
    lock_owner = _lock_owner(
        figures_dir,
        str(journal.get("session_id") or ""),
        attempt_id,
        lock_keys,
    )
    if any(
        key not in _HELD_LOCKS or _HELD_LOCKS[key][0] != lock_owner
        for key in lock_keys
    ):
        raise RuntimeError("publication transaction locks are not held by this producer")

    figures_fd: int | None = None
    attempts_fd: int | None = None
    attempt_fd: int | None = None
    jpg_fd: int | None = None
    pdf_fd: int | None = None
    try:
        figures_fd, attempts_fd, attempt_fd = _open_attempt_directory(
            Path(attempt_dir), create=False
        )
        _require_attempt_directory_binding(
            figures_fd, attempts_fd, attempt_fd, attempt_id
        )
        jpg_name = f"{figure_name}.jpg"
        pdf_name = f"{figure_name}.pdf"
        jpg_fd, _jpg_metadata, render_digest = _open_stable_staged_file(
            attempt_fd, jpg_name, label="reviewed staged publication JPG"
        )
        pdf_fd, _pdf_metadata, pdf_digest = _open_stable_staged_file(
            attempt_fd, pdf_name, label="reviewed staged publication PDF"
        )
        _require_regular_entry_binding(
            attempt_fd, jpg_name, jpg_fd, label="reviewed staged publication JPG"
        )
        _require_regular_entry_binding(
            attempt_fd, pdf_name, pdf_fd, label="reviewed staged publication PDF"
        )
        journal.update({
            "owner_pid": os.getpid(),
            "owner_process_token": _PROCESS_TOKEN,
            "new_record": _snapshot_record(published_record),
            "new_render_sha256": render_digest,
            "new_pdf_sha256": pdf_digest,
        })
        _persist_journal(journal_path, journal)
        _require_attempt_directory_binding(
            figures_fd, attempts_fd, attempt_fd, attempt_id
        )
        _require_regular_entry_binding(
            attempt_fd, jpg_name, jpg_fd, label="reviewed staged publication JPG"
        )
        _require_regular_entry_binding(
            attempt_fd, pdf_name, pdf_fd, label="reviewed staged publication PDF"
        )
        if (
            _digest_open_regular_file(
                jpg_fd, label="reviewed staged publication JPG"
            )
            != render_digest
            or _digest_open_regular_file(
                pdf_fd, label="reviewed staged publication PDF"
            )
            != pdf_digest
        ):
            raise ValueError("reviewed publication pair changed while it was frozen")
        verified = _read_verified_journal(journal_path)
        if (
            str(verified.get("new_render_sha256") or "") != render_digest
            or str(verified.get("new_pdf_sha256") or "") != pdf_digest
        ):
            raise ValueError("reviewed publication pair freeze was not durable")
        return verified
    finally:
        if pdf_fd is not None:
            os.close(pdf_fd)
        if jpg_fd is not None:
            os.close(jpg_fd)
        if attempt_fd is not None:
            os.close(attempt_fd)
        if attempts_fd is not None:
            os.close(attempts_fd)
        if figures_fd is not None:
            os.close(figures_fd)


def begin_figure_publish_transaction(
    *,
    attempt_dir: Path,
    staged_jpg: Path,
    staged_pdf: Path,
    final_jpg: Path,
    final_pdf: Path,
    published_record: Any,
    archived_record: Any = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Durably replace a validated pair, leaving its record uncommitted.

    The returned transaction remains recoverable until
    :func:`commit_figure_publish_transaction` writes its marker.  A normal I/O
    exception is rolled back here.  A process death or injected ``BaseException``
    is intentionally left to :func:`recover_figure_publish_transactions`.
    """
    try:
        figures_dir, attempt_id, safe_name = _validate_registration_paths(
            attempt_dir=attempt_dir,
            staged_jpg=staged_jpg,
            staged_pdf=staged_pdf,
            final_jpg=final_jpg,
            final_pdf=final_pdf,
            published_record=published_record,
        )
        transaction_dir, journal_path = _journal_paths(figures_dir, attempt_id)
        journal = _read_verified_journal(journal_path)
    except Exception as exc:
        figures_dir = _absolute(Path(final_jpg).parent)
        transaction_dir = figures_dir / TRANSACTION_ROOT / _safe_component(
            _value(published_record, "render_attempt_id", ""), "attempt"
        )
        _quarantine_transaction_dir(transaction_dir)
        failed_attempt_id = str(
            _value(published_record, "render_attempt_id", "") or ""
        )
        _release_publish_locks(
            failed_attempt_id,
            figures_dir=figures_dir,
        )
        return None, f"Could not verify publication transaction journal: {exc}"

    if (
        str(journal.get("new_record_id") or "")
        != str(_value(published_record, "id", "") or "")
        or str(journal.get("attempt_id") or "") != attempt_id
        or str(journal.get("figure_name") or "") != safe_name
    ):
        _quarantine_transaction_dir(transaction_dir)
        _release_publish_locks(
            attempt_id,
            keys=journal.get("lock_keys") or [],
            figures_dir=figures_dir,
            session_id=str(journal.get("session_id") or ""),
        )
        return None, "Could not publish a mismatched figure transaction"
    if any(
        _authenticated_marker_exists(transaction_dir, marker, journal)
        for marker in _TERMINAL_MARKERS
    ):
        _release_publish_locks(
            attempt_id,
            keys=journal.get("lock_keys") or [],
            figures_dir=figures_dir,
            session_id=str(journal.get("session_id") or ""),
        )
        return None, "Could not publish a terminalized figure attempt"

    lock_keys = _publish_lock_keys(figures_dir, Path(final_jpg), published_record)
    lock_error = _acquire_publish_locks(
        figures_dir,
        attempt_id,
        lock_keys,
        session_id=str(journal.get("session_id") or ""),
    )
    if lock_error:
        return None, lock_error

    attempt_figures_fd: int | None = None
    attempts_fd: int | None = None
    attempt_fd: int | None = None
    staged_jpg_fd: int | None = None
    staged_pdf_fd: int | None = None
    try:
        attempt_figures_fd, attempts_fd, attempt_fd = _open_attempt_directory(
            Path(attempt_dir), create=False
        )
        _require_attempt_directory_binding(
            attempt_figures_fd, attempts_fd, attempt_fd, attempt_id
        )
        expected_render_sha256 = str(journal.get("new_render_sha256") or "")
        expected_pdf_sha256 = str(journal.get("new_pdf_sha256") or "")
        if not expected_render_sha256 or not expected_pdf_sha256:
            raise ValueError("publication journal has no frozen pair digest")
        staged_jpg_fd, staged_jpg_metadata = _open_frozen_staged_file(
            attempt_fd,
            f"{safe_name}.jpg",
            expected_render_sha256,
            label="staged publication JPG",
        )
        staged_pdf_fd, staged_pdf_metadata = _open_frozen_staged_file(
            attempt_fd,
            f"{safe_name}.pdf",
            expected_pdf_sha256,
            label="staged publication PDF",
        )
        had_final_jpg = _regular_child_exists(
            attempt_figures_fd,
            f"{safe_name}.jpg",
            label="publication JPG",
        )
        had_final_pdf = _regular_child_exists(
            attempt_figures_fd,
            f"{safe_name}.pdf",
            label="publication PDF",
        )
        journal.update({
            "owner_pid": os.getpid(),
            "owner_process_token": _PROCESS_TOKEN,
            "status": "PREPARED",
            "completed_steps": [],
            "had_final_jpg": had_final_jpg,
            "had_final_pdf": had_final_pdf,
            "new_record_id": str(_value(published_record, "id", "")),
            "new_record": _snapshot_record(published_record),
            "old_record_id": str(_value(archived_record, "id", "") or ""),
            "old_record_before": _snapshot_record(archived_record),
        })
        _persist_journal(journal_path, journal)
        journal = _read_verified_journal(journal_path)
        backup_jpg = Path(journal["backup_jpg"])
        backup_pdf = Path(journal["backup_pdf"])
        archive_jpg = _path_from_journal(journal, "archive_jpg")
        archive_pdf = _path_from_journal(journal, "archive_pdf")

        _require_attempt_directory_binding(
            attempt_figures_fd, attempts_fd, attempt_fd, attempt_id
        )
        if had_final_jpg:
            _copy_durable(Path(final_jpg), backup_jpg)
            _complete_step(journal_path, journal, "backup_jpg")
        if had_final_pdf:
            _copy_durable(Path(final_pdf), backup_pdf)
            _complete_step(journal_path, journal, "backup_pdf")

        if archived_record is not None:
            if had_final_jpg and archive_jpg is not None:
                _copy_durable(backup_jpg, archive_jpg)
                _complete_step(journal_path, journal, "archive_jpg")
            if had_final_pdf and archive_pdf is not None:
                _copy_durable(backup_pdf, archive_pdf)
                _complete_step(journal_path, journal, "archive_pdf")

        _complete_step(
            journal_path, journal, "replace_jpg_started", status="JPG_REPLACE_INTENT"
        )
        _require_attempt_directory_binding(
            attempt_figures_fd, attempts_fd, attempt_fd, attempt_id
        )
        if (
            _digest_open_regular_file(
                staged_jpg_fd, label="staged publication JPG"
            )
            != expected_render_sha256
        ):
            raise ValueError("staged publication JPG changed before publish")
        _replace_staged_entry(
            attempt_fd,
            attempt_figures_fd,
            f"{safe_name}.jpg",
            f"{safe_name}.jpg",
        )
        _verify_published_entry(
            attempt_figures_fd,
            f"{safe_name}.jpg",
            staged_jpg_fd,
            staged_jpg_metadata,
            expected_render_sha256,
            label="published JPG",
        )
        os.fsync(staged_jpg_fd)
        os.fsync(attempt_figures_fd)
        _require_attempt_directory_binding(
            attempt_figures_fd, attempts_fd, attempt_fd, attempt_id
        )
        _complete_step(journal_path, journal, "replace_jpg", status="JPG_REPLACED")

        _complete_step(
            journal_path, journal, "replace_pdf_started", status="PDF_REPLACE_INTENT"
        )
        _require_attempt_directory_binding(
            attempt_figures_fd, attempts_fd, attempt_fd, attempt_id
        )
        if (
            _digest_open_regular_file(
                staged_pdf_fd, label="staged publication PDF"
            )
            != expected_pdf_sha256
        ):
            raise ValueError("staged publication PDF changed before publish")
        _replace_staged_entry(
            attempt_fd,
            attempt_figures_fd,
            f"{safe_name}.pdf",
            f"{safe_name}.pdf",
        )
        _verify_published_entry(
            attempt_figures_fd,
            f"{safe_name}.pdf",
            staged_pdf_fd,
            staged_pdf_metadata,
            expected_pdf_sha256,
            label="published PDF",
        )
        os.fsync(staged_pdf_fd)
        os.fsync(attempt_figures_fd)
        _require_attempt_directory_binding(
            attempt_figures_fd, attempts_fd, attempt_fd, attempt_id
        )
        _complete_step(journal_path, journal, "replace_pdf", status="FILES_REPLACED")
        journal = _read_verified_journal(journal_path)
        _verify_published_entry(
            attempt_figures_fd,
            f"{safe_name}.jpg",
            staged_jpg_fd,
            staged_jpg_metadata,
            expected_render_sha256,
            label="published JPG",
        )
        _verify_published_entry(
            attempt_figures_fd,
            f"{safe_name}.pdf",
            staged_pdf_fd,
            staged_pdf_metadata,
            expected_pdf_sha256,
            label="published PDF",
        )
        _require_attempt_directory_binding(
            attempt_figures_fd, attempts_fd, attempt_fd, attempt_id
        )
    except Exception as exc:
        rollback_figure_publish_transaction(
            journal_path, state=None, producer_status="failed"
        )
        return None, f"Could not publish validated figure render: {exc}"
    finally:
        if staged_pdf_fd is not None:
            os.close(staged_pdf_fd)
        if staged_jpg_fd is not None:
            os.close(staged_jpg_fd)
        if attempt_fd is not None:
            os.close(attempt_fd)
        if attempts_fd is not None:
            os.close(attempts_fd)
        if attempt_figures_fd is not None:
            os.close(attempt_figures_fd)

    return journal, None


def apply_publish_record_paths(
    transaction: dict[str, Any],
    published_record: Any,
    archived_record: Any = None,
) -> None:
    """Apply the file half of a transaction to its in-memory records."""
    verified = _read_verified_journal(Path(transaction["journal_path"]))
    if str(_value(published_record, "id", "")) != str(
        verified.get("new_record_id") or ""
    ):
        raise ValueError("published record does not match transaction identity")
    _set(published_record, "png_path", verified["final_jpg"])
    _set(published_record, "pdf_path", verified["final_pdf"])
    if archived_record is not None:
        if str(_value(archived_record, "id", "")) != str(
            verified.get("old_record_id") or ""
        ):
            raise ValueError("archived record does not match transaction identity")
        archive_jpg = str(verified.get("archive_jpg") or "")
        archive_pdf = str(verified.get("archive_pdf") or "")
        if archive_jpg:
            _set(archived_record, "png_path", archive_jpg)
        if archive_pdf:
            _set(archived_record, "pdf_path", archive_pdf)


def mark_publish_record_applied(
    transaction: dict[str, Any],
    *,
    published_record: Any,
    archived_record: Any = None,
) -> None:
    """Journal the exact record state that the session write must persist."""
    journal_path = Path(transaction["journal_path"])
    verified = _read_verified_journal(journal_path)
    if str(_value(published_record, "id", "")) != str(
        verified.get("new_record_id") or ""
    ):
        raise ValueError("published record does not match transaction identity")
    if archived_record is not None and str(_value(archived_record, "id", "")) != str(
        verified.get("old_record_id") or ""
    ):
        raise ValueError("archived record does not match transaction identity")
    verified["new_record"] = _snapshot_record(published_record)
    verified["old_record_after"] = _snapshot_record(archived_record)
    _complete_step(
        journal_path,
        verified,
        "record_applied",
        status="RECORD_APPLIED",
    )
    transaction.clear()
    transaction.update(_read_verified_journal(journal_path))


def commit_figure_publish_transaction(transaction: dict[str, Any]) -> None:
    """Commit after the record boundary has been durably persisted."""
    journal_path = Path(transaction["journal_path"])
    verified = _read_verified_journal(journal_path)
    verified["status"] = "RECORD_DURABLE"
    _persist_journal(journal_path, verified)
    _write_marker(journal_path.parent / COMMIT_MARKER)
    verified["status"] = "COMMITTED"
    try:
        _persist_journal(journal_path, verified)
    except Exception:
        # The fsynced marker is the commit point. Failure to rewrite the
        # descriptive status after it must never make the caller roll back a
        # transaction that recovery is required to forward-complete.
        logger.exception(
            "Publication transaction committed but final journal update failed: %s",
            journal_path,
        )
    attempt_clean = _cleanup_attempt(verified)
    _release_journal_locks(verified)
    if attempt_clean:
        _remove_transaction_dir(journal_path.parent)


def _path_from_journal(journal: Mapping[str, Any], key: str) -> Path | None:
    value = str(journal.get(key) or "")
    return Path(value) if value else None


def _release_journal_locks(journal: Mapping[str, Any]) -> None:
    final_jpg = _path_from_journal(journal, "final_jpg")
    if final_jpg is None:
        return
    _release_publish_locks(
        str(journal.get("attempt_id") or ""),
        keys=journal.get("lock_keys") or [],
        figures_dir=final_jpg.parent,
        session_id=str(journal.get("session_id") or ""),
    )


def _cleanup_attempt(journal: Mapping[str, Any]) -> bool:
    attempt_dir = _path_from_journal(journal, "attempt_dir")
    if attempt_dir is None:
        return True
    attempts_root = attempt_dir.parent
    if attempts_root.name != ".attempts":
        return False
    try:
        root_fd, parents = _open_hidden_root(attempts_root, create=False)
    except ValueError:
        # A missing attempts root means cleanup already completed. A present
        # non-directory/symlink root is untrusted and must fail closed.
        try:
            os.lstat(attempts_root)
        except FileNotFoundError:
            return True
        return False
    try:
        try:
            os.stat(attempt_dir.name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            return True
    finally:
        os.close(root_fd)
        for descriptor in reversed(parents):
            os.close(descriptor)
    if _remove_bounded_root_entry(attempt_dir, attempts_root):
        try:
            root_fd, parents = _open_hidden_root(attempts_root, create=False)
        except ValueError:
            return True
        try:
            if not os.listdir(root_fd):
                os.close(root_fd)
                root_fd = -1
                os.rmdir(attempts_root.name, dir_fd=parents[-1])
                os.fsync(parents[-1])
        except OSError:
            pass
        finally:
            if root_fd >= 0:
                os.close(root_fd)
            for descriptor in reversed(parents):
                os.close(descriptor)
        return True
    else:
        logger.warning("Could not clean publication attempt %s", attempt_dir)
        return False


def _remove_transaction_dir(transaction_dir: Path) -> bool:
    if _remove_bounded_root_entry(transaction_dir, transaction_dir.parent):
        return True
    else:
        logger.warning("Could not garbage-collect figure transaction %s", transaction_dir)
        return False


def _remove_revision_archive_dir(archive_dir: Path) -> bool:
    """Remove one derived revision attempt without following hidden roots."""
    archive_dir = _absolute(archive_dir)
    record_dir = archive_dir.parent
    revision_root = record_dir.parent
    figures_dir = revision_root.parent
    if (
        revision_root.name != ".revisions"
        or len(record_dir.name) != 64
        or any(char not in "0123456789abcdef" for char in record_dir.name)
        or archive_dir.name != _safe_component(archive_dir.name, "attempt")
    ):
        return False
    figures_fd: int | None = None
    revision_fd: int | None = None
    record_fd: int | None = None
    try:
        figures_fd = _open_directory_nofollow(figures_dir)
        revision_fd = _open_child_directory_nofollow(
            figures_fd, ".revisions", create=False
        )
        record_fd = _open_child_directory_nofollow(
            revision_fd, record_dir.name, create=False
        )
        metadata = os.stat(
            archive_dir.name, dir_fd=record_fd, follow_symlinks=False
        )
        if not stat.S_ISDIR(metadata.st_mode):
            return False
        shutil.rmtree(archive_dir.name, dir_fd=record_fd)
        os.fsync(record_fd)
        if not os.listdir(record_fd):
            os.close(record_fd)
            record_fd = None
            os.rmdir(record_dir.name, dir_fd=revision_fd)
            os.fsync(revision_fd)
        if not os.listdir(revision_fd):
            os.close(revision_fd)
            revision_fd = None
            os.rmdir(".revisions", dir_fd=figures_fd)
            os.fsync(figures_fd)
        return True
    except (OSError, ValueError):
        return False
    finally:
        if record_fd is not None:
            os.close(record_fd)
        if revision_fd is not None:
            os.close(revision_fd)
        if figures_fd is not None:
            os.close(figures_fd)


def _open_hidden_root(
    root: Path,
    *,
    create: bool = False,
) -> tuple[int, list[int]]:
    """Open one recognized hidden root and retain its trusted parent FDs."""
    root = _absolute(root)
    if root.name == ".attempts":
        figures_fd = _open_directory_nofollow(root.parent)
        try:
            root_fd = _open_child_directory_nofollow(
                figures_fd, root.name, create=create
            )
        except BaseException:
            os.close(figures_fd)
            raise
        return root_fd, [figures_fd]
    if root.name == TRANSACTION_ROOT:
        figures_fd, root_fd = _open_transaction_root(root.parent, create=create)
        return root_fd, [figures_fd]
    if root.name in {".locks", ".quarantine"} and root.parent.name == TRANSACTION_ROOT:
        figures_fd, transaction_fd = _open_transaction_root(
            root.parent.parent, create=create
        )
        try:
            root_fd = _open_child_directory_nofollow(
                transaction_fd, root.name, create=create
            )
        except BaseException:
            os.close(transaction_fd)
            os.close(figures_fd)
            raise
        return root_fd, [figures_fd, transaction_fd]
    raise ValueError("path is not a recognized hidden publication root")


def _remove_bounded_root_entry(path: Path, root: Path) -> bool:
    """Remove one direct child without following a symlink outside *root*."""
    path = _absolute(Path(path))
    root = _absolute(Path(root))
    if path.parent != root or path.name in {"", ".", ".."}:
        return False
    root_fd: int | None = None
    parents: list[int] = []
    try:
        root_fd, parents = _open_hidden_root(root, create=False)
        metadata = os.stat(path.name, dir_fd=root_fd, follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode) or stat.S_ISREG(metadata.st_mode):
            os.unlink(path.name, dir_fd=root_fd)
        elif stat.S_ISDIR(metadata.st_mode):
            shutil.rmtree(path.name, dir_fd=root_fd)
        else:
            return False
        os.fsync(root_fd)
        return True
    except (OSError, ValueError):
        return False
    finally:
        if root_fd is not None:
            os.close(root_fd)
        for descriptor in reversed(parents):
            os.close(descriptor)


def garbage_collect_figure_transaction_roots(
    figures_dir: Path,
    *,
    budget: int | None = None,
) -> int:
    """Bound maintenance strictly to hidden publication transaction roots."""
    figures_dir = _absolute(Path(figures_dir))
    transaction_root = figures_dir / TRANSACTION_ROOT
    now = time.time()
    remaining = _MAX_GC_ENTRIES if budget is None else max(0, int(budget))

    try:
        transaction_fd, transaction_parents = _open_hidden_root(
            transaction_root, create=False
        )
    except ValueError:
        return remaining

    quarantine_root = transaction_root / ".quarantine"
    try:
        quarantine_fd, quarantine_parents = _open_hidden_root(
            quarantine_root, create=False
        )
    except ValueError:
        quarantine_fd = None
        quarantine_parents = []
    if quarantine_fd is not None:
        try:
            quarantine_names = sorted(os.listdir(quarantine_fd))
        finally:
            os.close(quarantine_fd)
            for descriptor in reversed(quarantine_parents):
                os.close(descriptor)
        for name in quarantine_names:
            if remaining <= 0:
                break
            entry = quarantine_root / name
            try:
                stale = now - entry.lstat().st_mtime >= _QUARANTINE_RETENTION_SECONDS
            except OSError:
                continue
            if stale and _remove_bounded_root_entry(entry, quarantine_root):
                remaining -= 1

    attempts_root = figures_dir / ".attempts"
    active_attempts: set[str] = set()
    for name in os.listdir(transaction_fd):
        if name.startswith("."):
            continue
        try:
            child_fd = _open_child_directory_nofollow(
                transaction_fd, name, create=False
            )
        except ValueError:
            continue
        try:
            metadata = os.stat(
                JOURNAL_FILENAME, dir_fd=child_fd, follow_symlinks=False
            )
            if stat.S_ISREG(metadata.st_mode):
                active_attempts.add(name)
        except OSError:
            pass
        finally:
            os.close(child_fd)
    try:
        attempts_fd, attempts_parents = _open_hidden_root(
            attempts_root, create=False
        )
    except ValueError:
        attempts_fd = None
        attempts_parents = []
    if attempts_fd is not None:
        try:
            attempt_names = sorted(os.listdir(attempts_fd))
        finally:
            os.close(attempts_fd)
            for descriptor in reversed(attempts_parents):
                os.close(descriptor)
        for name in attempt_names:
            if remaining <= 0:
                break
            if name in active_attempts:
                continue
            entry = attempts_root / name
            try:
                stale = now - entry.lstat().st_mtime >= _STALE_ENTRY_SECONDS
            except OSError:
                continue
            if stale and _remove_bounded_root_entry(entry, attempts_root):
                remaining -= 1

    lock_root = transaction_root / ".locks"
    try:
        lock_fd, lock_parents = _open_hidden_root(lock_root, create=False)
    except ValueError:
        lock_fd = None
        lock_parents = []
    if lock_fd is not None:
        try:
            import fcntl
        except ImportError:  # pragma: no cover
            os.close(lock_fd)
            for descriptor in reversed(lock_parents):
                os.close(descriptor)
            return remaining
        for name in sorted(os.listdir(lock_fd)):
            if remaining <= 0:
                break
            if Path(name).suffix != ".lock":
                continue
            try:
                metadata = os.stat(name, dir_fd=lock_fd, follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode):
                    continue
                if now - metadata.st_mtime < _STALE_ENTRY_SECONDS:
                    continue
                descriptor = os.open(
                    name,
                    os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=lock_fd,
                )
                with os.fdopen(descriptor, "a+b") as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    os.unlink(name, dir_fd=lock_fd)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                os.fsync(lock_fd)
                remaining -= 1
            except (BlockingIOError, OSError):
                continue
        os.close(lock_fd)
        for descriptor in reversed(lock_parents):
            os.close(descriptor)
    os.close(transaction_fd)
    for descriptor in reversed(transaction_parents):
        os.close(descriptor)
    return remaining


def _restore_file(
    *,
    final_path: Path | None,
    backup_path: Path | None,
    was_replaced: bool,
    had_final: bool,
    new_digest: str | None,
) -> None:
    if final_path is None or not was_replaced:
        return
    if had_final and backup_path is not None:
        temporary = final_path.with_name(f".{final_path.name}.rollback-{uuid.uuid4().hex}")
        _copy_durable(backup_path, temporary)
        os.replace(temporary, final_path)
        _fsync_file(final_path)
        _fsync_dir(final_path.parent)
        return
    # Do not unlink bytes that no longer belong to this transaction.  A later
    # producer may have reused the path before recovery was invoked.
    if new_digest and _sha256(final_path) == new_digest:
        parent_fd, ancestors = _open_known_file_parent(final_path, create=False)
        try:
            metadata = os.stat(
                final_path.name, dir_fd=parent_fd, follow_symlinks=False
            )
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("publication rollback target is not regular")
            os.unlink(final_path.name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            _close_parent_chain(parent_fd, ancestors)


def _restore_record(record: Any, snapshot: Mapping[str, Any]) -> bool:
    mutated = False
    for key, value in snapshot.items():
        if key not in _TRANSACTION_RECORD_FIELDS:
            continue
        try:
            if _value(record, key) != value:
                _set(record, key, value)
                mutated = True
        except (AttributeError, TypeError, ValueError):
            logger.debug("Could not restore figure record field %s", key)
    return mutated


def _abandon_new_record(
    state: Any,
    journal: Mapping[str, Any],
    producer_status: str,
) -> bool:
    if state is None:
        return False
    records = list(_value(state, "publication_figures", []) or [])
    record_id = str(journal.get("new_record_id") or "")
    record = _find_record(records, record_id)
    mutated = False
    if record is None and journal.get("new_record"):
        try:
            from backend.models.schemas import PublicationFigure

            snapshot = dict(journal["new_record"])
            snapshot["png_base64"] = None
            record = PublicationFigure.model_validate(snapshot)
            records.append(record)
            _set(state, "publication_figures", records)
            mutated = True
        except Exception:
            logger.exception("Could not reconstruct abandoned figure record %s", record_id)
            return mutated
    if record is not None:
        for key, value in {
            "producer_status": producer_status,
            "png_base64": None,
            "png_path": "",
            "pdf_path": "",
            "render_sha256": None,
            "preview_sha256": None,
            "pdf_sha256": None,
            "is_published_head": False,
        }.items():
            if _value(record, key) != value:
                _set(record, key, value)
                mutated = True
    return mutated


def _restore_abandoned_records(
    state: Any,
    journal: Mapping[str, Any],
    producer_status: str,
) -> bool:
    if state is None:
        return False
    records = list(_value(state, "publication_figures", []) or [])
    old_id = str(journal.get("old_record_id") or "")
    old_record = _find_record(records, old_id) if old_id else None
    mutated = False
    if old_record is not None and journal.get("old_record_before"):
        mutated = _restore_record(
            old_record, journal["old_record_before"]
        ) or mutated
    return _abandon_new_record(state, journal, producer_status) or mutated


def rollback_figure_publish_transaction(
    journal_path: Path,
    *,
    state: Any = None,
    producer_status: str = "abandoned",
) -> bool:
    """Rollback one non-committed journal and terminalize its producer record."""
    try:
        journal = _read_verified_journal(Path(journal_path))
    except Exception:
        logger.exception("Could not read publication transaction %s", journal_path)
        transaction_dir = _absolute(Path(journal_path).parent)
        _release_publish_locks(
            transaction_dir.name,
            figures_dir=transaction_dir.parent.parent,
        )
        _quarantine_transaction_dir(transaction_dir)
        return False
    if _authenticated_marker_exists(
        Path(journal_path).parent, COMMIT_MARKER, journal
    ):
        logger.error("Refusing to roll back committed publication transaction %s", journal_path)
        _release_journal_locks(journal)
        return False
    completed = set(journal.get("completed_steps") or [])
    try:
        _restore_file(
            final_path=_path_from_journal(journal, "final_pdf"),
            backup_path=_path_from_journal(journal, "backup_pdf"),
            was_replaced=bool(
                {"replace_pdf_started", "replace_pdf"} & completed
            ),
            had_final=bool(journal.get("had_final_pdf")),
            new_digest=str(journal.get("new_pdf_sha256") or "") or None,
        )
        _restore_file(
            final_path=_path_from_journal(journal, "final_jpg"),
            backup_path=_path_from_journal(journal, "backup_jpg"),
            was_replaced=bool(
                {"replace_jpg_started", "replace_jpg"} & completed
            ),
            had_final=bool(journal.get("had_final_jpg")),
            new_digest=str(journal.get("new_render_sha256") or "") or None,
        )
    except Exception:
        logger.exception("Could not roll back publication transaction %s", journal_path)
        _release_journal_locks(journal)
        return False

    mutated = False
    if state is not None:
        mutated = _restore_abandoned_records(
            state, journal, producer_status
        ) or mutated

    try:
        archive_jpg = _path_from_journal(journal, "archive_jpg")
        archive_dir = archive_jpg.parent if archive_jpg is not None else None
        if archive_dir is not None:
            _remove_revision_archive_dir(archive_dir)
        _cleanup_attempt(journal)
        journal["status"] = producer_status.upper()
        _persist_journal(journal_path, journal)
        marker = FAILURE_MARKER if producer_status == "failed" else ABANDON_MARKER
        _unlink_transaction_regular_file_if_present(
            journal_path.parent, ROLLBACK_DURABLE_MARKER
        )
        _write_marker(journal_path.parent / marker)
    except Exception:
        logger.exception(
            "Could not persist terminal publication rollback %s", journal_path
        )
    finally:
        _release_journal_locks(journal)
    return mutated


def _record_matches_committed_files(record: Any, journal: Mapping[str, Any]) -> bool:
    if record is None or str(_value(record, "producer_status", "")) != "published":
        return False
    return bool(
        str(_value(record, "png_path", "")) == str(journal.get("final_jpg") or "")
        and str(_value(record, "pdf_path", "")) == str(journal.get("final_pdf") or "")
        and _sha256(Path(str(journal.get("final_jpg") or "")))
        == str(journal.get("new_render_sha256") or "")
        and _sha256(Path(str(journal.get("final_pdf") or "")))
        == str(journal.get("new_pdf_sha256") or "")
    )


def _apply_committed_records(state: Any, journal: Mapping[str, Any]) -> bool:
    if state is None:
        return False
    records = list(_value(state, "publication_figures", []) or [])
    new_id = str(journal.get("new_record_id") or "")
    new_record = _find_record(records, new_id)
    mutated = False
    if new_record is None and journal.get("new_record"):
        try:
            from backend.models.schemas import PublicationFigure

            new_record = PublicationFigure.model_validate(journal["new_record"])
            records.append(new_record)
            _set(state, "publication_figures", records)
            mutated = True
        except Exception:
            logger.exception("Could not reconstruct committed figure record %s", new_id)
            return mutated
    if new_record is not None and journal.get("new_record"):
        mutated = _restore_record(new_record, journal["new_record"]) or mutated
    old_id = str(journal.get("old_record_id") or "")
    old_record = _find_record(records, old_id) if old_id else None
    if old_record is not None and journal.get("old_record_after"):
        mutated = _restore_record(old_record, journal["old_record_after"]) or mutated
    return mutated


def recover_figure_publish_transactions(
    state: Any,
    figures_dir: Path,
    *,
    force: bool = False,
    applied_transaction_ids: set[str] | None = None,
) -> bool:
    """Recover every publication transaction visible to this workspace.

    A commit marker is authoritative.  Without one, a record that is already
    durable and hash-bound to both final files can be forward-committed; every
    other state is rolled back and terminalized as ``abandoned``.
    """
    figures_dir = _absolute(Path(figures_dir))
    root = figures_dir / TRANSACTION_ROOT
    try:
        root_fd, root_parents = _open_hidden_root(root, create=False)
    except ValueError:
        return False
    try:
        transaction_names = sorted(os.listdir(root_fd))
    finally:
        os.close(root_fd)
        for descriptor in reversed(root_parents):
            os.close(descriptor)
    mutated = False
    remaining_budget = _MAX_GC_ENTRIES
    for transaction_name in transaction_names:
        if transaction_name.startswith("."):
            continue
        if remaining_budget <= 0:
            break
        remaining_budget -= 1
        transaction_dir = root / transaction_name
        journal_path = transaction_dir / JOURNAL_FILENAME
        try:
            _figures_fd, _root_fd, transaction_fd = _open_transaction_directory(
                transaction_dir
            )
            try:
                journal_metadata = os.stat(
                    JOURNAL_FILENAME,
                    dir_fd=transaction_fd,
                    follow_symlinks=False,
                )
                has_regular_journal = stat.S_ISREG(journal_metadata.st_mode)
            except OSError:
                has_regular_journal = False
            finally:
                os.close(transaction_fd)
                os.close(_root_fd)
                os.close(_figures_fd)
        except ValueError:
            has_regular_journal = False
        if not has_regular_journal:
            if _transaction_creation_is_live(figures_dir, transaction_name):
                continue
            _quarantine_transaction_dir(transaction_dir)
            continue
        try:
            journal = _read_verified_journal(journal_path)
        except Exception:
            logger.exception("Quarantining unreadable publication journal %s", journal_path)
            _quarantine_transaction_dir(transaction_dir)
            continue
        journal_session_id = str(journal.get("session_id") or "")
        state_session_id = str(_value(state, "id", "") or "") if state else ""
        if journal_session_id and state_session_id and journal_session_id != state_session_id:
            logger.error(
                "Quarantining publication transaction for mismatched session %s",
                journal_session_id,
            )
            _quarantine_transaction_dir(transaction_dir)
            continue
        if (
            not force
            and str(journal.get("owner_process_token") or "") == _PROCESS_TOKEN
            and not any(
                _authenticated_marker_exists(transaction_dir, marker, journal)
                for marker in _TERMINAL_MARKERS
            )
        ):
            # Authenticated status/report requests can run while this process is
            # awaiting a reviewer.  Crash recovery must not mistake that live
            # hidden attempt for debris. A restarted process has a new token.
            continue
        abandoned = _authenticated_marker_exists(
            transaction_dir, ABANDON_MARKER, journal
        )
        failed = _authenticated_marker_exists(
            transaction_dir, FAILURE_MARKER, journal
        )
        committed = _authenticated_marker_exists(
            transaction_dir, COMMIT_MARKER, journal
        )
        if abandoned or failed:
            mutated = _restore_abandoned_records(
                state,
                journal,
                "failed" if failed else "abandoned",
            ) or mutated
            rollback_durable = _authenticated_marker_exists(
                transaction_dir, ROLLBACK_DURABLE_MARKER, journal
            )
            if not rollback_durable:
                # The journal is terminal but session.json may still contain
                # the pre-recovery pending/published record. Force the caller's
                # durable state rewrite even when its in-memory object already
                # happens to match the rollback.
                mutated = True
                if applied_transaction_ids is not None:
                    applied_transaction_ids.add(
                        str(journal.get("attempt_id") or transaction_dir.name)
                    )
            attempt_clean = _cleanup_attempt(journal)
            if rollback_durable and attempt_clean:
                _remove_transaction_dir(transaction_dir)
            continue
        if committed:
            record_mutated = _apply_committed_records(state, journal)
            mutated = record_mutated or mutated
            attempt_clean = _cleanup_attempt(journal)
            if record_mutated:
                if applied_transaction_ids is not None:
                    applied_transaction_ids.add(
                        str(journal.get("attempt_id") or transaction_dir.name)
                    )
            elif attempt_clean:
                _remove_transaction_dir(transaction_dir)
            continue

        attempt_id = str(journal.get("attempt_id") or "")
        lock_error = _acquire_publish_locks(
            figures_dir,
            attempt_id,
            _publish_lock_keys(
                figures_dir, Path(journal["final_jpg"]), journal["new_record"]
            ),
            session_id=str(journal.get("session_id") or ""),
        )
        if lock_error:
            # Another process still owns the public path or logical lineage.
            # Its non-terminal journal is live, not crash debris.
            continue

        records = list(_value(state, "publication_figures", []) or []) if state else []
        record = _find_record(records, str(journal.get("new_record_id") or ""))
        if (
            journal.get("status") in {"RECORD_APPLIED", "RECORD_DURABLE", "COMMITTED"}
            and _record_matches_committed_files(record, journal)
        ):
            # The session record crossed its fsync boundary but the process died
            # before the marker.  Completing is safer than rolling a durable
            # record back to files it no longer names.
            _write_marker(transaction_dir / COMMIT_MARKER)
            journal["status"] = "COMMITTED"
            _persist_journal(journal_path, journal)
            record_mutated = _apply_committed_records(state, journal)
            mutated = record_mutated or mutated
            attempt_clean = _cleanup_attempt(journal)
            _release_publish_locks(
                attempt_id,
                keys=journal.get("lock_keys") or [],
                figures_dir=figures_dir,
                session_id=str(journal.get("session_id") or ""),
            )
            if record_mutated:
                if applied_transaction_ids is not None:
                    applied_transaction_ids.add(attempt_id)
            elif attempt_clean:
                _remove_transaction_dir(transaction_dir)
        else:
            rolled_back = rollback_figure_publish_transaction(
                journal_path, state=state, producer_status="abandoned"
            )
            mutated = rolled_back or mutated
            refreshed = None
            try:
                refreshed = _read_verified_journal(journal_path)
            except Exception:
                pass
            if refreshed is not None and _authenticated_marker_exists(
                transaction_dir, ABANDON_MARKER, refreshed
            ):
                if applied_transaction_ids is not None:
                    applied_transaction_ids.add(attempt_id)
                # Even when the in-memory record already carried the same
                # terminal fields, session.json may still contain the durable
                # pending state.
                mutated = True
            else:
                _release_publish_locks(
                    attempt_id,
                    keys=journal.get("lock_keys") or [],
                    figures_dir=figures_dir,
                    session_id=str(journal.get("session_id") or ""),
                )
    garbage_collect_figure_transaction_roots(
        figures_dir, budget=remaining_budget
    )
    return mutated


def mark_figure_record_recovery_durable(
    figures_dir: Path,
    transaction_ids: Iterable[str],
) -> None:
    """Mark terminal record rollbacks only after session.json was fsynced."""
    figures_dir = _absolute(Path(figures_dir))
    root = figures_dir / TRANSACTION_ROOT
    try:
        root_fd, parent_fds = _open_hidden_root(root, create=False)
    except ValueError:
        return
    try:
        transaction_names = set(os.listdir(root_fd))
    finally:
        os.close(root_fd)
        for descriptor in reversed(parent_fds):
            os.close(descriptor)
    for transaction_id in sorted({str(value) for value in transaction_ids if value}):
        if transaction_id != _safe_component(transaction_id, "attempt"):
            continue
        if transaction_id not in transaction_names:
            continue
        transaction_dir = root / transaction_id
        try:
            figures_fd, transaction_root_fd, transaction_fd = (
                _open_transaction_directory(transaction_dir)
            )
            try:
                journal_metadata = os.stat(
                    JOURNAL_FILENAME,
                    dir_fd=transaction_fd,
                    follow_symlinks=False,
                )
                if not stat.S_ISREG(journal_metadata.st_mode):
                    raise ValueError("publication transaction journal is not regular")
            finally:
                os.close(transaction_fd)
                os.close(transaction_root_fd)
                os.close(figures_fd)
        except (OSError, ValueError):
            _quarantine_transaction_dir(transaction_dir)
            continue
        try:
            journal = _read_verified_journal(transaction_dir / JOURNAL_FILENAME)
        except Exception:
            _quarantine_transaction_dir(transaction_dir)
            continue
        terminal_rollback = any(
            _authenticated_marker_exists(transaction_dir, marker, journal)
            for marker in (ABANDON_MARKER, FAILURE_MARKER)
        )
        committed = _authenticated_marker_exists(
            transaction_dir, COMMIT_MARKER, journal
        )
        if not terminal_rollback and not committed:
            continue
        if terminal_rollback:
            _write_marker(transaction_dir / ROLLBACK_DURABLE_MARKER)
        if _cleanup_attempt(journal):
            _remove_transaction_dir(transaction_dir)
    garbage_collect_figure_transaction_roots(Path(figures_dir))
