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
    publish as _publish_owner,
    recovery as _recovery_owner,
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
    return _filesystem_owner._copy_durable(
        source, destination, bindings=sys.modules[__name__]
    )


def register_pending_figure_attempt(*, attempt_dir: Path, staged_jpg: Path, staged_pdf: Path, final_jpg: Path, final_pdf: Path, published_record: Any, session_id: str='') -> dict[str, Any]:
    """Persist the producer record while owning both publication locks."""
    return _publish_owner.register_pending_figure_attempt(attempt_dir=attempt_dir, staged_jpg=staged_jpg, staged_pdf=staged_pdf, final_jpg=final_jpg, final_pdf=final_pdf, published_record=published_record, session_id=session_id, bindings=sys.modules[__name__])


def refreeze_pending_figure_attempt(pending_transaction: Mapping[str, Any], *, attempt_dir: Path, staged_jpg: Path, staged_pdf: Path, final_jpg: Path, final_pdf: Path, published_record: Any) -> dict[str, Any]:
    """Authenticate and freeze the final reviewed bytes of a pending attempt.

    Revisions replace the JPG/PDF inside the already-journaled attempt.  Only
    the process that owns the original authenticated transaction and all of its
    locks may advance the signed digests to those final candidate bytes.
    """
    return _publish_owner.refreeze_pending_figure_attempt(pending_transaction, attempt_dir=attempt_dir, staged_jpg=staged_jpg, staged_pdf=staged_pdf, final_jpg=final_jpg, final_pdf=final_pdf, published_record=published_record, bindings=sys.modules[__name__])


def begin_figure_publish_transaction(*, attempt_dir: Path, staged_jpg: Path, staged_pdf: Path, final_jpg: Path, final_pdf: Path, published_record: Any, archived_record: Any=None) -> tuple[dict[str, Any] | None, str | None]:
    """Durably replace a validated pair, leaving its record uncommitted.

    The returned transaction remains recoverable until
    :func:`commit_figure_publish_transaction` writes its marker.  A normal I/O
    exception is rolled back here.  A process death or injected ``BaseException``
    is intentionally left to :func:`recover_figure_publish_transactions`.
    """
    return _publish_owner.begin_figure_publish_transaction(attempt_dir=attempt_dir, staged_jpg=staged_jpg, staged_pdf=staged_pdf, final_jpg=final_jpg, final_pdf=final_pdf, published_record=published_record, archived_record=archived_record, bindings=sys.modules[__name__])


def apply_publish_record_paths(transaction: dict[str, Any], published_record: Any, archived_record: Any=None) -> None:
    """Apply the file half of a transaction to its in-memory records."""
    return _publish_owner.apply_publish_record_paths(transaction, published_record, archived_record, bindings=sys.modules[__name__])


def mark_publish_record_applied(transaction: dict[str, Any], *, published_record: Any, archived_record: Any=None) -> None:
    """Journal the exact record state that the session write must persist."""
    return _publish_owner.mark_publish_record_applied(transaction, published_record=published_record, archived_record=archived_record, bindings=sys.modules[__name__])


def commit_figure_publish_transaction(transaction: dict[str, Any]) -> None:
    """Commit after the record boundary has been durably persisted."""
    return _publish_owner.commit_figure_publish_transaction(transaction, bindings=sys.modules[__name__])


def _path_from_journal(journal: Mapping[str, Any], key: str) -> Path | None:
    return _publish_owner._path_from_journal(journal, key, bindings=sys.modules[__name__])


def _release_journal_locks(journal: Mapping[str, Any]) -> None:
    return _publish_owner._release_journal_locks(journal, bindings=sys.modules[__name__])


def _cleanup_attempt(journal: Mapping[str, Any]) -> bool:
    return _publish_owner._cleanup_attempt(journal, bindings=sys.modules[__name__])


def _remove_transaction_dir(transaction_dir: Path) -> bool:
    return _publish_owner._remove_transaction_dir(transaction_dir, bindings=sys.modules[__name__])


def _remove_revision_archive_dir(archive_dir: Path) -> bool:
    """Remove one derived revision attempt without following hidden roots."""
    return _publish_owner._remove_revision_archive_dir(archive_dir, bindings=sys.modules[__name__])


def _open_hidden_root(root: Path, *, create: bool=False) -> tuple[int, list[int]]:
    """Open one recognized hidden root and retain its trusted parent FDs."""
    return _recovery_owner._open_hidden_root(root, create=create, bindings=sys.modules[__name__])


def _remove_bounded_root_entry(path: Path, root: Path) -> bool:
    """Remove one direct child without following a symlink outside *root*."""
    return _recovery_owner._remove_bounded_root_entry(path, root, bindings=sys.modules[__name__])


def garbage_collect_figure_transaction_roots(figures_dir: Path, *, budget: int | None=None) -> int:
    """Bound maintenance strictly to hidden publication transaction roots."""
    return _recovery_owner.garbage_collect_figure_transaction_roots(figures_dir, budget=budget, bindings=sys.modules[__name__])


def _restore_file(*, final_path: Path | None, backup_path: Path | None, was_replaced: bool, had_final: bool, new_digest: str | None) -> None:
    return _recovery_owner._restore_file(final_path=final_path, backup_path=backup_path, was_replaced=was_replaced, had_final=had_final, new_digest=new_digest, bindings=sys.modules[__name__])


def _restore_record(record: Any, snapshot: Mapping[str, Any]) -> bool:
    return _recovery_owner._restore_record(record, snapshot, bindings=sys.modules[__name__])


def _abandon_new_record(state: Any, journal: Mapping[str, Any], producer_status: str) -> bool:
    return _recovery_owner._abandon_new_record(state, journal, producer_status, bindings=sys.modules[__name__])


def _restore_abandoned_records(state: Any, journal: Mapping[str, Any], producer_status: str) -> bool:
    return _recovery_owner._restore_abandoned_records(state, journal, producer_status, bindings=sys.modules[__name__])


def rollback_figure_publish_transaction(journal_path: Path, *, state: Any=None, producer_status: str='abandoned') -> bool:
    """Rollback one non-committed journal and terminalize its producer record."""
    return _recovery_owner.rollback_figure_publish_transaction(journal_path, state=state, producer_status=producer_status, bindings=sys.modules[__name__])


def _record_matches_committed_files(record: Any, journal: Mapping[str, Any]) -> bool:
    return _recovery_owner._record_matches_committed_files(record, journal, bindings=sys.modules[__name__])


def _apply_committed_records(state: Any, journal: Mapping[str, Any]) -> bool:
    return _recovery_owner._apply_committed_records(state, journal, bindings=sys.modules[__name__])


def recover_figure_publish_transactions(state: Any, figures_dir: Path, *, force: bool=False, applied_transaction_ids: set[str] | None=None) -> bool:
    """Recover every publication transaction visible to this workspace.

    A commit marker is authoritative.  Without one, a record that is already
    durable and hash-bound to both final files can be forward-committed; every
    other state is rolled back and terminalized as ``abandoned``.
    """
    return _recovery_owner.recover_figure_publish_transactions(state, figures_dir, force=force, applied_transaction_ids=applied_transaction_ids, bindings=sys.modules[__name__])


def mark_figure_record_recovery_durable(figures_dir: Path, transaction_ids: Iterable[str]) -> None:
    """Mark terminal record rollbacks only after session.json was fsynced."""
    return _recovery_owner.mark_figure_record_recovery_durable(figures_dir, transaction_ids, bindings=sys.modules[__name__])
