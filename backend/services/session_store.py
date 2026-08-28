"""Session persistence: memory cache + atomic disk writes.

Extracted from backend/main.py. Public API:
  - save_session_to_disk(session_id, state)   # sync write, event-loop thread
  - async_save_session(session_id, state, force=False)  # throttled async write
  - load_session_from_disk(session_id) -> AnalysisState | None
  - ensure_session_loaded(session_id) -> AnalysisState | None
  - list_saved_sessions() -> list[dict]
  - update_session_cache(session_id, state)
  - touch_session(session_id)
  - pause_and_restart_ws(session_id, state)
"""
import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import (
    SESSION_DIR,
    get_session_states,
    get_session_metadata_path,
    resolve_session_metadata_path,
)
from backend.constants import WaitInputReason
from backend.models.schemas import AnalysisState, TaskStatus, DEFAULT_ROUND_WAIT_SECONDS
from backend.utils.blob_store import externalize_blobs, inline_blobs
from backend.utils.files import validate_session_id
from backend.utils.asyncio_tasks import run_in_owned_thread
from backend.utils.websocket_delivery import (
    json_safe_payload,
    strip_nested_binary_payload,
)
from backend.services.runtime_persistence_projection import (
    apply_runtime_persistence_projection,
    project_runtime_persistence,
)
from backend.state import (
    sessions, state_lock, session_access_times, last_save_times,
    session_locks,
)

logger = logging.getLogger(__name__)

_MIN_SAVE_INTERVAL = 5.0
_FORCE_SAVE_STATUSES = frozenset({
    TaskStatus.COMPLETED, TaskStatus.FAILED,
    TaskStatus.WAITING_INPUT, TaskStatus.WAITING_ANNOTATION,
    TaskStatus.GENERATING_REPORT, TaskStatus.PAUSED,
})

# Session-level recovery block when AnalysisState cannot be loaded for display.
# Checked by assert_recovery_clear(session_id=...) as a fail-closed fallback.
_recovery_blocked_sessions: set[str] = set()
_TERMINAL_STATUSES = frozenset({
    TaskStatus.FAILED,
    TaskStatus.COMPLETED,
})

_PUBLICATION_RECOVERY_FIELDS = frozenset({
    "png_path",
    "pdf_path",
    "png_base64",
    "render_sha256",
    "preview_sha256",
    "pdf_sha256",
    "producer_status",
    "is_published_head",
    "supersedes_id",
    "revision_of_id",
})


def _merge_publication_recovery(
    live_state: AnalysisState,
    before_recovery: AnalysisState,
    recovered_state: AnalysisState,
) -> None:
    """Apply only transaction-owned changes to an existing live state.

    The disk copy is the authority for crash journals, but not for unrelated
    in-memory edits made since its last throttled save. In particular,
    ``is_current`` and invalidation metadata must never be rolled back here.
    """
    before_by_id = {
        str(item.id): item for item in before_recovery.publication_figures
    }
    live_by_id = {str(item.id): item for item in live_state.publication_figures}
    for recovered in recovered_state.publication_figures:
        record_id = str(recovered.id)
        live = live_by_id.get(record_id)
        before = before_by_id.get(record_id)
        if live is None:
            # Only a record reconstructed by recovery is itself part of the
            # transaction delta.  An unchanged disk-only record must not
            # repopulate a live list that deliberately no longer contains it.
            if before is not None:
                continue
            copied = recovered.model_copy(deep=True)
            live_state.publication_figures.append(copied)
            live_by_id[record_id] = copied
            continue
        for field in _PUBLICATION_RECOVERY_FIELDS:
            recovered_value = getattr(recovered, field, None)
            if before is not None and getattr(before, field, None) == recovered_value:
                continue
            setattr(live, field, recovered_value)

    live_legacy_ids = {
        str(item.id) for item in (live_state.legacy_artifacts or [])
    }
    for recovered in recovered_state.legacy_artifacts or []:
        if str(recovered.id) in live_legacy_ids:
            continue
        live_state.legacy_artifacts.append(recovered.model_copy(deep=True))
        live_legacy_ids.add(str(recovered.id))
    recovered_version = recovered_state.artifact_registry_version
    if recovered_version is not None:
        live_version = live_state.artifact_registry_version
        live_state.artifact_registry_version = max(
            int(live_version or 0), int(recovered_version)
        )


def mark_recovery_blocked(
    session_id: str,
    state: AnalysisState | None,
    failure: Dict[str, Any] | None = None,
) -> None:
    """Mark session recovery-blocked (in-memory + state when available).

    Used when journal recovery returns mutations_blocked or raises unexpectedly.
    State is mutated for read-only display; callers persist when appropriate.
    """
    _recovery_blocked_sessions.add(session_id)
    if state is None:
        return
    state.recovery_blocked = True
    code = "RECOVERY_UNEXPECTED"
    message = "Snapshot recovery failed; mutations blocked"
    if isinstance(failure, dict):
        code = str(failure.get("code") or code)
        message = str(failure.get("message") or message)
    # Merge recovery_error into existing snapshot_transaction — do not wipe
    # restore/capture fields (transaction_id, status, snapshot_id, etc.).
    existing = state.snapshot_transaction if isinstance(state.snapshot_transaction, dict) else {}
    merged = dict(existing)
    merged["recovery_error"] = {"code": code, "message": message[:500]}
    state.snapshot_transaction = merged
    if state.status not in _TERMINAL_STATUSES:
        state.status = TaskStatus.PAUSED


def clear_recovery_blocked(
    session_id: str,
    state: AnalysisState | None = None,
) -> bool:
    """Clear recovery block after a clean successful recovery.

    Discards the session-level marker and, when state is available, clears
    ``recovery_blocked`` plus ``snapshot_transaction.recovery_error`` without
    removing other restore/capture fields. Returns True if state was mutated
    (caller should persist).
    """
    _recovery_blocked_sessions.discard(session_id)
    if state is None:
        return False
    mutated = False
    if getattr(state, "recovery_blocked", False):
        state.recovery_blocked = False
        mutated = True
    tx = state.snapshot_transaction
    if isinstance(tx, dict) and "recovery_error" in tx:
        cleaned = {k: v for k, v in tx.items() if k != "recovery_error"}
        state.snapshot_transaction = cleaned or None
        mutated = True
    return mutated


async def _persist_recovery_marker(session_id: str, state: AnalysisState) -> None:
    """Persist recovery_blocked / cleared marker to session.json (best-effort)."""
    try:
        await _persist_dump_under_lifecycle(session_id, state)
    except Exception:
        logger.exception(
            "Session %s: failed to persist recovery marker",
            session_id,
        )


def assert_recovery_clear(state: AnalysisState | None = None, *, session_id: str | None = None) -> None:
    """Raise if snapshot recovery left mutations blocked.

    Call before workflow construction, lifecycle start/resume, WebSocket resume,
    or kernel start. An unresolved/tampered APPLYING journal or unexpected
    recovery exception sets ``recovery_blocked`` and must prevent forward
    execution.
    """
    from backend.services.snapshot_v2.errors import SnapshotTransactionError

    sid = session_id
    if state is None and sid:
        state = sessions.get(sid)
    if state is not None:
        sid = sid or getattr(state, "id", None) or getattr(state, "session_id", None)
    if sid and sid in _recovery_blocked_sessions:
        raise SnapshotTransactionError(
            "RECOVERY_BLOCKED",
            "Unresolved snapshot restore journal blocks forward execution; "
            "recovery must complete before workflow/kernel resume.",
        )
    if state is not None and getattr(state, "recovery_blocked", False):
        raise SnapshotTransactionError(
            "RECOVERY_BLOCKED",
            "Unresolved snapshot restore journal blocks forward execution; "
            "recovery must complete before workflow/kernel resume.",
        )


def _write_session_json(session_id: str, json_str: str) -> None:
    try:
        session_file = get_session_metadata_path(session_id)
        session_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(session_file.parent, 0o700)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json.tmp", prefix=f"{session_id}_", dir=session_file.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json_str)
                f.flush()
                os.fsync(f.fileno())
            # Session metadata can contain encrypted credentials and private
            # analysis content.  App-layer ownership checks do not protect it
            # from other local OS users, so preserve mkstemp's owner-only mode.
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, session_file)
            os.chmod(session_file, 0o600)
            try:
                directory_fd = os.open(
                    str(session_file.parent),
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                # Some non-POSIX filesystems do not expose directory fsync. The
                # file itself is already fsynced; retain compatibility there.
                pass
            logger.info(f"Session {session_id} saved to disk")
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    except Exception:
        logger.exception(f"Failed to save session {session_id}")
        raise


def _blob_dir_for(session_id: str) -> Path:
    return get_session_metadata_path(session_id).parent / "blobs"


def _preserve_legacy_preview_digests(
    source_state: AnalysisState, durable_dump: dict
) -> None:
    """Carry a verifiable legacy preview identity across binary stripping."""
    if source_state.artifact_registry_version is not None:
        return
    source_figures = source_state.publication_figures
    durable_figures = durable_dump.get("publication_figures")
    if not isinstance(source_figures, list) or not isinstance(durable_figures, list):
        return
    from backend.services.report.artifacts import preview_render_sha256

    for source_figure, durable_figure in zip(source_figures, durable_figures):
        if not isinstance(durable_figure, dict):
            continue
        if str(durable_figure.get("preview_sha256") or "").strip():
            continue
        source_preview = (
            source_figure.get("png_base64")
            if isinstance(source_figure, dict)
            else getattr(source_figure, "png_base64", None)
        )
        digest = preview_render_sha256(source_preview)
        if digest:
            durable_figure["preview_sha256"] = digest


def _strip_legacy_blob_references(value: Any, *, realtime: bool = False) -> Any:
    """Replace legacy blob handles only inside durable realtime history."""
    if isinstance(value, str):
        return "[image]" if realtime and value.startswith("@scai-blob:") else value
    if isinstance(value, list):
        return [_strip_legacy_blob_references(item, realtime=realtime) for item in value]
    if isinstance(value, dict):
        return {
            key: _strip_legacy_blob_references(
                item,
                realtime=realtime or key == "realtime_messages",
            )
            for key, item in value.items()
        }
    return value


def _runtime_persistence_projection(
    session_id: str,
    state: AnalysisState,
) -> Optional[tuple[Dict[str, Any], Dict[str, Any]]]:
    return project_runtime_persistence(
        get_session_states(session_id) / f"{session_id}_state.json",
        session_id,
        state,
    )


def refresh_runtime_persistence_projection(
    session_id: str,
    state: AnalysisState,
) -> Dict[str, Any]:
    """Refresh the mutable live projection without creating a second authority."""
    return apply_runtime_persistence_projection(
        get_session_states(session_id) / f"{session_id}_state.json",
        session_id,
        state,
    )


def _durable_state_dump(
    state: AnalysisState,
    *,
    session_id: Optional[str] = None,
) -> dict:
    """Create a detached, sanitized JSON dump without mutating live state."""
    # Totalize arbitrary realtime ``Any`` content before Pydantic serialization
    # while retaining the AnalysisState serializer that seals persisted secrets.
    safe_messages = []
    for message in state.realtime_messages:
        safe_content = json_safe_payload(
            getattr(message, "content", None),
            strip_binary=True,
        )
        if hasattr(message, "model_copy"):
            safe_messages.append(message.model_copy(
                deep=False,
                update={"content": safe_content},
            ))
        elif isinstance(message, dict):
            safe_messages.append({**message, "content": safe_content})
        else:
            safe_messages.append(json_safe_payload(message, strip_binary=True))

    updates: Dict[str, Any] = {"realtime_messages": safe_messages}
    if session_id:
        projection = _runtime_persistence_projection(session_id, state)
        if projection is not None:
            updates["runtime_persistence"], updates["sandbox_state"] = projection
    prepared_state = state.model_copy(
        deep=False,
        update=updates,
    )
    sanitized_state = strip_nested_binary_payload(prepared_state)
    if not isinstance(sanitized_state, AnalysisState):
        raise TypeError("durable state copier returned an unexpected model")
    durable_dump = sanitized_state.model_dump(mode="json")
    _preserve_legacy_preview_digests(state, durable_dump)
    return _strip_legacy_blob_references(durable_dump)


def _externalize_and_write(session_id: str, dumped: dict) -> None:
    """Externalize blobs to disk and write the session JSON. Blocking I/O —
    async callers must invoke this via ``run_in_owned_thread`` so large base64
    blob writes do not stall the event loop."""
    from backend.services.session_lifecycle import (
        assert_dump_writable,
        lifecycle_file_lock,
    )

    with lifecycle_file_lock(session_id):
        assert_dump_writable(session_id, dumped)
        externalized = externalize_blobs(dumped, _blob_dir_for(session_id))
        _write_session_json(session_id, json.dumps(externalized, ensure_ascii=False))


async def _persist_dump_under_lifecycle(
    session_id: str, state: AnalysisState
) -> None:
    """Unthrottled internal write serialized against delete and stale saves."""
    from backend.services.session_lifecycle import get_lifecycle_lock

    dumped = _durable_state_dump(state, session_id=session_id)
    async with get_lifecycle_lock(session_id):
        await run_in_owned_thread(_externalize_and_write, session_id, dumped)


def save_session_to_disk(session_id: str, state: AnalysisState) -> None:
    """Synchronous startup/test entrypoint guarded by the lifecycle file lock."""
    _externalize_and_write(
        session_id,
        _durable_state_dump(state, session_id=session_id),
    )


async def _async_save_session_under_lifecycle(
    session_id: str,
    state: AnalysisState,
    force: bool = False,
) -> None:
    """Persist while the caller owns ``get_lifecycle_lock(session_id)``."""
    from backend.services.session_lifecycle import assert_generation_writable

    assert_generation_writable(session_id, state)
    if not force and hasattr(state, "status") and state.status in _FORCE_SAVE_STATUSES:
        force = True
    now = time.monotonic()
    if not force:
        last = last_save_times.get(session_id, 0)
        if now - last < _MIN_SAVE_INTERVAL:
            logger.debug("Throttled save for %s", session_id)
            return
    last_save_times[session_id] = now
    # model_dump on the event loop (snapshot the mutable model here to avoid
    # racing with in-place mutation); offload the blob writes + JSON write.
    dumped = _durable_state_dump(state, session_id=session_id)
    await run_in_owned_thread(_externalize_and_write, session_id, dumped)


async def async_save_session(session_id: str, state: AnalysisState, force: bool = False) -> None:
    from backend.services.session_lifecycle import get_lifecycle_lock

    async with get_lifecycle_lock(session_id):
        await _async_save_session_under_lifecycle(session_id, state, force=force)


def load_session_from_disk(session_id: str) -> Optional[AnalysisState]:
    try:
        session_file = resolve_session_metadata_path(session_id)
        if session_file and session_file.exists():
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data = _strip_legacy_blob_references(data)
            data = inline_blobs(data, session_file.parent / "blobs")
            if not data.get("user_id"):
                logger.warning(f"Session {session_id} has empty user_id, refusing to load")
                return None
            state = AnalysisState(**data)
            refresh_runtime_persistence_projection(session_id, state)
            logger.info(f"Session {session_id} loaded from disk")
            return state
    except (OSError, json.JSONDecodeError, ValueError, KeyError):
        logger.exception(f"Failed to load session {session_id}")
    return None


async def update_session_cache(session_id: str, state: AnalysisState) -> None:
    from backend.services.session_lifecycle import (
        adopt_persisted_generation,
        get_lifecycle_lock,
    )

    async with get_lifecycle_lock(session_id):
        adopt_persisted_generation(state)
        async with state_lock:
            sessions[session_id] = state
            session_access_times[session_id] = time.time()


async def touch_session(session_id: str) -> None:
    async with state_lock:
        session_access_times[session_id] = time.time()


async def _get_cached_session(session_id: str) -> Optional[AnalysisState]:
    """Quick lock to read from the in-memory cache."""
    async with state_lock:
        cached = sessions.get(session_id)
        if cached is not None:
            session_access_times[session_id] = time.time()
        return cached


async def ensure_session_loaded_for_read(
    session_id: str,
) -> Optional[AnalysisState]:
    """Load a session for read-only endpoints, preferring the canonical cache."""
    validate_session_id(session_id)
    from backend.services.session_lifecycle import (
        SessionGoneError,
        adopt_persisted_generation,
        get_lifecycle_lock,
        read_lifecycle_record,
    )

    async with get_lifecycle_lock(session_id):
        lifecycle = read_lifecycle_record(session_id)
        if lifecycle and lifecycle.get("status") == "deleted":
            raise SessionGoneError(
                session_id,
                str(lifecycle.get("owner_id") or ""),
                int(lifecycle.get("generation") or 0),
            )
        if lifecycle and lifecycle.get("status") == "importing":
            return None
        cached = await _get_cached_session(session_id)
        if cached is not None:
            adopt_persisted_generation(cached)
            refresh_runtime_persistence_projection(session_id, cached)
            return cached

    return await ensure_session_loaded(session_id)


async def _install_loaded_session(
    session_id: str,
    loaded: AnalysisState,
) -> AnalysisState:
    """Atomically install a loaded state if not already cached."""
    async with state_lock:
        existing = sessions.get(session_id)
        if existing is not None:
            session_access_times[session_id] = time.time()
            return existing
        sessions[session_id] = loaded
        session_access_times[session_id] = time.time()
        return loaded


async def _recover_snapshot_v2_on_load(
    session_id: str,
) -> tuple[bool, Optional[Dict[str, Any]]]:
    """Run enabled-mode Snapshot V2 journal recovery before session loading."""
    recovery_blocked = False
    recovery_failure: Optional[Dict[str, Any]] = None
    gc_result = None
    try:
        from backend.services.snapshot_v2.paths import SnapshotV2Paths
        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions
        from backend.config._config import get_session_dir, get_session_workspace, get_session_states

        paths = SnapshotV2Paths.for_session(session_id)
        session_dir = str(get_session_dir(session_id))
        workspace_root = str(get_session_workspace(session_id))
        states_root = str(get_session_states(session_id))
        result = await recover_snapshot_transactions(
            session_id=session_id,
            paths=paths,
            session_dir=session_dir,
            workspace_root=workspace_root,
            states_root=states_root,
        )
        if result.mutations_blocked:
            recovery_blocked = True
            recovery_failure = {
                "code": "RECOVERY_BLOCKED",
                "message": (
                    f"Recovery mutations blocked "
                    f"({len(result.failed_transactions)} failed transactions)"
                ),
            }
            logger.error(
                "Session %s: recovery failure — mutations blocked (%d failed transactions)",
                session_id, len(result.failed_transactions),
            )
        if result.session_state_rewritten:
            # Recovery rewrote session.json from the pre_restore snapshot (or
            # forward-completed the target), but it never touches the in-memory
            # cache — which still holds the AnalysisState the interrupted
            # restore installed via `_persist_restored_state`. Left in place,
            # `ensure_session_loaded` returns that object, clears its recovery
            # block and persists it, writing the *target* state back over the
            # file the rollback just restored: the rollback is silently
            # half-undone and the session is unblocked while its state, files,
            # notebook and checkpoint no longer agree. Evict so the next read
            # comes from the disk copy recovery just wrote.
            async with state_lock:
                sessions.pop(session_id, None)
                session_access_times.pop(session_id, None)
            logger.info(
                "Session %s: evicted cached state after recovery rewrote "
                "session.json", session_id,
            )
        gc_result = result.gc_result
        if result.recovered:
            logger.info(
                "Session %s: recovered %d restore transactions",
                session_id, len(result.recovered_transactions),
            )
            if result.errors:
                logger.warning(
                    "Session %s: recovery errors: %s",
                    session_id, result.errors,
                )
        if gc_result and gc_result.removed_count > 0:
            logger.info(
                "Session %s: GC cleaned %d objects (%d bytes)",
                session_id, gc_result.removed_count, gc_result.removed_bytes,
            )
    except Exception as exc:
        recovery_blocked = True
        recovery_failure = {
            "code": "RECOVERY_UNEXPECTED",
            "message": str(exc)[:500],
        }
        logger.exception(
            "Session %s: snapshot recovery failed; blocking mutations",
            session_id,
        )

    if recovery_blocked and recovery_failure is None:
        recovery_failure = {
            "code": "RECOVERY_BLOCKED",
            "message": "Snapshot recovery left mutations blocked",
        }
    return recovery_blocked, recovery_failure


def _snapshot_transaction_in_flight(session_id: str) -> bool:
    """True while an in-process snapshot transaction owns this session's journals.

    Journal recovery is *crash* recovery: a STAGED/APPLYING ``restore_apply``
    journal is taken as proof that the process which wrote it died, so recovery
    rolls the live session back to the pre_restore snapshot, removes the staging
    tree and deletes the journal. That inference only holds when nobody is
    mid-transaction — but this runs on EVERY authenticated request, including
    the frontend's 2-second poll of the restore's own status endpoint.

    ``SnapshotService.restore_snapshot`` holds the per-session lock from
    ``stage()`` (journal STAGED) through ``commit()``, and yields inside that
    window (closing the superseded checkpoint connection, the owned worker in
    ``async_save_session``). ``recover_snapshot_transactions`` has no awaits,
    so a request served at one of those yields runs the entire rollback and
    destroys the restore it was asked to report on.

    Deferring is safe: the journal stays on disk for a later load, and genuine
    crashes are covered by startup recovery, which runs before any request is
    served.
    """
    lock = session_locks.get(session_id)
    return lock is not None and lock.locked()


async def _recover_publication_transactions_on_load(
    session_id: str,
    state: AnalysisState,
) -> bool:
    """Recover figure pair/record journals from the latest durable state.

    The asyncio lock serializes callers in this process.  The synchronous helper
    then holds the shared lifecycle flock across latest-state load, transaction-ID
    merge, session fsync, and exact marker/GC.  Two workers therefore cannot each
    save a recovery derived from a stale pre-recovery AnalysisState.
    """
    from backend.services.session_lifecycle import get_lifecycle_lock

    async with get_lifecycle_lock(session_id):
        return await run_in_owned_thread(
            _recover_publication_transactions_under_file_lock,
            session_id,
            state,
        )


def _recover_publication_transactions_under_file_lock(
    session_id: str,
    state: AnalysisState,
) -> bool:
    """Synchronous cross-process publication recovery transaction."""
    from backend.config import get_session_workspace
    from backend.services.report.figure_transactions import (
        mark_figure_record_recovery_durable,
        recover_figure_publish_transactions,
    )
    from backend.services.report.figure_discovery import (
        backfill_legacy_publication_integrity,
        ensure_legacy_artifact_inventory,
    )
    from backend.services.session_lifecycle import (
        SessionGoneError,
        assert_generation_writable,
        lifecycle_file_lock,
        read_lifecycle_record,
    )

    with lifecycle_file_lock(session_id):
        lifecycle = read_lifecycle_record(session_id)
        if lifecycle and lifecycle.get("status") == "deleted":
            raise SessionGoneError(
                session_id,
                str(lifecycle.get("owner_id") or ""),
                int(lifecycle.get("generation") or 0),
            )
        latest = load_session_from_disk(session_id)
        recovery_state = latest if latest is not None else state.model_copy(deep=True)
        before_recovery = recovery_state.model_copy(deep=True)
        assert_generation_writable(session_id, recovery_state)
        applied_transaction_ids: set[str] = set()
        mutated = recover_figure_publish_transactions(
            recovery_state,
            get_session_workspace(session_id) / "figures",
            applied_transaction_ids=applied_transaction_ids,
        )
        registry_before = getattr(
            recovery_state, "artifact_registry_version", None
        )
        legacy_count_before = len(recovery_state.legacy_artifacts or [])
        backfill_legacy_publication_integrity(
            recovery_state, get_session_workspace(session_id)
        )
        ensure_legacy_artifact_inventory(
            recovery_state, get_session_workspace(session_id)
        )
        inventory_migrated = bool(
            registry_before != recovery_state.artifact_registry_version
            or legacy_count_before != len(recovery_state.legacy_artifacts or [])
        )
        if mutated or inventory_migrated:
            _externalize_and_write(session_id, _durable_state_dump(recovery_state))
        if mutated:
            mark_figure_record_recovery_durable(
                get_session_workspace(session_id) / "figures",
                applied_transaction_ids,
            )
        _merge_publication_recovery(state, before_recovery, recovery_state)
        return mutated or inventory_migrated


async def recover_publication_transactions_for_state(
    session_id: str,
    state: AnalysisState,
) -> bool:
    """Public lifecycle-safe entrypoint for an already loaded producer state."""
    return await _recover_publication_transactions_on_load(session_id, state)


async def ensure_session_loaded(session_id: str) -> Optional[AnalysisState]:
    validate_session_id(session_id)

    from backend.services.session_lifecycle import (
        SessionGoneError,
        adopt_persisted_generation,
        get_lifecycle_lock,
        read_lifecycle_record,
    )

    from backend.config import is_snapshot_v2_enabled

    snapshot_v2_enabled = is_snapshot_v2_enabled()
    # Distinguishes "recovery ran and found nothing" from "recovery was
    # deferred" — only the former may clear an existing recovery block.
    recovery_evaluated = False
    recovery_blocked = False
    recovery_failure = None
    # Recovery mutates journals, session.json, runtime files, and checkpoints.
    # It therefore shares deletion's lifecycle boundary; a load that passed an
    # early tombstone check must not recover artifacts after delete committed.
    async with get_lifecycle_lock(session_id):
        lifecycle = read_lifecycle_record(session_id)
        if lifecycle and lifecycle.get("status") == "deleted":
            raise SessionGoneError(
                session_id,
                str(lifecycle.get("owner_id") or ""),
                int(lifecycle.get("generation") or 0),
            )
        if lifecycle and lifecycle.get("status") == "importing":
            # The destination directory is claimed before extraction starts.
            # Do not expose a partially extracted state merely because its
            # session.json happened to arrive before the import commit marker.
            return None
        if snapshot_v2_enabled and not _snapshot_transaction_in_flight(session_id):
            recovery_evaluated = True
            recovery_blocked, recovery_failure = await _recover_snapshot_v2_on_load(
                session_id,
            )

    # Check cache before disk load
    cached = await _get_cached_session(session_id)
    if cached is not None:
        await _recover_publication_transactions_on_load(session_id, cached)
        if not snapshot_v2_enabled:
            from backend.services.snapshot_v2.feature_gate import (
                normalize_disabled_snapshot_state,
            )

            if normalize_disabled_snapshot_state(cached, workflow_active=False):
                await _persist_recovery_marker(session_id, cached)
            # The initial lifecycle check preceded the cache read and any
            # normalization write. Deletion may have committed in between;
            # linearize this cached return against the same generation guard
            # used by disk-loaded returns.
            async with get_lifecycle_lock(session_id):
                adopt_persisted_generation(cached)
                refresh_runtime_persistence_projection(session_id, cached)
            return cached
        # Apply recovery outcome to cached state and return
        if recovery_blocked:
            mark_recovery_blocked(session_id, cached, recovery_failure)
            await _persist_recovery_marker(session_id, cached)
        elif recovery_evaluated:
            if clear_recovery_blocked(session_id, cached):
                await _persist_recovery_marker(session_id, cached)
        async with get_lifecycle_lock(session_id):
            adopt_persisted_generation(cached)
            refresh_runtime_persistence_projection(session_id, cached)
        return cached

    # Load from disk outside state_lock in a joined, cancellation-safe worker.
    state = await run_in_owned_thread(load_session_from_disk, session_id)
    if state is None:
        if recovery_blocked:
            mark_recovery_blocked(session_id, None, recovery_failure)
        elif recovery_evaluated:
            clear_recovery_blocked(session_id, None)
        return None

    await _recover_publication_transactions_on_load(session_id, state)

    # A delete may have committed while disk I/O was in flight.  This second
    # generation check prevents the loaded object from reviving the cache.
    async with get_lifecycle_lock(session_id):
        adopt_persisted_generation(state)

    # Clean up a stale workflow-bound gate from a previous process. Every reason
    # listed here is resolved only by POST /{id}/llm-failure-action, which needs
    # the live workflow object that set it; after a restart there is none, so the
    # dialog the WebSocket re-opens would 404 on every button. `agent_disabled`
    # belongs with the other two — it was omitted, and it is the one gate that
    # therefore never self-healed.
    if state.wait_input_reason in (
        WaitInputReason.LLM_FAILURE.value,
        WaitInputReason.AGENT_DISABLED.value,
        WaitInputReason.MARKER_DB_MISSING.value,
    ):
        logger.warning(
            f"Session {session_id}: clearing stale "
            f"{state.wait_input_reason} gate from previous process"
        )
        info = state.llm_failure_info
        if info:
            node_name = info.get("node_name", "")
            logger.info(
                f"Session {session_id}: LLM failure for node '{node_name}' "
                f"was not resolved before server restart. Node marked as failed."
            )
        state.wait_input_reason = None
        state.llm_failure_info = None
        # Left behind, this still rides out on every state_sync as an open
        # agent-disabled dialog, which is the gate we just declared unresolvable.
        state.agent_disabled_info = None
        state.status = TaskStatus.PAUSED
        logger.info(
            f"Session {session_id}: Analysis paused due to an unresolved "
            f"workflow-bound gate before restart."
        )
        # Save the cleaned-up state so PAUSED status persists
        await _persist_dump_under_lifecycle(session_id, state)

    # Apply Snapshot V2 recovery or disabled-mode stale-state normalization.
    if not snapshot_v2_enabled:
        from backend.services.snapshot_v2.feature_gate import (
            normalize_disabled_snapshot_state,
        )

        if normalize_disabled_snapshot_state(state, workflow_active=False):
            await _persist_recovery_marker(session_id, state)
    elif recovery_blocked:
        mark_recovery_blocked(session_id, state, recovery_failure)
        await _persist_recovery_marker(session_id, state)
    elif recovery_evaluated and clear_recovery_blocked(session_id, state):
        # Same guard as the cached and not-found branches above. Omitting it
        # here read as "recovery reported nothing wrong", but a deferred pass
        # reports nothing at all: the lock-first endpoints (/resume, /pause,
        # /stop, prompts) hold the session lock before they load, so
        # _snapshot_transaction_in_flight always defers for them — and once the
        # session has been evicted, this is the branch they land on. Clearing
        # here discarded the process-wide marker and the persisted flag, and
        # assert_recovery_clear then waved through a session whose restore_apply
        # journal is still APPLYING. A block left standing costs one lock-free
        # load (any status poll) to clear once recovery really has run.
        await _persist_recovery_marker(session_id, state)

    # Reconcile AnalysisState.snapshots from verified V2 manifests
    # (skip when recovery is blocked — read-only display only)
    if snapshot_v2_enabled and not recovery_blocked:
        try:
            from backend.services.snapshot_v2.paths import SnapshotV2Paths
            from backend.services.snapshot_v2.service import SnapshotService
            svc = SnapshotService(session_id, SnapshotV2Paths.for_session(session_id))
            # Pass the loaded listing as the baseline: the reconcile is
            # authoritative for V2 projections only, and the state is not in the
            # cache yet, so it is the sole place the pre-cutover V1 records
            # (which have no manifest by construction) can be read from.
            state.snapshots = svc.rebuild_snapshots_from_manifests(state.snapshots)
        except Exception:
            logger.debug(
                "Session %s: snapshot listing reconcile skipped",
                session_id, exc_info=True,
            )

    # Install under the same lifecycle lock deletion uses.  Re-check after all
    # post-load work because deletion may have committed while it ran.
    async with get_lifecycle_lock(session_id):
        adopt_persisted_generation(state)
        return await _install_loaded_session(session_id, state)


def list_saved_sessions() -> List[Dict[str, Any]]:
    sessions_list: List[Dict[str, Any]] = []
    seen_ids: set = set()

    def _parse(session_file: Path, fallback_id: str = "") -> None:
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            sid = data.get("id", fallback_id)
            from backend.services.session_lifecycle import read_lifecycle_record

            lifecycle = read_lifecycle_record(sid)
            if lifecycle and lifecycle.get("status") in ("deleted", "importing"):
                return
            if sid in seen_ids:
                return
            seen_ids.add(sid)
            data_files = data.get("data_files", [])
            has_path = has_seq = has_bio = False
            for f in data_files:
                s = f.get("section")
                if s == "pathology": has_path = True
                elif s == "sequencing": has_seq = True
                elif s == "biochemistry": has_bio = True
            if data.get("biochemistry_indicators"):
                has_bio = True
            sessions_list.append({
                "id": sid,
                "user_id": data.get("user_id", ""),
                "name": data.get("name", "Unknown"),
                "status": data.get("status", "unknown"),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "current_round": data.get("current_round", 0),
                "waiting_input_started_at": data.get("waiting_input_started_at"),
                "round_wait_seconds": data.get("round_wait_seconds", DEFAULT_ROUND_WAIT_SECONDS),
                "has_pathology": has_path,
                "has_sequencing": has_seq,
                "has_biochemistry": has_bio,
            })
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to read session file {session_file}: {e}")

    try:
        for entry in SESSION_DIR.iterdir():
            if entry.is_dir():
                sf = entry / "session.json"
                if sf.exists():
                    _parse(sf, fallback_id=entry.name)
        for session_file in SESSION_DIR.glob("*.json"):
            if session_file.is_file():
                _parse(session_file, fallback_id=session_file.stem)
    except Exception:
        logger.exception("Failed to list sessions")

    sessions_list.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return sessions_list


async def pause_and_restart_ws(
    session_id: str,
    state: AnalysisState,
    _legacy_websockets: Optional[Dict[str, Any]] = None,
) -> None:
    state.status = TaskStatus.PAUSED
    await update_session_cache(session_id, state)
    await async_save_session(session_id, state)
    from backend.utils.websocket_delivery import close_analysis_websocket

    await close_analysis_websocket(session_id, code=4000)
