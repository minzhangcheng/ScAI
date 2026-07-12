"""Session persistence: memory cache + atomic disk writes.

Extracted from backend/main.py. Public API:
  - save_session_to_disk(session_id, state)   # sync write, event-loop thread
  - async_save_session(session_id, state, force=False)  # throttled async write
  - load_session_from_disk(session_id) -> AnalysisState | None
  - ensure_session_loaded(session_id) -> AnalysisState | None
  - list_saved_sessions() -> list[dict]
  - update_session_cache(session_id, state)
  - touch_session(session_id)
  - pause_and_restart_ws(session_id, state, websockets)
"""
import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import (
    SESSION_DIR,
    get_session_metadata_path,
    resolve_session_metadata_path,
)
from backend.constants import WaitInputReason
from backend.models.schemas import AnalysisState, TaskStatus, DEFAULT_ROUND_WAIT_SECONDS
from backend.utils.blob_store import externalize_blobs, inline_blobs
from backend.utils.files import validate_session_id
from backend.state import (
    sessions, state_lock, session_access_times, last_save_times,
)

logger = logging.getLogger(__name__)

_MIN_SAVE_INTERVAL = 5.0
_FORCE_SAVE_STATUSES = frozenset({
    TaskStatus.COMPLETED, TaskStatus.FAILED,
    TaskStatus.WAITING_INPUT, TaskStatus.WAITING_ANNOTATION,
    TaskStatus.GENERATING_REPORT,
})

# Session-level recovery block when AnalysisState cannot be loaded for display.
# Checked by assert_recovery_clear(session_id=...) as a fail-closed fallback.
_recovery_blocked_sessions: set[str] = set()
_TERMINAL_STATUSES = frozenset({
    TaskStatus.FAILED,
    TaskStatus.COMPLETED,
})


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
        dumped = state.model_dump(mode="json")
        await asyncio.to_thread(_externalize_and_write, session_id, dumped)
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
        session_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json.tmp", prefix=f"{session_id}_", dir=session_file.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json_str)
            # mkstemp hardcodes 0600; align with the other session files
            # (state.json/blobs are 0644) so the metadata isn't accidentally
            # owner-only. Isolation is enforced at the app layer, not via mode.
            os.chmod(tmp_path, 0o644)
            os.replace(tmp_path, session_file)
            logger.info(f"Session {session_id} saved to disk")
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    except Exception:
        logger.exception(f"Failed to save session {session_id}")


def _blob_dir_for(session_id: str) -> Path:
    return get_session_metadata_path(session_id).parent / "blobs"


def _externalize_and_write(session_id: str, dumped: dict) -> None:
    """Externalize blobs to disk and write the session JSON. Blocking I/O —
    async callers must invoke this via asyncio.to_thread so the large base64
    blob writes do not stall the event loop."""
    externalized = externalize_blobs(dumped, _blob_dir_for(session_id))
    _write_session_json(session_id, json.dumps(externalized, ensure_ascii=False))


def save_session_to_disk(session_id: str, state: AnalysisState) -> None:
    _externalize_and_write(session_id, state.model_dump(mode="json"))


async def async_save_session(session_id: str, state: AnalysisState, force: bool = False) -> None:
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
    dumped = state.model_dump(mode="json")
    await asyncio.to_thread(_externalize_and_write, session_id, dumped)


def load_session_from_disk(session_id: str) -> Optional[AnalysisState]:
    try:
        session_file = resolve_session_metadata_path(session_id)
        if session_file and session_file.exists():
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data = inline_blobs(data, session_file.parent / "blobs")
            if not data.get("user_id"):
                logger.warning(f"Session {session_id} has empty user_id, refusing to load")
                return None
            state = AnalysisState(**data)
            logger.info(f"Session {session_id} loaded from disk")
            return state
    except (OSError, json.JSONDecodeError, ValueError, KeyError):
        logger.exception(f"Failed to load session {session_id}")
    return None


async def update_session_cache(session_id: str, state: AnalysisState) -> None:
    async with state_lock:
        sessions[session_id] = state
        session_access_times[session_id] = time.time()


async def touch_session(session_id: str) -> None:
    async with state_lock:
        session_access_times[session_id] = time.time()


async def ensure_session_loaded(session_id: str) -> Optional[AnalysisState]:
    validate_session_id(session_id)

    # Run journal recovery before any session mutation
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

    async def _apply_recovery_outcome(cached: AnalysisState | None) -> None:
        """Mark blocked or clear after clean recovery; persist when state mutates."""
        if recovery_blocked:
            if cached is not None:
                mark_recovery_blocked(session_id, cached, recovery_failure)
                await _persist_recovery_marker(session_id, cached)
            else:
                mark_recovery_blocked(session_id, None, recovery_failure)
            return
        # Clean recovery: drop session-level block and clear persisted flags.
        if clear_recovery_blocked(session_id, cached) and cached is not None:
            await _persist_recovery_marker(session_id, cached)

    if session_id in sessions:
        async with state_lock:
            session_access_times[session_id] = time.time()
            cached = sessions.get(session_id)
            await _apply_recovery_outcome(cached)
            return sessions.get(session_id)
    async with state_lock:
        if session_id in sessions:
            session_access_times[session_id] = time.time()
            cached = sessions.get(session_id)
            # Same persist path as the first cache branch (shared helper).
            await _apply_recovery_outcome(cached)
            return sessions[session_id]
        state = await asyncio.to_thread(load_session_from_disk, session_id)
        if state:
            # Clean up stale LLM failure state from a previous process
            if state.wait_input_reason in (
                WaitInputReason.LLM_FAILURE.value,
                WaitInputReason.MARKER_DB_MISSING.value,
            ):
                logger.warning(
                    f"Session {session_id}: clearing stale LLM failure state from previous process"
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
                state.status = TaskStatus.PAUSED
                logger.info(
                    f"Session {session_id}: Analysis paused due to unresolved LLM failure before restart."
                )
                # Save the cleaned-up state so PAUSED status persists
                dumped = state.model_dump(mode="json")
                await asyncio.to_thread(_externalize_and_write, session_id, dumped)
            if recovery_blocked:
                mark_recovery_blocked(session_id, state, recovery_failure)
                await _persist_recovery_marker(session_id, state)
            else:
                if clear_recovery_blocked(session_id, state):
                    await _persist_recovery_marker(session_id, state)
            # Reconcile AnalysisState.snapshots from verified V2 manifests
            # (skip when recovery is blocked — read-only display only)
            if not recovery_blocked:
                try:
                    from backend.services.snapshot_v2.paths import SnapshotV2Paths
                    from backend.services.snapshot_v2.service import SnapshotService
                    svc = SnapshotService(session_id, SnapshotV2Paths.for_session(session_id))
                    state.snapshots = svc.rebuild_snapshots_from_manifests()
                except Exception:
                    logger.debug(
                        "Session %s: snapshot listing reconcile skipped",
                        session_id, exc_info=True,
                    )
            sessions[session_id] = state
            session_access_times[session_id] = time.time()
            return state
        if recovery_blocked:
            # State cannot be loaded — keep session-level marker for guards.
            mark_recovery_blocked(session_id, None, recovery_failure)
        else:
            clear_recovery_blocked(session_id, None)
    return None


def list_saved_sessions() -> List[Dict[str, Any]]:
    sessions_list: List[Dict[str, Any]] = []
    seen_ids: set = set()

    def _parse(session_file: Path, fallback_id: str = "") -> None:
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            sid = data.get("id", fallback_id)
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


async def pause_and_restart_ws(session_id: str, state: AnalysisState, websockets: Dict[str, Any]) -> None:
    state.status = TaskStatus.PAUSED
    await update_session_cache(session_id, state)
    await async_save_session(session_id, state)
    ws = websockets.get(session_id)
    if ws:
        try:
            await ws.close(4000, "workflow_restart")
        except Exception:
            logger.debug("WebSocket close failed during pause-and-restart", exc_info=True)
