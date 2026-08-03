"""Persistent owner-scoped lifecycle generations for session mutation safety."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import threading
import weakref
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

from backend.config import REPORTS_DIR, SESSION_DIR, STATE_DIR, UPLOAD_DIR, WORKSPACE_DIR
from backend.utils.files import (
    atomic_write_bytes,
    legacy_session_state_paths,
    validate_session_id,
)


class SessionGoneError(RuntimeError):
    """The requested generation was deleted and must not be resurrected."""

    def __init__(self, session_id: str, owner_id: str = "", generation: int = 0):
        self.session_id = session_id
        self.owner_id = owner_id
        self.generation = generation
        super().__init__(f"session {session_id} generation {generation} is deleted")


class SessionGenerationError(RuntimeError):
    """A stale in-memory object attempted to mutate another generation."""


_locks: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    weakref.WeakValueDictionary[str, asyncio.Lock],
] = weakref.WeakKeyDictionary()
_locks_guard = threading.Lock()
_file_lock_state = threading.local()
_T = TypeVar("_T")


def get_lifecycle_lock(session_id: str) -> asyncio.Lock:
    """Return this running event loop's lock for one session generation.

    ``asyncio.Lock`` binds to its loop only once an acquisition contends. A
    process-wide session-only cache therefore appears safe until a waiter binds
    the lock, after which a later ``asyncio.run`` receives a lock owned by the
    closed loop. Both registry levels are weak so completed loops and idle
    locks are collectible rather than accumulating per-session metadata.
    """
    validate_session_id(session_id)
    loop = asyncio.get_running_loop()
    with _locks_guard:
        loop_locks = _locks.get(loop)
        if loop_locks is None:
            loop_locks = weakref.WeakValueDictionary()
            _locks[loop] = loop_locks
        lock = loop_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            loop_locks[session_id] = lock
        return lock


def snapshot_operation_active(session_id: str) -> bool:
    """Return whether capture/restore owns or is reserving the session.

    Producer publication checks this while holding ``get_lifecycle_lock``;
    every production safe-point establishment uses
    ``establish_snapshot_operation`` and therefore takes the same barrier.
    Neither side can observe a free slot and publish concurrently with the
    other in this process.

    The registry and lock are process-local. Deployments must keep one worker
    authoritative for a session/data root; cross-process coordination would
    require a shared lease service.
    """
    from backend.services.snapshot_v2.safe_point import (
        get_snapshot_safe_point_manager,
    )

    safe_point = get_snapshot_safe_point_manager()
    return bool(
        safe_point.has_pending_request(session_id)
        or safe_point.is_paused(session_id)
    )


async def establish_snapshot_operation(
    session_id: str,
    reserve: Callable[[], Awaitable[_T]],
    *,
    validate: Callable[[], Any | Awaitable[Any]] | None = None,
    publish: Callable[[_T, int], Any] | None = None,
) -> _T:
    """Establish one pending/acquired safe point at the producer barrier.

    The critical section is intentionally narrow: generation validation,
    optional owner validation, the non-blocking SafePointManager mutation, and
    (when supplied) synchronous publication of the operation's owner handle.
    It must never contain acknowledgement waits, capture work, general session
    persistence, or release. Producers and deletion take the same lifecycle
    lock, so ``publish`` closes the reservation-to-registry visibility gap.
    """
    async with get_lifecycle_lock(session_id):
        assert_session_generation_active(session_id)
        if validate is not None:
            checked = validate()
            if inspect.isawaitable(checked):
                await checked
        reservation = await reserve()
        if (
            publish is not None
            and reservation is not None
            and reservation is not False
        ):
            record = read_lifecycle_record(session_id)
            generation = int((record or {}).get("generation") or 0)
            published = publish(reservation, generation)
            if inspect.isawaitable(published):
                raise TypeError(
                    "snapshot operation publication must be synchronous"
                )
        return reservation


def reset_lifecycle_locks() -> None:
    """Drop loop-local lock metadata during tests or process teardown."""
    with _locks_guard:
        _locks.clear()


@contextmanager
def lifecycle_file_lock(session_id: str):
    """Cross-thread/process guard used by synchronous persistence entrypoints."""
    import fcntl

    validate_session_id(session_id)
    held = getattr(_file_lock_state, "held", None)
    if held is None:
        held = {}
        _file_lock_state.held = held
    if held.get(session_id, 0) > 0:
        held[session_id] += 1
        try:
            yield
        finally:
            held[session_id] -= 1
        return

    path = SESSION_DIR / ".lifecycle" / ".locks" / f"{session_id}.lock"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        held[session_id] = 1
        try:
            yield
        finally:
            held.pop(session_id, None)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _owner_key(owner_id: str) -> str:
    return hashlib.sha256(str(owner_id).encode("utf-8")).hexdigest()


def _record_path(session_id: str, owner_id: str) -> Path:
    return SESSION_DIR / ".lifecycle" / _owner_key(owner_id) / f"{session_id}.json"


def _read_path(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_lifecycle_record(session_id: str) -> dict[str, Any] | None:
    """Return the stable lifecycle record without trusting the session directory."""
    validate_session_id(session_id)
    root = SESSION_DIR / ".lifecycle"
    if not root.is_dir():
        return None
    records = []
    for path in root.glob(f"*/{session_id}.json"):
        record = _read_path(path)
        if record and record.get("session_id") == session_id:
            records.append(record)
    if len(records) > 1:
        raise SessionGenerationError(
            f"multiple lifecycle owner records exist for session {session_id}"
        )
    return records[0] if records else None


def list_lifecycle_records(*, status: str | None = None) -> list[dict[str, Any]]:
    """List persistent lifecycle records for maintenance sweepers."""
    root = SESSION_DIR / ".lifecycle"
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in root.glob("*/*.json"):
        record = _read_path(path)
        if not record or not record.get("session_id"):
            continue
        if status is not None and record.get("status") != status:
            continue
        records.append(record)
    return records


def _write_record(record: dict[str, Any]) -> None:
    path = _record_path(str(record["session_id"]), str(record["owner_id"]))
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write_bytes(
        path,
        json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8"),
    )
    path.chmod(0o600)


def _deleted_generations(record: dict[str, Any] | None) -> list[int]:
    """Return the monotonic generations that this installation deleted."""
    if not record:
        return []
    values = {
        int(value)
        for value in (record.get("deleted_generations") or [])
        if isinstance(value, int) and value > 0
    }
    if record.get("status") == "deleted":
        generation = int(record.get("generation") or 0)
        if generation > 0:
            values.add(generation)
    return sorted(values)


def claim_import_generation(
    session_id: str,
    owner_id: str,
    session_dir: Path,
) -> int | None:
    """Atomically reserve an import destination and its new generation.

    A deleted ID may be reused only by its original owner, and only as the next
    lifecycle generation.  The ``importing`` record is written under the same
    cross-process lock as the directory claim so a tombstone sweeper cannot
    mistake the new tree for the deleted generation it was cleaning.
    """
    validate_session_id(session_id)
    # `cleanup_complete` is written immediately before the cleanup owner drops
    # its flock. A same-ID import must therefore acquire that same lease before
    # trusting the flag; otherwise it can claim g+1 in the tiny completion/
    # unlock window and its own failure cleanup collides with g's live lease.
    # Local import avoids a module-load cycle (session_deletion imports here).
    from backend.services.session_deletion import (
        _try_acquire_cleanup_lease_sync,
    )

    cleanup_lease = _try_acquire_cleanup_lease_sync(session_id)
    if cleanup_lease is None:
        return None
    try:
        with lifecycle_file_lock(session_id):
            record = read_lifecycle_record(session_id)
            if record is not None:
                if record.get("status") != "deleted":
                    return None
                if str(record.get("owner_id") or "") != owner_id:
                    return None
                # A tombstone is reusable only after every old-generation
                # producer, side surface, filesystem tree, and mutation-lock slot
                # has been cleaned. Missing is deliberately treated as incomplete:
                # legacy records need one idempotent sweeper pass before same-ID
                # reuse can be proven safe.
                if record.get("cleanup_complete") is not True:
                    return None
                generation = int(record.get("generation") or 0) + 1
                deleted_generations = _deleted_generations(record)
            else:
                generation = 1
                deleted_generations = []

            try:
                session_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
            except FileExistsError:
                return None
            except OSError:
                return None

            now = datetime.now(timezone.utc).isoformat()
            importing = {
                "session_id": session_id,
                "owner_id": owner_id,
                "generation": generation,
                "status": "importing",
                "created_at": now,
                "deleted_at": None,
                "deleted_generations": deleted_generations,
                "import_started_at": now,
            }
            try:
                _write_record(importing)
            except BaseException:
                # The directory is still empty at this boundary.  Release the
                # filesystem claim if the durable lifecycle reservation failed.
                try:
                    session_dir.rmdir()
                except OSError:
                    pass
                raise
            return generation
    finally:
        cleanup_lease.release()


def bind_import_generation(state) -> int:
    """Bind an imported AnalysisState to its server-reserved generation."""
    with lifecycle_file_lock(state.id):
        record = read_lifecycle_record(state.id)
        if record is None or record.get("status") != "importing":
            raise SessionGenerationError(
                f"session {state.id} has no active import reservation"
            )
        if str(record.get("owner_id") or "") != state.user_id:
            raise SessionGenerationError(
                "import owner does not match lifecycle owner"
            )
        state.lifecycle_generation = int(record.get("generation") or 0)
        return state.lifecycle_generation


def commit_import_generation(state) -> int:
    """Publish a fully saved import as the active lifecycle generation."""
    with lifecycle_file_lock(state.id):
        record = read_lifecycle_record(state.id)
        if record is None or record.get("status") != "importing":
            raise SessionGenerationError(
                f"session {state.id} has no import generation to commit"
            )
        if record.get("import_cancel_requested") is True:
            raise SessionGenerationError(
                f"session {state.id} import generation was cancelled"
            )
        generation = int(record.get("generation") or 0)
        if str(record.get("owner_id") or "") != state.user_id:
            raise SessionGenerationError(
                "import owner does not match lifecycle owner"
            )
        if int(getattr(state, "lifecycle_generation", 0) or 0) != generation:
            raise SessionGenerationError("imported session generation is stale")
        active = dict(record)
        active.update({
            "status": "active",
            "deleted_at": None,
            "import_completed_at": datetime.now(timezone.utc).isoformat(),
        })
        _write_record(active)
        return generation


def request_import_generation_cancellation(
    session_id: str,
    owner_id: str,
) -> int | None:
    """Durably prevent one importing generation from being committed.

    The record deliberately remains ``importing`` until its producer exits and
    calls ``cleanup_failed_import_generation`` under the cleanup lease. Turning
    it into an ordinary tombstone here would let the background sweeper delete
    its directory while extraction is still writing to it.
    """
    validate_session_id(session_id)
    with lifecycle_file_lock(session_id):
        record = read_lifecycle_record(session_id)
        if record is None or record.get("status") != "importing":
            return None
        if str(record.get("owner_id") or "") != owner_id:
            raise SessionGenerationError(
                "import owner does not match lifecycle owner"
            )
        cancelled = dict(record)
        cancelled.update({
            "import_cancel_requested": True,
            "import_cancel_requested_at": datetime.now(timezone.utc).isoformat(),
        })
        _write_record(cancelled)
        return int(cancelled.get("generation") or 0)


def abandon_import_generation(session_id: str) -> bool:
    """Tombstone a failed import reservation before its tree is removed."""
    validate_session_id(session_id)
    with lifecycle_file_lock(session_id):
        record = read_lifecycle_record(session_id)
        if record is None:
            # Compatibility with pre-lifecycle/faked allocators: the caller's
            # exclusive directory claim is still safe to release.
            return True
        if record.get("status") != "importing":
            return False
        generation = int(record.get("generation") or 0)
        deleted_generations = set(_deleted_generations(record))
        if generation > 0:
            deleted_generations.add(generation)
        failed = dict(record)
        failed.update({
            "status": "deleted",
            "deleted_at": datetime.now(timezone.utc).isoformat(),
            "deleted_generations": sorted(deleted_generations),
            "import_failed": True,
            "cleanup_complete": False,
            "cleanup_completed_at": None,
        })
        _write_record(failed)
        return True


def register_new_generation(state) -> int:
    """Register a server-issued ID; collisions and tombstones fail closed."""
    with lifecycle_file_lock(state.id):
        return _register_new_generation_locked(state)


def _register_new_generation_locked(state) -> int:
    record = read_lifecycle_record(state.id)
    if record is not None:
        if record.get("status") == "deleted":
            raise SessionGoneError(
                state.id,
                str(record.get("owner_id") or ""),
                int(record.get("generation") or 0),
            )
        raise SessionGenerationError(f"session id collision: {state.id}")
    # A pre-lifecycle session can still occupy the generated identifier.  New
    # creation must never silently adopt or overwrite it: only the load/import
    # paths are allowed to call ``adopt_persisted_generation``.
    legacy_candidates = (
        SESSION_DIR / state.id,
        SESSION_DIR / f"{state.id}.json",
        UPLOAD_DIR / state.id,
        WORKSPACE_DIR / state.id,
        REPORTS_DIR / state.id,
        *legacy_session_state_paths(STATE_DIR, state.id),
    )
    if any(path.exists() for path in legacy_candidates):
        raise SessionGenerationError(
            f"session id collision with persisted data: {state.id}"
        )
    generation = 1
    state.lifecycle_generation = generation
    _write_record({
        "session_id": state.id,
        "owner_id": state.user_id,
        "generation": generation,
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "deleted_at": None,
        "deleted_generations": [],
    })
    return generation


async def register_new_session_generation(state) -> int:
    """Linearize creation before any directory, cache, or metadata write."""
    async with get_lifecycle_lock(state.id):
        with lifecycle_file_lock(state.id):
            return _register_new_generation_locked(state)


def adopt_persisted_generation(state) -> int:
    """Adopt an existing disk session after restart, never a tombstone."""
    with lifecycle_file_lock(state.id):
        return _adopt_persisted_generation_locked(state)


def _adopt_persisted_generation_locked(state) -> int:
    record = read_lifecycle_record(state.id)
    if record is None:
        generation = max(1, int(getattr(state, "lifecycle_generation", 0) or 0))
        state.lifecycle_generation = generation
        _write_record({
            "session_id": state.id,
            "owner_id": state.user_id,
            "generation": generation,
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "deleted_at": None,
            "deleted_generations": [],
        })
        return generation
    owner_id = str(record.get("owner_id") or "")
    generation = int(record.get("generation") or 0)
    if record.get("status") == "deleted":
        raise SessionGoneError(state.id, owner_id, generation)
    if owner_id != state.user_id:
        raise SessionGenerationError("persisted session owner does not match lifecycle owner")
    state_generation = int(getattr(state, "lifecycle_generation", 0) or 0)
    if state_generation not in (0, generation):
        if state_generation in _deleted_generations(record):
            raise SessionGoneError(state.id, owner_id, state_generation)
        raise SessionGenerationError("persisted session generation is stale")
    state.lifecycle_generation = generation
    return generation


def assert_generation_writable(session_id: str, state) -> None:
    """Reject stale saves after deletion or generation replacement."""
    with lifecycle_file_lock(session_id):
        _assert_generation_writable_locked(session_id, state)


def _assert_generation_writable_locked(session_id: str, state) -> None:
    state_id = str(getattr(state, "id", "") or "")
    if state_id != session_id:
        raise SessionGenerationError(
            f"session id mismatch: expected {session_id}, state carries {state_id or '<empty>'}"
        )
    record = read_lifecycle_record(session_id)
    if record is None:
        _adopt_persisted_generation_locked(state)
        record = read_lifecycle_record(session_id)
    if record is None:
        raise SessionGenerationError(
            f"lifecycle record was not created for session {session_id}"
        )
    owner_id = str(record.get("owner_id") or "")
    generation = int(record.get("generation") or 0)
    if record.get("status") == "deleted":
        raise SessionGoneError(session_id, owner_id, generation)
    if owner_id != state.user_id:
        raise SessionGenerationError("session owner does not match lifecycle owner")
    state_generation = int(getattr(state, "lifecycle_generation", 0) or 0)
    if state_generation == 0:
        state.lifecycle_generation = generation
        state_generation = generation
    if state_generation != generation:
        if state_generation in _deleted_generations(record):
            raise SessionGoneError(session_id, owner_id, state_generation)
        raise SessionGenerationError("stale session generation cannot be saved")


def assert_dump_writable(session_id: str, dumped: dict[str, Any]) -> None:
    """Generation guard for already-serialized AnalysisState payloads."""
    class _StateView:
        id = str(dumped.get("id") or "")
        user_id = str(dumped.get("user_id") or "")
        lifecycle_generation = int(dumped.get("lifecycle_generation") or 0)

    view = _StateView()
    assert_generation_writable(session_id, view)
    dumped["lifecycle_generation"] = view.lifecycle_generation


def tombstone_generation(session_id: str, owner_id: str, generation: int) -> dict[str, Any]:
    """Persist deletion before removing any mutable session artifacts."""
    with lifecycle_file_lock(session_id):
        return _tombstone_generation_locked(
            session_id, owner_id, generation
        )


def _tombstone_generation_locked(
    session_id: str,
    owner_id: str,
    generation: int,
) -> dict[str, Any]:
    record = read_lifecycle_record(session_id)
    if record is not None:
        record_owner = str(record.get("owner_id") or "")
        if record_owner != owner_id:
            raise SessionGenerationError("session owner does not match lifecycle owner")
        current_generation = int(record.get("generation") or generation or 1)
        requested_generation = int(generation or 0)
        if (
            record.get("status") != "deleted"
            and requested_generation not in (0, current_generation)
        ):
            if requested_generation in _deleted_generations(record):
                raise SessionGoneError(
                    session_id, record_owner, requested_generation
                )
            raise SessionGenerationError(
                "stale session generation cannot be deleted"
            )
        generation = current_generation
    else:
        generation = max(1, int(generation or 0))
    deleted_generations = set(_deleted_generations(record))
    deleted_generations.add(generation)
    tombstone = {
        "session_id": session_id,
        "owner_id": owner_id,
        "generation": generation,
        "status": "deleted",
        "created_at": (record or {}).get("created_at"),
        "deleted_at": datetime.now(timezone.utc).isoformat(),
        "deleted_generations": sorted(deleted_generations),
        "cleanup_complete": bool((record or {}).get("cleanup_complete"))
        if (record or {}).get("status") == "deleted"
        else False,
        "cleanup_completed_at": (record or {}).get("cleanup_completed_at")
        if (record or {}).get("status") == "deleted"
        else None,
    }
    _write_record(tombstone)
    return tombstone


def complete_tombstone_cleanup(
    session_id: str,
    owner_id: str,
    generation: int,
) -> bool:
    """Publish proof that one exact deleted generation is fully cleaned.

    Returns ``False`` when the lifecycle record moved to another owner,
    generation, or status while cleanup was in progress. Callers must treat
    that as a stale cleanup and must not touch any current-generation surface.
    """
    validate_session_id(session_id)
    with lifecycle_file_lock(session_id):
        record = read_lifecycle_record(session_id)
        if not record or record.get("status") != "deleted":
            return False
        if str(record.get("owner_id") or "") != owner_id:
            return False
        if int(record.get("generation") or 0) != int(generation or 0):
            return False
        completed = dict(record)
        completed.update({
            "cleanup_complete": True,
            "cleanup_completed_at": datetime.now(timezone.utc).isoformat(),
        })
        _write_record(completed)
        return True


def assert_session_generation_active(
    session_id: str,
    *,
    expected_generation: int | None = None,
) -> None:
    """Reject commits after tombstone or lifecycle-generation replacement.

    ``expected_generation`` binds a long-running producer to the generation it
    observed when its owner handle was published. Generation ``0`` denotes the
    legacy/no-record state and remains valid only while no lifecycle record has
    appeared; it must not silently adopt a later same-ID generation.
    """
    with lifecycle_file_lock(session_id):
        record = read_lifecycle_record(session_id)
        if record and record.get("status") == "deleted":
            raise SessionGoneError(
                session_id,
                str(record.get("owner_id") or ""),
                int(record.get("generation") or 0),
            )
        if expected_generation is None:
            return
        current_generation = int((record or {}).get("generation") or 0)
        if current_generation != int(expected_generation or 0):
            if int(expected_generation or 0) in _deleted_generations(record):
                raise SessionGoneError(
                    session_id,
                    str((record or {}).get("owner_id") or ""),
                    int(expected_generation or 0),
                )
            raise SessionGenerationError(
                f"session {session_id} lifecycle generation changed from "
                f"{int(expected_generation or 0)} to {current_generation}"
            )


def tombstone_visible_to_owner(session_id: str, owner_id: str | None) -> bool:
    record = read_lifecycle_record(session_id)
    return bool(
        record
        and record.get("status") == "deleted"
        and owner_id
        and record.get("owner_id") == owner_id
    )


def is_session_tombstoned(session_id: str) -> bool:
    """Return whether every remaining producer must discard this generation."""
    record = read_lifecycle_record(session_id)
    return bool(record and record.get("status") == "deleted")
