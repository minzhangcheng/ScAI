"""Capture one decision snapshot. Part B spec 4.5.

The layer performs no serialisation. It asks for the ordinary save when the
namespace has moved past the newest committed generation, and then copies the
files that save committed, verifying them against the digests it signed.

Two hooks call in. Three of the four decisions are taken at the round boundary,
where the unconditional round-boundary save has just committed a generation
whose pair holds ``next = wait_for_user_input``. The fourth -- confirming the
cell annotation -- is taken inside the executor's ``annotate_clusters`` tool
call, minutes after the last step-boundary save, so it arrives with a
``prepare`` callable that writes the executor's position first and then needs a
save of its own. Both paths run the same code from that point on: the rule is
"capture the newest committed generation, and commit one first if the kernel
has moved past it".

What the pause costs the user is the workspace copy and nothing else. The dump
is pinned by hardlink -- microseconds regardless of size -- and copied after the
locks are released, and so are the inputs, which are append-only and therefore
need no fence at all.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from .gates import retention_rank
from .layout import (
    BLOBS_DIRNAME,
    CONTINUATION_NAME,
    MANIFEST_NAME,
    NOTEBOOKS_DIRNAME,
    PAIR_CHECKPOINT_RELPATH,
    PAIR_EXECUTOR_RELPATH,
    PAIR_STATE_RELPATH,
    RUNTIME_MANIFEST_RELPATH,
    RUNTIME_MANIFEST_SIG_RELPATH,
    RUNTIME_PKL_RELPATH,
    SESSION_STATE_NAME,
    STAGING_DIRNAME,
    WORKSPACE_DIRNAME,
    SnapshotsDisabled,
    failed_record_path,
    parse_snapshot_id,
    snapshot_dir,
    snapshot_id as build_snapshot_id,
    snapshot_storage_root,
    snapshots_root,
    staging_root,
    uploads_root,
)
from .manifest import (
    SnapshotManifestError,
    build_continuation,
    build_snapshot_manifest,
    read_snapshot_manifest,
    write_snapshot_manifest,
)

logger = logging.getLogger(__name__)

_CHUNK = 4 * 1024 * 1024

#: Where a capture holds the committed generation still. Under the session
#: directory rather than under the snapshot root, because a hardlink cannot
#: cross a filesystem and the snapshot root is normally a different mount.
PIN_DIRNAME = "snapshot-pins"

#: Attempts this process is still working on. The periodic sweep reclaims pins
#: and staging, and a pin is a hardlink to a whole dump -- so a sweep that ran
#: during the off-lock generation copy would delete the capture's own inode out
#: from under it. Process-local by design: after a restart nothing is in
#: flight, and everything left behind really is abandoned.
_LIVE_ATTEMPTS: set[str] = set()

#: Channels the capsule must not carry: the analysis state is restored from the
#: paired graph checkpoint, and the message list is rebuilt by the graph.
_UNCAPTURED_CHANNELS = frozenset({"analysis_state", "messages"})


class SnapshotAbandoned(RuntimeError):
    """The capture stopped making progress, or ran past its bound."""

    def __init__(self, reason: str, bytes_written: int = 0):
        self.reason = reason
        self.bytes_written = int(bytes_written)
        super().__init__(f"snapshot capture abandoned: {reason}")


class SnapshotPrecondition(RuntimeError):
    """There is no committed generation to capture at this decision."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"snapshot precondition unmet: {reason}")


async def _notify_snapshot_listing_changed(session_id: str) -> None:
    """Tell the active browser to invalidate its expensive snapshot listing.

    This is deliberately an ephemeral transport event, not a durable analysis
    message. Reconnect and foreground refreshes recover a notification that was
    sent while no browser owned the analysis WebSocket.
    """
    from backend.models.schemas import WebSocketMessage
    from backend.utils.websocket_delivery import send_analysis_json

    try:
        event = WebSocketMessage(
            type="snapshot_listing_changed",
            content={"session_id": session_id},
        )
        await send_analysis_json(session_id, event.model_dump(mode="json"))
    except Exception:
        # A UI invalidation must never turn a completed capture into a failure.
        logger.debug(
            "[%s] could not notify the snapshot listing consumer",
            session_id,
            exc_info=True,
        )


# --------------------------------------------------------------- copy helpers


def _digest_copy(source: Path, target: Path, progress: dict) -> tuple[str, int]:
    """Copy one file, hashing it on the way through.

    Hashing during the copy rather than after it halves the reads on a
    multi-gigabyte dump, and the counter it advances is what the watchdog
    watches: a hung mount stops moving it while a slow one does not.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with open(source, "rb") as reader, open(target, "wb") as writer:
        while True:
            chunk = reader.read(_CHUNK)
            if not chunk:
                break
            writer.write(chunk)
            digest.update(chunk)
            size += len(chunk)
            progress["bytes"] += len(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    return digest.hexdigest(), size


def _digest_file(path: Path, progress: dict) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as reader:
        while True:
            chunk = reader.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            progress["bytes"] += len(chunk)
    return digest.hexdigest()


def _copy_tree(
    root: Path,
    destination: Path,
    arc_prefix: str,
    progress: dict,
    skipped: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Copy a directory, refusing everything that is not an ordinary file.

    Snapshot V2's ``_walk_scope`` rules, restated because that code is gone: no
    symlink is followed at any level, FIFOs, devices and sockets are refused,
    and no entry may resolve outside the root. Refusing symlinked directories
    is what makes containment hold for the whole walk rather than only for its
    leaves; the explicit check is the second lock on the same door.

    A refused entry is recorded rather than dropped. A bundle quietly missing a
    file the analysis wrote is the coverage problem in a new place.
    """
    files: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return files
    real_root = os.path.realpath(root)
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            rel = os.path.relpath(directory, root).replace(os.sep, "/")
            skipped.append({"path": f"{arc_prefix}/{rel}", "reason": "unreadable"})
            continue
        for entry in entries:
            rel = entry.relative_to(root).as_posix()
            arcname = f"{arc_prefix}/{rel}"
            try:
                info = entry.lstat()
            except OSError:
                skipped.append({"path": arcname, "reason": "unreadable"})
                continue
            if stat.S_ISLNK(info.st_mode):
                skipped.append({"path": arcname, "reason": "symlink"})
                continue
            if not os.path.realpath(entry).startswith(real_root + os.sep):
                skipped.append({"path": arcname, "reason": "outside_root"})
                continue
            if stat.S_ISDIR(info.st_mode):
                stack.append(entry)
                continue
            if not stat.S_ISREG(info.st_mode):
                skipped.append({"path": arcname, "reason": "not_a_regular_file"})
                continue
            try:
                digest, size = _digest_copy(entry, destination / rel, progress)
            except OSError:
                logger.warning("Could not capture %s", entry, exc_info=True)
                skipped.append({"path": arcname, "reason": "unreadable"})
                continue
            files[arcname] = {"sha256": digest, "size": size}
    return files


# --------------------------------------------------------------- the watchdog


async def _watched(work: Callable[[], Any], *, progress: dict, session_id: str, what: str):
    """Run blocking copy work, abandoning it if the target stops moving.

    A single duration cap is the wrong instrument on its own: a legitimate
    50 GB workspace on 115 MB/s storage takes minutes, and any cap short enough
    to catch a hang would abort healthy captures on slow targets. So the bound
    is a stall watchdog with an absolute backstop under it.

    The worker is *not* cancelled when a bound expires. A thread blocked in an
    uninterruptible write cannot be, and a killable helper process would not
    help either -- SIGKILL does not release a process in uninterruptible sleep.
    What makes abandonment safe is that the worker can only ever write into
    this attempt's staging directory and never publishes: publication is the
    caller's, and the caller has given up by then.
    """
    import asyncio

    from backend.config import snapshot_max_seconds, snapshot_stall_seconds

    stall_seconds = snapshot_stall_seconds()
    max_seconds = snapshot_max_seconds()
    task = asyncio.create_task(asyncio.to_thread(work), name=f"snapshot-copy:{session_id}")
    started = time.monotonic()
    last_bytes = -1
    last_moved_at = started
    while True:
        done, _pending = await asyncio.wait({task}, timeout=1.0)
        if done:
            return task.result()
        now = time.monotonic()
        moved = int(progress.get("bytes") or 0)
        if moved != last_bytes:
            last_bytes, last_moved_at = moved, now
        if now - last_moved_at >= stall_seconds:
            _abandon(task, session_id, what)
            raise SnapshotAbandoned("stall", moved)
        if now - started >= max_seconds:
            _abandon(task, session_id, what)
            raise SnapshotAbandoned("timeout", moved)


def _abandon(task, session_id: str, what: str) -> None:
    """Let the worker run on, but never let its outcome go unread."""
    logger.warning(
        "[%s] abandoning the %s of a decision snapshot; the worker keeps its "
        "staging directory and publishes nothing", session_id, what,
    )
    task.add_done_callback(
        lambda finished: logger.info(
            "[%s] abandoned snapshot worker finished: %s",
            session_id,
            "error" if finished.exception() else "ok",
        )
    )


# --------------------------------------------------------------- reclamation


def _reclaim(root: Path) -> int:
    """Remove every attempt directory under *root* that is not in flight."""
    removed = 0
    if not root.is_dir():
        return removed
    for entry in root.iterdir():
        if not entry.is_dir() or entry.name in _LIVE_ATTEMPTS:
            continue
        shutil.rmtree(entry, ignore_errors=True)
        removed += 1
    return removed


def _reclaim_session(session_id: str) -> None:
    from backend.config import get_session_dir

    _reclaim(get_session_dir(session_id) / PIN_DIRNAME)
    try:
        _reclaim(staging_root(session_id))
    except SnapshotsDisabled:
        pass


def reclaim_abandoned_capture_work() -> int:
    """Sweep pins and staging left by captures that never published.

    Run from the periodic cleanup as well as from the next capture. A session
    abandoned mid-capture that never reaches another decision would otherwise
    hold its pin until the session was deleted -- and a pin is a hardlink to a
    whole dump, so that is gigabytes, not bookkeeping.
    """
    from backend.config import SESSION_DIR

    removed = 0
    if SESSION_DIR.is_dir():
        for session in SESSION_DIR.iterdir():
            if session.is_dir():
                removed += _reclaim(session / PIN_DIRNAME)
    try:
        root = snapshot_storage_root()
    except SnapshotsDisabled:
        return removed
    if root.is_dir():
        for session in root.iterdir():
            if session.is_dir():
                removed += _reclaim(session / STAGING_DIRNAME)
    return removed


# --------------------------------------------------------------- preconditions


def _persistence(sandbox) -> dict[str, Any]:
    record = getattr(sandbox.notebook_state, "runtime_persistence", None)
    return dict(record) if isinstance(record, dict) else {}


def _has_committed_generation(sandbox, persistence: dict[str, Any]) -> bool:
    return bool(
        sandbox.notebook_state.has_session_dump()
        and persistence.get("committed_source")
        and persistence.get("committed_digest")
    )


def _namespace_has_moved(sandbox, persistence: dict[str, Any]) -> bool:
    """Whether a cell has run since the newest committed generation.

    This is the whole of the freshness rule, and it is what lets one code path
    serve both hooks. At a round boundary the save has just committed, so it is
    normally false and no second dump is written. Inside the executor's
    annotation call it is true, which is exactly the save that decision needs.

    It also closes a race the round-boundary hook would otherwise have: the
    save and this capture are separate awaits, and a chat-delegated cell can
    take the execution lock in between.
    """
    if not _has_committed_generation(sandbox, persistence):
        return True
    return int(persistence.get("current_revision") or 0) != int(
        persistence.get("committed_revision") or 0
    )


def _pair_dir(sandbox, revision: int) -> Path:
    from backend.utils.sandbox_parts.regular_save import GENERATION_PAIR_DIRNAME

    return Path(sandbox.notebook_state.state_dir) / GENERATION_PAIR_DIRNAME / str(revision)


def _require_capturable_generation(sandbox, persistence: dict[str, Any]) -> int:
    if not _has_committed_generation(sandbox, persistence):
        raise SnapshotPrecondition("no_generation")
    if persistence.get("last_attempt_status") == "failed":
        raise SnapshotPrecondition("save_failed")
    revision = int(persistence.get("committed_revision") or 0)
    pair = _pair_dir(sandbox, revision)
    if not (pair / "state.json").is_file() or not (pair / "checkpoints.sqlite").is_file():
        raise SnapshotPrecondition("no_pair")
    return revision


# --------------------------------------------------------------- the pin


def _pin_generation(sandbox, session_id: str, revision: int, pin_dir: Path) -> dict[str, Path]:
    """Hold the committed generation's inodes, whatever the next save does.

    ``os.link`` costs microseconds regardless of size, and every file below is
    published by rename or written to a fresh inode rather than modified in
    place -- so a pinned inode cannot change under the capture, and the next
    commit's ``os.replace`` and the pair pruner's unlink both leave it intact.

    Names are preserved, because ``_verify_pkl_integrity`` finds a dump's
    signature by the dump's own filename.

    A filesystem that refuses the link falls back to copying under the caller's
    lock, which is the cost this design exists to avoid but is still correct.
    """
    from backend.utils.runtime_manifest import (
        runtime_manifest_path,
        runtime_manifest_sig_path,
    )

    states = Path(sandbox.notebook_state.state_dir)
    manifest = runtime_manifest_path(session_id, states_dir=states)
    pair = _pair_dir(sandbox, revision)
    sources: list[tuple[str, Path, bool]] = [
        ("pkl", Path(sandbox.notebook_state.session_path), True),
        ("runtime_manifest", manifest, True),
        ("runtime_manifest_sig", runtime_manifest_sig_path(manifest), True),
        ("pair_state", pair / "state.json", True),
        ("pair_checkpoint", pair / "checkpoints.sqlite", True),
        ("pair_executor", pair / "executor_checkpoint.json", False),
    ]
    pin_dir.mkdir(parents=True, exist_ok=True)
    pinned: dict[str, Path] = {}
    for kind, source, required in sources:
        if not source.is_file():
            if required:
                raise SnapshotPrecondition(f"missing_{kind}")
            continue
        target = pin_dir / source.name
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        pinned[kind] = target
    return pinned


def _verify_pinned_generation(session_id: str, pinned: dict[str, Path], revision: int) -> dict:
    """Three checks, the same three the export path runs before it ships a dump.

    The pin proves the bytes are the ones that were committed; these prove the
    commit was sound. Running them on the pinned copies rather than on the live
    files is deliberate: a check on a file a later save can replace proves
    nothing about what ends up in the bundle.
    """
    from backend.utils.runtime_manifest import read_runtime_manifest
    from backend.utils.sandbox import _integrity_dir, _verify_pkl_integrity

    if not _verify_pkl_integrity(pinned["pkl"], _integrity_dir(session_id)):
        raise SnapshotPrecondition("dump_signature_invalid")
    manifest = read_runtime_manifest(pinned["runtime_manifest"])
    if int(manifest.get("revision") or -1) != int(revision):
        raise SnapshotPrecondition("manifest_revision_mismatch")
    return manifest


# --------------------------------------------------------------- staging


def _stage_session_files(
    *,
    session_id: str,
    sandbox,
    staging: Path,
    capsule: dict[str, Any],
    progress: dict,
    skipped: list[dict[str, str]],
) -> dict[str, Any]:
    """Everything the capture must take while the analysis is held.

    The workspace is here rather than off the lock on purpose. Pinning it the
    way the dump is pinned would cost no pause, but an in-place rewrite by
    analysis code during the gate truncates the shared inode, so the bundle
    could hold half of one decision and half of the next.
    """
    from backend.config import get_session_dir

    files: dict[str, dict[str, Any]] = {}
    session_dir = get_session_dir(session_id)

    payload = json.dumps(capsule, sort_keys=True, separators=(",", ":")).encode("utf-8")
    staging.mkdir(parents=True, exist_ok=True)
    (staging / CONTINUATION_NAME).write_bytes(payload)
    files[CONTINUATION_NAME] = {
        "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload),
    }

    state_file = session_dir / "session.json"
    if not state_file.is_file():
        raise SnapshotPrecondition("no_session_state")
    digest, size = _digest_copy(state_file, staging / SESSION_STATE_NAME, progress)
    files[SESSION_STATE_NAME] = {"sha256": digest, "size": size}

    # Without the blob sidecars the restored state carries dangling
    # "@scai-blob:<hash>" tokens that surface as raw text in the UI.
    files.update(_copy_tree(
        session_dir / BLOBS_DIRNAME, staging / BLOBS_DIRNAME, BLOBS_DIRNAME,
        progress, skipped,
    ))

    # Immutable AnnData checkpoint blobs referenced by the captured catalog.
    # They are copied under the same session-relative layout import/restore
    # already uses; a missing/corrupt blob is reconciled as unavailable rather
    # than silently advertised as loadable.
    files.update(_copy_tree(
        session_dir / "adata-checkpoints" / "blobs",
        staging / "adata-checkpoints" / "blobs",
        "adata-checkpoints/blobs",
        progress,
        skipped,
    ))

    states = Path(sandbox.notebook_state.state_dir)
    for notebook in sorted(states.glob("*.ipynb")):
        digest, size = _digest_copy(
            notebook, staging / NOTEBOOKS_DIRNAME / notebook.name, progress,
        )
        files[f"{NOTEBOOKS_DIRNAME}/{notebook.name}"] = {"sha256": digest, "size": size}

    files.update(_copy_tree(
        Path(sandbox.notebook_state.working_dir),
        staging / WORKSPACE_DIRNAME,
        WORKSPACE_DIRNAME,
        progress,
        skipped,
    ))
    return files


def _stage_uploads(session_id: str, progress: dict) -> dict[str, dict[str, Any]]:
    """The analysis inputs, stored once per session and shared by every snapshot.

    Off the execution lock, unlike the workspace: uploads are append-only --
    the upload route writes a temporary and renames -- so a file already there
    is the same file and nothing can rewrite one under the capture. Holding the
    analysis for tens of gigabytes of input on the first capture of a session
    would be a pause paid for nothing.
    """
    from backend.config import get_session_uploads

    source = get_session_uploads(session_id)
    shared = uploads_root(session_id)
    uploads: dict[str, dict[str, Any]] = {}
    if not source.is_dir():
        return uploads
    for path in sorted(source.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(source).as_posix()
        target = shared / rel
        if target.is_file():
            uploads[rel] = {
                "sha256": _digest_file(target, progress),
                "size": target.stat().st_size,
            }
            continue
        digest, size = _digest_copy(path, target, progress)
        uploads[rel] = {"sha256": digest, "size": size}
    return uploads


def _existing_dump(session_id: str, digest: str, progress: dict) -> Path | None:
    """An earlier snapshot's dump with these exact bytes, re-verified, or None.

    Re-hashed rather than trusted: a digest recorded when that bundle was
    written says what its dump was, not what it is, and linking on that basis
    would make one corrupt file quietly become two.
    """
    if not digest:
        return None
    try:
        root = snapshots_root(session_id)
    except SnapshotsDisabled:
        return None
    if not root.is_dir():
        return None
    for entry in sorted(root.iterdir(), reverse=True):
        if not entry.is_dir():
            continue
        try:
            parse_snapshot_id(entry.name)
            manifest = read_snapshot_manifest(entry / MANIFEST_NAME)
        except (ValueError, SnapshotManifestError):
            continue
        if manifest.get("pkl_sha256") != digest:
            continue
        candidate = entry / RUNTIME_PKL_RELPATH
        if candidate.is_file() and _digest_file(candidate, progress) == digest:
            return candidate
    return None


def _stage_generation(
    *,
    session_id: str,
    staging: Path,
    pinned: dict[str, Path],
    runtime_manifest: dict[str, Any],
    progress: dict,
) -> dict[str, dict[str, Any]]:
    """The dump and its pair, off the critical path.

    The dump is shared with an earlier snapshot of the same session when the
    bytes are identical, which they are whenever two decisions are reached
    without the kernel executing anything in between. The filesystem reference
    counts it, so deleting either snapshot leaves the other readable.
    """
    files: dict[str, dict[str, Any]] = {}
    declared = str(runtime_manifest.get("pkl_sha256") or "")

    target = staging / RUNTIME_PKL_RELPATH
    target.parent.mkdir(parents=True, exist_ok=True)
    shared = _existing_dump(session_id, declared, progress)
    linked = False
    if shared is not None:
        try:
            os.link(shared, target)
            linked = True
        except OSError:
            logger.info("Could not share an identical dump; copying instead")
    if linked:
        digest, size = declared, target.stat().st_size
    else:
        digest, size = _digest_copy(pinned["pkl"], target, progress)
    if digest != declared:
        raise SnapshotPrecondition("dump_digest_mismatch")
    files[RUNTIME_PKL_RELPATH] = {"sha256": digest, "size": size}

    for kind, relpath in (
        ("runtime_manifest", RUNTIME_MANIFEST_RELPATH),
        ("runtime_manifest_sig", RUNTIME_MANIFEST_SIG_RELPATH),
        ("pair_state", PAIR_STATE_RELPATH),
        ("pair_checkpoint", PAIR_CHECKPOINT_RELPATH),
        ("pair_executor", PAIR_EXECUTOR_RELPATH),
    ):
        source = pinned.get(kind)
        if source is None:
            continue
        digest, size = _digest_copy(source, staging / relpath, progress)
        files[relpath] = {"sha256": digest, "size": size}
    return files


# --------------------------------------------------------------- the capture


def _record_failure(session_id: str, snapshot: str | None, reason: str, written: int) -> None:
    """Say which decision has no restore point, and why.

    Unsigned, and read by nothing that restores: a record that could influence
    a restore would need the protection the manifest has. This exists so the
    listing can show a gap rather than an absence.
    """
    from backend.utils.files import atomic_write_bytes

    if not snapshot:
        return
    try:
        atomic_write_bytes(
            failed_record_path(session_id, snapshot),
            json.dumps({
                "reason": reason,
                "at": datetime.now(timezone.utc).isoformat(),
                "bytes_written": int(written),
            }, sort_keys=True).encode("utf-8"),
        )
    except (OSError, ValueError, SnapshotsDisabled):
        logger.warning(
            "[%s] could not record the failed capture %s", session_id, snapshot,
            exc_info=True,
        )


def _capsule(analysis_state, workflow_state, gate: str) -> dict[str, Any]:
    phase = getattr(analysis_state, "analysis_phase", None)
    keys = [
        key for key in (workflow_state or {})
        if key not in _UNCAPTURED_CHANNELS and not str(key).startswith("_")
    ]
    return build_continuation(
        gate=gate,
        analysis_phase=str(getattr(phase, "value", phase) or ""),
        current_round=int(getattr(analysis_state, "current_round", 0) or 0),
        state_version=int(getattr(analysis_state, "state_version", 0) or 0),
        workflow_state_keys=keys,
    )


def _provisional_id(analysis_state, sandbox, gate: str) -> str | None:
    """The identifier a failure is recorded under, or None if it cannot be named.

    The revision is the one committed when the failure is noticed, which is not
    always the one the capture would have produced. That is why the listing
    drops a failure record whose identifier also names a published bundle
    rather than showing the same decision as both captured and missing.
    """
    try:
        return build_snapshot_id(
            round_number=int(getattr(analysis_state, "current_round", 0) or 0),
            gate=gate,
            revision=int(_persistence(sandbox).get("committed_revision") or 0),
        )
    except (ValueError, AttributeError, TypeError):
        return None


async def capture_decision_snapshot(
    workflow,
    analysis_state,
    *,
    gate: str,
    workflow_state: dict | None = None,
    prepare: Callable[[], Awaitable[None]] | None = None,
) -> str | None:
    """Capture the point immediately before one decision. Never raises.

    Spec 6.3's third invariant: a capture that fails, stalls or times out
    leaves the analysis exactly as it would have been. The honest bound is on
    time, not on outcome -- a slow target lengthens the moment before the
    dialog opens by the workspace copy, and the watchdog is what stops
    "lengthens" from becoming "forever".
    """
    from backend.config import snapshots_enabled

    if not snapshots_enabled():
        return None
    sandbox = getattr(workflow, "sandbox", None)
    if sandbox is None or getattr(sandbox, "km", None) is None:
        return None
    session_id = str(getattr(analysis_state, "id", "") or "")
    if not session_id:
        return None

    attempt = uuid.uuid4().hex
    progress = {"bytes": 0}
    _LIVE_ATTEMPTS.add(attempt)
    captured: str | None = None
    try:
        captured = await _capture(
            workflow=workflow, sandbox=sandbox, analysis_state=analysis_state,
            session_id=session_id, gate=gate, workflow_state=workflow_state,
            prepare=prepare, attempt=attempt, progress=progress,
        )
    except SnapshotAbandoned as abandoned:
        _record_failure(
            session_id, _provisional_id(analysis_state, sandbox, gate),
            abandoned.reason, abandoned.bytes_written,
        )
    except SnapshotPrecondition as unmet:
        logger.info(
            "[%s] no decision snapshot at %s: %s", session_id, gate, unmet.reason,
        )
        _record_failure(
            session_id, _provisional_id(analysis_state, sandbox, gate),
            unmet.reason, progress["bytes"],
        )
    except Exception:
        logger.warning(
            "[%s] decision snapshot at %s failed", session_id, gate, exc_info=True,
        )
        _record_failure(
            session_id, _provisional_id(analysis_state, sandbox, gate),
            "error", progress["bytes"],
        )
    finally:
        _LIVE_ATTEMPTS.discard(attempt)
        _reclaim_session(session_id)
        await _notify_snapshot_listing_changed(session_id)
    return captured


async def _capture(
    *, workflow, sandbox, analysis_state, session_id, gate, workflow_state,
    prepare, attempt, progress,
) -> str:
    from backend.config import get_session_dir
    from backend.constants import VERSION
    from backend.services.save_cadence import runtime_save_due
    from backend.services.session_store import async_save_session
    from backend.utils.files import fsync_directory

    started = time.monotonic()
    _reclaim_session(session_id)
    staging = staging_root(session_id) / attempt
    pin_dir = get_session_dir(session_id) / PIN_DIRNAME / attempt
    skipped: list[dict[str, str]] = []

    async with sandbox._exec_lock:
        if prepare is not None:
            # The executor's position, written before the save so the pair the
            # commit captures is the one this decision belongs to.
            await prepare()
        persistence = _persistence(sandbox)
        # The cadence module owns the question of whether a boundary saves; the
        # revision test owns whether there is anything new to save. A decision
        # boundary answers the first unconditionally, so in practice this saves
        # exactly when the kernel has moved past its newest generation.
        needs_generation = _namespace_has_moved(sandbox, persistence)
        if needs_generation and runtime_save_due(sandbox, "decision_boundary"):
            if await sandbox._save_session_locked() is not True:
                raise SnapshotPrecondition("save_failed")
            sandbox.sync_runtime_persistence_state(analysis_state)
            persistence = _persistence(sandbox)

        revision = _require_capturable_generation(sandbox, persistence)
        # Forced: session.json is written through a five-second throttle that
        # RUNNING does not bypass, so an unforced write would capture a
        # projection up to five seconds behind this decision.  Reconcile blob
        # availability first so the captured state never advertises an
        # available checkpoint whose bytes are missing/corrupt.
        try:
            from backend.services.adata_checkpoints import (
                reconcile_catalog_availability,
            )

            reconcile_catalog_availability(
                analysis_state, verify_digests=True, runtime_available=True,
            )
        except Exception:
            logger.exception(
                "decision snapshot catalog reconciliation failed"
            )
        await async_save_session(session_id, analysis_state, force=True)

        pinned = _pin_generation(sandbox, session_id, revision, pin_dir)
        runtime_manifest = _verify_pinned_generation(session_id, pinned, revision)

        capsule = _capsule(analysis_state, workflow_state, gate)
        files = await _watched(
            lambda: _stage_session_files(
                session_id=session_id, sandbox=sandbox, staging=staging,
                capsule=capsule, progress=progress, skipped=skipped,
            ),
            progress=progress, session_id=session_id, what="session capture",
        )

    uploads = await _watched(
        lambda: _stage_uploads(session_id, progress),
        progress=progress, session_id=session_id, what="input capture",
    )
    files.update(await _watched(
        lambda: _stage_generation(
            session_id=session_id, staging=staging, pinned=pinned,
            runtime_manifest=runtime_manifest, progress=progress,
        ),
        progress=progress, session_id=session_id, what="generation copy",
    ))

    snapshot = build_snapshot_id(
        round_number=int(getattr(analysis_state, "current_round", 0) or 0),
        gate=gate,
        revision=revision,
    )
    manifest = build_snapshot_manifest(
        snapshot_id=snapshot,
        source_session_id=session_id,
        source_session_name=str(getattr(analysis_state, "name", "") or ""),
        source_lifecycle_generation=int(
            getattr(analysis_state, "lifecycle_generation", 0) or 0
        ),
        scai_version=VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        round_number=int(getattr(analysis_state, "current_round", 0) or 0),
        gate=gate,
        retention_rank=retention_rank(gate),
        runtime_revision=revision,
        pkl_digest=str(runtime_manifest.get("pkl_sha256") or ""),
        files=files,
        uploads=uploads,
        coverage={
            "saved_names": len(runtime_manifest.get("saved_names") or []),
            "skipped_names": runtime_manifest.get("skipped_names") or [],
            "carried_definitions": runtime_manifest.get("carried_definitions") or [],
        },
        executor_checkpoint_present=PAIR_EXECUTOR_RELPATH in files,
        bytes_total=sum(int(entry["size"]) for entry in files.values()),
        capture_seconds=time.monotonic() - started,
        skipped_entries=skipped,
    )
    # Written last, so a bundle that appears in the listing is one whose
    # manifest describes every file already beside it.
    write_snapshot_manifest(staging / MANIFEST_NAME, manifest)

    published = snapshot_dir(session_id, snapshot)
    published.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(published, ignore_errors=True)
    os.replace(staging, published)
    fsync_directory(published.parent)
    shutil.rmtree(pin_dir, ignore_errors=True)
    # A decision that failed once and succeeded on a later attempt is not a gap.
    failed_record_path(session_id, snapshot).unlink(missing_ok=True)
    logger.info(
        "[%s] captured decision snapshot %s (%d bytes, %.1fs)",
        session_id, snapshot, manifest["bytes_total"], manifest["capture_seconds"],
    )
    # After publication, and never before it: a capture that failed must not
    # cost an existing restore point, and the one just written is the caller's.
    from .prune import prune_for_free_space

    try:
        prune_for_free_space(keep_session=session_id, keep=snapshot)
    except Exception:
        logger.warning(
            "[%s] could not reclaim space after a capture", session_id, exc_info=True,
        )
    return snapshot
