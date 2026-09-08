"""Capture one verified runtime/state/checkpoint generation for portable export."""
from __future__ import annotations

import hashlib
import json
import logging
import tarfile
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ResumableExportUnavailable(RuntimeError):
    """A results-only export remains available when this capture cannot resume."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(detail)


def live_sandbox(session_id: str) -> Optional[Any]:
    from backend.state import chat_sandboxes, workflows

    owner = workflows.get(session_id)
    sandbox = getattr(owner, "sandbox", None)
    if sandbox is not None and getattr(sandbox, "km", None) is not None:
        return sandbox
    sandbox = chat_sandboxes.get(session_id)
    if sandbox is not None and getattr(sandbox, "km", None) is not None:
        return sandbox
    return None


def _refuse(reason: str, detail: str):
    return ResumableExportUnavailable(
        reason, detail + " Export without runtime to keep the results, or retry when the session is idle.",
    )


def _pairing(session_id: str, state) -> dict:
    from backend.utils.runtime_pairing import capture_pairing

    try:
        pairing = capture_pairing(session_id, state)
        if pairing is None:
            raise ValueError("state pairing is absent")
        return pairing
    except Exception as exc:
        raise _refuse("boundary_unavailable", "The state/checkpoint identity could not be read.") from exc


def _digest(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def verify_committed_generation(session_id: str) -> dict:
    """Require an existing signed dump and its matching signed manifest."""
    from backend.utils.runtime_apply import _live_pkl, verify_live_pkl_or_raise
    from backend.utils.runtime_manifest import read_runtime_manifest, runtime_manifest_path

    try:
        pkl = _live_pkl(session_id)
        if not pkl.is_file():
            raise ValueError("committed runtime is absent")
        verify_live_pkl_or_raise(session_id)
        manifest = read_runtime_manifest(runtime_manifest_path(session_id))
        with pkl.open("rb") as stream:
            if _digest(stream) != manifest["pkl_sha256"]:
                raise ValueError("runtime digest does not match its manifest")
        return manifest
    except Exception as exc:
        raise _refuse("generation_unverified", "The saved runtime is absent or failed its integrity check.") from exc


def assert_boundary_held(session_id: str, state, before: dict) -> None:
    if before != _pairing(session_id, state):
        raise _refuse("boundary_moved", "The analysis advanced while its archive was being built.")


async def _flush_locked(sandbox, state) -> None:
    try:
        if await sandbox._save_session_locked() is not True:
            raise ValueError("runtime save did not commit")
        sandbox.sync_runtime_persistence_state(state)
        # The notebook is part of the same frozen execution boundary.
        sandbox.notebook_state.save()
    except Exception as exc:
        raise _refuse("flush_failed", "The current runtime and notebook could not be saved.") from exc


@asynccontextmanager
async def frozen_resumable_export(session_id: str, state):
    """Hold execution/save/shutdown exclusion until archive capture has drained.

    Cold exports reserve the existing runtime-start barrier, without starting a
    kernel. Lifecycle locks are held only for reservation/release; state saves
    and export cancellation need to acquire those locks independently.
    """
    from backend.services.session_lifecycle import get_lifecycle_lock
    from backend.state import (
        chat_sandboxes, workflows, report_workflows, initializing_sessions,
        claim_initializing_session, release_initializing_session,
    )
    from backend.utils.asyncio_tasks import await_with_deferred_cancellation
    import asyncio

    token = None
    async with get_lifecycle_lock(session_id):
        sandbox = live_sandbox(session_id)
        if sandbox is None:
            if (session_id in initializing_sessions or session_id in workflows
                    or session_id in report_workflows or session_id in chat_sandboxes):
                raise _refuse("runtime_busy", "The runtime is starting or closing.")
            token = claim_initializing_session(session_id)
    try:
        if sandbox is None:
            yield await asyncio.to_thread(_capture_descriptor, session_id, state)
        else:
            async with sandbox._exec_lock:
                if (live_sandbox(session_id) is not sandbox
                        or not getattr(sandbox, "_initialized", False)
                        or getattr(sandbox, "_starting", False)
                        or getattr(sandbox, "_restarting", False)
                        or getattr(sandbox, "_kernel_dead", False)
                        or getattr(sandbox, "_session_restore_failed", False)):
                    raise _refuse("runtime_busy", "The live runtime changed before capture.")
                await _flush_locked(sandbox, state)
                yield await asyncio.to_thread(_capture_descriptor, session_id, state)
    finally:
        if token is not None:
            async def release():
                async with get_lifecycle_lock(session_id):
                    release_initializing_session(session_id, token)
            await await_with_deferred_cancellation(asyncio.create_task(release()))


def _capture_descriptor(session_id: str, state) -> dict:
    """Verify the generation and fingerprint the notebooks. Blocking on purpose.

    Two full passes over the dump -- the HMAC and the SHA-256 the manifest must
    match -- plus a digest per notebook. Measured at 0.92 s for a 973 MiB dump,
    and it scales, so both callers hand it to a worker thread rather than
    stalling the event loop for every other session in the process.

    The freeze is what makes running it off-loop safe: the lifecycle claim and,
    on the live path, the execution lock are held across the call, so nothing
    can save or start a kernel while it reads.
    """
    from backend.config import get_session_states
    from backend.utils.runtime_apply import state_runtime_coherence

    manifest = verify_committed_generation(session_id)
    pairing = _pairing(session_id, state)
    if manifest.get("pairing") != pairing or state_runtime_coherence(state) != "current":
        raise _refuse("generation_mismatch", "The committed runtime does not match the current state and checkpoint.")
    notebooks = {}
    for path in get_session_states(session_id).glob("*.ipynb"):
        with path.open("rb") as stream:
            notebooks[path.name] = _digest(stream)
    return {"pairing": pairing, "manifest": manifest, "notebooks": notebooks}


async def flush_for_resumable_export(session_id: str, state) -> dict:
    """Compatibility flush; archive callers must retain frozen_resumable_export."""
    async with frozen_resumable_export(session_id, state) as descriptor:
        return descriptor["pairing"]


def verify_captured_archive(tar_path: str, session_id: str, descriptor: dict) -> None:
    """Check required captured members, including the actual SQLite backup.

    Generic archive helpers tolerate disappearing files for results-only export.
    A resumable archive cannot silently inherit that tolerance.
    """
    from backend.utils.runtime_manifest import runtime_manifest_path
    from backend.utils.runtime_pairing import checkpoint_identity

    prefix = f"{session_id}/states/"
    try:
        with tarfile.open(tar_path, "r:gz") as archive:
            def member(name):
                stream = archive.extractfile(name)
                if stream is None:
                    raise ValueError(f"missing archive member: {name}")
                return stream

            manifest = json.load(member(prefix + runtime_manifest_path(session_id).name))
            if manifest != descriptor["manifest"]:
                raise ValueError("captured runtime manifest changed")
            with member(prefix + f"{session_id}_session.pkl") as stream:
                if _digest(stream) != manifest["pkl_sha256"]:
                    raise ValueError("captured runtime digest changed")
            state = json.load(member(f"{session_id}/sessions/{session_id}.json"))
            if state.get("state_version") != descriptor["pairing"]["state_version"]:
                raise ValueError("captured state version changed")
            for name, digest in descriptor["notebooks"].items():
                with member(prefix + name) as stream:
                    if _digest(stream) != digest:
                        raise ValueError("captured notebook changed")
            checkpoint_name = prefix + "checkpoints.sqlite"
            checkpoint = None
            if checkpoint_name in archive.getnames():
                with tempfile.TemporaryDirectory(prefix="scai-export-check-") as directory:
                    target = Path(directory) / "checkpoints.sqlite"
                    with member(checkpoint_name) as source, target.open("wb") as dest:
                        import shutil
                        shutil.copyfileobj(source, dest, 1024 * 1024)
                    checkpoint = checkpoint_identity(session_id, directory)
            if checkpoint != descriptor["pairing"]["checkpoint"]:
                raise ValueError("captured checkpoint changed")
    except Exception as exc:
        raise _refuse("capture_unverified", "The archive did not capture one complete runtime generation.") from exc
