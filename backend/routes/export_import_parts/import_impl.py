# Import job execution and archive extraction.
# Moved verbatim out of backend/routes/export_import.py.
import asyncio
import io
import json
import logging
import os
import re
import shutil
import tarfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional
import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from backend.config import (
    DATA_BASE_DIR, REPORTS_DIR, SESSION_DIR, STATE_DIR, UPLOAD_DIR, WORKSPACE_DIR,
    UPLOAD_CHUNK_SIZE, UPLOAD_TEMP_DIR,
    get_session_dir, get_session_metadata_path, get_session_reports,
    get_session_states, get_session_uploads, get_session_workspace, resolve_session_metadata_path,
)
from backend.models.schemas import AnalysisState, APIResponse
from backend.services.reproducibility import REPRODUCIBILITY_DIR_NAME
from backend.services.snapshot_v2.feature_gate import (
    is_snapshot_v2_enabled,
    require_snapshot_v2_enabled,
)
from backend.services.session_store import (
    async_save_session, ensure_session_loaded,
    update_session_cache as _update_session_cache,
)
from backend.state import (
    import_jobs as _import_jobs,
    import_jobs_lock as _import_jobs_lock,
    prepared_exports as _prepared_exports,
    prepared_exports_lock as _prepared_exports_lock,
)
from backend.utils.auth import require_session_ownership
from backend.utils.blob_store import inline_blobs
from backend.utils.files import slugify_session_name, validate_path_within, validate_session_id, temp_file_on_error
from backend.utils.secret_redaction import strip_analysis_state_secrets
from starlette.background import BackgroundTask
from starlette.responses import FileResponse

from .core import ImportCancelled, _IMPORT_PROGRESS_UPDATE_INTERVAL_S, _is_disabled_snapshot_v2_member, is_sandbox_pickle


logger = logging.getLogger(__name__)


def _mark_runtime_availability_after_import(state: AnalysisState) -> None:
    """Record whether the imported session actually has a kernel to resume from.

    ``runtime_restore_status`` rides along in session.json, so a session exported
    mid-analysis arrives claiming ``complete`` even when the archive carried no
    pkl (``include_runtime=false``, or ``trust=false`` on the way in). The
    workflow would then resume against an empty kernel and the bioinformatics
    agent would fail on a missing ``adata`` with no explanation.

    Marking it ``unavailable`` routes resume through the existing
    ``is_forward_resume_blocked`` gate, which tells the user to re-run from their
    data. Completed sessions are deliberately unaffected — that gate exempts
    them, so reports stay viewable and re-analysis still works.
    """
    from backend.utils.runtime_apply import _live_pkl

    if state.current_round <= 0:
        # Never executed anything; a fresh kernel is the correct starting point.
        return
    if _live_pkl(state.id).exists():
        return
    state.runtime_restore_status = "unavailable"
    state.runtime_restore_warning = (
        "This session was imported without kernel runtime state, so in-memory "
        "values from the original run (adata and friends) are gone. Re-run the "
        "analysis from the imported data to recompute them."
    )
    logger.info(
        "[%s] Imported without a live sandbox pkl; marked runtime unavailable "
        "(round=%s).",
        state.id,
        state.current_round,
    )


def _resign_imported_live_pkl(session_id: str) -> bool:
    """Re-sign an imported live sandbox pkl with *this* installation's key.

    The HMAC sidecar lives in ``<DATA_BASE_DIR>/integrity/<session_id>/``, outside
    the session directory, so it is never part of a portable archive — and the
    signing key is installation-scoped anyway, so shipping it would not help.
    Without a local signature ``verify_live_pkl_or_raise`` refuses to rehydrate
    the kernel and sandbox startup fails outright, which is strictly worse than
    having no pkl at all.

    Signing here mirrors what Snapshot V2 restore does in
    ``runtime_apply.apply_runtime_pkl``: content arrives verified by other means
    (the caller passed ``trust=true``), and integrity for local storage is
    re-established under the destination key.

    Returns True when a signature was written.
    """
    from backend.utils.runtime_apply import _live_pkl
    from backend.utils.sandbox import _integrity_dir, _write_pkl_signature

    pkl = _live_pkl(session_id)
    if not pkl.exists():
        return False
    _write_pkl_signature(pkl, _integrity_dir(session_id))
    logger.warning(
        "[%s] Re-signed imported sandbox pkl under this installation's key; "
        "the archive's origin is trusted on the caller's authority (trust=true).",
        session_id,
    )
    return True


async def _process_import_job(tmp_path_str: str, import_job_id: str, user_id: str, trust: bool = False):
    """Background task: extract tar.gz archive and load session."""
    logger.info(f"Import {import_job_id}: starting extraction from {tmp_path_str}")
    archive_size = Path(tmp_path_str).stat().st_size if Path(tmp_path_str).exists() else 0
    # Move into the extracting phase; preserve cancel flag set during upload.
    async with _import_jobs_lock:
        prev = _import_jobs.get(import_job_id, {})
        created_at = prev.get("created_at", time.time())
        _import_jobs[import_job_id] = {
            "status": "extracting",
            "phase": "extracting",
            "created_at": created_at,
            "bytes_total": archive_size,
            "bytes_processed": 0,
            "files_total": 0,
            "files_processed": 0,
            "current_file": "",
            "cancel_requested": prev.get("cancel_requested", False),
        }
    extracted_session_id: Optional[str] = None
    try:
        state = await asyncio.to_thread(_import_tar_gz, tmp_path_str, user_id, import_job_id, trust)
        extracted_session_id = state.id

        validate_session_id(state.id)

        await async_save_session(state.id, state)
        await _update_session_cache(state.id, state)
        async with _import_jobs_lock:
            _import_jobs[import_job_id] = {
                "status": "completed",
                "phase": "completed",
                "created_at": created_at,
                "session_id": state.id,
                "name": state.name,
            }
        logger.info(f"Import {import_job_id}: completed, session_id={state.id}")
    except ImportCancelled as e:
        logger.info(f"Import {import_job_id}: cancelled by user ({e})")
        # Best-effort cleanup: drop the half-extracted session directory.
        if extracted_session_id:
            await asyncio.to_thread(_cleanup_partial_session, extracted_session_id)
        async with _import_jobs_lock:
            _import_jobs[import_job_id] = {"status": "cancelled", "phase": "cancelled", "error": "cancelled by user", "created_at": created_at}
    except HTTPException as e:
        logger.error(f"Import {import_job_id}: failed (HTTPException): {e.detail}")
        async with _import_jobs_lock:
            _import_jobs[import_job_id] = {"status": "failed", "phase": "failed", "error": e.detail, "created_at": created_at}
    except Exception as e:
        logger.exception(f"Import {import_job_id}: failed with exception")
        async with _import_jobs_lock:
            _import_jobs[import_job_id] = {"status": "failed", "phase": "failed", "error": str(e), "created_at": created_at}
    finally:
        Path(tmp_path_str).unlink(missing_ok=True)

def _cleanup_partial_session(session_id: str) -> None:
    """Remove a partially-extracted session directory on cancel.

    Only touches the session-scoped directory under sessions/<id>/, leaving
    other sessions intact. Idempotent — safe if directory does not exist.
    """
    try:
        validate_session_id(session_id)
    except Exception:
        return
    try:
        sess_dir = get_session_dir(session_id)
        if sess_dir.exists():
            shutil.rmtree(sess_dir, ignore_errors=True)
            logger.info(f"Cleaned up partial session directory: {sess_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up partial session {session_id}: {e}")

def _resolve_import_dest(relative_to_base: str, session_id: str) -> Path:
    """Map an archive-relative path to the destination path in the new session-centric layout."""
    if relative_to_base.startswith("states/"):
        fname = relative_to_base.split("/", 1)[1]
        return get_session_states(session_id) / fname
    if relative_to_base.startswith(f"uploads/{session_id}/"):
        rest = relative_to_base.split("/", 2)[2] if relative_to_base.count("/") >= 2 else ""
        return get_session_uploads(session_id) / rest
    if relative_to_base.startswith(f"workspace/{session_id}/"):
        rest = relative_to_base.split("/", 2)[2] if relative_to_base.count("/") >= 2 else ""
        return get_session_workspace(session_id) / rest
    if relative_to_base.startswith(f"reports/{session_id}/"):
        rest = relative_to_base.split("/", 2)[2] if relative_to_base.count("/") >= 2 else ""
        return get_session_reports(session_id) / rest
    if relative_to_base.startswith("blobs/"):
        # Session-level externalized blob store sidecars.
        fname = relative_to_base.split("/", 1)[1]
        return get_session_dir(session_id) / "blobs" / fname
    if relative_to_base.startswith("snapshots/"):
        # Snapshot sidecars: state json, notebook ipynb, compressed pkl,
        # AND per-snapshot blob subdirs `<snap_id>_blobs/<hash>.b64`. The
        # original code only handled top-level files; rglob in the export
        # now also brings in those nested blob files.
        rest = relative_to_base.split("/", 1)[1]
        return get_session_dir(session_id) / "snapshots" / rest
    return DATA_BASE_DIR / relative_to_base

def _import_tar_gz(file_path: str, user_id: str, import_job_id: Optional[str] = None, trust: bool = False) -> AnalysisState:
    """Import from tar.gz archive file on disk with path remapping.

    Uses streaming iteration and chunked file writes to minimise memory usage,
    which is critical for large archives (20+ GB) on memory-constrained systems.

    Runs inside ``asyncio.to_thread``. Progress is written to ``_import_jobs[import_job_id]``
    via atomic single-key dict assignments (GIL-safe; no async-lock needed from a thread).
    Cancellation is cooperative: the cancel endpoint sets ``cancel_requested`` on the same
    dict and this loop checks it between members.
    """
    data_base_str = str(DATA_BASE_DIR)

    session_id = None
    old_base_dir = ""
    state_json_data = None
    # Only a pkl written by *this* import gets re-signed; a pre-existing live pkl
    # left over from another session must keep whatever signature it already had.
    wrote_sandbox_pkl = False

    archive_size = 0
    try:
        archive_size = os.path.getsize(file_path)
    except OSError:
        pass
    if import_job_id:
        job = _import_jobs.get(import_job_id)
        if job is not None:
            job["bytes_total"] = archive_size

    last_progress_ts = 0.0
    files_processed = 0
    bytes_processed = 0

    def _maybe_check_cancel():
        if not import_job_id:
            return
        job = _import_jobs.get(import_job_id)
        if job is not None and job.get("cancel_requested"):
            raise ImportCancelled("cancelled by user")

    def _maybe_update_progress(current_name: str, force: bool = False):
        nonlocal last_progress_ts
        if not import_job_id:
            return
        now = time.monotonic()
        if not force and now - last_progress_ts < _IMPORT_PROGRESS_UPDATE_INTERVAL_S:
            return
        last_progress_ts = now
        job = _import_jobs.get(import_job_id)
        if job is None:
            return
        # Atomic single-key writes — no lock needed.
        job["files_processed"] = files_processed
        job["bytes_processed"] = bytes_processed
        job["current_file"] = current_name

    with tarfile.open(file_path, mode='r:gz') as tf:
        for member in tf:
            _maybe_check_cancel()
            # Reject link entries (symlink/hardlink) and device/special files BEFORE
            # the isfile() filter: such members would otherwise be silently skipped,
            # but an attacker may craft them to point at /etc/passwd or other sessions.
            if member.issym() or member.islnk():
                raise HTTPException(
                    status_code=400,
                    detail=f"Archive contains link entries (rejected): {member.name}",
                )
            if member.isdev() or member.ischr() or member.isfifo():
                raise HTTPException(
                    status_code=400,
                    detail=f"Archive contains device/special entries: {member.name}",
                )

            if not member.isfile():
                continue

            if _is_disabled_snapshot_v2_member(member.name):
                logger.warning(
                    "Skipping Snapshot V2 member from ordinary import while "
                    "the subsystem is disabled: %s",
                    member.name,
                )
                continue

            if '..' in member.name or member.name.startswith('/'):
                logger.warning(f"Skipping archive entry with suspicious path: {member.name}")
                continue

            parts = member.name.split('/', 1)
            if len(parts) < 2:
                continue

            if session_id is None:
                session_id = parts[0]

            relative_to_base = parts[1]

            if relative_to_base == 'manifest.json':
                f = tf.extractfile(member)
                if f:
                    manifest_data = json.loads(f.read().decode('utf-8'))
                    session_id = manifest_data.get("session_id", session_id)
                    old_base_dir = manifest_data.get("data_base_dir", "")
                continue

            if relative_to_base == f"sessions/{session_id}.json":
                f = tf.extractfile(member)
                if f:
                    state_json_data = f.read()
                continue

            validate_session_id(session_id)

            dest_path = _resolve_import_dest(relative_to_base, session_id)
            validate_path_within(dest_path, DATA_BASE_DIR)
            # SECURITY (S4): sandbox pickles can execute arbitrary code via dill.load.
            # Drop them unless the caller passed trust=true to take responsibility.
            is_pkl_member = is_sandbox_pickle(dest_path.name)
            if is_pkl_member:
                if not trust:
                    logger.warning(f"Skipping sandbox pickle from imported archive (untrusted): {member.name}")
                    continue
                logger.warning(f"Restoring sandbox pickle (trust=true): {member.name}")
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            f = tf.extractfile(member)
            if not f:
                continue

            if relative_to_base.startswith("snapshots/") and dest_path.suffix == ".json":
                content = f.read().decode("utf-8")
                if old_base_dir and old_base_dir != data_base_str:
                    content = content.replace(old_base_dir, data_base_str)
                content = content.replace(
                    str(UPLOAD_DIR / session_id), str(get_session_uploads(session_id)))
                content = content.replace(
                    str(WORKSPACE_DIR / session_id), str(get_session_workspace(session_id)))
                content = content.replace(
                    str(REPORTS_DIR / session_id), str(get_session_reports(session_id)))
                dest_path.write_text(content, encoding="utf-8")
            else:
                with open(dest_path, 'wb') as out_f:
                    shutil.copyfileobj(f, out_f, length=UPLOAD_CHUNK_SIZE)
                if is_pkl_member:
                    wrote_sandbox_pkl = True

            # Progress accounting: count uncompressed member size (matches what
            # the user sees on disk). bytes_total is compressed archive size, so
            # bytes_processed can exceed it for highly compressible payloads —
            # the frontend should treat the ratio as a heuristic, not a hard %.
            files_processed += 1
            bytes_processed += getattr(member, "size", 0) or 0
            _maybe_update_progress(relative_to_base)

    # Final progress flush so the poll sees 100% before completion is reported.
    if import_job_id:
        job = _import_jobs.get(import_job_id)
        if job is not None:
            job["files_processed"] = files_processed
            job["bytes_processed"] = bytes_processed
            job["files_total"] = files_processed
            job["current_file"] = ""

    if wrote_sandbox_pkl and session_id:
        _resign_imported_live_pkl(session_id)

    if state_json_data is None:
        raise HTTPException(status_code=400, detail="Session state JSON not found in archive")

    # Path remapping: replace old data_base_dir with current one
    state_json_str = state_json_data.decode('utf-8')
    if old_base_dir and old_base_dir != data_base_str:
        state_json_str = state_json_str.replace(old_base_dir, data_base_str)

    # Remap legacy paths to new session-centric paths
    state_json_str = state_json_str.replace(
        str(UPLOAD_DIR / session_id), str(get_session_uploads(session_id)))
    state_json_str = state_json_str.replace(
        str(WORKSPACE_DIR / session_id), str(get_session_workspace(session_id)))
    state_json_str = state_json_str.replace(
        str(REPORTS_DIR / session_id), str(get_session_reports(session_id)))

    state_data = json.loads(state_json_str)
    state_data["user_id"] = user_id

    # Write the externalized JSON (with @scai-blob tokens intact) to disk.
    # This keeps session.json consistent with the extracted blob sidecars.
    # async_save_session will re-externalize and overwrite, but writing here
    # first ensures a valid session.json exists even if the process crashes
    # before that step — the externalized form is always loadable.
    # Re-serialize from state_data (not state_json_str) so the updated
    # user_id is reflected on disk.
    session_file = get_session_metadata_path(session_id)
    session_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    session_file.parent.chmod(0o700)
    session_file.write_text(json.dumps(state_data, ensure_ascii=False), encoding="utf-8")
    session_file.chmod(0o600)

    # Inline blob references so the in-memory AnalysisState carries the
    # original content rather than dangling @scai-blob tokens.
    state_data = inline_blobs(state_data, get_session_dir(session_id) / "blobs")
    state = AnalysisState(**state_data)
    _mark_runtime_availability_after_import(state)

    return state

def _import_zip(content: bytes, user_id: str, trust: bool = False) -> AnalysisState:
    """Legacy zip import (backward compatibility).

    See :func:`import_state` for the ``trust`` semantics.
    """
    zip_buffer = io.BytesIO(content)

    with zipfile.ZipFile(zip_buffer, 'r') as zf:
        namelist = zf.namelist()
        if not namelist:
            raise HTTPException(status_code=400, detail="Empty zip file")

        session_id = namelist[0].split('/')[0]
        validate_session_id(session_id)

        # Extract state.json
        state_json_path = f"{session_id}/state.json"
        if state_json_path not in namelist:
            raise HTTPException(status_code=400, detail="state.json not found in archive")

        state_json_str = zf.read(state_json_path).decode('utf-8')

        # Remap legacy paths to new session-centric paths
        state_json_str = state_json_str.replace(
            str(UPLOAD_DIR / session_id), str(get_session_uploads(session_id)))
        state_json_str = state_json_str.replace(
            str(WORKSPACE_DIR / session_id), str(get_session_workspace(session_id)))
        state_json_str = state_json_str.replace(
            str(REPORTS_DIR / session_id), str(get_session_reports(session_id)))

        state_data = json.loads(state_json_str)
        state_data["user_id"] = user_id
        state_data = inline_blobs(state_data, get_session_dir(session_id) / "blobs")
        state = AnalysisState(**state_data)

        # Extract notebook if exists
        states_dir = get_session_states(session_id)
        states_dir.mkdir(parents=True, exist_ok=True)
        notebook_path = f"{session_id}/notebook.ipynb"
        if notebook_path in namelist:
            (states_dir / f"{session_id}.ipynb").write_bytes(zf.read(notebook_path))

        # SECURITY (S4): sandbox pickles can execute arbitrary code via dill.load.
        # Drop them unless the caller passed trust=true to take responsibility.
        pkl_path = f"{session_id}/session.pkl"
        if pkl_path in namelist:
            if trust:
                logger.warning(f"Restoring sandbox pickle from zip (trust=true): {pkl_path}")
                (states_dir / f"{session_id}_session.pkl").write_bytes(zf.read(pkl_path))
                _resign_imported_live_pkl(session_id)
            else:
                logger.warning(f"Skipping sandbox pickle from imported zip (untrusted): {pkl_path}")

        # Extract metadata if exists
        metadata_path = f"{session_id}/metadata.json"
        if metadata_path in namelist:
            (states_dir / f"{session_id}_state.json").write_bytes(zf.read(metadata_path))

        # Extract workspace files with path traversal protection
        ws_dir = get_session_workspace(session_id)
        workspace_parent = ws_dir.resolve()
        for wf in namelist:
            if wf.startswith(f"{session_id}/workspace/") and not wf.endswith('/'):
                relative_path = wf.replace(f"{session_id}/workspace/", "")
                dest_path = (ws_dir / relative_path).resolve()
                # Robust traversal check via Path.is_relative_to (3.9+). Reject
                # rather than silently skip — a malicious archive should fail loudly.
                if not dest_path.is_relative_to(workspace_parent):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Archive entry escapes target directory: {wf}",
                    )
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_bytes(zf.read(wf))

    _mark_runtime_availability_after_import(state)
    return state
