"""Export/Import routes for session tar.gz/zip archives.

Extracted from backend/main.py to decouple export-import handlers.
"""
import asyncio
import io
import json
import logging
import os
import shutil
import tarfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile

from backend.config import (
    DATA_BASE_DIR, REPORTS_DIR, SESSION_DIR, STATE_DIR, UPLOAD_DIR, WORKSPACE_DIR,
    UPLOAD_CHUNK_SIZE, UPLOAD_TEMP_DIR,
    get_session_dir, get_session_metadata_path, get_session_reports,
    get_session_states, get_session_uploads, get_session_workspace, resolve_session_metadata_path,
)
from backend.models.schemas import AnalysisState, APIResponse
from backend.services.reproducibility import REPRODUCIBILITY_DIR_NAME
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
from backend.utils.files import validate_path_within, validate_session_id
from backend.utils.upload import stream_save_file
from starlette.background import BackgroundTask
from starlette.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["export_import"])


# ---------------------------------------------------------------------------
# Helper functions (used by export handlers)
# ---------------------------------------------------------------------------

def _add_snapshots_to_tar(tf, session_id: str):
    """Add snapshot files to tar archive."""
    snapshots_dir = get_session_dir(session_id) / "snapshots"
    if snapshots_dir.exists():
        for snap_file in snapshots_dir.iterdir():
            if snap_file.is_file():
                tf.add(str(snap_file), arcname=f"{session_id}/snapshots/{snap_file.name}")


def _add_reproducibility_to_tar(tf, session_id: str):
    """Add reproducibility bundle files to tar archive."""
    ws_dir = get_session_workspace(session_id)
    if not ws_dir.exists():
        ws_dir = WORKSPACE_DIR / session_id
    repro_dir = ws_dir / REPRODUCIBILITY_DIR_NAME
    if repro_dir.exists():
        for file_path in sorted(repro_dir.iterdir()):
            if file_path.is_file():
                tf.add(
                    str(file_path),
                    arcname=f"{session_id}/{REPRODUCIBILITY_DIR_NAME}/{file_path.name}",
                )


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@router.api_route("/api/session/{session_id}/export", methods=["GET", "HEAD"])
async def export_state(session_id: str, request: Request):
    """Export complete session as tar.gz including all uploads, workspace, reports, and state files.

    Uses a temporary file on disk to avoid loading the entire archive into memory.
    """
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    # Save latest state to disk first
    await async_save_session(session_id, state)

    # Write tar.gz to a temp file on disk (not in memory)
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False, dir=str(UPLOAD_TEMP_DIR))
    tmp_path = tmp.name
    tmp.close()

    def _build_tar():
        with tarfile.open(tmp_path, mode='w:gz') as tf:
            def _add_bytes(arcname: str, data: bytes):
                info = tarfile.TarInfo(name=arcname)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

            # Manifest
            manifest = json.dumps({
                "session_id": session_id,
                "version": "1.0",
                "data_base_dir": str(DATA_BASE_DIR)
            }, indent=2).encode("utf-8")
            _add_bytes(f"{session_id}/manifest.json", manifest)

            # 1. session.json
            session_file = resolve_session_metadata_path(session_id)
            if session_file and session_file.exists():
                tf.add(str(session_file), arcname=f"{session_id}/sessions/{session_id}.json")

            # 2. states (checkpoints, notebook, pkl)
            states_dir = get_session_states(session_id)
            if states_dir.exists():
                for state_file in states_dir.iterdir():
                    if state_file.is_file():
                        tf.add(str(state_file), arcname=f"{session_id}/states/{state_file.name}")
            # Legacy fallback
            for state_file in STATE_DIR.glob(f"{session_id}*"):
                if state_file.is_file():
                    tf.add(str(state_file), arcname=f"{session_id}/states/{state_file.name}")

            # 3. uploads (recursive)
            upload_dir = get_session_uploads(session_id)
            if not upload_dir.exists():
                upload_dir = UPLOAD_DIR / session_id
            if upload_dir.exists():
                for file_path in upload_dir.rglob("*"):
                    if file_path.is_file():
                        rel = file_path.relative_to(upload_dir)
                        tf.add(str(file_path), arcname=f"{session_id}/uploads/{session_id}/{rel}")

            # 4. workspace (recursive)
            workspace_dir = get_session_workspace(session_id)
            if not workspace_dir.exists():
                workspace_dir = WORKSPACE_DIR / session_id
            if workspace_dir.exists():
                for file_path in workspace_dir.rglob("*"):
                    if file_path.is_file():
                        rel = file_path.relative_to(workspace_dir)
                        tf.add(str(file_path), arcname=f"{session_id}/workspace/{session_id}/{rel}")

            # 5. reports (recursive)
            reports_dir = get_session_reports(session_id)
            if not reports_dir.exists():
                reports_dir = REPORTS_DIR / session_id
            if reports_dir.exists():
                for file_path in reports_dir.rglob("*"):
                    if file_path.is_file():
                        rel = file_path.relative_to(reports_dir)
                        tf.add(str(file_path), arcname=f"{session_id}/reports/{session_id}/{rel}")

            # 6. reproducibility bundle
            _add_reproducibility_to_tar(tf, session_id)

            # 7. snapshots (JSON state files + session pkl backups)
            _add_snapshots_to_tar(tf, session_id)

    await asyncio.to_thread(_build_tar)

    # Build safe filename
    safe_name = "".join(c if c.isalnum() or c in '-_ ' else '' for c in state.name).strip().replace(' ', '_')
    if not safe_name:
        safe_name = "session"
    filename = f"{safe_name}_{session_id[:8]}.tar.gz"

    # FileResponse streams from disk; BackgroundTask deletes the temp file after response is sent
    return FileResponse(
        tmp_path,
        media_type="application/gzip",
        filename=filename,
        background=BackgroundTask(os.unlink, tmp_path)
    )


@router.post("/api/session/{session_id}/export/prepare")
async def export_prepare(session_id: str, request: Request):
    """Phase 1: Build the tar.gz archive on disk and return metadata.

    The frontend calls this first (showing a spinner), then triggers
    the streaming download via GET /export/download once ready.
    """
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    # Save latest state to disk first
    await async_save_session(session_id, state)

    # Build tar to a stable temp file
    export_dir = get_session_reports(session_id)
    export_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(c if c.isalnum() or c in '-_ ' else '' for c in state.name).strip().replace(' ', '_')
    if not safe_name:
        safe_name = "session"
    filename = f"{safe_name}_{session_id[:8]}.tar.gz"
    tar_path = export_dir / filename

    def _build_tar():
        with tarfile.open(str(tar_path), mode='w:gz') as tf:
            def _add_bytes(arcname: str, data: bytes):
                info = tarfile.TarInfo(name=arcname)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

            manifest = json.dumps({
                "session_id": session_id,
                "version": "2.0",
                "data_base_dir": str(DATA_BASE_DIR)
            }, indent=2).encode("utf-8")
            _add_bytes(f"{session_id}/manifest.json", manifest)

            # session.json
            session_file = resolve_session_metadata_path(session_id)
            if session_file and session_file.exists():
                tf.add(str(session_file), arcname=f"{session_id}/sessions/{session_id}.json")

            # states
            states_dir = get_session_states(session_id)
            if states_dir.exists():
                for sf in states_dir.iterdir():
                    if sf.is_file():
                        tf.add(str(sf), arcname=f"{session_id}/states/{sf.name}")
            for sf in STATE_DIR.glob(f"{session_id}*"):
                if sf.is_file():
                    tf.add(str(sf), arcname=f"{session_id}/states/{sf.name}")

            # uploads
            upload_dir = get_session_uploads(session_id)
            if not upload_dir.exists():
                upload_dir = UPLOAD_DIR / session_id
            if upload_dir.exists():
                for file_path in upload_dir.rglob("*"):
                    if file_path.is_file():
                        rel = file_path.relative_to(upload_dir)
                        tf.add(str(file_path), arcname=f"{session_id}/uploads/{session_id}/{rel}")

            # workspace
            workspace_dir = get_session_workspace(session_id)
            if not workspace_dir.exists():
                workspace_dir = WORKSPACE_DIR / session_id
            if workspace_dir.exists():
                for file_path in workspace_dir.rglob("*"):
                    if file_path.is_file():
                        rel = file_path.relative_to(workspace_dir)
                        tf.add(str(file_path), arcname=f"{session_id}/workspace/{session_id}/{rel}")

            # reports
            rpts_dir = get_session_reports(session_id)
            if not rpts_dir.exists():
                rpts_dir = REPORTS_DIR / session_id
            if rpts_dir.exists():
                for file_path in rpts_dir.rglob("*"):
                    if file_path.is_file():
                        if file_path.name == filename:
                            continue
                        rel = file_path.relative_to(rpts_dir)
                        tf.add(str(file_path), arcname=f"{session_id}/reports/{session_id}/{rel}")

            # reproducibility bundle
            _add_reproducibility_to_tar(tf, session_id)

            # snapshots (JSON state files + session pkl backups)
            _add_snapshots_to_tar(tf, session_id)

    await asyncio.to_thread(_build_tar)

    size_bytes = tar_path.stat().st_size

    # Store reference for the download endpoint
    async with _prepared_exports_lock:
        _prepared_exports[session_id] = {
            "path": str(tar_path),
            "filename": filename,
            "size": size_bytes,
        }

    return {
        "status": "ready",
        "size_bytes": size_bytes,
        "filename": filename,
    }


@router.get("/api/session/{session_id}/export/download")
async def export_download(session_id: str, request: Request):
    """Phase 2: Stream the pre-built tar.gz to the browser.

    The file is deleted after the response is fully sent.
    """
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    async with _prepared_exports_lock:
        info = _prepared_exports.pop(session_id, None)
    if not info or not os.path.exists(info["path"]):
        raise HTTPException(status_code=404, detail="Export not prepared. Call POST /export/prepare first.")

    return FileResponse(
        info["path"],
        media_type="application/gzip",
        filename=info["filename"],
        background=BackgroundTask(os.unlink, info["path"])
    )


@router.post("/api/session/import")
async def import_state(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...), import_id: Optional[str] = Form(None)):
    """Import session from tar.gz/zip/json archive, restoring all files to their directories.

    For tar.gz: streams upload to temp file, then extracts in a background task and returns
    an import_id immediately to avoid proxy timeouts on large archives.
    The frontend can pre-generate import_id so that even if a proxy timeout prevents the
    response from reaching the client, it can still poll for status.
    For zip/json: processed inline (small files).
    """
    user_id = getattr(request.state, "user_id", "")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required to import sessions")

    filename = file.filename or ""

    if filename.endswith(('.tar.gz', '.tgz')):
        # ---- tar.gz: save to disk, then extract in background to avoid proxy timeouts ----
        # Register import job BEFORE saving file so frontend can poll even if proxy times out
        import_job_id = import_id or str(uuid.uuid4())
        async with _import_jobs_lock:
            _import_jobs[import_job_id] = {"status": "uploading", "created_at": __import__("time").time()}

        tmp_path = UPLOAD_TEMP_DIR / f"import_{uuid.uuid4().hex}.tar.gz"
        try:
            await stream_save_file(file, tmp_path)
        except Exception as e:
            logger.error(f"Import {import_job_id}: stream_save_file failed: {e}")
            async with _import_jobs_lock:
                _import_jobs[import_job_id] = {"status": "failed", "error": f"Upload failed: {str(e)}"}
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

        async with _import_jobs_lock:
            _import_jobs[import_job_id] = {"status": "pending"}
        background_tasks.add_task(_process_import_job, str(tmp_path), import_job_id, user_id)

        return APIResponse(
            success=True,
            message="File uploaded, extraction in progress",
            data={"import_id": import_job_id, "status": "pending"}
        )

    elif filename.endswith('.zip'):
        # ---- Legacy zip import (typically small) ----
        content = await file.read()
        state = _import_zip(content, user_id)

    else:
        # ---- Plain JSON (backward compatibility) ----
        content = await file.read()
        state_data = json.loads(content.decode('utf-8'))
        state_data["user_id"] = user_id
        state = AnalysisState(**state_data)

    # Validate imported session ID (zip/json path)
    validate_session_id(state.id)

    # Save state to disk and load into memory
    await async_save_session(state.id, state)
    await _update_session_cache(state.id, state)

    return APIResponse(
        success=True,
        message="Session imported successfully",
        session_id=state.id,
        data={"name": state.name}
    )


async def _process_import_job(tmp_path_str: str, import_job_id: str, user_id: str):
    """Background task: extract tar.gz archive and load session."""
    logger.info(f"Import {import_job_id}: starting extraction from {tmp_path_str}")
    try:
        state = await asyncio.to_thread(_import_tar_gz, tmp_path_str, user_id)

        validate_session_id(state.id)

        await async_save_session(state.id, state)
        await _update_session_cache(state.id, state)
        async with _import_jobs_lock:
            _import_jobs[import_job_id] = {"status": "completed", "session_id": state.id, "name": state.name}
        logger.info(f"Import {import_job_id}: completed, session_id={state.id}")
    except HTTPException as e:
        logger.error(f"Import {import_job_id}: failed (HTTPException): {e.detail}")
        async with _import_jobs_lock:
            _import_jobs[import_job_id] = {"status": "failed", "error": e.detail}
    except Exception as e:
        logger.exception(f"Import {import_job_id}: failed with exception")
        async with _import_jobs_lock:
            _import_jobs[import_job_id] = {"status": "failed", "error": str(e)}
    finally:
        Path(tmp_path_str).unlink(missing_ok=True)


@router.get("/api/session/import/{import_id}")
async def get_import_status(import_id: str):
    """Poll the status of a background tar.gz import job."""
    async with _import_jobs_lock:
        if import_id not in _import_jobs:
            raise HTTPException(status_code=404, detail="Import job not found")
        job = _import_jobs[import_id].copy()
    return {"import_id": import_id, **job}


# ---------------------------------------------------------------------------
# Import helper functions
# ---------------------------------------------------------------------------

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
    if relative_to_base.startswith("snapshots/"):
        fname = relative_to_base.split("/", 1)[1]
        return get_session_dir(session_id) / "snapshots" / fname
    return DATA_BASE_DIR / relative_to_base


def _import_tar_gz(file_path: str, user_id: str) -> AnalysisState:
    """Import from tar.gz archive file on disk with path remapping.

    Uses streaming iteration and chunked file writes to minimise memory usage,
    which is critical for large archives (20+ GB) on memory-constrained systems.
    """
    data_base_str = str(DATA_BASE_DIR)

    session_id = None
    old_base_dir = ""
    state_json_data = None

    with tarfile.open(file_path, mode='r:gz') as tf:
        for member in tf:
            if not member.isfile():
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
    state = AnalysisState(**state_data)

    # Write session file to new location
    session_file = get_session_metadata_path(state.id)
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text(state.model_dump_json(indent=2), encoding='utf-8')

    return state


def _import_zip(content: bytes, user_id: str) -> AnalysisState:
    """Legacy zip import (backward compatibility)"""
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

        state_data = json.loads(zf.read(state_json_path).decode('utf-8'))
        state_data["user_id"] = user_id
        state = AnalysisState(**state_data)

        # Extract notebook if exists
        states_dir = get_session_states(session_id)
        states_dir.mkdir(parents=True, exist_ok=True)
        notebook_path = f"{session_id}/notebook.ipynb"
        if notebook_path in namelist:
            (states_dir / f"{session_id}.ipynb").write_bytes(zf.read(notebook_path))

        # Extract session.pkl if exists (for sandbox restore)
        pkl_path = f"{session_id}/session.pkl"
        if pkl_path in namelist:
            (states_dir / f"{session_id}_session.pkl").write_bytes(zf.read(pkl_path))

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
                if not str(dest_path).startswith(str(workspace_parent) + os.sep):
                    logger.warning(f"Skipping zip entry with path traversal: {wf}")
                    continue
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_bytes(zf.read(wf))

    return state
