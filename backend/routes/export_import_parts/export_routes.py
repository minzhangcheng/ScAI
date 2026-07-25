# Export endpoints.
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

from .core import _build_export_filename, router
from .tar_builders import _add_portable_state_json, _add_reproducibility_to_tar, _add_session_blobs_to_tar, _add_snapshots_to_tar, _add_states_to_tar


@router.api_route("/api/session/{session_id}/export", methods=["GET", "HEAD"])
async def export_state(session_id: str, request: Request, include_runtime: bool = False):
    """Export complete session as tar.gz including all uploads, workspace, reports, and state files.

    Uses a temporary file on disk to avoid loading the entire archive into memory.

    Query params:
        include_runtime: When True, include sandbox session pickles
            (live kernel variable dumps + per-snapshot ``session.pkl.zst``).
            These can be multi-GB and are only loadable on the same trusted
            instance — every import path drops them unless ``trust=true`` is
            passed. Default False produces smaller, portable archives.
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
                _add_portable_state_json(
                    tf, session_file, f"{session_id}/sessions/{session_id}.json"
                )

            # 2. states (checkpoints, notebook, pkl)
            _add_states_to_tar(tf, session_id, include_runtime=include_runtime)

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

            # 7. session-level externalized blob store (must accompany session.json)
            _add_session_blobs_to_tar(tf, session_id)

            # 8. snapshots (JSON state files + session pkl backups)
            _add_snapshots_to_tar(tf, session_id, include_runtime=include_runtime)

    async with temp_file_on_error(tmp_path):
        await asyncio.to_thread(_build_tar)

    # Build safe filename (consistent format with export_prepare)
    filename = _build_export_filename(session_id, state.name)

    # FileResponse streams from disk; BackgroundTask deletes the temp file after response is sent
    return FileResponse(
        tmp_path,
        media_type="application/gzip",
        filename=filename,
        background=BackgroundTask(Path(tmp_path).unlink, missing_ok=True)
    )

@router.post("/api/session/{session_id}/export/prepare")
async def export_prepare(session_id: str, request: Request, include_runtime: bool = False):
    """Phase 1: Build the tar.gz archive on disk and return metadata.

    The frontend calls this first (showing a spinner), then triggers
    the streaming download via GET /export/download once ready.

    Query params:
        include_runtime: Include sandbox session pickles. See ``export_state``
            for the trade-off. Default False.
    """
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    # Save latest state to disk first
    await async_save_session(session_id, state)

    # Build tar to a stable temp file
    export_dir = UPLOAD_TEMP_DIR
    export_dir.mkdir(parents=True, exist_ok=True)

    filename = _build_export_filename(session_id, state.name)
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
                _add_portable_state_json(
                    tf, session_file, f"{session_id}/sessions/{session_id}.json"
                )

            # states
            _add_states_to_tar(tf, session_id, include_runtime=include_runtime)

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
                        rel = file_path.relative_to(rpts_dir)
                        tf.add(str(file_path), arcname=f"{session_id}/reports/{session_id}/{rel}")

            # reproducibility bundle
            _add_reproducibility_to_tar(tf, session_id)

            # session-level externalized blob store
            _add_session_blobs_to_tar(tf, session_id)

            # snapshots (JSON state files + session pkl backups)
            _add_snapshots_to_tar(tf, session_id, include_runtime=include_runtime)

    async with temp_file_on_error(str(tar_path)):
        await asyncio.to_thread(_build_tar)

    size_bytes = tar_path.stat().st_size

    # Store reference for the download endpoint (V1-specific key)
    v1_key = f"v1-{session_id}"
    async with _prepared_exports_lock:
        _prepared_exports[v1_key] = {
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
async def export_download(session_id: str, request: Request, token: str | None = None):
    """Phase 2: Stream the pre-built tar.gz to the browser.

    The file is deleted after the response is fully sent.

    Query params:
        token: Export token from POST /export-v2. When absent, falls back
            to legacy session_id key (V1 prepared export path).
    """
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    # Token-based (V2) or legacy session_id key (V1 with v1- prefix)
    key = token or f"v1-{session_id}"
    async with _prepared_exports_lock:
        info = _prepared_exports.pop(key, None)
    if not info or not os.path.exists(info["path"]):
        raise HTTPException(status_code=404, detail="Export not prepared. Call POST /export/prepare first.")

    return FileResponse(
        info["path"],
        media_type="application/gzip",
        filename=info["filename"],
        background=BackgroundTask(os.unlink, info["path"])
    )

@router.post(
    "/api/session/{session_id}/export-v2",
    dependencies=[Depends(require_snapshot_v2_enabled)],
)
async def export_v2(
    session_id: str,
    request: Request,
    include_runtime: bool = True,
    snapshot_ids: str | None = None,
):
    """Export verified V2 snapshot objects as a tar.gz archive.

    Unlike the legacy ``/export`` endpoint, V2 export streams verified
    snapshot objects (blobs) from the V2 object store into a portable
    archive.  Base/workspace objects always export; runtime objects
    (adata variables, dill residuals) only with ``include_runtime=true``.

    Query params:
        include_runtime: Include runtime snapshot objects (default True).
        snapshot_ids: Comma-separated list of snapshot IDs to export
            (default: all committed snapshots).
    """
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    from backend.services.snapshot_v2.paths import SnapshotV2Paths
    from backend.services.snapshot_v2.service import SnapshotService
    from backend.services.snapshot_v2.errors import (
        SnapshotV2Error,
        snapshot_error_detail,
    )

    paths = SnapshotV2Paths.for_session(session_id)

    # Parse optional snapshot_ids
    parsed_ids: list[str] | None = None
    if snapshot_ids:
        parsed_ids = [s.strip() for s in snapshot_ids.split(",") if s.strip()]

    service = SnapshotService(session_id, paths)
    try:
        export_path = await service.export_snapshots(
            include_runtime=include_runtime,
            snapshot_ids=parsed_ids,
        )
    except SnapshotV2Error as exc:
        err = snapshot_error_detail(exc)
        raise HTTPException(
            status_code=err["status_code"],
            detail={key: value for key, value in err.items() if key != "status_code"},
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Move to stable location for download
    export_dir = UPLOAD_TEMP_DIR
    export_dir.mkdir(parents=True, exist_ok=True)
    export_token = uuid.uuid4().hex[:24]
    filename = f"v2-export-{session_id[:8]}-{uuid.uuid4().hex[:8]}.tar.gz"
    final_path = export_dir / filename
    shutil.move(str(export_path), str(final_path))

    size_bytes = final_path.stat().st_size

    # Token-keyed prepared export prevents V1/V2 collisions
    async with _prepared_exports_lock:
        _prepared_exports[export_token] = {
            "path": str(final_path),
            "filename": filename,
            "size": size_bytes,
            "session_id": session_id,
        }

    return {
        "status": "ready",
        "export_token": export_token,
        "download_url": f"/api/session/{session_id}/export/download?token={export_token}",
        "filename": filename,
        "size_bytes": size_bytes,
    }
