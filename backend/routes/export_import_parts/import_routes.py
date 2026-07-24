# Import endpoints and job status.
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

from .core import ImportCancelled, MAX_INLINE_IMPORT_BYTES, _IMPORT_UPLOAD_PROGRESS_BYTES, check_inline_import_size, router

from .import_impl import _import_zip, _process_import_job


logger = logging.getLogger(__name__)


@router.post(
    "/api/session/{session_id}/import-v2",
    dependencies=[Depends(require_snapshot_v2_enabled)],
)
async def import_v2(
    session_id: str,
    request: Request,
):
    """Import a V2 snapshot export archive (tar.gz) into a session.

    Accepts a raw tar.gz body (not multipart) uploaded as
    ``application/gzip`` with an ``X-Filename`` header identifying
    the archive.  Uses the path *session_id* as the destination
    session — all snapshot/session IDs in the archive are rebound
    to this session via a deterministic mapping.

    Headers:
        X-Trust: ``"true"`` to restore runtime objects (dill/h5ad).
            Default ``false`` — runtime is marked unavailable and
            no dill objects are written to the destination store.

    Returns the ``ImportResult`` payload.
    """
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required to import V2 snapshots")
    # Ownership check (also rejects missing sessions for non-auth callers that
    # slipped past middleware with a stale user_id).
    await require_session_ownership(session_id=session_id, user_id=user_id)

    trust = (request.headers.get("x-trust") or "").lower() in ("true", "1", "yes")

    # Stream body to a temp file
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False, dir=str(UPLOAD_TEMP_DIR))
    tmp_path = tmp.name
    tmp.close()

    try:
        buffer_size = 0
        with open(tmp_path, "wb") as f:
            async for chunk in request.stream():
                if not chunk:
                    continue
                f.write(chunk)
                buffer_size += len(chunk)

        if buffer_size == 0:
            raise HTTPException(status_code=400, detail="Empty upload")

        from backend.services.snapshot_v2.paths import SnapshotV2Paths
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.services.snapshot_v2.errors import (
            SnapshotV2Error,
            snapshot_error_detail,
        )

        paths = SnapshotV2Paths.for_session(session_id)
        service = SnapshotService(session_id, paths)
        try:
            result = await service.import_snapshots(tmp_path, trust_runtime=trust)
        except SnapshotV2Error as exc:
            err = snapshot_error_detail(exc)
            raise HTTPException(
                status_code=err["status_code"],
                detail={key: value for key, value in err.items() if key != "status_code"},
            ) from exc

        if not result.success:
            raise HTTPException(
                status_code=422,
                detail=f"Import failed: {'; '.join(result.errors)}",
            )

        return result.model_dump(mode="json")

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("V2 import failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Import failed: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

@router.post("/api/session/import")
async def import_state(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Import session from tar.gz/zip/json archive, restoring all files to their directories.

    **Raw octet-stream upload** — the request body is the archive bytes directly,
    not multipart/form-data. This avoids Starlette's SpooledTemporaryFile from
    making a redundant copy of multi-GB archives (each multipart upload spool
    + re-copy doubled disk I/O, adding ~50 min for a 73 GB archive).

    Headers:
        X-Filename: original filename, used to dispatch on extension
            (``.tar.gz``/``.tgz`` → background extract; ``.zip``/``.json`` → inline).
        X-Import-Id: optional client-generated UUID for the import job. Letting
            the client pick it means the frontend can poll for status even if
            the response is lost to a proxy timeout.
        X-Trust: ``"true"`` opts in to restoring sandbox session pickles
            (``*_session.pkl``). ``dill.load`` is arbitrary code execution, so
            only pass for archives the caller fully trusts. Default false.

    For tar.gz: streams the body to disk under ``UPLOAD_TEMP_DIR`` (single write
    pass), then extracts in a background task and returns an ``import_id``.
    For zip/json: streams to an in-memory buffer with a size cap, then processes
    inline.
    """
    import_start_time = time.time()
    user_id = getattr(request.state, "user_id", "")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required to import sessions")

    filename = (request.headers.get("x-filename") or "").strip()
    if not filename:
        logger.warning("Import rejected: missing X-Filename header from client %s", request.client)
        raise HTTPException(status_code=400, detail="X-Filename header is required")
    # Reject obviously malicious filenames before they touch disk paths.
    if '/' in filename or '\\' in filename or '\x00' in filename:
        logger.warning("Import rejected: bad X-Filename %r from client %s", filename, request.client)
        raise HTTPException(status_code=400, detail="X-Filename must not contain path separators")

    import_id = request.headers.get("x-import-id") or None
    trust = (request.headers.get("x-trust") or "").lower() in ("true", "1", "yes")

    _cl_hdr = request.headers.get("content-length")
    content_length = int(_cl_hdr) if _cl_hdr and _cl_hdr.isdigit() else 0

    logger.info(
        "Import request received: filename=%r, content_length=%s, trust=%s, client=%s, "
        "user_id=%s, import_id=%s, x-forwarded-for=%s",
        filename, content_length, trust, request.client,
        user_id[:8] if user_id else "", import_id,
        request.headers.get("x-forwarded-for", "N/A"),
    )

    if filename.endswith(('.tar.gz', '.tgz')):
        # ---- tar.gz: stream body directly to disk, then extract in background ----
        import_job_id = import_id or str(uuid.uuid4())
        async with _import_jobs_lock:
            _import_jobs[import_job_id] = {
                "status": "uploading",
                "phase": "uploading",
                "created_at": time.time(),
                "bytes_total": content_length,
                "bytes_processed": 0,
                "files_total": 0,
                "files_processed": 0,
                "current_file": "",
                "cancel_requested": False,
            }

        tmp_path = UPLOAD_TEMP_DIR / f"import_{uuid.uuid4().hex}.tar.gz"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        bytes_written = 0
        last_progress_update = 0
        try:
            async with aiofiles.open(tmp_path, 'wb') as f:
                async for chunk in request.stream():
                    if not chunk:
                        continue
                    # Cooperative cancel during the upload phase.
                    if _import_jobs.get(import_job_id, {}).get("cancel_requested"):
                        raise ImportCancelled("cancelled during upload")
                    await f.write(chunk)
                    bytes_written += len(chunk)
                    if bytes_written - last_progress_update >= _IMPORT_UPLOAD_PROGRESS_BYTES:
                        job = _import_jobs.get(import_job_id)
                        if job is not None:
                            job["bytes_processed"] = bytes_written
                        last_progress_update = bytes_written
        except ImportCancelled:
            logger.info(f"Import {import_job_id}: cancelled during upload")
            async with _import_jobs_lock:
                created_at = _import_jobs.get(import_job_id, {}).get("created_at", time.time())
                _import_jobs[import_job_id] = {"status": "cancelled", "phase": "cancelled", "error": "cancelled by user", "created_at": created_at}
            tmp_path.unlink(missing_ok=True)
            return APIResponse(success=False, message="Import cancelled", data={"import_id": import_job_id, "status": "cancelled"})
        except Exception as e:
            logger.error("Import %s: upload failed after %.1fs: %s", import_job_id, time.time() - import_start_time, e, exc_info=True)
            async with _import_jobs_lock:
                created_at = _import_jobs.get(import_job_id, {}).get("created_at", time.time())
                _import_jobs[import_job_id] = {"status": "failed", "phase": "failed", "error": f"Upload failed: {str(e)}", "created_at": created_at}
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

        async with _import_jobs_lock:
            job = _import_jobs.get(import_job_id, {})
            cancel_flag = job.get("cancel_requested", False)
            created_at = job.get("created_at", time.time())
            archive_size = bytes_written
            _import_jobs[import_job_id] = {
                "status": "pending",
                "phase": "pending",
                "created_at": created_at,
                "bytes_total": archive_size,
                "bytes_processed": archive_size,
                "files_total": 0,
                "files_processed": 0,
                "current_file": "",
                "cancel_requested": cancel_flag,
            }
        if cancel_flag:
            tmp_path.unlink(missing_ok=True)
            async with _import_jobs_lock:
                _import_jobs[import_job_id] = {"status": "cancelled", "phase": "cancelled", "error": "cancelled by user", "created_at": created_at}
            return APIResponse(success=False, message="Import cancelled", data={"import_id": import_job_id, "status": "cancelled"})

        background_tasks.add_task(_process_import_job, str(tmp_path), import_job_id, user_id, trust)

        logger.info(
            "Import %s: upload complete (%d bytes, %.1fs), returning 200 with import_id",
            import_job_id, bytes_written, time.time() - import_start_time,
        )

        return APIResponse(
            success=True,
            message="File uploaded, extraction in progress",
            data={"import_id": import_job_id, "status": "pending"}
        )

    # ---- zip / json: stream to in-memory buffer with size cap ----
    check_inline_import_size(content_length if content_length > 0 else None)
    buf = bytearray()
    total = 0
    async for chunk in request.stream():
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_INLINE_IMPORT_BYTES:
            raise HTTPException(status_code=413, detail=f"Import file too large (max {MAX_INLINE_IMPORT_BYTES} bytes)")
        buf.extend(chunk)
    content = bytes(buf)

    if filename.endswith('.zip'):
        state = _import_zip(content, user_id, trust=trust)
    else:
        # Plain JSON
        state_data = json.loads(content.decode('utf-8'))
        state_data["user_id"] = user_id
        state = AnalysisState(**state_data)

    validate_session_id(state.id)
    await async_save_session(state.id, state)
    await _update_session_cache(state.id, state)

    return APIResponse(
        success=True,
        message="Session imported successfully",
        session_id=state.id,
        data={"name": state.name}
    )

@router.get("/api/session/import/{import_id}")
async def get_import_status(import_id: str):
    """Poll the status of a background tar.gz import job.

    Response carries phase/bytes/files progress fields so the client can render
    a live progress bar instead of guessing whether the job is still alive.
    """
    async with _import_jobs_lock:
        if import_id not in _import_jobs:
            raise HTTPException(status_code=404, detail="Import job not found")
        job = _import_jobs[import_id].copy()
    return {"import_id": import_id, **job}

@router.post("/api/session/import/{import_id}/cancel")
async def cancel_import(import_id: str):
    """Request cancellation of an in-progress tar.gz import.

    Cooperative cancellation: sets a flag that the upload loop and extraction
    worker check between chunks/members. Already-completed or already-failed
    jobs are left as-is.
    """
    async with _import_jobs_lock:
        job = _import_jobs.get(import_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Import job not found")
        terminal = job.get("status") in ("completed", "failed", "cancelled")
        if terminal:
            return {"import_id": import_id, "status": job.get("status"), "cancelled": False}
        job["cancel_requested"] = True
    return {"import_id": import_id, "status": "cancel_requested", "cancelled": True}
