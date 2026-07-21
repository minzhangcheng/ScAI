"""Export/Import routes for session tar.gz/zip archives.

Extracted from backend/main.py to decouple export-import handlers.
"""
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


class ImportCancelled(Exception):
    """Raised inside the extraction worker when the user requests cancellation."""
    pass


# Throttle progress writes from the extraction worker so we don't update the
# job dict on every file (some archives have tens of thousands of small files).
_IMPORT_PROGRESS_UPDATE_INTERVAL_S = 0.5
# Throttle upload-phase progress updates (every ~64 MB).
_IMPORT_UPLOAD_PROGRESS_BYTES = 64 * 1024 * 1024

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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["export_import"])

# S3: 100 MB cap for inline zip/json imports to prevent OOM
MAX_INLINE_IMPORT_BYTES = 100 * 1024 * 1024


def check_inline_import_size(content_length: Optional[int], max_bytes: int = MAX_INLINE_IMPORT_BYTES) -> None:
    """Fast-reject oversized inline imports based on the Content-Length header."""
    if content_length is not None and content_length > max_bytes:
        raise HTTPException(status_code=413, detail=f"Import file too large (max {max_bytes} bytes)")


def is_sandbox_pickle(name: str) -> bool:
    """True if an archive entry is a sandbox session pickle.

    SECURITY: dill.load on imported pickle is arbitrary code execution, so these
    files are skipped during import. Matches 'session.pkl', '<id>_session.pkl',
    and '<id>_session.pkl.zst' (compressed snapshot format).
    """
    base = name.rsplit("/", 1)[-1]
    return bool(re.match(r".*_session\.pkl(\.zst)?$", base) or base == "session.pkl")


_SNAPSHOT_V2_SESSION_ROOTS = frozenset({
    "v2",
    "objects",
    "staging",
    "journals",
    "committed",
    "integrity",
})
_SNAPSHOT_V2_OBJECT_MEMBER = re.compile(r"objects/[0-9a-f]{64}$")


def _is_disabled_snapshot_v2_member(name: str) -> bool:
    """Return whether a tar member belongs to Snapshot V2 storage.

    Explicit V2 exports use flat ``objects/`` and ``manifests/`` roots, while
    ordinary session archives may carry session-prefixed V2 storage roots.
    Neither form may be materialized through the ordinary import path when the
    subsystem is disabled.
    """
    if is_snapshot_v2_enabled():
        return False

    normalized = name.strip("/")
    if normalized == "snapshot-v2-export.json":
        return True
    if _SNAPSHOT_V2_OBJECT_MEMBER.fullmatch(normalized):
        return True
    if normalized.startswith("manifests/"):
        return True

    parts = normalized.split("/")
    return len(parts) >= 2 and parts[1] in _SNAPSHOT_V2_SESSION_ROOTS


# P3-7: cap export filename's slug portion to keep total filename below
# filesystem limits (most FS allow 255 bytes). When truncation is needed,
# append a short UUID suffix to preserve uniqueness across long names.
MAX_EXPORT_NAME = 200


def _build_export_filename(session_id: str, name: str, ext: str = ".tar.gz") -> str:
    """Build a length-capped, unique export filename."""
    safe_name = slugify_session_name(name)
    if len(safe_name) > MAX_EXPORT_NAME:
        suffix = uuid.uuid4().hex[:8]
        safe_name = f"{safe_name[:MAX_EXPORT_NAME]}_{suffix}"
    return f"export_{session_id[:8]}_{safe_name}{ext}"


# ---------------------------------------------------------------------------
# Helper functions (used by export handlers)
# ---------------------------------------------------------------------------

def _add_json_bytes(tf, arcname: str, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    info.mode = 0o600
    tf.addfile(info, io.BytesIO(data))


def _add_portable_state_json(tf, source: Path, arcname: str) -> None:
    """Add state JSON without installation-bound or plaintext credentials."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    _add_json_bytes(tf, arcname, strip_analysis_state_secrets(payload))

def _add_snapshots_to_tar(tf, session_id: str, include_runtime: bool = False):
    """Add snapshot files to tar archive.

    Recurses so per-snapshot blob subdirectories (<snap_id>_blobs/<hash>.b64)
    are included alongside the top-level state.json / notebook.ipynb /
    session.pkl.zst files. Without recursion `inline_blobs()` on the imported
    snapshot would dangle, leaving raw "@scai-blob:<hash>" tokens in restored
    state.

    When ``include_runtime`` is False (default), sandbox session pickles
    (``*_session.pkl`` / ``*_session.pkl.zst``) are skipped. These can be
    multi-GB per snapshot and are only useful for in-memory kernel restore on
    the *same* trusted instance — they cannot be re-loaded by import on any
    other instance because ``dill.load`` is arbitrary code execution.
    """
    snapshots_dir = get_session_dir(session_id) / "snapshots"
    if not snapshots_dir.exists():
        return
    for snap_file in snapshots_dir.rglob("*"):
        if snap_file.is_file():
            if not include_runtime and is_sandbox_pickle(snap_file.name):
                continue
            rel = snap_file.relative_to(snapshots_dir)
            arcname = f"{session_id}/snapshots/{rel.as_posix()}"
            if snap_file.suffix.casefold() == ".json":
                _add_portable_state_json(tf, snap_file, arcname)
            else:
                tf.add(str(snap_file), arcname=arcname)


def _add_states_to_tar(tf, session_id: str, include_runtime: bool = False):
    """Add files under sessions/<id>/states/ to the tar.

    Skips the live sandbox pickle and LangGraph checkpoint unless
    ``include_runtime`` is true. Checkpoints contain installation-key-encrypted
    credentials and therefore belong only in same-instance runtime exports;
    notebooks remain portable and are always included.
    """
    def _add(state_file: Path):
        if not include_runtime and is_sandbox_pickle(state_file.name):
            return
        if not include_runtime and state_file.name.startswith("checkpoints.sqlite"):
            return
        tf.add(str(state_file), arcname=f"{session_id}/states/{state_file.name}")

    states_dir = get_session_states(session_id)
    if states_dir.exists():
        for sf in states_dir.iterdir():
            if sf.is_file():
                _add(sf)
    # Legacy fallback (flat STATE_DIR layout)
    for sf in STATE_DIR.glob(f"{session_id}*"):
        if sf.is_file():
            _add(sf)


def _add_session_blobs_to_tar(tf, session_id: str):
    """Add the session-level externalized blob store to tar archive.

    The blob_store offloads any string >= 1024 chars from session.json into
    <session_dir>/blobs/<sha256>.b64 sidecars. Without these files the
    imported session.json contains dangling "@scai-blob:<hash>" tokens that
    surface as raw text in the UI (e.g. inside the messages history).
    """
    blob_dir = get_session_dir(session_id) / "blobs"
    if not blob_dir.exists():
        return
    for blob_file in blob_dir.iterdir():
        if blob_file.is_file():
            tf.add(str(blob_file), arcname=f"{session_id}/blobs/{blob_file.name}")


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
            if is_sandbox_pickle(dest_path.name):
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

    return state
