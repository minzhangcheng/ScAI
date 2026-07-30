# Export endpoints.
# Moved verbatim out of backend/routes/export_import.py.
import asyncio
import io
import json
import logging
from urllib.parse import quote
import os
import re
import secrets
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
    EXPORT_STREAM_PREFIX, PREPARED_EXPORT_PREFIX, SNAPSHOT_EXPORT_PREFIX,
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
from starlette.responses import FileResponse, Response

from .core import (
    ARCHIVE_FORMAT_VERSION,
    _build_export_filename,
    check_export_disk_space,
    count_v2_snapshot_records,
    new_export_token,
    router,
)
from .tar_builders import (
    _add_json_bytes,
    _add_portable_state_json,
    _add_reproducibility_to_tar,
    _add_session_blobs_to_tar,
    _add_session_logs_to_tar,
    _add_snapshots_to_tar,
    _add_states_to_tar,
    _add_tree_to_tar,
)

logger = logging.getLogger(__name__)


def _build_session_tar(tar_path: str, session_id: str, include_runtime: bool = False) -> None:
    """Write the portable session archive for *session_id* to *tar_path*.

    Both export endpoints share this so their archives cannot drift apart; they
    used to hold byte-identical copies of the packing logic that nonetheless
    stamped different manifest versions.

    ``dereference=True`` matters for importability, not for size: the import
    side rejects every symlink and hardlink member outright, so a single
    ``os.symlink`` or ``cp -l`` left behind by analysis code inside the
    workspace would otherwise produce an archive that can never be imported.
    """
    with tarfile.open(tar_path, mode="w:gz", dereference=True) as tf:
        def _add_bytes(arcname: str, data: bytes):
            info = tarfile.TarInfo(name=arcname)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        # 1. session.json is read first -- though still written after the
        # manifest -- because the manifest states how many Snapshot V2 records
        # travel without their object storage, which only this payload knows.
        session_file = resolve_session_metadata_path(session_id)
        state_payload = None
        if session_file and session_file.exists():
            state_payload = json.loads(session_file.read_text(encoding="utf-8"))

        # Manifest first: the importer latches the session id from the first
        # member it sees and reconciles it against this file.
        manifest = json.dumps({
            "session_id": session_id,
            "version": ARCHIVE_FORMAT_VERSION,
            "data_base_dir": str(DATA_BASE_DIR),
            # An ordinary archive packs no V2 object storage at all, so a
            # destination could not otherwise tell "the source had no snapshots"
            # from "the export dropped them". This is that difference, stated.
            "snapshot_v2_records_without_payload": count_v2_snapshot_records(state_payload),
        }, indent=2).encode("utf-8")
        _add_bytes(f"{session_id}/manifest.json", manifest)

        if state_payload is not None:
            _add_json_bytes(
                tf,
                f"{session_id}/sessions/{session_id}.json",
                strip_analysis_state_secrets(state_payload),
            )

        # 2. states (checkpoints, notebook, pkl)
        _add_states_to_tar(tf, session_id, include_runtime=include_runtime)

        # 3. uploads (recursive)
        upload_dir = get_session_uploads(session_id)
        if not upload_dir.exists():
            upload_dir = UPLOAD_DIR / session_id
        _add_tree_to_tar(tf, upload_dir, f"{session_id}/uploads/{session_id}")

        # 4. workspace (recursive)
        workspace_dir = get_session_workspace(session_id)
        if not workspace_dir.exists():
            workspace_dir = WORKSPACE_DIR / session_id
        _add_tree_to_tar(tf, workspace_dir, f"{session_id}/workspace/{session_id}")

        # 5. reports (recursive)
        reports_dir = get_session_reports(session_id)
        if not reports_dir.exists():
            reports_dir = REPORTS_DIR / session_id
        _add_tree_to_tar(tf, reports_dir, f"{session_id}/reports/{session_id}")

        # 6. reproducibility bundle
        _add_reproducibility_to_tar(tf, session_id)

        # 7. session-level externalized blob store (must accompany session.json)
        _add_session_blobs_to_tar(tf, session_id)

        # 8. snapshots (JSON state files + session pkl backups)
        _add_snapshots_to_tar(tf, session_id, include_runtime=include_runtime)

        # 9. LLM interaction logs (the detailed report's usage numbers)
        _add_session_logs_to_tar(tf, session_id)


async def _pack_session_archive(
    session_id: str,
    state,
    include_runtime: bool,
    *,
    prefix: str,
) -> tuple[str, str]:
    """Pack *session_id* into a temp archive and return (tar_path, filename).

    Both export endpoints share the preconditions so they cannot drift apart:
    the disk-headroom check, the forced save the packer depends on, and the
    prefixed temp file that a failed build has to clean up.

    Save latest state to disk first. Forced: the packer reads session.json off
    the disk, and RUNNING is not one of the statuses that bypass the 5s save
    throttle — so a session exported while it is analysing would ship a
    session.json up to five seconds stale. A `current_round` that lands back at
    0 that way also makes the importer skip its runtime-availability marking
    entirely, producing an archive that claims a kernel it lacks.

    The temp-file prefix is what makes a leak from an aborted download
    sweepable — without it the name is indistinguishable from an import
    upload's temp file.
    """
    check_export_disk_space()

    await async_save_session(session_id, state, force=True)

    export_dir = UPLOAD_TEMP_DIR
    export_dir.mkdir(parents=True, exist_ok=True)

    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        prefix=prefix, suffix=".tar.gz", delete=False, dir=str(export_dir)
    )
    tar_path = tmp.name
    tmp.close()

    async with temp_file_on_error(tar_path):
        await asyncio.to_thread(_build_session_tar, tar_path, session_id, include_runtime)

    return tar_path, _build_export_filename(session_id, state.name)


@router.api_route("/api/session/{session_id}/export", methods=["GET", "HEAD"])
async def export_state(session_id: str, request: Request, include_runtime: bool = False):
    """Export complete session as tar.gz including all uploads, workspace, reports, and state files.

    Uses a temporary file on disk to avoid loading the entire archive into memory.

    Requires authentication: the archive carries every uploaded matrix, figure,
    report and message of the session, so it is not one of the browser-embedded
    resources that may be fetched without a token.

    Query params:
        include_runtime: When True, include sandbox session pickles (live kernel
            variable dumps + per-snapshot ``session.pkl.zst``) and the LangGraph
            checkpoint. These can be multi-GB and are only loadable by an import
            that passes ``trust=true``. Default False produces smaller archives
            that cannot resume analysis.
    """
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    if request.method == "HEAD":
        # Answer the metadata question without paying for the answer: this route
        # is registered for HEAD as well as GET, and with no branch here a HEAD
        # forced a session save and a full multi-GB pack whose body starlette then
        # discarded. Placed after validate_session_id/require_session_ownership so
        # HEAD keeps GET's exact 400/401/404 surface and cannot become an
        # existence oracle, and before the save/pack, which is the point.
        # Content-Disposition is percent-encoded because export names keep CJK
        # characters and a header must be latin-1.
        head_filename = _build_export_filename(session_id, state.name)
        return Response(
            status_code=200,
            media_type="application/gzip",
            headers={
                "Content-Disposition":
                    f"attachment; filename*=utf-8''{quote(head_filename)}"
            },
        )

    # Write tar.gz to a temp file on disk (not in memory).
    tmp_path, filename = await _pack_session_archive(
        session_id, state, include_runtime, prefix=EXPORT_STREAM_PREFIX
    )

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

    The frontend calls this first (showing a spinner), then triggers the
    streaming download via GET /export/download using the returned token.

    The archive is written to a unique temp path. It used to be written to a
    path derived from the session id and name alone, so a second prepare for the
    same session — trivially reached by clicking again after a proxy timeout, or
    from a second tab — truncated the archive the first one was still writing
    and produced a download that no importer could read.

    Query params:
        include_runtime: Include sandbox session pickles and the LangGraph
            checkpoint. See ``export_state`` for the trade-off. Default False.
    """
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    tar_path, filename = await _pack_session_archive(
        session_id, state, include_runtime, prefix=PREPARED_EXPORT_PREFIX
    )

    size_bytes = Path(tar_path).stat().st_size

    export_token = new_export_token()
    async with _prepared_exports_lock:
        _prepared_exports[export_token] = {
            "path": str(tar_path),
            "filename": filename,
            "size": size_bytes,
            "session_id": session_id,
            "user_id": user_id,
            "created_at": time.time(),
        }

    return {
        "status": "ready",
        "size_bytes": size_bytes,
        "filename": filename,
        "export_token": export_token,
        "download_url": f"/api/session/{session_id}/export/download?token={export_token}",
    }

@router.get("/api/session/{session_id}/export/download")
async def export_download(session_id: str, request: Request, token: str | None = None):
    """Phase 2: Stream the pre-built tar.gz to the browser.

    The file is deleted after the response is fully sent.

    The browser fetches this through a plain ``<a download>`` navigation, which
    carries no Authorization header, so the unguessable one-shot ``token``
    minted by ``/export/prepare`` is the credential. It is bound to the session
    it was issued for; a caller that *does* present a token is additionally
    checked for ownership so a leaked token cannot be redeemed by a signed-in
    account that does not own the session.
    """
    validate_session_id(session_id)
    if not token:
        raise HTTPException(
            status_code=400,
            detail="Export token is required. Call POST /export/prepare first.",
        )

    async with _prepared_exports_lock:
        info = _prepared_exports.get(token)
        # Peek only. Binding, existence and ownership must all pass before the
        # token is consumed: popping first means any rejected request destroys
        # the owner's prepared archive, which then has to be packed again.
        if info is not None and info.get("session_id") != session_id:
            info = None

    if not info or not os.path.exists(info["path"]):
        raise HTTPException(status_code=404, detail="Export not prepared. Call POST /export/prepare first.")

    user_id = getattr(request.state, "user_id", None)
    if user_id:
        # Authenticated caller: ownership still decides.
        await require_session_ownership(session_id=session_id, user_id=user_id)

    async with _prepared_exports_lock:
        # Consume last. A racing second redemption finds it gone and 404s, so
        # the one-shot guarantee is unchanged.
        if _prepared_exports.pop(token, None) is None:
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
            # Build the archive where it will live, so publishing it below is a
            # rename on the same filesystem instead of a multi-GB copy.
            staging_dir=UPLOAD_TEMP_DIR,
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
    export_token = new_export_token()
    filename = f"{SNAPSHOT_EXPORT_PREFIX}{session_id[:8]}-{uuid.uuid4().hex[:8]}.tar.gz"
    final_path = export_dir / filename
    # Off the event loop regardless: a deployment that overrides the staging
    # directory, or a same-name-different-device layout, would otherwise block
    # every other request and WebSocket ping for the length of the copy.
    await asyncio.to_thread(shutil.move, str(export_path), str(final_path))

    size_bytes = final_path.stat().st_size

    # Token-keyed prepared export prevents V1/V2 collisions
    async with _prepared_exports_lock:
        _prepared_exports[export_token] = {
            "path": str(final_path),
            "filename": filename,
            "size": size_bytes,
            "session_id": session_id,
            "user_id": user_id,
            "created_at": time.time(),
        }

    return {
        "status": "ready",
        "export_token": export_token,
        "download_url": f"/api/session/{session_id}/export/download?token={export_token}",
        "filename": filename,
        "size_bytes": size_bytes,
    }
