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
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from backend.config import (
    DATA_BASE_DIR, REPORTS_DIR, SESSION_DIR, STATE_DIR, UPLOAD_DIR, WORKSPACE_DIR,
    UPLOAD_CHUNK_SIZE, UPLOAD_TEMP_DIR,
    EXPORT_STREAM_PREFIX, PREPARED_EXPORT_PREFIX,
    get_session_dir, get_session_metadata_path, get_session_reports,
    get_session_states, get_session_uploads, get_session_workspace, resolve_session_metadata_path,
)
from backend.models.schemas import AnalysisState, APIResponse
from backend.services.reproducibility import REPRODUCIBILITY_DIR_NAME
from backend.services.session_store import (
    async_save_session, ensure_session_loaded,
    update_session_cache as _update_session_cache,
)
from backend.services.session_lifecycle import (
    assert_generation_writable,
    get_lifecycle_lock,
)
from backend.services.export_lifecycle import (
    LifecycleFileResponse,
    abandon_export_record,
    bind_export_path,
    claim_export_stream,
    create_export_ownership_marker,
    generation_bound_export_prefix,
    publish_export_ready,
    reserve_export_record,
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
from backend.utils.asyncio_tasks import (
    await_with_deferred_cancellation,
    run_in_owned_thread,
)
from backend.utils.secret_redaction import strip_analysis_state_secrets
from starlette.responses import Response

from .core import (
    ARCHIVE_FORMAT_VERSION,
    _build_export_filename,
    check_export_disk_space,
    new_export_token,
    router,
)
from .tar_builders import (
    _add_json_bytes,
    _add_portable_state_json,
    _add_reproducibility_to_tar,
    _add_session_blobs_to_tar,
    _add_session_logs_to_tar,
    _add_reports_to_tar,
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
        _add_reports_to_tar(tf, reports_dir, f"{session_id}/reports/{session_id}")

        # 6. reproducibility bundle
        _add_reproducibility_to_tar(tf, session_id)

        # 7. session-level externalized blob store (must accompany session.json)
        _add_session_blobs_to_tar(tf, session_id)

        # 8. snapshots (JSON state files + session pkl backups)

        # 9. LLM interaction logs (the detailed report's usage numbers)
        _add_session_logs_to_tar(tf, session_id)


async def _pack_session_archive(
    session_id: str,
    state,
    include_runtime: bool,
    *,
    prefix: str,
    export_info: dict[str, Any],
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

    save_task = asyncio.create_task(
        async_save_session(session_id, state, force=True),
        name=f"export-state-save:{session_id}:{export_info['token']}",
    )
    await await_with_deferred_cancellation(save_task)

    export_dir = UPLOAD_TEMP_DIR
    export_dir.mkdir(parents=True, exist_ok=True)

    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        prefix=generation_bound_export_prefix(prefix, export_info),
        suffix=".tar.gz",
        delete=False,
        dir=str(export_dir),
    )
    tar_path = tmp.name
    tmp.close()
    # No await separates creation from this in-record ownership publication.
    # If deletion wins the following lifecycle lock, it can still name and
    # reclaim the just-created archive after cancelling this task.
    export_info["path"] = tar_path
    export_info["path_owned"] = True
    create_export_ownership_marker(tar_path)
    export_info["ownership_marker_required"] = True
    export_info["ownership_marker_owned"] = True
    await bind_export_path(export_info, tar_path)

    await run_in_owned_thread(
        _build_session_tar,
        tar_path,
        session_id,
        include_runtime,
    )

    return tar_path, _build_export_filename(session_id, state.name)


async def _retire_failed_export(info: dict[str, Any]) -> None:
    """Drain exact archive cleanup without replacing the primary failure."""
    try:
        await abandon_export_record(info)
    except BaseException:
        logger.exception(
            "Failed to retire export owner %s for session %s",
            info.get("token"),
            info.get("session_id"),
        )


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

    export_token = new_export_token()
    export_info = await reserve_export_record(
        export_token,
        session_id=session_id,
        user_id=user_id,
        state=state,
        kind="direct",
    )
    try:
        tmp_path, filename = await _pack_session_archive(
            session_id,
            state,
            include_runtime,
            prefix=EXPORT_STREAM_PREFIX,
            export_info=export_info,
        )
        await publish_export_ready(
            export_info,
            filename=filename,
            size=Path(tmp_path).stat().st_size,
        )
        claimed = await claim_export_stream(export_token, session_id)
        if claimed is None:
            raise HTTPException(status_code=410, detail="Export was retired")
    except BaseException:
        await _retire_failed_export(export_info)
        raise

    return LifecycleFileResponse(
        tmp_path,
        media_type="application/gzip",
        filename=filename,
        export_info=claimed,
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

    export_token = new_export_token()
    export_info = await reserve_export_record(
        export_token,
        session_id=session_id,
        user_id=user_id,
        state=state,
        kind="prepared",
    )
    try:
        tar_path, filename = await _pack_session_archive(
            session_id,
            state,
            include_runtime,
            prefix=PREPARED_EXPORT_PREFIX,
            export_info=export_info,
        )
        size_bytes = Path(tar_path).stat().st_size
        await publish_export_ready(
            export_info,
            filename=filename,
            size=size_bytes,
        )
    except BaseException:
        await _retire_failed_export(export_info)
        raise

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

    info = await claim_export_stream(token, session_id)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail="Export not prepared. Call POST /export/prepare first.",
        )
    if not os.path.exists(info["path"]):
        await _retire_failed_export(info)
        raise HTTPException(
            status_code=404,
            detail="Export not prepared. Call POST /export/prepare first.",
        )

    return LifecycleFileResponse(
        info["path"],
        media_type="application/gzip",
        filename=info["filename"],
        export_info=info,
    )
