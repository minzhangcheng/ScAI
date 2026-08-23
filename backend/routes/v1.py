"""V1 API routes: batch analysis API with chunked upload, TTL, and lifecycle management."""
import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List

from fastapi import APIRouter, Request, HTTPException, UploadFile, File, status, Depends
from pydantic import ValidationError

from backend.models.config_schema import ConfigV1
from backend.utils.auth import require_session_ownership
from backend.utils.config_resolver import resolve_config_for_user
from backend.utils.files import validate_path_within
from backend.services.config_io import apply_config_to_session, export_session_to_config
from backend.services.upload import (
    cleanup_session_uploads,
    complete_upload,
    init_upload,
    read_chunk_with_limit,
    upload_chunk,
)
from backend.services.report.paths import resolve_published_report_dir
from backend.config import (
    UPLOAD_CHUNK_SIZE,
    UPLOAD_TEMP_DIR,
    EXPORT_STREAM_PREFIX,
    ensure_session_dirs,
    get_session_dir,
)
from backend.services.export_lifecycle import (
    OwnedArchiveFileResponse,
    SecureArchiveError,
    build_secure_zip,
    create_export_ownership_marker,
    unlink_export_archive_and_marker,
)
from backend.utils.asyncio_tasks import run_in_owned_bulk_thread

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["v1"])

_V1_EXPORT_RESERVE_BYTES = 64 * 1024 * 1024


def _v1_report_artifact_is_public(relative_path: Path) -> bool:
    """Hide publication metadata, staging trees, and sibling lineages."""
    return bool(
        relative_path.parts
        and relative_path.parts[0] != "branches"
        and not any(part.startswith(".") for part in relative_path.parts)
    )


# ── Helpers ───────────────────────────────────────────────────────────────

def _placeholder_manifest(state) -> List[Dict[str, Any]]:
    """Return the upload bindings in the same order as the config files."""
    return [
        {
            "id": data_file.id,
            "filename": data_file.filename,
            "section": data_file.section,
            "group_id": data_file.group_id,
            "sample_number": data_file.sample_number,
        }
        for data_file in state.data_files
    ]


async def _create_pending_session(request: Request):
    """Parse ConfigV1, create a pending session, and return its state."""
    from backend.services.session_store import update_session_cache as _update_session_cache, async_save_session

    user_id = getattr(request.state, "user_id", "")
    try:
        body = await request.json()
        config = ConfigV1.model_validate(body)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"errors": e.errors()},
        )

    state = apply_config_to_session(config, user_id=user_id)
    session_id = state.id
    from backend.services.session_lifecycle import register_new_session_generation
    from backend.services.session_deletion import delete_session_generation

    await register_new_session_generation(state)
    try:
        ensure_session_dirs(session_id)
        await _update_session_cache(session_id, state)
        await async_save_session(session_id, state)
    except BaseException:
        # Preserve a stable tombstone for a generated ID whose initialization
        # failed; recreating under the same ID could revive stale client state.
        try:
            await asyncio.shield(
                delete_session_generation(session_id, state, user_id)
            )
        except Exception:
            logger.exception(
                "Failed to clean up partially-created V1 session %s", session_id
            )
        raise
    return state


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/analyses")
async def create_analysis(request: Request):
    """Create a new pending analysis from a ConfigV1 JSON.
    Returns job_id and the placeholder manifest for the separate uploads.
    """
    state = await _create_pending_session(request)
    return {
        "job_id": state.id,
        "status": "pending",
        "files": _placeholder_manifest(state),
    }


@router.post("/analyses/from-config")
async def create_from_config(request: Request):
    """Create a new pending session from a ConfigV1 JSON (for web UI import).
    Returns session_id. The session is pending — user must upload files before starting.
    """
    state = await _create_pending_session(request)
    return {
        "id": state.id,
        "status": "pending",
        "files": _placeholder_manifest(state),
    }


@router.post("/analyses/{session_id}/files/init")
async def init_file_upload(session_id: str, request: Request):
    """Initialize a chunked file upload for a session's placeholder file."""
    return await init_upload(session_id, request)


@router.post("/analyses/{session_id}/files/chunk/{upload_id}/{chunk_index}")
async def upload_chunk_endpoint(
    session_id: str,
    upload_id: str,
    chunk_index: int,
    request: Request,
    file: UploadFile = File(...),
):
    """Upload a single chunk of a file."""
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    chunk_data = await read_chunk_with_limit(
        file,
        max_bytes=UPLOAD_CHUNK_SIZE,
    )
    return await upload_chunk(session_id, upload_id, chunk_index, chunk_data)


@router.post("/analyses/{session_id}/files/complete/{upload_id}")
async def complete_file_upload(session_id: str, upload_id: str, request: Request):
    """Complete a file upload: assemble chunks, verify SHA256, move to session dir."""
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    return await complete_upload(session_id, upload_id)


@router.post("/analyses/{session_id}/start")
async def start_analysis(session_id: str, request: Request):
    """Start a pending analysis. Idempotent: running/completed/failed return current status."""
    from backend.services.workflow_runner import (
        SessionWorkflowConflictError,
        start_workflow_if_needed,
    )
    from backend.services.session_store import async_save_session, update_session_cache as _update_session_cache
    from backend.models.schemas import TaskStatus

    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    config = await resolve_config_for_user(user_id)

    if state.status in {TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.GENERATING_REPORT}:
        return {"job_id": session_id, "status": state.status.value}

    if state.status not in {TaskStatus.PENDING, TaskStatus.PAUSED}:
        raise HTTPException(status_code=409, detail=f"Cannot start from status {state.status.value}")

    if config is None:
        raise HTTPException(status_code=400, detail="No API key configured. Please configure your LLM API key in Settings before starting analysis.")

    unbound: List[str] = []
    stale: List[str] = []
    for data_file in state.data_files:
        if not data_file.file_path:
            unbound.append(data_file.filename)
        elif not Path(data_file.file_path).exists():
            stale.append(data_file.filename)
    if unbound or stale:
        problems = []
        if unbound:
            problems.append(
                "declared in the configuration but never uploaded: "
                + ", ".join(unbound)
            )
        if stale:
            problems.append(
                "bound to a path that no longer exists: " + ", ".join(stale)
            )
        raise HTTPException(
            status_code=400,
            detail=(
                "Cannot start: these data files are "
                + "; ".join(problems)
                + ". Upload them, or drop them with "
                "DELETE /api/session/{session_id}/data/{file_id}."
            ),
        )

    async def _prepare_start() -> None:
        state.status = TaskStatus.RUNNING
        await _update_session_cache(session_id, state)
        await async_save_session(session_id, state)

    try:
        started = await start_workflow_if_needed(
            session_id,
            expected_state=state,
            prepare=_prepare_start,
        )
    except SessionWorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not started:
        raise HTTPException(
            status_code=409,
            detail="Another workflow owner became active for this session",
        )
    return {"job_id": session_id, "status": "running"}


@router.delete("/analyses/{session_id}")
async def delete_analysis(session_id: str, request: Request):
    """Delete an analysis. Cannot delete running sessions (returns 409)."""
    user_id = getattr(request.state, "user_id", "")
    from backend.services.session_lifecycle import tombstone_visible_to_owner
    from backend.services.session_deletion import (
        SessionProducerStopError,
        cleanup_tombstoned_generation,
        delete_session_generation,
    )

    if tombstone_visible_to_owner(session_id, user_id):
        stats = await cleanup_tombstoned_generation(session_id, user_id)
        return {
            "deleted": int(stats.get("pending_producers") or 0) == 0,
            "job_id": session_id,
            "pending_cleanup": int(stats.get("pending_producers") or 0) > 0,
        }
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    try:
        await delete_session_generation(session_id, state, user_id)
    except SessionProducerStopError:
        return {
            "deleted": False,
            "job_id": session_id,
            "pending_cleanup": True,
        }

    return {"deleted": True, "job_id": session_id}


@router.get("/analyses/{session_id}")
async def get_analysis(session_id: str, request: Request):
    """Get analysis status, progress, and artifacts index."""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    reports_dir = resolve_published_report_dir(
        session_id,
        getattr(state, "restore_branch_id", None),
    )
    artifacts: List[Dict[str, Any]] = []
    if reports_dir.exists():
        for f in sorted(reports_dir.rglob("*")):
            relative = f.relative_to(reports_dir)
            if (
                f.is_file()
                and not f.is_symlink()
                and _v1_report_artifact_is_public(relative)
            ):
                artifacts.append({
                    "name": str(relative),
                    "size": f.stat().st_size,
                })

    return {
        "job_id": session_id,
        "status": state.status.value,
        "current_round": state.current_round,
        "max_rounds": state.max_rounds,
        "name": state.name,
        "created_at": state.created_at.isoformat() if state.created_at else "",
        "updated_at": state.updated_at.isoformat() if state.updated_at else "",
        "artifacts": artifacts,
    }


@router.get("/analyses/{session_id}/artifacts/{name:path}")
async def download_artifact(session_id: str, name: str, request: Request):
    """Download a single artifact file."""
    from fastapi.responses import FileResponse

    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    reports_dir = resolve_published_report_dir(
        session_id,
        getattr(state, "restore_branch_id", None),
    )
    relative = Path(name)
    if not _v1_report_artifact_is_public(relative):
        raise HTTPException(status_code=404, detail="Artifact not found")
    file_path = validate_path_within(reports_dir / name, reports_dir)

    if file_path.is_symlink() or not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(file_path, filename=file_path.name)


@router.get("/analyses/{session_id}/export")
async def export_analysis(session_id: str, request: Request):
    """Download the full session as a zip."""
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    session_dir = get_session_dir(session_id)
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session directory not found")

    UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=EXPORT_STREAM_PREFIX,
        suffix=".zip",
        dir=str(UPLOAD_TEMP_DIR),
    )
    tmp = Path(tmp_name)
    metadata = os.fstat(fd)
    owned_identity = (int(metadata.st_dev), int(metadata.st_ino))
    marker_owned = False

    def cleanup_stage() -> None:
        try:
            unlink_export_archive_and_marker(
                tmp,
                expected_identity=owned_identity,
                remove_marker=marker_owned,
            )
        except OSError as exc:
            logger.warning("Failed to clean V1 export archive %s: %s", tmp, exc)
        finally:
            os.close(fd)

    try:
        await run_in_owned_bulk_thread(
            build_secure_zip,
            session_dir,
            tmp,
            destination_fd=fd,
            recursive=True,
            include=lambda _relative: True,
            max_files=None,
            max_source_bytes=None,
            reserve_bytes=_V1_EXPORT_RESERVE_BYTES,
        )
        current_metadata = os.fstat(fd)
        current_identity = (
            int(current_metadata.st_dev),
            int(current_metadata.st_ino),
        )
        if current_identity != owned_identity:
            raise OSError("V1 export staging descriptor changed during construction")
        owned_identity = current_identity
    except SecureArchiveError as exc:
        cleanup_stage()
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except BaseException:
        cleanup_stage()
        raise

    try:
        create_export_ownership_marker(
            tmp,
            expected_identity=owned_identity,
        )
        marker_owned = True
    except BaseException:
        cleanup_stage()
        raise

    try:
        return OwnedArchiveFileResponse(
            tmp,
            file_fd=fd,
            owned_identity=owned_identity,
            marker_owned=marker_owned,
            filename=f"scai-{session_id}.zip",
            media_type="application/zip",
        )
    except BaseException:
        cleanup_stage()
        raise


@router.get("/analyses/{session_id}/config/export")
async def export_config_v1(session_id: str, request: Request):
    """Export session configuration as ConfigV1 JSON."""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    try:
        config_dict = export_session_to_config(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export config: {str(e)}")

    return config_dict
