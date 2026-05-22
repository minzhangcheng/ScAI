"""V1 API routes: batch analysis API with chunked upload, TTL, and lifecycle management."""
import asyncio
import logging
import zipfile
from pathlib import Path
from typing import Dict, Any, List

from fastapi import APIRouter, Request, HTTPException, UploadFile, File, status, Depends
from pydantic import ValidationError

from backend.models.config_schema import ConfigV1
from backend.utils.auth import require_session_ownership
from backend.utils.config_resolver import resolve_config_for_user
from backend.utils.files import validate_path_within
from backend.services.config_io import apply_config_to_session, export_session_to_config
from backend.services.upload import init_upload, upload_chunk, complete_upload, cleanup_session_uploads
from backend.config import (
    get_session_dir, ensure_session_dirs,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["v1"])


# ── Helpers ───────────────────────────────────────────────────────────────

async def _create_pending_session(request: Request) -> str:
    """Parse ConfigV1 from request body, create pending session, return session_id."""
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
    ensure_session_dirs(session_id)
    await _update_session_cache(session_id, state)
    await async_save_session(session_id, state)
    return session_id


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/analyses")
async def create_analysis(request: Request):
    """Create a new pending analysis from a ConfigV1 JSON.
    Returns job_id. Files must be uploaded separately, then call /start.
    """
    session_id = await _create_pending_session(request)
    return {"job_id": session_id, "status": "pending"}


@router.post("/analyses/from-config")
async def create_from_config(request: Request):
    """Create a new pending session from a ConfigV1 JSON (for web UI import).
    Returns session_id. The session is pending — user must upload files before starting.
    """
    session_id = await _create_pending_session(request)
    return {"id": session_id, "status": "pending"}


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

    chunk_data = await file.read()
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
    from backend.services.workflow_runner import start_workflow_if_needed
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

    missing = [df.filename for df in state.data_files if not df.file_path]
    if missing:
        logger.warning(f"Starting analysis {session_id} with missing files: {missing}")

    state.status = TaskStatus.RUNNING
    await _update_session_cache(session_id, state)
    await async_save_session(session_id, state)

    await start_workflow_if_needed(session_id)
    return {"job_id": session_id, "status": "running"}


@router.delete("/analyses/{session_id}")
async def delete_analysis(session_id: str, request: Request):
    """Delete an analysis. Cannot delete running sessions (returns 409)."""
    from backend.state import sessions, workflows, state_lock as _state_lock, initializing_sessions as _initializing_sessions
    from backend.state import websockets
    import shutil

    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    async with _state_lock:
        if session_id in workflows or session_id in _initializing_sessions:
            raise HTTPException(status_code=409, detail="Cannot delete a running analysis. Stop it first.")
        sessions.pop(session_id, None)
        websockets.pop(session_id, None)
        from backend.state import release_session_lock_slot
        release_session_lock_slot(session_id)

    await cleanup_session_uploads(session_id)

    session_dir = get_session_dir(session_id)
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)

    from backend.utils.auth import get_db
    db = await get_db()
    await db.delete_session_record(session_id)

    return {"deleted": True, "job_id": session_id}


@router.get("/analyses/{session_id}")
async def get_analysis(session_id: str, request: Request):
    """Get analysis status, progress, and artifacts index."""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    reports_dir = (get_session_dir(session_id) / "reports")
    artifacts: List[Dict[str, Any]] = []
    if reports_dir.exists():
        for f in sorted(reports_dir.rglob("*")):
            if f.is_file():
                artifacts.append({
                    "name": str(f.relative_to(reports_dir)),
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
    await require_session_ownership(session_id=session_id, user_id=user_id)

    reports_dir = (get_session_dir(session_id) / "reports")
    file_path = validate_path_within(reports_dir / name, reports_dir)

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(file_path, filename=file_path.name)


def _zip_session_dir(session_dir: Path, tmp: Path) -> None:
    """Synchronous helper to zip a session directory (runs in thread pool)."""
    with zipfile.ZipFile(str(tmp), 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in session_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(session_dir))


@router.get("/analyses/{session_id}/export")
async def export_analysis(session_id: str, request: Request):
    """Download the full session as a zip."""
    from fastapi.responses import FileResponse

    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    session_dir = get_session_dir(session_id)
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session directory not found")

    export_dir = session_dir / "workspace"
    if export_dir.exists():
        for f in export_dir.glob("*.zip"):
            return FileResponse(f, filename=f.name, media_type="application/zip")

    import tempfile
    from starlette.background import BackgroundTask
    tmp = Path(tempfile.mkstemp(suffix=".zip")[1])
    try:
        await asyncio.to_thread(_zip_session_dir, session_dir, tmp)
        return FileResponse(
            str(tmp),
            filename=f"scai-{session_id}.zip",
            media_type="application/zip",
            background=BackgroundTask(tmp.unlink, missing_ok=True),
        )
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")


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