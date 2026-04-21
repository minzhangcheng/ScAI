"""File serving routes for filesystem access."""
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from backend.config import SESSION_DIR, REPORTS_DIR, UPLOAD_DIR, STATE_DIR, WORKSPACE_DIR
from backend.utils.agent_llm import validate_file_path_ownership as _validate_file_path_ownership

router = APIRouter(prefix="/api", tags=["files"])

_ALLOWED_DIRS = [
    Path(SESSION_DIR).resolve(),
    Path(REPORTS_DIR).resolve(),
    Path(UPLOAD_DIR).resolve(),
    Path(STATE_DIR).resolve(),
    Path(WORKSPACE_DIR).resolve(),
]


def _resolve_within_allowed_dirs(path: str) -> Path:
    """Resolve a file path and verify it stays within allowed data directories.
    Raises HTTPException if path traversal is detected or file not found.
    """
    try:
        real_path = Path(path).resolve()
    except (ValueError, OSError):
        raise HTTPException(status_code=400, detail="Invalid path")

    for allowed_dir in _ALLOWED_DIRS:
        try:
            real_path.relative_to(allowed_dir)
            return real_path
        except ValueError:
            continue

    raise HTTPException(status_code=403, detail="Access denied")


@router.get("/file")
async def get_file(path: str, request: Request):
    """Serve a file from the filesystem — ownership-checked by session_id prefix."""
    await _validate_file_path_ownership(path, request)

    real_path = _resolve_within_allowed_dirs(path)

    if not real_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if not real_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    return FileResponse(str(real_path))


@router.get("/download")
async def download_file(path: str, request: Request):
    """Download a file — ownership-checked by session_id prefix."""
    await _validate_file_path_ownership(path, request)

    real_path = _resolve_within_allowed_dirs(path)

    if not real_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    filename = real_path.name
    return FileResponse(str(real_path), filename=filename, media_type="application/octet-stream")
