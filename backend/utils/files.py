"""File path safety utilities — no imports from backend.main."""
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import HTTPException


def safe_filename(filename: str) -> str:
    """Extract safe base filename, preventing path traversal.
    Returns empty string for invalid input.
    """
    name = Path(filename).name
    if not name or name.startswith('.'):
        return ""
    return name


def validate_session_id(session_id: str) -> None:
    """Validate session ID to prevent path traversal (whitelist approach)."""
    if not session_id or len(session_id) > 128 or not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")


def slugify_session_name(name: str) -> str:
    """Sanitize a session name into a safe filename slug."""
    safe = re.sub(r'\s+', '_', "".join(c if c.isalnum() or c in '-_ ' else '' for c in name).strip())
    return safe if safe else "session"


def validate_path_within(path: Path, parent: Path) -> Path:
    """Validate that resolved path stays within expected parent directory."""
    resolved = path.resolve()
    parent_resolved = parent.resolve()
    if resolved != parent_resolved and not resolved.is_relative_to(parent_resolved):
        raise HTTPException(status_code=400, detail="Invalid path: path traversal detected")
    return resolved


@asynccontextmanager
async def temp_file_on_error(tmp_path: str) -> AsyncIterator[None]:
    """Context manager that deletes *tmp_path* if the body raises an exception.

    Usage::

        async with temp_file_on_error(tmp_path):
            await asyncio.to_thread(_build_tar, ...)
        # If we reach here the file is intact; caller is responsible for
        # scheduling its eventual deletion (e.g. via FileResponse background).
    """
    try:
        yield
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise