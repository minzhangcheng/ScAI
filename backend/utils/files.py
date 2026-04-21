"""File path safety utilities — no imports from backend.main."""
import re
from pathlib import Path

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


def validate_path_within(path: Path, parent: Path) -> Path:
    """Validate that resolved path stays within expected parent directory."""
    resolved = path.resolve()
    parent_resolved = parent.resolve()
    if resolved != parent_resolved and not resolved.is_relative_to(parent_resolved):
        raise HTTPException(status_code=400, detail="Invalid path: path traversal detected")
    return resolved