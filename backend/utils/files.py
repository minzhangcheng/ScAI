"""File path safety utilities — no imports from backend.main."""
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Set

from fastapi import HTTPException

logger = logging.getLogger(__name__)


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


def resolve_figure_in_workspace(
    basename: str,
    workspace_dir: Path,
    seen_stems: Optional[Set[str]] = None,
) -> Optional[Path]:
    """Locate a figure file by basename within the workspace directory tree.

    Searches in priority order:
    1. workspace_dir / basename  (root)
    2. workspace_dir / "figures" / basename  (figures subdir)
    3. workspace_dir.rglob(basename)  (deep search — catches results/<dir>/figures/)

    Returns the first existing match, or None.  When multiple deep matches
    exist, the sorted-first is chosen deterministically and a warning is logged.

    If *seen_stems* is provided and the file's stem is already present, returns
    None (caller already collected a figure with that stem).
    """
    stem = os.path.splitext(basename)[0]
    if seen_stems is not None and stem in seen_stems:
        return None

    for candidate in [workspace_dir / basename, workspace_dir / "figures" / basename]:
        if candidate.exists():
            return candidate

    matches = [
        m for m in sorted(workspace_dir.rglob(basename))
        if m.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning("Multiple workspace files named %s; using %s", basename, matches[0])
    return matches[0]