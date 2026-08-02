"""File path safety, durable I/O, and atomic write utilities — no imports from backend.main."""
import logging
import os
import re
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from fastapi import HTTPException

logger = logging.getLogger(__name__)


# Flat state files written before the session-centric directory migration.
# Keep this exact: session IDs may contain underscores and one valid ID may be
# a prefix of another, so a glob such as ``f"{session_id}*"`` crosses session
# ownership boundaries.
LEGACY_SESSION_STATE_SUFFIXES = (
    ".ipynb",
    "_state.json",
    "_session.pkl",
    "_bioinfo_checkpoint.json",
    "_report_checkpoint.json",
)


def legacy_session_state_paths(
    state_dir: str | Path,
    session_id: str,
) -> tuple[Path, ...]:
    """Return the exact legacy flat-state paths owned by *session_id*."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", session_id or ""):
        return ()
    root = Path(state_dir)
    return tuple(root / f"{session_id}{suffix}" for suffix in LEGACY_SESSION_STATE_SUFFIXES)


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
    except BaseException:
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


# ── Durable atomic I/O primitives ───────────────────────────────────


def fsync_directory(path: Path) -> None:
    """Fsync *path* (must be a directory) to ensure its metadata is durable."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes, *, mode: int | None = None) -> None:
    """Atomically write *data* to *path* using a unique temp file and fsync(dir).

    On failure the temp file is removed.  Creates parent directories as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
        fsync_directory(path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Atomically write UTF-8 text to *path*."""
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Path, payload: Any, *, canonical: bool = False) -> None:
    """Atomically write JSON to *path*.

    When *canonical* is True, use sorted keys and compact separators
    (matching the canonical manifest serialisation pattern).
    """
    import json
    if canonical:
        data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    else:
        data = json.dumps(payload)
    atomic_write_text(path, data)


def atomic_copy_file(src: Path, dst: Path) -> None:
    """Atomically copy *src* to *dst*, fsyncing the destination directory.

    Reads *src*, writes to a unique temp next to *dst*, then renames.
    """
    data = src.read_bytes()
    atomic_write_bytes(dst, data)
