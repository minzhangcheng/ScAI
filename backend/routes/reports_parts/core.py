# Shared router, generation-pending state, and report path/cache helpers.
# Moved verbatim out of backend/routes/reports.py.
"""Report routes: exploration-report, annotation-report, detailed-report, report/*, 
publication-figures, publication-tables, figure/{id}/pdf, figure/{id}/png, integrated-report.

Extracted from backend/main.py to decouple report generation and serving route handlers.
"""
import asyncio
import uuid
import json
import logging
import math
import os
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter


from backend.state import detailed_reports
from backend.services.report.paths import resolve_report_target_lineage

logger = logging.getLogger(__name__)

REPORT_ARCHIVE_PREFIX = "previous_report_v"


def _should_skip_in_report_zip(file_path: Path) -> bool:
    """Exclude report archives and leftover partial archive files."""
    name = file_path.name
    if name.startswith(REPORT_ARCHIVE_PREFIX) and name.endswith(".zip"):
        return True
    if name.endswith(".zip.tmp"):
        return True
    return False

def _branch_query(branch_id: Optional[str]) -> str:
    if not branch_id:
        return ""
    return f"?branch_id={quote(branch_id, safe='')}"

def _report_cache_key(session_id: str, branch_id: Optional[str]) -> str:
    target_lineage_id = resolve_report_target_lineage(branch_id)
    if target_lineage_id == "main":
        return session_id
    return f"{session_id}:{target_lineage_id}"

def _report_cache_key_for_lineage(session_id: str, target_lineage_id: str) -> str:
    if target_lineage_id == "main":
        return session_id
    return f"{session_id}:{target_lineage_id}"

router = APIRouter(prefix="/api/session", tags=["reports"])

# Note: _report_generation_pending is process-local. In multi-worker deployments
# (e.g., uvicorn --workers N), cross-worker concurrency is not prevented. This is
# consistent with the existing chat_task_lock pattern in chat.py. For multi-worker
# production, a Redis-based lock would be needed — out of scope for this fix.
_report_generation_pending: dict[str, str] = {}

_pending_lock = asyncio.Lock()

async def _try_begin_generation(session_id: str) -> str | None:
    """Atomically check-and-set _report_generation_pending.

    Returns an opaque ownership token if generation was started, or ``None``
    when another owner is already pending. The token prevents an old request's
    cleanup from clearing a marker installed after same-ID reuse.
    """
    async with _pending_lock:
        if session_id in _report_generation_pending:
            return None
        token = uuid.uuid4().hex
        _report_generation_pending[session_id] = token
        return token

def clear_generation_pending(session_id: str, token: str | None = None) -> None:
    """Remove session_id from the report-generation-pending set.

    Public helper so that external cleanup code (session deletion, periodic
    cleanup) does not need to import the private ``_report_generation_pending``
    mapping directly. Lifecycle deletion intentionally omits ``token`` to clear
    the current generation; producer finalizers must provide their token and
    therefore compare-and-clear only their own marker.
    """
    if token is None or _report_generation_pending.get(session_id) == token:
        _report_generation_pending.pop(session_id, None)

def clear_detailed_report_cache(session_id: str) -> None:
    """Remove all in-memory detailed report cache entries for a session."""
    detailed_reports.pop(session_id, None)
    prefix = f"{session_id}:"
    for key in list(detailed_reports.keys()):
        if key.startswith(prefix):
            detailed_reports.pop(key, None)

def _sanitize_nan(obj):
    """Recursively replace NaN/Infinity float values with None for JSON compliance."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj

def _load_report_json(path: Path) -> dict:
    """Load a JSON report, handling NaN/Infinity via parse_constant (not regex on raw text)."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    data = json.loads(content, parse_constant=lambda c: None)
    return _sanitize_nan(data)

def _next_archive_version(reports_dir: Path) -> int:
    """Determine the next archive version number as max(existing) + 1."""
    if not reports_dir.exists():
        return 1
    existing_versions = []
    for p in reports_dir.glob(f"{REPORT_ARCHIVE_PREFIX}*.zip"):
        try:
            version = int(p.stem.split("_v")[-1])
            existing_versions.append(version)
        except (ValueError, IndexError):
            continue
    return max(existing_versions, default=0) + 1

def _create_report_archive(reports_dir: Path, version: int) -> Path:
    """Create a versioned zip archive of the reports directory, excluding existing archives.
    
    Uses atomic write (tmp file + os.replace) to avoid self-inclusion race.
    """
    archive_path = reports_dir / f"{REPORT_ARCHIVE_PREFIX}{version}.zip"
    tmp_path = archive_path.with_suffix(".zip.tmp")
    
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED) as zf:
        for file_path in reports_dir.rglob("*"):
            if file_path.is_file():
                if _should_skip_in_report_zip(file_path):
                    continue
                arcname = file_path.relative_to(reports_dir)
                zf.write(file_path, arcname)
    
    os.replace(tmp_path, archive_path)
    return archive_path

def _enforce_archive_retention(reports_dir: Path, max_archives: int = 5):
    """Keep only the most recent max_archives archives, delete the rest sorted by mtime."""
    archives = sorted(
        reports_dir.glob(f"{REPORT_ARCHIVE_PREFIX}*.zip"),
        key=lambda p: p.stat().st_mtime,
    )
    while len(archives) > max_archives:
        oldest = archives.pop(0)
        oldest.unlink(missing_ok=True)
