# Shared router/logger, size limits, and snapshot-member filters.
# Moved verbatim out of backend/routes/export_import.py.
import asyncio
import io
import json
import logging
import os
import re
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
from starlette.responses import FileResponse


class ImportCancelled(Exception):
    """Raised inside the extraction worker when the user requests cancellation."""
    pass

# Throttle progress writes from the extraction worker so we don't update the
# job dict on every file (some archives have tens of thousands of small files).
_IMPORT_PROGRESS_UPDATE_INTERVAL_S = 0.5

# Throttle upload-phase progress updates (every ~64 MB).
_IMPORT_UPLOAD_PROGRESS_BYTES = 64 * 1024 * 1024

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["export_import"])

# S3: 100 MB cap for inline zip/json imports to prevent OOM
MAX_INLINE_IMPORT_BYTES = 100 * 1024 * 1024

def check_inline_import_size(content_length: Optional[int], max_bytes: int = MAX_INLINE_IMPORT_BYTES) -> None:
    """Fast-reject oversized inline imports based on the Content-Length header."""
    if content_length is not None and content_length > max_bytes:
        raise HTTPException(status_code=413, detail=f"Import file too large (max {max_bytes} bytes)")

def is_sandbox_pickle(name: str) -> bool:
    """True if an archive entry is a sandbox session pickle.

    SECURITY: dill.load on imported pickle is arbitrary code execution, so these
    files are skipped during import. Matches 'session.pkl', '<id>_session.pkl',
    and '<id>_session.pkl.zst' (compressed snapshot format).
    """
    base = name.rsplit("/", 1)[-1]
    return bool(re.match(r".*_session\.pkl(\.zst)?$", base) or base == "session.pkl")

_SNAPSHOT_V2_SESSION_ROOTS = frozenset({
    "v2",
    "objects",
    "staging",
    "journals",
    "committed",
    "integrity",
})

_SNAPSHOT_V2_OBJECT_MEMBER = re.compile(r"objects/[0-9a-f]{64}$")

def _is_disabled_snapshot_v2_member(name: str) -> bool:
    """Return whether a tar member belongs to Snapshot V2 storage.

    Explicit V2 exports use flat ``objects/`` and ``manifests/`` roots, while
    ordinary session archives may carry session-prefixed V2 storage roots.
    Neither form may be materialized through the ordinary import path when the
    subsystem is disabled.
    """
    if is_snapshot_v2_enabled():
        return False

    normalized = name.strip("/")
    if normalized == "snapshot-v2-export.json":
        return True
    if _SNAPSHOT_V2_OBJECT_MEMBER.fullmatch(normalized):
        return True
    if normalized.startswith("manifests/"):
        return True

    parts = normalized.split("/")
    return len(parts) >= 2 and parts[1] in _SNAPSHOT_V2_SESSION_ROOTS

# P3-7: cap export filename's slug portion to keep total filename below
# filesystem limits (most FS allow 255 bytes). When truncation is needed,
# append a short UUID suffix to preserve uniqueness across long names.
MAX_EXPORT_NAME = 200

def _build_export_filename(session_id: str, name: str, ext: str = ".tar.gz") -> str:
    """Build a length-capped, unique export filename."""
    safe_name = slugify_session_name(name)
    if len(safe_name) > MAX_EXPORT_NAME:
        suffix = uuid.uuid4().hex[:8]
        safe_name = f"{safe_name[:MAX_EXPORT_NAME]}_{suffix}"
    return f"export_{session_id[:8]}_{safe_name}{ext}"
