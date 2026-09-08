# Shared router/logger, size limits, and snapshot-member filters.
# Moved verbatim out of backend/routes/export_import.py.
import asyncio
import io
import json
import logging
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
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from backend.config import (
    DATA_BASE_DIR, REPORTS_DIR, SESSION_DIR, STATE_DIR, UPLOAD_DIR, WORKSPACE_DIR,
    PREPARED_EXPORT_TTL, UPLOAD_CHUNK_SIZE, UPLOAD_TEMP_DIR,
    get_session_dir, get_session_metadata_path, get_session_reports,
    get_session_states, get_session_uploads, get_session_workspace, resolve_session_metadata_path,
)
from backend.models.schemas import AnalysisState, APIResponse
from backend.services.reproducibility import REPRODUCIBILITY_DIR_NAME
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
from backend.utils.runtime_apply import RUNTIME_H5AD_MANIFEST_FILENAME
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

# tar.gz imports are deliberately uncapped -- real sessions run to tens of GB --
# so the protection against an oversized or highly compressible archive is a
# free-space floor rather than a size limit. Without it, extraction fills the
# data volume and every other upload and analysis on the installation starts
# failing on ENOSPC, with a multi-GB half-session left behind.
MIN_FREE_DISK_BYTES = 1024 * 1024 * 1024


def free_disk_bytes(path) -> int:
    """Bytes still available on the volume holding *path* (0 if unknown)."""
    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return 0

def check_export_disk_space() -> None:
    """Refuse to build an export archive on a volume that is already near full.

    An export is a second full-size copy of the session, written to the same data
    volume it came from, and nothing sized it before this. Import has refused to
    start under this floor from the beginning; without the mirror check on the way
    out, one export is what fills the disk and starts failing every other upload
    and analysis on the installation.
    """
    free = free_disk_bytes(UPLOAD_TEMP_DIR)
    if free and free < MIN_FREE_DISK_BYTES:
        raise HTTPException(
            status_code=507,
            detail=(
                "Not enough free disk space to build the export "
                f"({free // (1024 * 1024)} MB left). Free space and try again."
            ),
        )


def import_job_record(
    status: str,
    *,
    user_id: str,
    created_at: float,
    error: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build one entry of the import-job registry.

    ``phase`` always mirrors ``status``; the status endpoint spreads the record
    verbatim, so a field the completed record grows has to be reachable from the
    cancel and failure records too. Written in one place for that reason.
    """
    record: Dict[str, Any] = {
        "status": status,
        "phase": status,
        "user_id": user_id,
        "created_at": created_at,
    }
    if error is not None:
        record["error"] = error
    record.update(extra)
    return record

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

def is_runtime_generation_name(name: str) -> bool:
    """True for any file that belongs to the committed runtime generation.

    The dump, its signed manifest and that manifest's signature are one thing.
    Shipping the manifest without the dump lands a foreign-signed file in the
    destination, where `state_runtime_coherence` reads it, fails the HMAC and
    reports "inconsistent" -- for a session whose actual condition is that it
    carries no runtime at all. Export and import share this predicate so the
    parts cannot separate.
    """
    base = name.rsplit("/", 1)[-1]
    if is_sandbox_pickle(base):
        return True
    return bool(re.match(r".*_session\.runtime\.json(\.sig)?$", base))


def is_runtime_h5ad_name(name: str) -> bool:
    """True if a filename belongs to the live-runtime h5ad sidecar set.

    ``runtime_h5ad_manifest.json`` plus the ``*.h5ad`` files it names are runtime,
    not data: a Snapshot V2 restore writes them so a kernel can come back up
    holding ``adata`` when the dill residual alone cannot carry it. Export and
    import share this so the two halves are always dropped together -- a manifest
    whose sidecars are missing makes every later kernel start raise
    "h5ad sidecar missing", which is a brick rather than a degrade.

    Case-folded for the same reason as the checkpoint filter: the readers spell
    these names in lower case.
    """
    base = name.rsplit("/", 1)[-1]
    return base == RUNTIME_H5AD_MANIFEST_FILENAME or base.casefold().endswith(".h5ad")

_RETIRED_SNAPSHOT_SESSION_ROOTS = frozenset({
    "v2",
    "objects",
    "staging",
    "journals",
    "committed",
    "integrity",
})

_RETIRED_SNAPSHOT_OBJECT_MEMBER = re.compile(r"objects/[0-9a-f]{64}$")

def _is_retired_snapshot_member(name: str) -> bool:
    """Return whether a tar member belongs to the retired snapshot storage.

    Archives written by an older build carry it in two shapes: explicit
    snapshot exports use flat ``objects/`` and ``manifests/`` roots, ordinary
    session archives use session-prefixed storage roots. Nothing can
    materialize either any more, so both are skipped rather than rejected --
    an archive built before the removal must still import its session.
    """
    # Empty and "." segments are dropped before anything is compared: a member
    # named "./<sid>/v2/..." or "<sid>//objects/..." names the same path as the
    # plain form, and matching on the raw string let both bypass the session-root
    # rule below.
    parts = [p for p in name.split("/") if p not in ("", ".")]
    normalized = "/".join(parts)
    if normalized == "snapshot-v2-export.json":
        return True
    if _RETIRED_SNAPSHOT_OBJECT_MEMBER.fullmatch(normalized):
        return True
    if normalized.startswith("manifests/"):
        return True

    return len(parts) >= 2 and parts[1] in _RETIRED_SNAPSHOT_SESSION_ROOTS

# P3-7: cap export filename's slug portion to keep total filename below
# filesystem limits (most FS allow 255 *bytes*). When truncation is needed,
# append a short UUID suffix to preserve uniqueness across long names.
MAX_EXPORT_NAME = 200

# Manifest version stamped into every session archive this build writes. Both
# export endpoints share it; they used to disagree (1.0 vs 2.0) for byte-identical
# archives, which made the field useless for diagnosing a bad import.
ARCHIVE_FORMAT_VERSION = "2.0"


def new_export_token() -> str:
    """Mint the one-shot credential that authorizes an export download.

    The browser downloads through a plain ``<a download>`` navigation and cannot
    attach an Authorization header, so this token stands in for one. It must be
    unguessable: the previous scheme keyed prepared exports by ``v1-<session_id>``,
    which anyone holding a (deliberately auth-optional) report URL could derive.
    """
    return secrets.token_urlsafe(32)


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """Trim *text* to at most *max_bytes* UTF-8 bytes without splitting a character."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _build_export_filename(session_id: str, name: str, ext: str = ".tar.gz") -> str:
    """Build a length-capped, unique export filename.

    The cap counts bytes, not characters: filesystem name limits are in bytes, so
    a session named with ~78 CJK characters produced a 234-byte name that passed a
    200-*character* check and then failed ``tarfile.open`` with ENAMETOOLONG —
    leaving that session permanently un-exportable through the UI.
    """
    safe_name = slugify_session_name(name)
    truncated = _truncate_utf8(safe_name, MAX_EXPORT_NAME)
    if truncated != safe_name:
        suffix = uuid.uuid4().hex[:8]
        safe_name = f"{truncated}_{suffix}"
    return f"export_{session_id[:8]}_{safe_name}{ext}"
