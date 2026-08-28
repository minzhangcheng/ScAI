# Import job execution and archive extraction.
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
    get_session_dir, get_session_logs, get_session_metadata_path, get_session_reports,
    get_session_states, get_session_uploads, get_session_workspace, resolve_session_metadata_path,
)
from backend.models.schemas import AnalysisState, APIResponse
from backend.services.reproducibility import REPRODUCIBILITY_DIR_NAME
from backend.services.report.output_generation import (
    validate_and_rebind_imported_reports,
)
from backend.services.snapshot_v2.checkpoint_paths import (
    CHECKPOINT_SQLITE_FILENAME,
    checkpoint_sqlite_path,
)
from backend.services.snapshot_v2.feature_gate import (
    is_snapshot_v2_enabled,
    normalize_disabled_snapshot_state,
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
from backend.utils.blob_store import externalize_blobs, inline_blobs
from backend.utils.files import slugify_session_name, validate_path_within, validate_session_id, temp_file_on_error
from backend.utils.secret_redaction import strip_analysis_state_secrets
from starlette.background import BackgroundTask
from starlette.responses import FileResponse

from .core import (
    ARCHIVE_FORMAT_VERSION,
    ImportCancelled,
    MIN_FREE_DISK_BYTES,
    _IMPORT_PROGRESS_UPDATE_INTERVAL_S,
    _is_disabled_snapshot_v2_member,
    free_disk_bytes,
    import_job_record,
    is_runtime_h5ad_name,
    is_sandbox_pickle,
)


logger = logging.getLogger(__name__)

def _session_id_is_taken(session_id: str, user_id: str | None = None) -> bool:
    """Whether *session_id* already names a session on this installation.

    Checks disk and the in-memory registries, because a session evicted from
    ``sessions`` still owns its directory, and a session mid-creation may own a
    workflow before its directory is fully populated.
    """
    from backend.state import sessions as _live_sessions, workflows as _live_workflows

    from backend.services.session_lifecycle import read_lifecycle_record

    lifecycle = read_lifecycle_record(session_id)
    if lifecycle is not None and not (
        lifecycle.get("status") == "deleted"
        and user_id is not None
        and str(lifecycle.get("owner_id") or "") == user_id
    ):
        return True
    if get_session_dir(session_id).exists():
        return True
    return session_id in _live_sessions or session_id in _live_workflows


def _claim_session_id(session_id: str, user_id: str) -> bool:
    """Take exclusive ownership of *session_id* by creating its directory.

    ``mkdir(exist_ok=False)`` is the claim, so two imports racing for the same
    id cannot both decide it is free: exactly one creates the directory and the
    other falls through to a fresh id. Checking-then-creating would let both
    extract into the same tree and interleave two sessions' files.
    """
    if _session_id_is_taken(session_id, user_id):
        return False
    from backend.services.session_lifecycle import claim_import_generation

    generation = claim_import_generation(
        session_id,
        user_id,
        get_session_dir(session_id),
    )
    if generation is None:
        return False
    return True


def _allocate_import_session_id(archive_session_id: str, user_id: str) -> str:
    """Pick the session id an archive should be extracted into, and claim it.

    Reusing the archive's own id is what makes an import round-trip on the same
    installation (re-import to restore a backup) *destructive*: the extraction
    writes straight over the live session's files, and ``session.json`` is then
    rewritten with the importer's ``user_id`` — which on a shared installation
    silently transfers another user's session. Importing is therefore defined as
    "create a session", never "overwrite one": on any collision the archive is
    materialized under a fresh id and every embedded path is rebound to it.
    """
    if _claim_session_id(archive_session_id, user_id):
        return archive_session_id
    new_id = str(uuid.uuid4())
    while not _claim_session_id(new_id, user_id):  # pragma: no cover - uuid4 collision
        new_id = str(uuid.uuid4())
    logger.warning(
        "Import target %s already exists on this installation; extracting into "
        "a new session %s instead of overwriting it.",
        archive_session_id,
        new_id,
    )
    return new_id
