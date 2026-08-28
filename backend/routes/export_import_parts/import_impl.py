# Import job execution and archive extraction.
# Moved verbatim out of backend/routes/export_import.py.
import asyncio
import io
import json
import logging
import os
import re
import shutil
import sys
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

from .import_archive_policy import (
    _SUPPORTED_ARCHIVE_VERSIONS,
    _drop_kernelless_checkpoints,
    _is_checkpoint_member,
    _is_runtime_h5ad_member,
    _is_unsafe_member_name,
    _rebind_session_id_values,
    _rebind_states_member_name,
    _remap_paths_in_text,
    _resolve_import_dest,
)
from .import_identity import (
    _allocate_import_session_id,
    _claim_session_id,
    _session_id_is_taken,
)
from .import_state import (
    _arm_restore_intent_after_import,
    _cleanup_partial_session,
    _finalize_imported_state,
    _mark_runtime_availability_after_import,
    _normalize_imported_status,
    _resign_imported_live_pkl,
)
from . import import_tar, import_zip

async def _process_import_job(tmp_path_str: str, import_job_id: str, user_id: str, trust: bool = False):
    """Background task: extract tar.gz archive and load session."""
    logger.info(f"Import {import_job_id}: starting extraction from {tmp_path_str}")
    archive_size = Path(tmp_path_str).stat().st_size if Path(tmp_path_str).exists() else 0
    # Move into the extracting phase; preserve cancel flag set during upload.
    async with _import_jobs_lock:
        prev = _import_jobs.get(import_job_id, {})
        created_at = prev.get("created_at", time.time())
        _import_jobs[import_job_id] = import_job_record(
            "extracting",
            user_id=user_id,
            created_at=created_at,
            bytes_total=archive_size,
            bytes_processed=0,
            files_total=0,
            files_processed=0,
            current_file="",
            cancel_requested=prev.get("cancel_requested", False),
        )
    # Populated by the extraction worker the moment the destination id is known,
    # so a cancel or a mid-archive failure can still remove what it created.
    # The worker only ever writes into a session directory it allocated itself
    # (see _allocate_import_session_id), which is what makes this cleanup safe.
    outcome: Dict[str, Any] = {}
    try:
        state = await asyncio.to_thread(
            _import_tar_gz, tmp_path_str, user_id, import_job_id, trust, outcome
        )

        validate_session_id(state.id)

        # Forced: this is the write that turns the extracted tree into a listable
        # session, and PAUSED does not bypass the save throttle on its own.
        await async_save_session(state.id, state, force=True)
        await _update_session_cache(state.id, state)
        from backend.services.session_lifecycle import commit_import_generation

        commit_import_generation(state)
        async with _import_jobs_lock:
            _import_jobs[import_job_id] = import_job_record(
                "completed",
                user_id=user_id,
                created_at=created_at,
                session_id=state.id,
                name=state.name,
                # Zeros are filtered so the common case stays an empty dict; the
                # endpoint spreads this record verbatim.
                skipped={
                    k: v for k, v in (outcome.get("skipped") or {}).items() if v
                },
                skipped_examples=outcome.get("skipped_examples") or [],
            )
        logger.info(f"Import {import_job_id}: completed, session_id={state.id}")
    except ImportCancelled as e:
        logger.info(f"Import {import_job_id}: cancelled by user ({e})")
        await asyncio.to_thread(_cleanup_partial_session, outcome.get("session_id"))
        async with _import_jobs_lock:
            _import_jobs[import_job_id] = import_job_record("cancelled", user_id=user_id, created_at=created_at, error="cancelled by user")
    except HTTPException as e:
        logger.error(f"Import {import_job_id}: failed (HTTPException): {e.detail}")
        await asyncio.to_thread(_cleanup_partial_session, outcome.get("session_id"))
        async with _import_jobs_lock:
            _import_jobs[import_job_id] = import_job_record("failed", user_id=user_id, created_at=created_at, error=e.detail)
    except Exception as e:
        logger.exception(f"Import {import_job_id}: failed with exception")
        await asyncio.to_thread(_cleanup_partial_session, outcome.get("session_id"))
        async with _import_jobs_lock:
            _import_jobs[import_job_id] = import_job_record("failed", user_id=user_id, created_at=created_at, error=str(e))
    finally:
        Path(tmp_path_str).unlink(missing_ok=True)

def _import_tar_gz(
    file_path: str,
    user_id: str,
    import_job_id: Optional[str] = None,
    trust: bool = False,
    outcome: Optional[Dict[str, Any]] = None,
) -> AnalysisState:
    return import_tar.import_tar_gz(
        file_path,
        user_id,
        import_job_id=import_job_id,
        trust=trust,
        outcome=outcome,
        bindings=sys.modules[__name__],
    )


def _import_zip(content: bytes, user_id: str, trust: bool = False) -> AnalysisState:
    return import_zip.import_zip(
        content,
        user_id,
        trust=trust,
        bindings=sys.modules[__name__],
    )
