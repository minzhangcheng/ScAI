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

from .import_archive_policy import _remap_paths_in_text, _rebind_session_id_values
from .import_state import _finalize_imported_state, _resign_imported_live_pkl

def import_zip(
    content: bytes,
    user_id: str,
    trust: bool = False,
    *,
    bindings,
) -> AnalysisState:
    """Legacy zip import (backward compatibility).

    See :func:`import_state` for the ``trust`` semantics.
    """
    # The allocator below takes ownership of the destination id by
    # creating sessions/<id>/ with mkdir(exist_ok=False). The tar.gz path
    # releases that claim on every failure (_process_import_job calls
    # _cleanup_partial_session on cancel, HTTPException and generic
    # errors); this one did not, so a truncated or malformed archive left
    # an ownerless directory with no session.json — invisible in the UI,
    # never reclaimed, and holding the archive's own id forever, so a
    # later valid import of the same archive was silently redirected to a
    # fresh uuid.
    session_id: Optional[str] = None
    try:
        zip_buffer = io.BytesIO(content)

        with zipfile.ZipFile(zip_buffer, 'r') as zf:
            namelist = zf.namelist()
            if not namelist:
                raise HTTPException(status_code=400, detail="Empty zip file")

            archive_session_id = namelist[0].split('/')[0]
            validate_session_id(archive_session_id)
            session_id = bindings._allocate_import_session_id(
                archive_session_id, user_id,
            )

            # Extract state.json
            state_json_path = f"{archive_session_id}/state.json"
            if state_json_path not in namelist:
                raise HTTPException(status_code=400, detail="state.json not found in archive")

            state_json_str = _remap_paths_in_text(
                zf.read(state_json_path).decode('utf-8'),
                "",
                str(DATA_BASE_DIR),
                archive_session_id,
                session_id,
            )

            state_data = json.loads(state_json_str)
            state_data = _rebind_session_id_values(state_data, archive_session_id, session_id)
            state_data["id"] = session_id
            state_data["user_id"] = user_id

            session_dir = get_session_dir(session_id)

            def _extract_tree(prefix: str, dest_root: Path) -> None:
                """Copy every member under *prefix* into *dest_root*, traversal-checked."""
                dest_root_resolved = dest_root.resolve()
                for name in namelist:
                    if not name.startswith(prefix) or name.endswith('/'):
                        continue
                    relative_path = name[len(prefix):]
                    if is_sandbox_pickle(relative_path) and not trust:
                        logger.warning(f"Skipping sandbox pickle from imported zip (untrusted): {name}")
                        continue
                    dest_path = (dest_root / relative_path).resolve()
                    # Robust traversal check via Path.is_relative_to (3.9+). Reject
                    # rather than silently skip — a malicious archive should fail loudly.
                    if not dest_path.is_relative_to(dest_root_resolved):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Archive entry escapes target directory: {name}",
                        )
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    # Streamed rather than read whole: these trees carry the
                    # uploaded matrices, which decompress to multiples of the
                    # 100 MB the zip itself is capped at.
                    with zf.open(name) as src, open(dest_path, "wb") as out_f:
                        shutil.copyfileobj(src, out_f, UPLOAD_CHUNK_SIZE)

            # Extract notebook if exists
            states_dir = get_session_states(session_id)
            states_dir.mkdir(parents=True, exist_ok=True)
            notebook_path = f"{archive_session_id}/notebook.ipynb"
            if notebook_path in namelist:
                (states_dir / f"{session_id}.ipynb").write_bytes(zf.read(notebook_path))

            # SECURITY (S4): sandbox pickles can execute arbitrary code via dill.load.
            # Drop them unless the caller passed trust=true to take responsibility.
            wrote_sandbox_pkl = False
            pkl_path = f"{archive_session_id}/session.pkl"
            if pkl_path in namelist:
                if trust:
                    logger.warning(f"Restoring sandbox pickle from zip (trust=true): {pkl_path}")
                    (states_dir / f"{session_id}_session.pkl").write_bytes(zf.read(pkl_path))
                    wrote_sandbox_pkl = True
                else:
                    logger.warning(f"Skipping sandbox pickle from imported zip (untrusted): {pkl_path}")

            # Extract metadata if exists
            metadata_path = f"{archive_session_id}/metadata.json"
            if metadata_path in namelist:
                (states_dir / f"{session_id}_state.json").write_bytes(zf.read(metadata_path))

            # Workspace, uploads, reports and the blob sidecars. Without the last
            # three the restored session points at files that never arrived and
            # renders raw "@scai-blob:<hash>" tokens in its history.
            _extract_tree(f"{archive_session_id}/workspace/", get_session_workspace(session_id))
            _extract_tree(f"{archive_session_id}/uploads/", get_session_uploads(session_id))
            _extract_tree(f"{archive_session_id}/reports/", get_session_reports(session_id))
            _extract_tree(f"{archive_session_id}/blobs/", session_dir / "blobs")

        validate_and_rebind_imported_reports(get_session_reports(session_id))
        if wrote_sandbox_pkl:
            _resign_imported_live_pkl(session_id)

        state = AnalysisState(**inline_blobs(state_data, get_session_dir(session_id) / "blobs"))
        _finalize_imported_state(state, imported_runtime=wrote_sandbox_pkl)
        return state

    except BaseException:
        bindings._cleanup_partial_session(session_id)
        raise
