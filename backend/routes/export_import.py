"""Export/Import routes for session tar.gz/zip archives.

Extracted from backend/main.py to decouple export-import handlers.
"""
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

# Kept verbatim from the pre-split module, as every other facade/parts split in
# this repo does, so `backend.routes.export_import.<name>` stays a live attribute
# for outside importers -- test_export_cleanup.py patches `asyncio.to_thread` and
# test_snapshot_v2_feature_gate.py reads DATA_BASE_DIR through this module. The
# second header below the re-exports is the rest of the same original block.

# Implementations live in ./export_import_parts/; re-exported here so existing
# importers and tests keep resolving them from this module.
from .export_import_parts.core import (  # noqa: F401
    ImportCancelled,
    _IMPORT_PROGRESS_UPDATE_INTERVAL_S,
    _IMPORT_UPLOAD_PROGRESS_BYTES,
    logger,
    router,
    MAX_INLINE_IMPORT_BYTES,
    MIN_FREE_DISK_BYTES,
    check_inline_import_size,
    free_disk_bytes,
    is_sandbox_pickle,
    _SNAPSHOT_V2_SESSION_ROOTS,
    _SNAPSHOT_V2_OBJECT_MEMBER,
    _is_disabled_snapshot_v2_member,
    MAX_EXPORT_NAME,
    ARCHIVE_FORMAT_VERSION,
    PREPARED_EXPORT_TTL,
    new_export_token,
    _truncate_utf8,
    _build_export_filename,
)
from .export_import_parts.tar_builders import (  # noqa: F401
    _add_json_bytes,
    _add_portable_state_json,
    _add_snapshots_to_tar,
    _add_states_to_tar,
    _add_session_blobs_to_tar,
    _add_session_logs_to_tar,
    _add_reproducibility_to_tar,
    _add_tree_to_tar,
    _safe_add_file,
)
from .export_import_parts.export_routes import (  # noqa: F401
    export_state,
    export_prepare,
    export_download,
    export_v2,
    _build_session_tar,
)
from .export_import_parts.import_routes import (  # noqa: F401
    import_v2,
    import_state,
    get_import_status,
    cancel_import,
)
from .export_import_parts.import_impl import (  # noqa: F401
    _process_import_job,
    _cleanup_partial_session,
    _resolve_import_dest,
    _import_tar_gz,
    _import_zip,
    _allocate_import_session_id,
    _arm_restore_intent_after_import,
    _claim_session_id,
    _is_checkpoint_member,
    _is_unsafe_member_name,
    _mark_runtime_availability_after_import,
    _normalize_imported_status,
    _rebind_session_id_values,
    _rebind_states_member_name,
    _remap_paths_in_text,
    _session_id_is_taken,
)


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
