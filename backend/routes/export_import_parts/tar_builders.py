# Archive member builders for state, snapshots, blobs, and reproducibility.
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

from .core import is_sandbox_pickle


def _add_json_bytes(tf, arcname: str, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    info.mode = 0o600
    tf.addfile(info, io.BytesIO(data))

def _add_portable_state_json(tf, source: Path, arcname: str) -> None:
    """Add state JSON without installation-bound or plaintext credentials."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    _add_json_bytes(tf, arcname, strip_analysis_state_secrets(payload))

def _add_snapshots_to_tar(tf, session_id: str, include_runtime: bool = False):
    """Add snapshot files to tar archive.

    Recurses so per-snapshot blob subdirectories (<snap_id>_blobs/<hash>.b64)
    are included alongside the top-level state.json / notebook.ipynb /
    session.pkl.zst files. Without recursion `inline_blobs()` on the imported
    snapshot would dangle, leaving raw "@scai-blob:<hash>" tokens in restored
    state.

    When ``include_runtime`` is False (default), sandbox session pickles
    (``*_session.pkl`` / ``*_session.pkl.zst``) are skipped. These can be
    multi-GB per snapshot and are only useful for in-memory kernel restore on
    the *same* trusted instance — they cannot be re-loaded by import on any
    other instance because ``dill.load`` is arbitrary code execution.
    """
    snapshots_dir = get_session_dir(session_id) / "snapshots"
    if not snapshots_dir.exists():
        return
    for snap_file in snapshots_dir.rglob("*"):
        if snap_file.is_file():
            if not include_runtime and is_sandbox_pickle(snap_file.name):
                continue
            rel = snap_file.relative_to(snapshots_dir)
            arcname = f"{session_id}/snapshots/{rel.as_posix()}"
            if snap_file.suffix.casefold() == ".json":
                _add_portable_state_json(tf, snap_file, arcname)
            else:
                tf.add(str(snap_file), arcname=arcname)

def _add_states_to_tar(tf, session_id: str, include_runtime: bool = False):
    """Add files under sessions/<id>/states/ to the tar.

    Skips the live sandbox pickle and LangGraph checkpoint unless
    ``include_runtime`` is true; notebooks remain portable and are always
    included.

    The checkpoint travels with the pkl rather than on its own because the two
    are only useful together: the checkpoint says which graph node to resume at,
    and the pkl holds the kernel values that node expects to find. Importing one
    without the other resumes into an empty kernel, which
    ``_mark_runtime_availability_after_import`` deliberately blocks.

    (Credentials sealed inside the checkpoint are no longer a reason to withhold
    it — :func:`backend.utils.secret_redaction.open_secret` degrades a value it
    cannot decrypt to empty instead of failing the load.)
    """
    def _add(state_file: Path):
        if not include_runtime and is_sandbox_pickle(state_file.name):
            return
        if not include_runtime and state_file.name.startswith("checkpoints.sqlite"):
            return
        tf.add(str(state_file), arcname=f"{session_id}/states/{state_file.name}")

    states_dir = get_session_states(session_id)
    if states_dir.exists():
        for sf in states_dir.iterdir():
            if sf.is_file():
                _add(sf)
    # Legacy fallback (flat STATE_DIR layout)
    for sf in STATE_DIR.glob(f"{session_id}*"):
        if sf.is_file():
            _add(sf)

def _add_session_blobs_to_tar(tf, session_id: str):
    """Add the session-level externalized blob store to tar archive.

    The blob_store offloads any string >= 1024 chars from session.json into
    <session_dir>/blobs/<sha256>.b64 sidecars. Without these files the
    imported session.json contains dangling "@scai-blob:<hash>" tokens that
    surface as raw text in the UI (e.g. inside the messages history).
    """
    blob_dir = get_session_dir(session_id) / "blobs"
    if not blob_dir.exists():
        return
    for blob_file in blob_dir.iterdir():
        if blob_file.is_file():
            tf.add(str(blob_file), arcname=f"{session_id}/blobs/{blob_file.name}")

def _add_reproducibility_to_tar(tf, session_id: str):
    """Add reproducibility bundle files to tar archive."""
    ws_dir = get_session_workspace(session_id)
    if not ws_dir.exists():
        ws_dir = WORKSPACE_DIR / session_id
    repro_dir = ws_dir / REPRODUCIBILITY_DIR_NAME
    if repro_dir.exists():
        for file_path in sorted(repro_dir.iterdir()):
            if file_path.is_file():
                tf.add(
                    str(file_path),
                    arcname=f"{session_id}/{REPRODUCIBILITY_DIR_NAME}/{file_path.name}",
                )
