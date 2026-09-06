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
from backend.utils.checkpoint_paths import (
    CHECKPOINT_SQLITE_FILENAME,
    checkpoint_sqlite_path,
)
from backend.services.legacy_state_sanitizer import sanitize_legacy_state
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
    _is_retired_snapshot_member,
    free_disk_bytes,
    import_job_record,
    is_runtime_h5ad_name,
    is_sandbox_pickle,
)


logger = logging.getLogger(__name__)

_SUPPORTED_ARCHIVE_VERSIONS = frozenset({"1.0", ARCHIVE_FORMAT_VERSION})


def _is_unsafe_member_name(name: str) -> bool:
    """Reject archive names that could escape the destination directory.

    Segment-wise rather than substring-wise: ``a/../b`` escapes and must go,
    while ``DEG_A..B.csv`` is an ordinary filename an analysis really produces —
    the old ``'..' in name`` test dropped those silently, so the restored
    session was missing files the archive actually contained.
    """
    if not name or name.startswith("/") or "\\" in name or "\x00" in name:
        return True
    return any(segment == ".." for segment in name.split("/"))


def _rebind_states_member_name(name: str, src_id: str, session_id: str) -> str:
    """Rename a ``states/`` member after the destination session.

    ``states/`` is the one archive root whose session id lives in the *filename*
    rather than in a path segment the caller strips — ``<id>.ipynb``,
    ``<id>_session.pkl`` and the ``<id>_*_checkpoint.json`` family are all
    looked up by name against the id of the session that owns them. An import
    that had to allocate a fresh id used to write them under the archive's name,
    where nothing ever looks: the notebook was silently lost, and the pkl sat
    next to a ``_live_pkl`` that reported no kernel at all.

    The suffix must start at a ``.`` or ``_`` boundary. Session ids are uuid4
    and therefore all the same length, so one cannot really prefix another — but
    ids only have to satisfy ``validate_session_id``, and a hand-built archive
    declaring ``abc`` would otherwise rewrite an unrelated ``abcd_session.pkl``.
    """
    if src_id == session_id or not name.startswith(src_id):
        return name
    rest = name[len(src_id):]
    if rest and rest[0] not in "._":
        return name
    return f"{session_id}{rest}"


def _is_checkpoint_member(relative_to_base: str) -> bool:
    """Whether a member is the LangGraph checkpoint database (or its WAL/SHM).

    Case-folded: the lookup in ``lifecycle.run_analysis`` spells the name in
    lower case, so on a case-insensitive volume an archive carrying
    ``CHECKPOINTS.SQLITE`` would resolve there while slipping past this filter.
    """
    if not relative_to_base.startswith("states/"):
        return False
    rest = relative_to_base[len("states/"):]
    return "/" not in rest and rest.casefold().startswith(CHECKPOINT_SQLITE_FILENAME)


def _is_runtime_h5ad_member(relative_to_base: str) -> bool:
    """Whether a member is part of the live-runtime h5ad sidecar set.

    ``states/runtime_h5ad_manifest.json`` plus the ``states/*.h5ad`` files it
    names are runtime, not data: a Snapshot V2 restore writes them so a kernel
    can come back up holding ``adata`` when the dill residual alone cannot carry
    it. They match neither ``is_sandbox_pickle`` nor ``_is_checkpoint_member``, so
    until now they crossed installations in an ordinary ``include_runtime=false``
    export and were written without the caller ever asserting trust.

    The name test itself is ``is_runtime_h5ad_name``, shared with the export
    side so the two cannot classify the set differently.
    """
    if not relative_to_base.startswith("states/"):
        return False
    rest = relative_to_base[len("states/"):]
    if "/" in rest:
        return False
    return is_runtime_h5ad_name(rest)


def _drop_kernelless_checkpoints(session_id: str, notify=None) -> None:
    """Delete checkpoints that describe a kernel this import did not bring.

    Both checkpoint families name a point mid-run and assume the values that
    point produced are still in memory. Kept without the live pkl they are worse
    than absent, because each one outranks the ``explicit_start`` intent that
    ``_arm_restore_intent_after_import`` arms for exactly this situation:

    * ``states/checkpoints.sqlite`` — ``workflow.resume()`` calls
      ``aget_state()`` first and continues the graph the moment a row for this
      thread exists; it consults ``restore_start_intent`` only in its "no
      checkpoint found" tail. It travels under ``include_runtime`` while the pkl
      needs ``trust`` as well, so a runtime export imported without trust — onto
      a *free* id, where the in-loop drop above does not fire — resumes the
      source machine's graph node against an empty kernel.
    * ``states/<id>_bioinfo_checkpoint.json`` — matched by neither
      ``is_sandbox_pickle`` nor ``_is_checkpoint_member``, so it travels in an
      ordinary export too. ``has_resumable_checkpoint`` then makes the
      bioinformatics node restore the old conversation and discard the freshly
      built instruction, including the kernel-is-empty notice
      ``runtime_recompute_pending`` exists to deliver.

    Dropping them is what makes "re-run from your data" mean that. A trusted
    runtime import keeps both, and keeps exact continuation with them.

    The ``report`` checkpoint family is deliberately NOT touched: report
    generation resumes from the round results in the state, not from anything in
    the kernel, so it is still resumable on a session that arrived without one.
    """
    states_dir = get_session_states(session_id)
    if not states_dir.exists():
        return
    for path in sorted(states_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name.casefold()
        if not (
            name.startswith(CHECKPOINT_SQLITE_FILENAME)
            or name.endswith("_bioinfo_checkpoint.json")
        ):
            continue
        try:
            path.unlink()
        except OSError as exc:
            logger.warning(
                "Could not drop kernel-less checkpoint %s: %s", path, exc
            )
            continue
        logger.warning(
            "Imported without a live sandbox pkl; dropped %s so the session "
            "starts from its restored round instead of resuming against an "
            "empty kernel.",
            path.name,
        )
        if notify is not None:
            notify("checkpoint_dropped", path.name)


def _resolve_import_dest(
    relative_to_base: str,
    session_id: str,
    archive_session_id: Optional[str] = None,
) -> Optional[Path]:
    """Map an archive-relative path to a destination inside the session directory.

    Returns ``None`` for anything outside the known session-centric layout, and
    the caller skips those members. The previous catch-all
    (``DATA_BASE_DIR / relative_to_base``) let a crafted archive place a member
    anywhere under the data root — ``app.db`` (the account and API-key store),
    ``.jwt_secret`` (the token signing key) and other users'
    ``sessions/<id>/session.json`` were all reachable, because the only guard
    was ``validate_path_within(dest, DATA_BASE_DIR)``.

    *archive_session_id* is the id embedded in the archive's own member names;
    it differs from *session_id* whenever the import had to allocate a fresh
    destination id (see :func:`_allocate_import_session_id`).
    """
    src_id = archive_session_id or session_id

    def _rest(prefix: str) -> str:
        return relative_to_base[len(prefix):]

    if relative_to_base.startswith("states/"):
        return get_session_states(session_id) / _rebind_states_member_name(
            _rest("states/"), src_id, session_id
        )
    if relative_to_base.startswith(f"uploads/{src_id}/"):
        return get_session_uploads(session_id) / _rest(f"uploads/{src_id}/")
    if relative_to_base.startswith(f"workspace/{src_id}/"):
        return get_session_workspace(session_id) / _rest(f"workspace/{src_id}/")
    if relative_to_base.startswith(f"reports/{src_id}/"):
        return get_session_reports(session_id) / _rest(f"reports/{src_id}/")
    if relative_to_base.startswith("blobs/"):
        # Session-level externalized blob store sidecars.
        return get_session_dir(session_id) / "blobs" / _rest("blobs/")
    if relative_to_base.startswith("snapshots/"):
        # Snapshot sidecars: state json, notebook ipynb, compressed pkl,
        # AND per-snapshot blob subdirs `<snap_id>_blobs/<hash>.b64`. The
        # original code only handled top-level files; rglob in the export
        # now also brings in those nested blob files.
        return get_session_dir(session_id) / "snapshots" / _rest("snapshots/")
    if relative_to_base.startswith("logs/"):
        return get_session_logs(session_id) / _rest("logs/")
    if relative_to_base.startswith(f"{REPRODUCIBILITY_DIR_NAME}/"):
        # The bundle lives under the workspace; the export also emits it at the
        # archive root, and archives already in users' hands carry that form.
        return (
            get_session_workspace(session_id)
            / REPRODUCIBILITY_DIR_NAME
            / _rest(f"{REPRODUCIBILITY_DIR_NAME}/")
        )
    return None


def _remap_paths_in_text(
    text: str,
    old_base_dir: str,
    data_base_str: str,
    archive_session_id: str,
    session_id: str,
) -> str:
    """Rewrite absolute paths baked into a state/snapshot JSON document.

    Two rules, both anchored so they cannot corrupt unrelated text:

    * the source installation's data root is replaced only where it starts a
      path — i.e. at a delimiter and followed by ``/``. An unanchored
      ``str.replace`` of a short root such as the Docker image's ``/data``
      rewrote every ``https://host/data/...`` URL and every ``/dataset/...``
      literal in the document as well;
    * the legacy flat layouts keep their existing three replacements.

    When the destination had to allocate a different session id, the session
    directory prefix and any standalone occurrence of the old id are rebound too.
    """
    def _anchored_replace(haystack: str, old: str, new: str) -> str:
        old = old.rstrip("/")
        new = new.rstrip("/")
        if not old or old == new or len(old) < 2:
            return haystack
        pattern = re.compile(r"(?<![A-Za-z0-9_.\-/])" + re.escape(old) + r"(?=/)")
        return pattern.sub(new.replace("\\", "\\\\"), haystack)

    if old_base_dir and old_base_dir != data_base_str:
        if os.path.isabs(old_base_dir):
            text = _anchored_replace(text, old_base_dir, data_base_str)
        else:
            logger.warning(
                "Archive recorded a relative data_base_dir (%r); skipping the "
                "path remap rather than rewriting arbitrary substrings.",
                old_base_dir,
            )

    # Legacy flat layout -> session-centric layout.
    text = _anchored_replace(text, str(UPLOAD_DIR / archive_session_id), str(get_session_uploads(session_id)))
    text = _anchored_replace(text, str(WORKSPACE_DIR / archive_session_id), str(get_session_workspace(session_id)))
    text = _anchored_replace(text, str(REPORTS_DIR / archive_session_id), str(get_session_reports(session_id)))

    if session_id != archive_session_id:
        text = _anchored_replace(
            text, str(get_session_dir(archive_session_id)), str(get_session_dir(session_id))
        )
    return text


def _rebind_session_id_values(value: Any, old_id: str, new_id: str) -> Any:
    """Replace standalone occurrences of *old_id* with *new_id* in a state tree.

    Only exact string values are rewritten; paths that merely *contain* the id
    were already handled by the anchored prefix rewrite, and a substring rule
    here would corrupt free text.
    """
    if old_id == new_id:
        return value
    if isinstance(value, str):
        return new_id if value == old_id else value
    if isinstance(value, dict):
        return {k: _rebind_session_id_values(v, old_id, new_id) for k, v in value.items()}
    if isinstance(value, list):
        return [_rebind_session_id_values(v, old_id, new_id) for v in value]
    return value
