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

from dataclasses import dataclass, field

from .import_archive_policy import (
    _SUPPORTED_ARCHIVE_VERSIONS,
    _drop_kernelless_checkpoints,
    _is_checkpoint_member,
    _is_runtime_h5ad_member,
    _is_unsafe_member_name,
    _rebind_session_id_values,
    _remap_paths_in_text,
    _resolve_import_dest,
)
from .import_state import _finalize_imported_state, _resign_imported_live_pkl

@dataclass
class ImportTarContext:
    file_path: str
    user_id: str
    import_job_id: Optional[str]
    outcome: Optional[Dict[str, Any]]
    bindings: Any
    archive_session_id: Optional[str] = None
    session_id: Optional[str] = None
    skipped: Dict[str, int] = field(default_factory=lambda: {
        "snapshot_v2": 0,
        "unsafe_path": 0,
        "off_layout": 0,
        "untrusted_pickle": 0,
        "untrusted_runtime_h5ad": 0,
        "checkpoint_dropped": 0,
    })
    skipped_examples: list[str] = field(default_factory=list)
    last_progress_ts: float = 0.0

    def __post_init__(self) -> None:
        if self.outcome is not None:
            self.outcome["skipped"] = self.skipped
            self.outcome["skipped_examples"] = self.skipped_examples
        try:
            archive_size = os.path.getsize(self.file_path)
        except OSError:
            archive_size = 0
        if self.import_job_id:
            job = _import_jobs.get(self.import_job_id)
            if job is not None:
                job["bytes_total"] = archive_size

    def note_skip(self, kind: str, name: str) -> None:
        self.skipped[kind] += 1
        if len(self.skipped_examples) < 20:
            self.skipped_examples.append(f"{kind}: {name}")

    def check_cancel(self) -> None:
        if not self.import_job_id:
            return
        job = _import_jobs.get(self.import_job_id)
        if job is not None and job.get("cancel_requested"):
            raise ImportCancelled("cancelled by user")

    def check_disk_space(self) -> None:
        free = self.bindings.free_disk_bytes(DATA_BASE_DIR)
        if free and free < MIN_FREE_DISK_BYTES:
            raise HTTPException(
                status_code=507,
                detail=(
                    "Not enough free disk space to finish the import "
                    f"({free // (1024 * 1024)} MB left). The partially extracted "
                    "session has been removed; free space and try again."
                ),
            )

    def update_progress(
        self,
        current_name: str,
        files_processed: int,
        bytes_processed: int,
        force: bool = False,
    ) -> None:
        if not self.import_job_id:
            return
        now = time.monotonic()
        if not force and now - self.last_progress_ts < _IMPORT_PROGRESS_UPDATE_INTERVAL_S:
            return
        self.last_progress_ts = now
        job = _import_jobs.get(self.import_job_id)
        if job is not None:
            job["files_processed"] = files_processed
            job["bytes_processed"] = bytes_processed
            job["current_file"] = current_name

    def bind_session(self, candidate: str) -> None:
        if self.archive_session_id is not None:
            return
        validate_session_id(candidate)
        self.archive_session_id = candidate
        self.session_id = self.bindings._allocate_import_session_id(
            candidate, self.user_id,
        )
        if self.outcome is not None:
            self.outcome["session_id"] = self.session_id
            self.outcome["archive_session_id"] = self.archive_session_id
        if self.import_job_id:
            job = _import_jobs.get(self.import_job_id)
            if job is not None:
                job["session_id"] = self.session_id

    def finish_progress(self, files_processed: int, bytes_processed: int) -> None:
        if not self.import_job_id:
            return
        job = _import_jobs.get(self.import_job_id)
        if job is not None:
            job["files_processed"] = files_processed
            job["bytes_processed"] = bytes_processed
            job["files_total"] = files_processed
            job["current_file"] = ""


def import_tar_gz(
    file_path: str,
    user_id: str,
    import_job_id: Optional[str] = None,
    trust: bool = False,
    outcome: Optional[Dict[str, Any]] = None,
    *,
    bindings,
) -> AnalysisState:
    """Import from tar.gz archive file on disk with path remapping.

    Uses streaming iteration and chunked file writes to minimise memory usage,
    which is critical for large archives (20+ GB) on memory-constrained systems.

    Runs inside ``asyncio.to_thread``. Progress is written to ``_import_jobs[import_job_id]``
    via atomic single-key dict assignments (GIL-safe; no async-lock needed from a thread).
    Cancellation is cooperative: the cancel endpoint sets ``cancel_requested`` on the same
    dict and this loop checks it between members.

    *outcome* receives ``session_id`` as soon as a destination is allocated so the
    caller can clean up a half-written session on cancel or failure.
    """
    from backend.utils.runtime_apply import _live_pkl, runtime_h5ad_manifest_path

    context = ImportTarContext(
        file_path=file_path,
        user_id=user_id,
        import_job_id=import_job_id,
        outcome=outcome,
        bindings=bindings,
    )
    data_base_str = str(DATA_BASE_DIR)
    archive_session_id: Optional[str] = None
    session_id: Optional[str] = None
    old_base_dir = ""
    state_json_data = None
    wrote_sandbox_pkl = False
    wrote_runtime_h5ad_manifest = False
    skipped = context.skipped
    files_processed = 0
    bytes_processed = 0

    context.check_disk_space()

    with tarfile.open(file_path, mode='r:gz') as tf:
        for member in tf:
            context.check_cancel()
            # Reject link entries (symlink/hardlink) and device/special files BEFORE
            # the isfile() filter: such members would otherwise be silently skipped,
            # but an attacker may craft them to point at /etc/passwd or other sessions.
            if member.issym() or member.islnk():
                raise HTTPException(
                    status_code=400,
                    detail=f"Archive contains link entries (rejected): {member.name}",
                )
            if member.isdev() or member.ischr() or member.isfifo():
                raise HTTPException(
                    status_code=400,
                    detail=f"Archive contains device/special entries: {member.name}",
                )

            if not member.isfile():
                continue

            if _is_disabled_snapshot_v2_member(member.name):
                logger.warning(
                    "Skipping Snapshot V2 member from ordinary import while "
                    "the subsystem is disabled: %s",
                    member.name,
                )
                context.note_skip("snapshot_v2", member.name)
                continue

            if _is_unsafe_member_name(member.name):
                logger.warning(f"Skipping archive entry with suspicious path: {member.name}")
                context.note_skip("unsafe_path", member.name)
                continue

            parts = member.name.split('/', 1)
            if len(parts) < 2:
                continue

            context.bind_session(parts[0])
            archive_session_id = context.archive_session_id
            session_id = context.session_id

            relative_to_base = parts[1]

            if relative_to_base == 'manifest.json':
                f = tf.extractfile(member)
                if f:
                    manifest_data = json.loads(f.read().decode('utf-8'))
                    manifest_session_id = manifest_data.get("session_id", "")
                    if manifest_session_id and manifest_session_id != archive_session_id:
                        # The top-level directory and the manifest disagree. Every
                        # member seen so far was resolved against the directory
                        # name, so silently switching now would scatter the archive
                        # across two destinations.
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                "Archive is inconsistent: manifest session_id "
                                f"{manifest_session_id!r} does not match the archive "
                                f"directory {archive_session_id!r}"
                            ),
                        )
                    old_base_dir = manifest_data.get("data_base_dir", "")
                    manifest_version = str(manifest_data.get("version", ""))
                    if manifest_version and manifest_version not in _SUPPORTED_ARCHIVE_VERSIONS:
                        logger.warning(
                            "Archive declares manifest version %s, which this build "
                            "does not know; importing on a best-effort basis.",
                            manifest_version,
                        )
                continue

            if relative_to_base == f"sessions/{archive_session_id}.json":
                f = tf.extractfile(member)
                if f:
                    state_json_data = f.read()
                continue

            # The checkpoint database is keyed by thread_id, which is the session
            # id of the run that wrote it, and its blobs carry a serialized
            # AnalysisState holding the same id. Under a reallocated id none of
            # that can be renamed into place, so it is dropped: the file would
            # otherwise satisfy the `has_checkpoint` test in lifecycle.run_analysis
            # while holding no row this session can resume from, which withholds
            # the explicit_start intent and leaves Start executing nothing.
            if session_id != archive_session_id and _is_checkpoint_member(relative_to_base):
                logger.warning(
                    "Import allocated a new session id; dropping the checkpoint "
                    "database from the archive (%s) so the session starts from "
                    "its restored round instead of a thread it cannot resume.",
                    member.name,
                )
                context.note_skip("checkpoint_dropped", member.name)
                continue

            dest_path = _resolve_import_dest(relative_to_base, session_id, archive_session_id)
            if dest_path is None:
                logger.warning(
                    "Skipping archive entry outside the session layout: %s",
                    member.name,
                )
                context.note_skip("off_layout", member.name)
                continue
            validate_path_within(dest_path, get_session_dir(session_id))
            # SECURITY (S4): sandbox pickles can execute arbitrary code via dill.load.
            # Drop them unless the caller passed trust=true to take responsibility.
            is_pkl_member = is_sandbox_pickle(dest_path.name)
            if is_pkl_member:
                if not trust:
                    logger.warning(f"Skipping sandbox pickle from imported archive (untrusted): {member.name}")
                    context.note_skip("untrusted_pickle", member.name)
                    continue
                logger.warning(f"Restoring sandbox pickle (trust=true): {member.name}")
            # The h5ad sidecar set is the other half of the live runtime, and the
            # X-Trust contract already says so ("restore runtime objects
            # (dill/h5ad)"). It carries no code, but it does bind identifier-named
            # globals into the kernel, and an ordinary export has no business
            # shipping a multi-GB AnnData it also declares absent.
            is_runtime_h5ad_member = _is_runtime_h5ad_member(relative_to_base)
            if is_runtime_h5ad_member and not trust:
                logger.warning(
                    "Skipping runtime h5ad sidecar from imported archive "
                    "(untrusted): %s",
                    member.name,
                )
                context.note_skip("untrusted_runtime_h5ad", member.name)
                continue
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            f = tf.extractfile(member)
            if not f:
                continue

            if relative_to_base.startswith("snapshots/") and dest_path.suffix == ".json":
                content = f.read().decode("utf-8")
                content = _remap_paths_in_text(
                    content, old_base_dir, data_base_str, archive_session_id, session_id
                )
                dest_path.write_text(content, encoding="utf-8")
            else:
                with open(dest_path, 'wb') as out_f:
                    shutil.copyfileobj(f, out_f, length=UPLOAD_CHUNK_SIZE)
                # Only the live kernel pkl answers "can this session resume".
                # `is_sandbox_pickle` also matches the per-snapshot
                # `<snap_id>_session.pkl.zst` backups, and counting one of those
                # let a session with no live kernel at all claim it had one.
                if is_pkl_member and dest_path == _live_pkl(session_id):
                    wrote_sandbox_pkl = True
                # A restore-written h5ad set is a live runtime in its own right:
                # a "complete" target with no dill residual leaves no pkl at all,
                # and the manifest is then the only carrier of `adata`.
                if is_runtime_h5ad_member and dest_path == runtime_h5ad_manifest_path(session_id):
                    wrote_runtime_h5ad_manifest = True

            # Progress accounting: count uncompressed member size (matches what
            # the user sees on disk). bytes_total is compressed archive size, so
            # bytes_processed can exceed it for highly compressible payloads —
            # the frontend should treat the ratio as a heuristic, not a hard %.
            files_processed += 1
            bytes_processed += getattr(member, "size", 0) or 0
            if files_processed % 256 == 0:
                context.check_disk_space()
            context.update_progress(relative_to_base, files_processed, bytes_processed)

    context.finish_progress(files_processed, bytes_processed)

    # Only decidable here: which members arrived is known after the walk, and
    # the checkpoints must be gone before `_arm_restore_intent_after_import`
    # tests for them below.
    imported_live_runtime = wrote_sandbox_pkl or wrote_runtime_h5ad_manifest
    if session_id:
        if wrote_sandbox_pkl:
            # Validation can retire both the dump and its h5ad fallback.
            imported_live_runtime = _resign_imported_live_pkl(session_id)
        if not imported_live_runtime:
            _drop_kernelless_checkpoints(session_id, notify=context.note_skip)
        validate_and_rebind_imported_reports(get_session_reports(session_id))

    if state_json_data is None:
        if skipped["snapshot_v2"]:
            # Every member a Snapshot V2 export emits is filtered above when the
            # subsystem is off, so the archive is intact and this installation is
            # the constraint. "State JSON not found" blamed the file for an
            # operator setting. The env var deliberately stays out of the detail:
            # this text reaches end users, and the operator-facing wording already
            # lives in config/_config.py and main.py.
            raise HTTPException(
                status_code=400,
                detail=(
                    "This archive carries only Snapshot V2 snapshot data "
                    f"({skipped['snapshot_v2']} member(s)) and no session state. "
                    "Snapshot V2 is not enabled on this server, so the archive "
                    "cannot be imported here; import it on a server with "
                    "Snapshot V2 enabled."
                ),
            )
        raise HTTPException(status_code=400, detail="Session state JSON not found in archive")

    state_data = json.loads(state_json_data.decode('utf-8'))
    state_data = _rebind_session_id_values(state_data, archive_session_id, session_id)
    state_data["id"] = session_id
    state_data["user_id"] = user_id

    # Validate BEFORE anything reaches session.json. A payload this build cannot
    # load must not leave a directory behind that `list_saved_sessions` happily
    # lists from raw JSON but `load_session_from_disk` then refuses — that
    # combination produces a sidebar entry which can be neither opened nor
    # deleted (both paths go through require_session_ownership and 404).
    #
    # Blob references are inlined for the in-memory state, and only then is the
    # document remapped -- the source installation's data root replaced with
    # ours. Remapping before inlining reaches nothing that lives in a sidecar:
    # strings >= 1 KB (long code history, report segments) are copied
    # byte-for-byte into blobs/*.b64 and would keep the source machine's
    # absolute paths.
    inlined = inline_blobs(state_data, get_session_dir(session_id) / "blobs")
    inlined = json.loads(
        _remap_paths_in_text(
            json.dumps(inlined, ensure_ascii=False),
            old_base_dir,
            data_base_str,
            archive_session_id,
            session_id,
        )
    )
    state = AnalysisState(**inlined)
    _finalize_imported_state(state, imported_runtime=imported_live_runtime)
    if imported_live_runtime:
        from .import_state import validate_imported_pairing
        validate_imported_pairing(state)

    # Write the externalized JSON (with @scai-blob tokens intact) to disk.
    # This keeps session.json consistent with the extracted blob sidecars.
    # async_save_session will re-externalize and overwrite, but writing here
    # ensures a valid session.json exists even if the process crashes before
    # that step — the externalized form is always loadable.
    session_file = get_session_metadata_path(session_id)
    session_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    session_file.parent.chmod(0o700)
    session_file.write_text(
        json.dumps(
            externalize_blobs(
                state.model_dump(mode="json"), get_session_dir(session_id) / "blobs"
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    session_file.chmod(0o600)

    return state
