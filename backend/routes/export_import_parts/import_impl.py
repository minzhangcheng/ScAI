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

# Archive layouts this build knows how to read. Unknown values are imported on a
# best-effort basis with a warning rather than rejected, so an archive from a
# newer build is not simply unopenable.
_SUPPORTED_ARCHIVE_VERSIONS = frozenset({"1.0", ARCHIVE_FORMAT_VERSION})


def _mark_runtime_availability_after_import(
    state: AnalysisState,
    imported_pkl: Optional[bool] = None,
) -> None:
    """Record whether the imported session actually has a kernel to resume from.

    ``runtime_restore_status`` rides along in session.json, so a session exported
    mid-analysis arrives claiming ``complete`` even when the archive carried no
    pkl (``include_runtime=false``, or ``trust=false`` on the way in). The
    workflow would then resume against an empty kernel and the bioinformatics
    agent would fail on a missing ``adata`` with no explanation.

    Marking it ``unavailable`` routes resume through the existing
    ``is_forward_resume_blocked`` gate, which offers the user a re-run from their
    data. Completed sessions are deliberately unaffected — that gate exempts
    them, so reports stay viewable and re-analysis still works.

    *imported_pkl* says whether **this** import wrote the live pkl. A pkl that
    was merely lying in the destination belongs to some other run, and trusting
    it would mark a session resumable against a kernel that does not match the
    state just imported. ``None`` keeps the historical "whatever is on disk"
    answer for callers that cannot distinguish the two.
    """
    from backend.utils.runtime_apply import _live_pkl

    if state.current_round <= 0:
        # Never executed anything; a fresh kernel is the correct starting point.
        return
    if imported_pkl if imported_pkl is not None else _live_pkl(state.id).exists():
        return
    state.runtime_restore_status = "unavailable"
    state.runtime_restore_warning = (
        "This session was imported without kernel runtime state, so in-memory "
        "values from the original run (adata and friends) are gone. Re-run the "
        "analysis from the imported data to recompute them."
    )
    logger.info(
        "[%s] Imported without a live sandbox pkl; marked runtime unavailable "
        "(round=%s).",
        state.id,
        state.current_round,
    )


def _normalize_imported_status(state: AnalysisState) -> None:
    """Land an imported session in a status this installation can back up.

    A session exported mid-analysis arrives claiming ``running``, but nothing
    here owns that run: no workflow object, no kernel, no WebSocket. The
    frontend reads the status literally and hides the very controls that would
    start one — including the "re-run from your data" button that a runtime-less
    import depends on — so the session looks permanently busy and the user has
    to stop it before anything else is reachable.

    A wait gate whose only exits run through that missing workflow object is the
    same problem wearing a different status, and a worse one because the dialog
    comes back: ``POST /{id}/llm-failure-action`` resolves ``llm_failure``,
    ``agent_disabled`` and ``marker_db_missing`` through ``workflows[session_id]``
    and raises ``404 Workflow not found`` when it is absent, while the WebSocket
    parks such a session in its HITL-hold branch and returns *before* registering
    a workflow — after re-opening the dialog. ``ensure_session_loaded`` clears
    two of the three on a disk load, but an import primes the cache, so nothing
    clears them here.
    """
    from backend.constants import WaitInputReason
    from backend.models.schemas import TaskStatus

    # GENERATING_REPORT is the same claim with a different label, and the
    # frontend has no branch for it at all: it is neither paused, nor waiting,
    # nor running, so an imported session sitting in it renders no controls
    # whatsoever.
    unowned = {TaskStatus.RUNNING, TaskStatus.GENERATING_REPORT}
    if state.status in unowned:
        previous = state.status
        state.status = TaskStatus.PAUSED
        logger.info(
            "[%s] Imported while marked %s; no workflow owns it here, so it "
            "lands paused.",
            state.id,
            previous.value,
        )

    # Checked on the reason rather than the status so a stale pairing cannot
    # smuggle the gate through: an unresolvable gate is cleared either way, and
    # only a session still parked on it needs its status moved.
    workflow_bound_reasons = {
        WaitInputReason.LLM_FAILURE.value,
        WaitInputReason.AGENT_DISABLED.value,
        WaitInputReason.MARKER_DB_MISSING.value,
    }
    if state.wait_input_reason in workflow_bound_reasons:
        previous_reason = state.wait_input_reason
        state.wait_input_reason = None
        state.llm_failure_info = None
        state.agent_disabled_info = None
        if state.status == TaskStatus.WAITING_INPUT:
            state.status = TaskStatus.PAUSED
        logger.info(
            "[%s] Imported while waiting on the %s gate, whose actions need a "
            "live workflow this installation has none of; cleared the gate and "
            "landed paused.",
            state.id,
            previous_reason,
        )


def _arm_restore_intent_after_import(state: AnalysisState) -> None:
    """Give a checkpoint-less imported session a way to re-enter the graph.

    ``resume()`` deliberately refuses to invent a fresh ``initialize_round``
    run, so a session with no usable checkpoint moves only when
    ``restore_start_intent`` says where to go. ``POST /run`` sets it — but the
    UI only offers that button while the runtime is unavailable. An import that
    carried a kernel under a reallocated id has neither: its checkpoint was
    dropped (it is keyed by the archive's thread id) yet its runtime is
    ``complete``, so the user is left with Resume, which never sets the intent,
    and the graph holds without executing anything on every attempt.

    Arming it here, where the archive's contents are known, also covers the
    Resume path rather than only the button that happens to call ``/run``.
    """
    if state.current_round <= 0:
        # Never executed; a normal start already begins at initialize_round.
        return
    if getattr(state, "restore_resume_kind", None) is not None:
        # An exact continuation target rides along in the state; honour it.
        return

    from backend.utils.runtime_apply import _live_pkl

    checkpoint = checkpoint_sqlite_path(get_session_states(state.id))
    if checkpoint.exists() and _live_pkl(state.id).exists():
        # Checkpoint plus the kernel it was written against: resume() can
        # continue exactly, and must not be diverted into a fresh round.
        return

    state.restore_start_intent = "explicit_start"
    logger.info(
        "[%s] Imported without a resumable checkpoint (round=%s); armed "
        "explicit_start so Resume re-enters at initialize_round.",
        state.id,
        state.current_round,
    )


def _finalize_imported_state(
    state: AnalysisState,
    *,
    imported_runtime: Optional[bool] = None,
) -> None:
    """Apply every import-time fixup to *state*, in the one order that is correct.

    ``normalize_disabled_snapshot_state`` has to run BEFORE
    ``_normalize_imported_status``: it picks the next status from the failure's
    ``retry_context``, and it passes ``generating_report`` straight through — the
    one status the frontend renders no controls for, and precisely what
    ``_normalize_imported_status`` exists to rewrite. So the status normalizer
    must have the last word.

    Without this, a session imported from an installation that had Snapshot V2
    enabled kept every V2-only blocker (``snapshot_failure``,
    ``pending_required_snapshot``, ``wait_input_reason``, ``recovery_blocked``,
    ``snapshot_transaction.recovery_error``) even though nothing here can ever
    answer them: a gate-off destination repaired them on the first authenticated
    read, but only there, and a destination with the gate ON has no load path that
    clears them at all — leaving ``/run`` at 409 and ``/resume`` refused, pointing
    the user at snapshot actions whose objects this archive never carried.
    """
    _mark_runtime_availability_after_import(state, imported_pkl=imported_runtime)
    if not is_snapshot_v2_enabled():
        # Nothing can own a session that is being created right now.
        normalize_disabled_snapshot_state(state, workflow_active=False)
    _normalize_imported_status(state)
    _arm_restore_intent_after_import(state)


def _resign_imported_live_pkl(session_id: str) -> bool:
    """Re-sign an imported live sandbox pkl with *this* installation's key.

    The HMAC sidecar lives in ``<DATA_BASE_DIR>/integrity/<session_id>/``, outside
    the session directory, so it is never part of a portable archive — and the
    signing key is installation-scoped anyway, so shipping it would not help.
    Without a local signature ``verify_live_pkl_or_raise`` refuses to rehydrate
    the kernel and sandbox startup fails outright, which is strictly worse than
    having no pkl at all.

    Signing here mirrors what Snapshot V2 restore does in
    ``runtime_apply.apply_runtime_pkl``: content arrives verified by other means
    (the caller passed ``trust=true``), and integrity for local storage is
    re-established under the destination key.

    Returns True when a signature was written.
    """
    from backend.utils.runtime_apply import _live_pkl
    from backend.utils.sandbox import _integrity_dir, _write_pkl_signature

    pkl = _live_pkl(session_id)
    if not pkl.exists():
        return False
    _write_pkl_signature(pkl, _integrity_dir(session_id))
    logger.warning(
        "[%s] Re-signed imported sandbox pkl under this installation's key; "
        "the archive's origin is trusted on the caller's authority (trust=true).",
        session_id,
    )
    return True


def _session_id_is_taken(session_id: str) -> bool:
    """Whether *session_id* already names a session on this installation.

    Checks disk and the in-memory registries, because a session evicted from
    ``sessions`` still owns its directory, and a session mid-creation may own a
    workflow before its directory is fully populated.
    """
    from backend.state import sessions as _live_sessions, workflows as _live_workflows

    if get_session_dir(session_id).exists():
        return True
    return session_id in _live_sessions or session_id in _live_workflows


def _claim_session_id(session_id: str) -> bool:
    """Take exclusive ownership of *session_id* by creating its directory.

    ``mkdir(exist_ok=False)`` is the claim, so two imports racing for the same
    id cannot both decide it is free: exactly one creates the directory and the
    other falls through to a fresh id. Checking-then-creating would let both
    extract into the same tree and interleave two sessions' files.
    """
    if _session_id_is_taken(session_id):
        return False
    try:
        get_session_dir(session_id).mkdir(parents=True, exist_ok=False, mode=0o700)
    except FileExistsError:
        return False
    except OSError as exc:
        logger.warning("Cannot claim session directory for %s: %s", session_id, exc)
        return False
    return True


def _allocate_import_session_id(archive_session_id: str) -> str:
    """Pick the session id an archive should be extracted into, and claim it.

    Reusing the archive's own id is what makes an import round-trip on the same
    installation (re-import to restore a backup) *destructive*: the extraction
    writes straight over the live session's files, and ``session.json`` is then
    rewritten with the importer's ``user_id`` — which on a shared installation
    silently transfers another user's session. Importing is therefore defined as
    "create a session", never "overwrite one": on any collision the archive is
    materialized under a fresh id and every embedded path is rebound to it.
    """
    if _claim_session_id(archive_session_id):
        return archive_session_id
    new_id = str(uuid.uuid4())
    while not _claim_session_id(new_id):  # pragma: no cover - uuid4 collision
        new_id = str(uuid.uuid4())
    logger.warning(
        "Import target %s already exists on this installation; extracting into "
        "a new session %s instead of overwriting it.",
        archive_session_id,
        new_id,
    )
    return new_id


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

def _cleanup_partial_session(session_id: Optional[str]) -> None:
    """Remove a partially-extracted session directory on cancel or failure.

    Only touches the session-scoped directory under sessions/<id>/, leaving
    other sessions intact. Idempotent — safe if directory does not exist.

    The caller must only pass an id the import allocated for itself; a partial
    archive must never be a way to delete a session that already existed.
    """
    if not session_id:
        return
    try:
        validate_session_id(session_id)
    except Exception:
        return
    try:
        sess_dir = get_session_dir(session_id)
        if sess_dir.exists():
            shutil.rmtree(sess_dir, ignore_errors=True)
            logger.info(f"Cleaned up partial session directory: {sess_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up partial session {session_id}: {e}")


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


def _import_tar_gz(
    file_path: str,
    user_id: str,
    import_job_id: Optional[str] = None,
    trust: bool = False,
    outcome: Optional[Dict[str, Any]] = None,
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

    data_base_str = str(DATA_BASE_DIR)

    archive_session_id: Optional[str] = None
    session_id: Optional[str] = None
    old_base_dir = ""
    state_json_data = None
    # Only a pkl written by *this* import gets re-signed; a pre-existing live pkl
    # left over from another session must keep whatever signature it already had.
    wrote_sandbox_pkl = False
    # Same rule for the h5ad sidecar set: a manifest merely lying in the
    # destination belongs to some other run and says nothing about this archive.
    wrote_runtime_h5ad_manifest = False

    # What the walk threw away. Every one of these decisions used to reach only
    # the log, so a user whose archive lost its checkpoint, its V2 payload or a
    # member outside the layout was told "Import successful" and nothing else.
    # Bound into `outcome` up front and mutated in place, so a cancel or a
    # mid-archive failure can still read the partial summary.
    skipped: Dict[str, int] = {
        "snapshot_v2": 0,
        "unsafe_path": 0,
        "off_layout": 0,
        "untrusted_pickle": 0,
        "untrusted_runtime_h5ad": 0,
        "checkpoint_dropped": 0,
    }
    skipped_examples: list[str] = []
    if outcome is not None:
        outcome["skipped"] = skipped
        outcome["skipped_examples"] = skipped_examples

    def _note_skip(kind: str, name: str) -> None:
        skipped[kind] += 1
        # An off-layout or hostile archive can hold tens of thousands of members
        # and this list is echoed on every status poll.
        if len(skipped_examples) < 20:
            skipped_examples.append(f"{kind}: {name}")

    archive_size = 0
    try:
        archive_size = os.path.getsize(file_path)
    except OSError:
        pass
    if import_job_id:
        job = _import_jobs.get(import_job_id)
        if job is not None:
            job["bytes_total"] = archive_size

    last_progress_ts = 0.0
    files_processed = 0
    bytes_processed = 0

    def _maybe_check_cancel():
        if not import_job_id:
            return
        job = _import_jobs.get(import_job_id)
        if job is not None and job.get("cancel_requested"):
            raise ImportCancelled("cancelled by user")

    def _check_disk_space():
        free = free_disk_bytes(DATA_BASE_DIR)
        if free and free < MIN_FREE_DISK_BYTES:
            raise HTTPException(
                status_code=507,
                detail=(
                    "Not enough free disk space to finish the import "
                    f"({free // (1024 * 1024)} MB left). The partially extracted "
                    "session has been removed; free space and try again."
                ),
            )

    def _maybe_update_progress(current_name: str, force: bool = False):
        nonlocal last_progress_ts
        if not import_job_id:
            return
        now = time.monotonic()
        if not force and now - last_progress_ts < _IMPORT_PROGRESS_UPDATE_INTERVAL_S:
            return
        last_progress_ts = now
        job = _import_jobs.get(import_job_id)
        if job is None:
            return
        # Atomic single-key writes — no lock needed.
        job["files_processed"] = files_processed
        job["bytes_processed"] = bytes_processed
        job["current_file"] = current_name

    def _bind_session(candidate: str) -> None:
        """Latch the archive id and allocate the destination id exactly once."""
        nonlocal archive_session_id, session_id
        if archive_session_id is not None:
            return
        validate_session_id(candidate)
        archive_session_id = candidate
        session_id = _allocate_import_session_id(candidate)
        if outcome is not None:
            outcome["session_id"] = session_id
            outcome["archive_session_id"] = archive_session_id

    _check_disk_space()

    with tarfile.open(file_path, mode='r:gz') as tf:
        for member in tf:
            _maybe_check_cancel()
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
                _note_skip("snapshot_v2", member.name)
                continue

            if _is_unsafe_member_name(member.name):
                logger.warning(f"Skipping archive entry with suspicious path: {member.name}")
                _note_skip("unsafe_path", member.name)
                continue

            parts = member.name.split('/', 1)
            if len(parts) < 2:
                continue

            _bind_session(parts[0])

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
                _note_skip("checkpoint_dropped", member.name)
                continue

            dest_path = _resolve_import_dest(relative_to_base, session_id, archive_session_id)
            if dest_path is None:
                logger.warning(
                    "Skipping archive entry outside the session layout: %s",
                    member.name,
                )
                _note_skip("off_layout", member.name)
                continue
            validate_path_within(dest_path, get_session_dir(session_id))
            # SECURITY (S4): sandbox pickles can execute arbitrary code via dill.load.
            # Drop them unless the caller passed trust=true to take responsibility.
            is_pkl_member = is_sandbox_pickle(dest_path.name)
            if is_pkl_member:
                if not trust:
                    logger.warning(f"Skipping sandbox pickle from imported archive (untrusted): {member.name}")
                    _note_skip("untrusted_pickle", member.name)
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
                _note_skip("untrusted_runtime_h5ad", member.name)
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
                _check_disk_space()
            _maybe_update_progress(relative_to_base)

    # Final progress flush so the poll sees 100% before completion is reported.
    if import_job_id:
        job = _import_jobs.get(import_job_id)
        if job is not None:
            job["files_processed"] = files_processed
            job["bytes_processed"] = bytes_processed
            job["files_total"] = files_processed
            job["current_file"] = ""

    # Only decidable here: which members arrived is known after the walk, and
    # the checkpoints must be gone before `_arm_restore_intent_after_import`
    # tests for them below.
    imported_live_runtime = wrote_sandbox_pkl or wrote_runtime_h5ad_manifest
    if session_id:
        if wrote_sandbox_pkl:
            _resign_imported_live_pkl(session_id)
        if not imported_live_runtime:
            _drop_kernelless_checkpoints(session_id, notify=_note_skip)

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


def _import_zip(content: bytes, user_id: str, trust: bool = False) -> AnalysisState:
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
            session_id = _allocate_import_session_id(archive_session_id)

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
                    _resign_imported_live_pkl(session_id)
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

        state = AnalysisState(**inline_blobs(state_data, get_session_dir(session_id) / "blobs"))
        _finalize_imported_state(state, imported_runtime=wrote_sandbox_pkl)
        return state

    except BaseException:
        _cleanup_partial_session(session_id)
        raise
