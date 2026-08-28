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
    # Lifecycle generations belong to this installation, not to the archive.
    # Bind the imported state to the destination reservation (generation 1 for
    # a new ID, or the next generation after a same-ID deletion).
    from backend.services.session_lifecycle import bind_import_generation

    bind_import_generation(state)


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
    from backend.services.session_deletion import (
        cleanup_failed_import_generation,
    )

    try:
        if cleanup_failed_import_generation(session_id):
            logger.info(
                "Cleaned up partial import generation for %s", session_id,
            )
    except Exception as e:
        logger.warning(f"Failed to clean up partial session {session_id}: {e}")
