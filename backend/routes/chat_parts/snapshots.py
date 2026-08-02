# Post-chat snapshot capture.
# Moved verbatim out of backend/routes/chat.py.
"""Chat routes for PI Agent free-form chat with analysis sessions."""
import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse

from backend.config import get_session_workspace, get_session_reports, get_session_uploads, get_session_states, PI_AGENT_NAME
from backend.models.schemas import WebSocketMessage, AgentType, DataFile, AnalysisState
from backend.services.session_store import ensure_session_loaded, async_save_session, update_session_cache
from backend.state import (
    chat_tasks, chat_task_lock, chat_sandboxes,
    websockets, workflows, state_lock,
)
from backend.utils.config_resolver import resolve_config_for_user
from backend.utils.auth import require_session_ownership
from backend.utils.files import safe_filename, validate_session_id
from backend.utils.upload import stream_save_file
from backend.utils.settings_provider import AgentSettingsProvider
from backend.utils.sandbox import _TPL_NOOP_INSTALL


logger = logging.getLogger(__name__)


async def capture_post_chat_snapshot(session_id: str, state: AnalysisState):
    """Capture a required post-chat snapshot at an exact safe point.

    Active workflow path: request the shared cooperative safe point, wait for
    acknowledgement, freeze the dynamic capsule from the live descriptor, and
    capture with the workflow sandbox. Idle/terminal path: prove no-kernel
    from persisted lifecycle state and an explicit descriptor — never hard-code
    ``kernel_present=False`` before determining lifecycle.

    Only cancels/releases a safe point this call owns. Foreign holds
    (restore or another capture) fail closed and are never stolen.
    """
    import asyncio

    from backend.config._config import default_config, get_session_workspace
    from backend.services.snapshot_v2.continuation import ContinuationRegistry
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.services.snapshot_v2.lineage import capture_lineage_from_state
    from backend.services.snapshot_v2.models import SnapshotCaptureRequest
    from backend.services.snapshot_v2.paths import SnapshotV2Paths
    from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager
    from backend.services.snapshot_v2.service import SnapshotService
    from backend.services.session_lifecycle import (
        SessionGoneError,
        establish_snapshot_operation,
    )
    from backend.services.workflow_runner import session_has_any_workflow_owner
    from backend.state import sessions, workflows

    paths = SnapshotV2Paths.for_session(session_id)
    service = SnapshotService(session_id, paths)
    sp = get_snapshot_safe_point_manager()
    workspace_root = str(get_session_workspace(session_id))
    workflow = None
    # True only when THIS call created the pending request or acquired the pause.
    owned = False
    # Requester token retained through finally for cancel/release ownership.
    owner_token = f"chat:{session_id}:{uuid.uuid4().hex[:12]}"

    # Compute idle eligibility before the reservation.  The lifecycle barrier
    # below rechecks whether a workflow owner appeared before it publishes the
    # hold, so these fields cannot turn a build/publication gap into an idle
    # snapshot.
    from backend.models.schemas import TaskStatus

    status = getattr(state, "status", None)
    terminal = status in (
        TaskStatus.COMPLETED,
        getattr(TaskStatus, "FAILED", None),
    ) or str(status) in ("completed", "failed")
    inactive = status in (
        TaskStatus.PENDING,
        TaskStatus.PAUSED,
        TaskStatus.WAITING_INPUT,
        TaskStatus.WAITING_ANNOTATION,
        TaskStatus.WAITING_PLAN_REVIEW,
        TaskStatus.WAITING_SPECIES_INPUT,
    ) or terminal or status is None
    descriptor = {
        "kernel_present": False,
        "current_round": getattr(state, "current_round", 0) or 0,
    }
    if terminal:
        descriptor["status"] = "completed"
        descriptor["continuation_kind"] = "terminal"
    else:
        descriptor["inactive"] = bool(inactive)

    async def _reserve() -> str:
        nonlocal owned, workflow

        async with state_lock:
            current = sessions.get(session_id)
            if current is not None and current is not state:
                raise RequiredSnapshotBlocked(
                    "LIVE_SANDBOX_REQUIRED",
                    "Session generation changed before post-chat capture",
                )
            workflow = workflows.get(session_id)
            other_owner = (
                workflow is None and session_has_any_workflow_owner(session_id)
            )

        if workflow is not None:
            request = await sp.request_safe_point(session_id, owner_token)
            owned = request is not False and request is not None
            if not owned:
                raise RequiredSnapshotBlocked(
                    "LIVE_SANDBOX_REQUIRED",
                    "Post-chat capture cannot steal an existing safe-point hold",
                )
            return "active"

        if other_owner:
            raise RequiredSnapshotBlocked(
                "LIVE_SANDBOX_REQUIRED",
                "Post-chat capture cannot classify an active producer as idle",
            )
        if not (terminal or inactive):
            raise RequiredSnapshotBlocked(
                "LIVE_SANDBOX_REQUIRED",
                "Post-chat capture without workflow cannot prove idle/terminal",
            )
        acquired = await sp.acquire(
            session_id, descriptor, owner_token=owner_token,
        )
        owned = bool(acquired)
        if not owned:
            raise RequiredSnapshotBlocked(
                "LIVE_SANDBOX_REQUIRED",
                "Post-chat capture cannot steal an existing safe-point hold",
            )
        return "idle"

    try:
        try:
            reservation_kind = await establish_snapshot_operation(
                session_id, _reserve,
            )
        except SessionGoneError as exc:
            raise RequiredSnapshotBlocked(
                "LIVE_SANDBOX_REQUIRED",
                "Session was deleted before post-chat snapshot reservation",
            ) from exc

        if reservation_kind == "active":

            # Wait for boundary acknowledgement.
            if not sp.is_paused(session_id):
                ack_event = sp.get_pending_event(session_id)
                if ack_event is not None:
                    timeout = default_config.get_snapshot_safe_point_timeout_seconds()
                    try:
                        await asyncio.wait_for(ack_event.wait(), timeout=timeout)
                    except asyncio.TimeoutError as exc:
                        raise RequiredSnapshotBlocked(
                            "LIVE_SANDBOX_REQUIRED",
                            "Timed out waiting for post-chat safe-point "
                            "acknowledgement",
                        ) from exc

            # Require a successful workflow acknowledgement — never force-acquire.
            if not sp.is_paused(session_id):
                raise RequiredSnapshotBlocked(
                    "LIVE_SANDBOX_REQUIRED",
                    "Post-chat safe point was not acknowledged by the "
                    "active workflow",
                )

            live_sp = sp.get_descriptor(session_id) or {}
            graph_state = dict(getattr(workflow, "_current_graph_state", None) or {})
            sandbox = getattr(workflow, "sandbox", None)
            kernel_present = (
                sandbox is not None and getattr(sandbox, "km", None) is not None
            )
            if "kernel_present" not in live_sp:
                live_sp = {
                    **live_sp,
                    "kernel_present": kernel_present,
                }
            # Prefer graph next_node for dynamic capsule when descriptor lacks it.
            if "next_node" not in live_sp and graph_state.get("next_node"):
                live_sp = {**live_sp, "next_node": graph_state["next_node"]}
            if "current_round" not in live_sp:
                live_sp = {
                    **live_sp,
                    "current_round": getattr(state, "current_round", 0) or 0,
                }

            # Refresh descriptor under the held (owned) safe point for capture.
            if not await sp.update_descriptor(session_id, owner_token, live_sp):
                raise RequiredSnapshotBlocked(
                    "LIVE_SANDBOX_REQUIRED",
                    "Post-chat safe point was released before descriptor refresh",
                )

            workflow_state = {**graph_state, **live_sp}
            capsule = ContinuationRegistry(service.object_store).freeze(
                "post_chat",
                state,
                workflow_state,
                live_safe_point=live_sp,
            )
            request = SnapshotCaptureRequest(
                session_id=session_id,
                reason="post_chat",
                phase="post_chat",
                capsule=capsule,
                round_number=getattr(state, "current_round", 0) or 0,
                **capture_lineage_from_state(state),
            )
            return await service.capture_required_snapshot(
                state,
                request,
                sandbox=sandbox,
                workflow_state=workflow_state,
                workspace_root=workspace_root,
                safe_point_owner_token=owner_token,
            )

        # The idle hold was published atomically with the no-owner proof.
        live_sp = dict(descriptor)
        workflow_state = dict(descriptor)
        capsule = ContinuationRegistry(service.object_store).freeze(
            "post_chat",
            state,
            workflow_state,
            live_safe_point=live_sp,
        )
        request = SnapshotCaptureRequest(
            session_id=session_id,
            reason="post_chat",
            phase="post_chat",
            capsule=capsule,
            round_number=getattr(state, "current_round", 0) or 0,
            **capture_lineage_from_state(state),
        )
        return await service.capture_required_snapshot(
            state,
            request,
            sandbox=None,
            workflow_state=workflow_state,
            workspace_root=workspace_root,
            safe_point_owner_token=owner_token,
        )
    finally:
        if owned:
            await sp.cancel_safe_point_request(session_id, owner_token)

async def _create_chat_snapshot(session_id: str, state: AnalysisState) -> None:
    """Required post-chat snapshot via SnapshotService (fail closed).

    Delegates to :func:`capture_post_chat_snapshot`. On failure, persists the
    same blocking snapshot_failure state so a later run/resume refuses to
    start until Retry, Manage, or Terminate.
    """
    from backend.config import is_snapshot_v2_enabled

    if not is_snapshot_v2_enabled():
        return None

    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.services.session_store import async_save_session, update_session_cache

    try:
        await capture_post_chat_snapshot(session_id, state)
    except RequiredSnapshotBlocked:
        # Persist blocking failure so run/resume refuses until HITL action.
        try:
            await update_session_cache(session_id, state)
            await async_save_session(session_id, state)
        except Exception:
            logger.warning(
                "[%s] Failed to persist post-chat snapshot failure state",
                session_id,
                exc_info=True,
            )
        raise
