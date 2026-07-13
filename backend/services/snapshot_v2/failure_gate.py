"""Centralized snapshot failure HITL gate.

All snapshot producers enter the failure gate through this module,
replacing ad-hoc manual construction across core.py, chat.py,
reporting.py, and pre_restore.py.

Retry dispatch (via ``retry_snapshot_producer``):
  post_report        → capture-only retry through workflow
  chat               → re-invoke post-chat capture
  pre_restore        → re-invoke pre-restore capture
  workflow_boundary  → re-invoke workflow-boundary capture

Terminate delegates to lifecycle stop (single kernel teardown).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from backend.constants import WaitInputReason
from backend.models.schemas import TaskStatus
from backend.services.snapshot_v2.errors import (
    SNAPSHOT_FAILURE_ACTIONS,
    RequiredSnapshotBlocked,
)
from backend.services.snapshot_v2.models import RetryOutcome, SnapshotRetryContext


SnapshotFailureActions = tuple[str, ...]

SNAPSHOT_FAILURE_GATE_ACTIONS: SnapshotFailureActions = SNAPSHOT_FAILURE_ACTIONS

RetryProducerFn = Callable[
    [str, Any, SnapshotRetryContext],
    Awaitable[RetryOutcome],
]


@dataclass
class SnapshotRetryDependencies:
    """Injectable producer retry callbacks for HTTP-request dispatch."""

    retry_post_report: RetryProducerFn
    retry_chat_capture: RetryProducerFn
    retry_pre_restore_capture: RetryProducerFn
    retry_workflow_boundary_capture: RetryProducerFn


# Phases captured via workflow auto-snapshot boundaries (producer=workflow_boundary).
# post_report is intentionally excluded — it has its own producer.
_WORKFLOW_BOUNDARY_PHASES: frozenset[str] = frozenset({
    "pre_execution_stage1",
    "pre_execution_stage2",
    "pre_execution_full",
    "pre_annotation",
    "post_annotation",
    "pre_direction",
    "post_direction",
    "pre_round_wait",
    "pre_integration_gate",
    "pre_major_revision",
})

# Producers that must never be overwritten by workflow_boundary typing.
_NON_WORKFLOW_BOUNDARY_PRODUCERS: frozenset[str] = frozenset({
    "post_report",
    "chat",
    "pre_restore",
})


def resolve_workflow_boundary_phase(
    error: RequiredSnapshotBlocked | Exception | None = None,
    *,
    phase: str | None = None,
) -> str | None:
    """Resolve the real workflow-boundary capture phase for retry context.

    Prefer an explicit *phase*, then ``error.details["phase"]`` /
    ``error.phase``. Never invent ``"workflow_boundary"`` as a phase name —
    that is a producer id, not a ContinuationRegistry phase.
    """
    if phase:
        p = str(phase).strip()
        if p and p != "workflow_boundary":
            return p
        if p == "workflow_boundary":
            return None
    details = getattr(error, "details", None) or {}
    if isinstance(details, dict):
        raw = details.get("phase")
        if raw:
            p = str(raw).strip()
            if p and p != "workflow_boundary":
                return p
    raw_attr = getattr(error, "phase", None)
    if raw_attr:
        p = str(raw_attr).strip()
        if p and p != "workflow_boundary":
            return p
    return None


def workflow_boundary_retry_context(
    error: RequiredSnapshotBlocked | Exception | None = None,
    *,
    phase: str | None = None,
    previous_status: str = "running",
) -> dict[str, Any]:
    """Build typed workflow_boundary retry_context with the real phase."""
    resolved = resolve_workflow_boundary_phase(error, phase=phase)
    snapshot_request: dict[str, Any] = {}
    if resolved:
        snapshot_request["phase"] = resolved
    return {
        "producer": "workflow_boundary",
        "resume_policy": "graph_checkpoint",
        "snapshot_request": snapshot_request,
        "previous_status": previous_status,
    }


def post_report_retry_context(
    error: RequiredSnapshotBlocked | Exception | None = None,
    *,
    previous_status: str = "generating_report",
    snapshot_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build typed post_report retry_context (never producer=workflow_boundary)."""
    req: dict[str, Any] = dict(snapshot_request or {})
    if not req.get("phase"):
        resolved = resolve_workflow_boundary_phase(error)
        req["phase"] = resolved or "post_report"
    # Never invent producer id as phase.
    if str(req.get("phase") or "").strip() == "workflow_boundary":
        req["phase"] = "post_report"
    return {
        "producer": "post_report",
        "resume_policy": "complete",
        "snapshot_request": req,
        "previous_status": previous_status,
    }


def _workflow_boundary_context_incomplete(retry_context: Any) -> bool:
    """True when workflow_boundary retry_context is missing or incomplete.

    Non-workflow_boundary producers (post_report / chat / pre_restore) are
    never treated as incomplete for WB purposes — callers must not overwrite
    them with workflow_boundary typing.
    """
    if not isinstance(retry_context, dict):
        return True
    producer = retry_context.get("producer")
    if producer in _NON_WORKFLOW_BOUNDARY_PRODUCERS:
        return False
    if producer is not None and producer != "workflow_boundary":
        return False
    if producer != "workflow_boundary":
        # Missing producer (bare gate) is incomplete for WB typing.
        return True
    req = retry_context.get("snapshot_request")
    if not isinstance(req, dict):
        return True
    phase = req.get("phase")
    if not phase or not str(phase).strip():
        return True
    if str(phase).strip() == "workflow_boundary":
        return True
    return False


def _post_report_context_incomplete(retry_context: Any) -> bool:
    """True when post_report retry_context is missing or incomplete."""
    if not isinstance(retry_context, dict):
        return True
    if retry_context.get("producer") != "post_report":
        return True
    req = retry_context.get("snapshot_request")
    if not isinstance(req, dict):
        return True
    phase = req.get("phase")
    if not phase or not str(phase).strip():
        return True
    if str(phase).strip() == "workflow_boundary":
        return True
    return False


def ensure_post_report_retry_context(
    state: Any,
    error: RequiredSnapshotBlocked | Exception | None = None,
    *,
    previous_status: str = "generating_report",
    snapshot_request: dict[str, Any] | None = None,
) -> None:
    """Ensure failure gate has typed post_report retry_context.

    Mirrors chat / workflow-boundary ensure helpers: bare service gates and
    incomplete contexts are upgraded; never invents phase ``workflow_boundary``.
    """
    typed = post_report_retry_context(
        error,
        previous_status=previous_status,
        snapshot_request=snapshot_request,
    )
    existing = getattr(state, "snapshot_failure", None)
    if not existing:
        code = getattr(error, "code", None) or "REQUIRED_SNAPSHOT_BLOCKED"
        message = (str(error) if error is not None else "") or "REQUIRED_SNAPSHOT_BLOCKED"
        snapshot_id = getattr(error, "snapshot_id", None) if error is not None else None
        enter_snapshot_failure_gate(
            state,
            {
                "code": code,
                "message": message,
                "allowed_actions": list(SNAPSHOT_FAILURE_GATE_ACTIONS),
                "snapshot_id": snapshot_id,
            },
            retry_context=typed,
        )
        return

    payload = dict(existing)
    if "allowed_actions" not in payload:
        payload["allowed_actions"] = list(SNAPSHOT_FAILURE_GATE_ACTIONS)
    if _post_report_context_incomplete(payload.get("retry_context")):
        payload["retry_context"] = typed
    state.snapshot_failure = payload
    state.status = TaskStatus.WAITING_INPUT
    state.wait_input_reason = WaitInputReason.SNAPSHOT_FAILURE.value


def ensure_workflow_boundary_retry_context(
    state: Any,
    error: RequiredSnapshotBlocked | Exception,
    *,
    previous_status: str = "running",
    phase: str | None = None,
) -> None:
    """Ensure failure gate has typed retry_context for run/resume catch sites.

    ``SnapshotService.capture_required_snapshot`` often sets a bare
    ``snapshot_failure`` (no ``retry_context``) before raising
    ``RequiredSnapshotBlocked``. Core catch sites call this helper for every
    block — including post_report — so typing must be producer-aware:

    - phase / existing producer ``post_report`` → post_report context
    - never overwrite chat / pre_restore / post_report with workflow_boundary
    - workflow_boundary typing only for real boundary phases
    - never invent phase ``"workflow_boundary"``
    """
    resolved = resolve_workflow_boundary_phase(error, phase=phase)
    existing = getattr(state, "snapshot_failure", None)
    existing_rc = (
        existing.get("retry_context")
        if isinstance(existing, dict)
        else None
    )
    existing_producer = (
        existing_rc.get("producer")
        if isinstance(existing_rc, dict)
        else None
    )

    # Preserve / upgrade non-WB producers; never clobber with workflow_boundary.
    if existing_producer in _NON_WORKFLOW_BOUNDARY_PRODUCERS:
        if existing_producer == "post_report" or resolved == "post_report":
            ensure_post_report_retry_context(
                state,
                error,
                previous_status=(
                    existing_rc.get("previous_status")
                    if isinstance(existing_rc, dict)
                    and existing_rc.get("previous_status")
                    else "generating_report"
                ),
                snapshot_request=(
                    dict(existing_rc.get("snapshot_request") or {})
                    if isinstance(existing_rc, dict)
                    else None
                ),
            )
        else:
            # chat / pre_restore already typed — keep status on gate.
            if isinstance(existing, dict):
                state.snapshot_failure = dict(existing)
            state.status = TaskStatus.WAITING_INPUT
            state.wait_input_reason = WaitInputReason.SNAPSHOT_FAILURE.value
        return

    if resolved == "post_report":
        ensure_post_report_retry_context(
            state,
            error,
            previous_status="generating_report",
        )
        return

    # Only apply workflow_boundary for real boundary phases (or bare unknown
    # phase that is not post_report). Never invent phase "workflow_boundary".
    if resolved is not None and resolved not in _WORKFLOW_BOUNDARY_PHASES:
        # Non-boundary known phase without a dedicated producer path: leave
        # existing gate as-is rather than mislabeling as workflow_boundary.
        if existing:
            payload = dict(existing)
            if "allowed_actions" not in payload:
                payload["allowed_actions"] = list(SNAPSHOT_FAILURE_GATE_ACTIONS)
            state.snapshot_failure = payload
            state.status = TaskStatus.WAITING_INPUT
            state.wait_input_reason = WaitInputReason.SNAPSHOT_FAILURE.value
            return

    typed = workflow_boundary_retry_context(
        error,
        phase=resolved,
        previous_status=previous_status,
    )
    if not existing:
        code = getattr(error, "code", None) or "REQUIRED_SNAPSHOT_BLOCKED"
        message = str(error) or type(error).__name__
        snapshot_id = getattr(error, "snapshot_id", None)
        enter_snapshot_failure_gate(
            state,
            {
                "code": code,
                "message": message,
                "allowed_actions": list(SNAPSHOT_FAILURE_GATE_ACTIONS),
                "snapshot_id": snapshot_id,
            },
            retry_context=typed,
        )
        return

    payload = dict(existing)
    if "allowed_actions" not in payload:
        payload["allowed_actions"] = list(SNAPSHOT_FAILURE_GATE_ACTIONS)
    if _workflow_boundary_context_incomplete(payload.get("retry_context")):
        payload["retry_context"] = typed
    state.snapshot_failure = payload
    state.status = TaskStatus.WAITING_INPUT
    state.wait_input_reason = WaitInputReason.SNAPSHOT_FAILURE.value


def enter_snapshot_failure_gate(
    state: Any,
    failure: dict[str, Any],
    retry_context: dict[str, Any] | None = None,
) -> None:
    """Set the snapshot failure gate on *state*.

    Replaces all ad-hoc manual construction.  Sets:

      * ``status`` → ``WAITING_INPUT``
      * ``wait_input_reason`` → ``"snapshot_failure"``
      * ``snapshot_failure`` → *failure* payload (with ``allowed_actions``)
      * ``snapshot_failure["retry_context"]`` → *retry_context* (when provided)

    The caller is responsible for WS emit and disk persistence (either
    already done by ``capture_required_snapshot`` before raising
    ``RequiredSnapshotBlocked``, or done explicitly by the caller).
    """
    state.status = TaskStatus.WAITING_INPUT
    state.wait_input_reason = WaitInputReason.SNAPSHOT_FAILURE.value
    payload = dict(failure)
    if "allowed_actions" not in payload:
        payload["allowed_actions"] = list(SNAPSHOT_FAILURE_GATE_ACTIONS)
    if retry_context is not None:
        payload["retry_context"] = dict(retry_context)
    state.snapshot_failure = payload


def enter_chat_snapshot_failure_gate(
    state: Any,
    error: RequiredSnapshotBlocked | Exception,
    previous_status: TaskStatus | str,
) -> None:
    """Enter the failure gate with a typed chat producer retry context.

    Both delegated-chat and ordinary-chat handlers must use this helper
    instead of constructing ``state.snapshot_failure`` directly.
    """
    if isinstance(previous_status, TaskStatus):
        prev = previous_status.value
    else:
        prev = str(previous_status)

    code = getattr(error, "code", None) or "REQUIRED_SNAPSHOT_BLOCKED"
    message = str(error) or type(error).__name__
    snapshot_id = getattr(error, "snapshot_id", None)

    enter_snapshot_failure_gate(
        state,
        {
            "code": code,
            "message": message,
            "allowed_actions": list(SNAPSHOT_FAILURE_GATE_ACTIONS),
            "snapshot_id": snapshot_id,
        },
        retry_context={
            "producer": "chat",
            "resume_policy": "stable_status",
            "snapshot_request": {"phase": "post_chat"},
            "previous_status": prev,
        },
    )


def retry_post_report(state: Any) -> TaskStatus:
    """Clear failure gate and return to GENERATING_REPORT for capture-only retry.

    The caller should then invoke ``workflow.retry_pending_required_snapshot()``
    or the equivalent capture path.
    """
    state.snapshot_failure = None
    state.wait_input_reason = None
    state.status = TaskStatus.GENERATING_REPORT
    return TaskStatus.GENERATING_REPORT


def retry_workflow_boundary(state: Any) -> None:
    """Clear failure gate and keep RUNNING (workflow retries from checkpoint)."""
    state.snapshot_failure = None
    state.wait_input_reason = None
    state.status = TaskStatus.RUNNING


def retry_pre_restore(state: Any) -> None:
    """Clear failure gate so the restore coordinator can retry."""
    state.snapshot_failure = None
    state.wait_input_reason = None


def retry_chat(
    state: Any,
    previous_status: TaskStatus | str | None = None,
) -> None:
    """Clear failure gate and restore *previous_status*.

    Falls back to COMPLETED when no previous status is available.
    """
    state.snapshot_failure = None
    state.wait_input_reason = None
    if previous_status is not None:
        try:
            state.status = (
                TaskStatus(previous_status)
                if isinstance(previous_status, str)
                else previous_status
            )
        except Exception:
            state.status = TaskStatus.COMPLETED
    else:
        state.status = TaskStatus.COMPLETED


def terminate_snapshot_failure(state: Any) -> None:
    """Clear failure gate, mark STOPPED, and record an audit reason.

    Does NOT create a second kernel teardown — the caller drives the single
    workflow teardown. Clears retryability by dropping the failure gate.
    """
    state.snapshot_failure = None
    state.wait_input_reason = None
    state.status = TaskStatus.STOPPED
    state.termination_reason = "snapshot_failure"
    if hasattr(state, "pending_required_snapshot"):
        state.pending_required_snapshot = None


def _status_value(state: Any) -> str | None:
    status = getattr(state, "status", None)
    if status is None:
        return None
    return getattr(status, "value", str(status))


def _reenter_gate_on_failure(
    state: Any,
    *,
    code: str,
    message: str,
    context: SnapshotRetryContext,
    snapshot_id: str | None = None,
) -> RetryOutcome:
    enter_snapshot_failure_gate(
        state,
        {
            "code": code,
            "message": message,
            "allowed_actions": list(SNAPSHOT_FAILURE_GATE_ACTIONS),
            "snapshot_id": snapshot_id,
        },
        retry_context=context.model_dump(mode="json"),
    )
    return RetryOutcome(
        success=False,
        still_blocked=True,
        message="Snapshot retry failed; still blocked",
        status=_status_value(state),
        wait_input_reason=getattr(state, "wait_input_reason", None),
        snapshot_failure=getattr(state, "snapshot_failure", None),
    )


async def default_retry_post_report(
    session_id: str,
    state: Any,
    context: SnapshotRetryContext,
) -> RetryOutcome:
    """Capture-only post-report retry during the HTTP request.

    Capture runs while still gated. Gate is cleared only after a successful
    commit; on failure the typed context is restored and WAITING_INPUT remains.
    """
    from datetime import datetime

    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.state import workflows, websockets, state_lock as _state_lock

    temporary_workflow = False
    workflow = None
    try:
        async with _state_lock:
            workflow = workflows.get(session_id)
        if workflow is None or not hasattr(
            workflow, "retry_pending_required_snapshot",
        ):
            from backend.services.workflow_runner import (
                _build_workflow_for_session,
            )

            workflow, built_state = await _build_workflow_for_session(
                session_id,
                fail_on_missing_config=False,
            )
            if workflow is None or not hasattr(
                workflow, "retry_pending_required_snapshot",
            ):
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Workflow is not available to recapture the "
                        "post-report snapshot; snapshot failure gate "
                        "left intact"
                    ),
                )
            attach_state = built_state if built_state is not None else state
            workflow._current_analysis_state = attach_state
            if getattr(workflow, "state", None) is None:
                try:
                    workflow.state = attach_state
                except Exception:
                    pass
            temporary_workflow = True

        # Keep gate until capture commits; only set GENERATING_REPORT status
        # as the capture target status without clearing snapshot_failure first.
        try:
            state.status = TaskStatus.GENERATING_REPORT
            await workflow.retry_pending_required_snapshot()
        except RequiredSnapshotBlocked as exc:
            return _reenter_gate_on_failure(
                state,
                code=exc.code,
                message=str(exc),
                context=context,
                snapshot_id=getattr(exc, "snapshot_id", None),
            )
        except Exception as exc:
            import logging

            logging.getLogger(__name__).exception(
                "[%s] Post-report snapshot retry raised",
                session_id,
            )
            return _reenter_gate_on_failure(
                state,
                code="REQUIRED_SNAPSHOT_RETRY_ERROR",
                message=str(exc) or type(exc).__name__,
                context=context,
            )

        if getattr(state, "snapshot_failure", None):
            return _reenter_gate_on_failure(
                state,
                code=(state.snapshot_failure or {}).get(
                    "code", "REQUIRED_SNAPSHOT_BLOCKED",
                ),
                message=(state.snapshot_failure or {}).get(
                    "message", "Snapshot retry failed; still blocked",
                ),
                context=context,
                snapshot_id=(state.snapshot_failure or {}).get("snapshot_id"),
            )

        # Success: clear gate + finalize report.
        state.status = TaskStatus.COMPLETED
        state.wait_input_reason = None
        state.snapshot_failure = None
        ws = websockets.get(session_id)
        if ws is not None:
            try:
                await ws.send_json({
                    "type": "completed",
                    "agent": "pi",
                    "content": "Analysis completed",
                    "timestamp": datetime.now().isoformat(),
                })
                await ws.send_json({
                    "type": "state_sync",
                    "agent": None,
                    "content": {
                        "change_type": "analysis_complete",
                        "status": TaskStatus.COMPLETED.value,
                        "wait_input_reason": None,
                        "snapshot_failure": None,
                    },
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception:
                import logging

                logging.getLogger(__name__).warning(
                    "[%s] Failed to emit post-report retry completion",
                    session_id,
                    exc_info=True,
                )
        return RetryOutcome(
            success=True,
            still_blocked=False,
            message="Snapshot recaptured; report finalized",
            status=_status_value(state),
            wait_input_reason=None,
            snapshot_failure=None,
        )
    finally:
        if temporary_workflow and workflow is not None:
            try:
                await workflow.close()
            except Exception:
                import logging

                logging.getLogger(__name__).warning(
                    "[%s] Error closing temporary post-report retry workflow",
                    session_id,
                    exc_info=True,
                )
            async with _state_lock:
                if workflows.get(session_id) is workflow:
                    workflows.pop(session_id, None)


async def default_retry_chat_capture(
    session_id: str,
    state: Any,
    context: SnapshotRetryContext,
) -> RetryOutcome:
    """Re-invoke the post-chat capture producer during the HTTP request.

    Capture runs while still gated. Gate is cleared only after a successful
    commit; if failure remains after capture, return still_blocked.
    """
    from backend.routes.chat import capture_post_chat_snapshot
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked

    prev = context.previous_status
    try:
        await capture_post_chat_snapshot(session_id, state)
    except RequiredSnapshotBlocked as exc:
        return _reenter_gate_on_failure(
            state,
            code=exc.code,
            message=str(exc),
            context=context,
            snapshot_id=getattr(exc, "snapshot_id", None),
        )
    except Exception as exc:
        return _reenter_gate_on_failure(
            state,
            code="REQUIRED_SNAPSHOT_RETRY_ERROR",
            message=str(exc) or type(exc).__name__,
            context=context,
        )

    # Align with pre_restore / workflow_boundary: residual gate means blocked.
    if getattr(state, "snapshot_failure", None):
        return _reenter_gate_on_failure(
            state,
            code=(state.snapshot_failure or {}).get(
                "code", "REQUIRED_SNAPSHOT_BLOCKED",
            ),
            message=(state.snapshot_failure or {}).get(
                "message", "Snapshot retry failed; still blocked",
            ),
            context=context,
            snapshot_id=(state.snapshot_failure or {}).get("snapshot_id"),
        )

    # Success: clear gate + restore previous status.
    retry_chat(state, prev)
    return RetryOutcome(
        success=True,
        still_blocked=False,
        message="Snapshot recaptured; chat retry complete",
        status=_status_value(state),
        wait_input_reason=None,
        snapshot_failure=None,
    )


async def default_retry_pre_restore_capture(
    session_id: str,
    state: Any,
    context: SnapshotRetryContext,
) -> RetryOutcome:
    """Re-invoke pre-restore capture during the HTTP request.

    Capture runs while still gated. Gate is cleared only after a successful
    capture commit. On failure the same retry context is restored and the
    session remains WAITING_INPUT.
    """
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.services.snapshot_v2.models import (
        IdleCapsule,
        SnapshotCaptureRequest,
    )
    from backend.services.snapshot_v2.paths import SnapshotV2Paths
    from backend.services.snapshot_v2.service import SnapshotService

    try:
        req_data = dict(context.snapshot_request or {})
        req_data.setdefault("session_id", session_id)
        req_data.setdefault("reason", "pre_restore")
        req_data.setdefault("phase", "pre_restore")
        if not req_data.get("continuation") and not req_data.get("capsule"):
            req_data["continuation"] = IdleCapsule(
                graph_version="2.0",
            ).model_dump(mode="json")
        request = SnapshotCaptureRequest.model_validate(req_data)
        paths = SnapshotV2Paths.for_session(session_id)
        service = SnapshotService(session_id, paths)
        await service.retry_capture(state, request)
    except RequiredSnapshotBlocked as exc:
        return _reenter_gate_on_failure(
            state,
            code=exc.code,
            message=str(exc),
            context=context,
            snapshot_id=getattr(exc, "snapshot_id", None),
        )
    except Exception as exc:
        return _reenter_gate_on_failure(
            state,
            code="REQUIRED_SNAPSHOT_RETRY_ERROR",
            message=str(exc) or type(exc).__name__,
            context=context,
        )

    if getattr(state, "snapshot_failure", None):
        return _reenter_gate_on_failure(
            state,
            code=(state.snapshot_failure or {}).get(
                "code", "REQUIRED_SNAPSHOT_BLOCKED",
            ),
            message=(state.snapshot_failure or {}).get(
                "message", "Snapshot retry failed; still blocked",
            ),
            context=context,
            snapshot_id=(state.snapshot_failure or {}).get("snapshot_id"),
        )

    # Success: clear gate after commit.
    retry_pre_restore(state)
    return RetryOutcome(
        success=True,
        still_blocked=False,
        message="Snapshot recaptured; restore retry ready",
        status=_status_value(state),
        wait_input_reason=getattr(state, "wait_input_reason", None),
        snapshot_failure=None,
    )


async def default_retry_workflow_boundary_capture(
    session_id: str,
    state: Any,
    context: SnapshotRetryContext,
) -> RetryOutcome:
    """Re-invoke workflow-boundary capture during the HTTP request.

    Uses the real phase from ``context.snapshot_request`` only — never invents
    ``"workflow_boundary"`` as a phase name. Capture runs while still gated;
    the gate is cleared only after a successful commit.
    """
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.state import workflows, state_lock as _state_lock

    phase = (context.snapshot_request or {}).get("phase")
    if not phase or str(phase).strip() == "workflow_boundary":
        return _reenter_gate_on_failure(
            state,
            code="UNKNOWN_PHASE",
            message=(
                "Workflow-boundary retry is missing the real capture phase; "
                "cannot re-enter ContinuationRegistry with producer id"
            ),
            context=context,
        )

    async with _state_lock:
        workflow = workflows.get(session_id)

    try:
        if workflow is not None and hasattr(workflow, "_create_auto_snapshot"):
            await workflow._create_auto_snapshot(state, str(phase).strip())
        else:
            return _reenter_gate_on_failure(
                state,
                code="WORKFLOW_UNAVAILABLE",
                message=(
                    "Workflow is not available to recapture the "
                    "workflow-boundary snapshot"
                ),
                context=context,
            )
    except RequiredSnapshotBlocked as exc:
        return _reenter_gate_on_failure(
            state,
            code=exc.code,
            message=str(exc),
            context=context,
            snapshot_id=getattr(exc, "snapshot_id", None),
        )
    except Exception as exc:
        return _reenter_gate_on_failure(
            state,
            code="REQUIRED_SNAPSHOT_RETRY_ERROR",
            message=str(exc) or type(exc).__name__,
            context=context,
        )

    if getattr(state, "snapshot_failure", None):
        return _reenter_gate_on_failure(
            state,
            code=(state.snapshot_failure or {}).get(
                "code", "REQUIRED_SNAPSHOT_BLOCKED",
            ),
            message=(state.snapshot_failure or {}).get(
                "message", "Snapshot retry failed; still blocked",
            ),
            context=context,
            snapshot_id=(state.snapshot_failure or {}).get("snapshot_id"),
        )

    # Success: clear gate + keep RUNNING for graph checkpoint resume.
    retry_workflow_boundary(state)
    return RetryOutcome(
        success=True,
        still_blocked=False,
        message="Snapshot recaptured; workflow retry ready",
        status=_status_value(state),
        wait_input_reason=None,
        snapshot_failure=None,
    )


def default_snapshot_retry_dependencies() -> SnapshotRetryDependencies:
    """Production dependency wiring for producer-dispatched retry."""
    return SnapshotRetryDependencies(
        retry_post_report=default_retry_post_report,
        retry_chat_capture=default_retry_chat_capture,
        retry_pre_restore_capture=default_retry_pre_restore_capture,
        retry_workflow_boundary_capture=default_retry_workflow_boundary_capture,
    )


async def retry_snapshot_producer(
    session_id: str,
    state: Any,
    context: SnapshotRetryContext,
    dependencies: SnapshotRetryDependencies,
) -> RetryOutcome:
    """Dispatch Retry to exactly the failed capture producer.

    Never defaults a missing producer; callers must validate context first.
    """
    from backend.services.snapshot_v2.errors import SnapshotTransactionError

    if context.producer == "post_report":
        return await dependencies.retry_post_report(session_id, state, context)
    if context.producer == "chat":
        return await dependencies.retry_chat_capture(session_id, state, context)
    if context.producer == "pre_restore":
        return await dependencies.retry_pre_restore_capture(
            session_id, state, context,
        )
    if context.producer == "workflow_boundary":
        return await dependencies.retry_workflow_boundary_capture(
            session_id, state, context,
        )
    raise SnapshotTransactionError(
        "UNKNOWN_RETRY_PRODUCER",
        str(context.producer),
    )
