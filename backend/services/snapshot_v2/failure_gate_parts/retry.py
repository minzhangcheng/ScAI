"""Snapshot producer retry implementations and dispatch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from backend.constants import WaitInputReason
from backend.models.schemas import TaskStatus
from backend.services.snapshot_v2.errors import (
    SNAPSHOT_FAILURE_ACTIONS,
    SNAPSHOT_FAILURE_ACTIONS_MEMORY,
    SNAPSHOT_FAILURE_ACTIONS_SKIPPABLE,
    SNAPSHOT_MEMORY_INSUFFICIENT_CODE,
    RequiredSnapshotBlocked,
)
from backend.services.snapshot_v2.models import RetryOutcome, SnapshotRetryContext
from backend.utils.error_parsing import sanitize_error_payload

SnapshotFailureActions = tuple[str, ...]
SNAPSHOT_FAILURE_GATE_ACTIONS: SnapshotFailureActions = SNAPSHOT_FAILURE_ACTIONS
SNAPSHOT_SKIPPABLE_GATE_ACTIONS: SnapshotFailureActions = SNAPSHOT_FAILURE_ACTIONS_SKIPPABLE

from .context import (
    SnapshotRetryDependencies,
    _actions_for_code,
    _status_value,
    enter_snapshot_failure_gate,
    retry_chat,
    retry_pre_restore,
    retry_workflow_boundary,
)

_SKIPPABLE_PRODUCERS: frozenset[str] = frozenset({
    "workflow_boundary", "post_report", "chat",
})


def _actions_for_producer(
    producer: str | None, code: str | None = None,
) -> list[str]:
    """Base actions plus Skip for producers where skipping is safe.

    *code* lets a memory-preflight refusal override the producer-derived set;
    see :func:`_actions_for_code`.
    """
    base = (
        SNAPSHOT_SKIPPABLE_GATE_ACTIONS
        if producer in _SKIPPABLE_PRODUCERS
        else SNAPSHOT_FAILURE_GATE_ACTIONS
    )
    return _actions_for_code(code, base)


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
            "allowed_actions": _actions_for_producer(context.producer, code),
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
    from backend.state import workflows, state_lock as _state_lock

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
        from backend.utils.websocket_delivery import send_analysis_json
        await send_analysis_json(session_id, {
                    "type": "completed",
                    "agent": "pi",
                    "content": "Analysis completed",
                    "timestamp": datetime.now().isoformat(),
        })
        await send_analysis_json(session_id, {
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
            close_complete = False
            try:
                close_result = await workflow.close()
                close_complete = close_result is not False
            except Exception:
                import logging

                logging.getLogger(__name__).warning(
                    "[%s] Error closing temporary post-report retry workflow",
                    session_id,
                    exc_info=True,
                )
            async with _state_lock:
                if close_complete and workflows.get(session_id) is workflow:
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
    """Queue a new restore job linked to the terminal failed transaction.

    A pre-restore capture failure makes the original restore job terminal.
    Retry therefore revalidates the preserved restore request and enqueues a
    distinct job; it never publishes a standalone pre-restore snapshot or
    resurrects the old journal.
    """
    from backend.services.snapshot_v2.jobs import RestoreJobManager
    from backend.services.snapshot_v2.models import SnapshotRestoreRequest
    from backend.services.snapshot_v2.paths import SnapshotV2Paths
    from backend.services.snapshot_v2.preflight import RestorePreflight

    try:
        if not context.restore_transaction_id or not context.restore_request:
            return _reenter_gate_on_failure(
                state,
                code="MISSING_RESTORE_RETRY_CONTEXT",
                message=(
                    "Pre-restore Retry is missing the original restore "
                    "transaction or request"
                ),
                context=context,
            )
        request = SnapshotRestoreRequest.model_validate(context.restore_request)
        paths = SnapshotV2Paths.for_session(session_id)
        preflight = RestorePreflight(paths).run(request.snapshot_id, request)
        if preflight.status != "ready":
            return _reenter_gate_on_failure(
                state,
                code="RESTORE_RETRY_PREFLIGHT_FAILED",
                message=str(
                    preflight.compatibility_errors or preflight.status
                ),
                context=context,
            )
        manager = RestoreJobManager(paths=paths, session_id=session_id)
        record = await manager.enqueue(
            session_id,
            preflight,
            request,
            retry_of_transaction_id=context.restore_transaction_id,
        )
    except Exception as exc:
        return _reenter_gate_on_failure(
            state,
            code="RESTORE_RETRY_ENQUEUE_FAILED",
            message=str(exc) or type(exc).__name__,
            context=context,
        )

    # Enqueue is the acceptance point. The new restore job reacquires its own
    # safe point and re-enters the full preflight/capture/apply state machine.
    transaction_payload = record.model_dump(mode="json")
    retry_pre_restore(state)
    state.snapshot_transaction = transaction_payload
    return RetryOutcome(
        success=True,
        still_blocked=False,
        message="Restore retry queued",
        status=_status_value(state),
        wait_input_reason=getattr(state, "wait_input_reason", None),
        snapshot_failure=None,
        transaction_id=record.transaction_id,
        snapshot_transaction=transaction_payload,
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

    if workflow is None or not hasattr(workflow, "_create_auto_snapshot"):
        # No live workflow is the normal case here, not an anomaly:
        # RequiredSnapshotBlocked unwinds the whole graph invocation, so
        # run()/resume() return, the WebSocket handler's finally closes the
        # workflow and drops it from ``backend.state.workflows``, and a
        # reconnect while WAITING_INPUT takes the HITL-hold branch that never
        # builds a replacement. Failing closed here made Retry unwinnable.
        #
        # Rebuilding a workflow (as post_report does) is not an option either:
        # the capsule is frozen from the live ``_current_graph_state`` and the
        # live kernel, both destroyed with the torn-down workflow, so an
        # in-process recapture would commit a snapshot naming a boundary it
        # cannot restore to.
        #
        # So Retry means what ``resume_policy="graph_checkpoint"`` says: clear
        # the gate and let the graph re-drive from its checkpoint. The resume
        # re-hits the same boundary and re-runs the capture against live state
        # -- the mechanism Skip already relies on, minus the skip marker, so
        # nothing passes the boundary without either a snapshot or a new gate.
        retry_workflow_boundary(state)
        return RetryOutcome(
            success=True,
            still_blocked=False,
            message="Snapshot retry armed; workflow resumes to the boundary",
            status=_status_value(state),
            wait_input_reason=None,
            snapshot_failure=None,
        )

    try:
        await workflow._create_auto_snapshot(state, str(phase).strip())
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

__all__ = [name for name in globals() if not name.startswith("__")]
