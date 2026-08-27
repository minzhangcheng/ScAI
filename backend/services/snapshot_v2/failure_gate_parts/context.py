"""Snapshot-failure actions, contexts, and state mutations."""
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


def _status_value(state: Any) -> str | None:
    status = getattr(state, "status", None)
    if status is None:
        return None
    return getattr(status, "value", str(status))

def _actions_for_code(
    code: str | None, base: SnapshotFailureActions,
) -> list[str]:
    """Gate actions for a failure *code*, defaulting to the producer's *base* set.

    A memory-preflight refusal replaces the base set. Retry is dropped because
    the preflight is deterministic — re-running it measures the same objects
    against the same headroom and refuses again, so the gate could never clear.
    Manage is dropped because it reclaims disk, which does not free kernel
    memory. Proceed takes retry's place as the "capture it anyway" escape hatch.

    Skip only survives when *base* already allowed it: pre_restore forbids
    skipping (its snapshot is the sole safety point before a restore overwrites
    live state), and a memory refusal must not smuggle that option back in.
    """
    if code != SNAPSHOT_MEMORY_INSUFFICIENT_CODE:
        return list(base)
    actions = list(SNAPSHOT_FAILURE_ACTIONS_MEMORY)
    if "skip" not in base:
        actions = [action for action in actions if action != "skip"]
    return actions


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
                "allowed_actions": _actions_for_code(code, SNAPSHOT_SKIPPABLE_GATE_ACTIONS),
                "snapshot_id": snapshot_id,
            },
            retry_context=typed,
        )
        return

    payload = dict(existing)

    # Producer-aware, like ensure_workflow_boundary_retry_context. This helper is
    # reachable from generate_report's entry guard with ANY gate already armed,
    # and it used to rewrite both fields unconditionally. That converted a
    # workflow_boundary — or a pre_restore — gate into a post_report one, which
    # is wrong twice over: Retry then dispatches to the post_report producer and
    # can never satisfy the original boundary, and the action set gains a Skip
    # that pre_restore explicitly forbids (skip_snapshot_producer refuses it).
    existing_producer = str(
        (payload.get("retry_context") or {}).get("producer") or ""
    )
    if existing_producer and existing_producer != "post_report":
        return

    # post_report is skippable: reports are already generated, so continuing
    # without the post-report snapshot only forfeits that restore point.
    payload["allowed_actions"] = _actions_for_code(
        payload.get("code"), SNAPSHOT_SKIPPABLE_GATE_ACTIONS
    )
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
                payload["allowed_actions"] = _actions_for_code(
                    payload.get("code"), SNAPSHOT_FAILURE_GATE_ACTIONS
                )
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
                "allowed_actions": _actions_for_code(code, SNAPSHOT_SKIPPABLE_GATE_ACTIONS),
                "snapshot_id": snapshot_id,
            },
            retry_context=typed,
        )
        return

    payload = dict(existing)
    # Workflow-boundary captures are skippable: the analysis proceeds past the
    # boundary without that restore point.
    payload["allowed_actions"] = _actions_for_code(
        payload.get("code"), SNAPSHOT_SKIPPABLE_GATE_ACTIONS
    )
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
    # Arriving at a gate means the capture attempt that preceded it did not
    # succeed, so any "capture anyway" approval still outstanding was for that
    # attempt and is now spent. The user is about to be asked again, and
    # answering Proceed re-arms it — nothing is lost, and an approval can no
    # longer drift onto a capture it was never given for.
    clear_memory_preflight_override(state)
    payload = sanitize_error_payload(dict(failure), max_chars=3000)
    declared = payload.get("allowed_actions")
    if not declared:
        payload["allowed_actions"] = _actions_for_code(
            payload.get("code"), SNAPSHOT_FAILURE_GATE_ACTIONS
        )
    elif payload.get("code") == SNAPSHOT_MEMORY_INSUFFICIENT_CODE:
        # A memory refusal's action set is decided by the code, never by the
        # caller: retry re-runs a deterministic check against the same kernel
        # and manage reclaims disk, neither of which can clear this gate. Only
        # the caller's skip policy is preserved — pre_restore forbids skipping
        # and a memory refusal must not smuggle it back in. This also repairs a
        # payload that reached us carrying a stale or defaulted set, which is
        # how the gate lost "Capture anyway" on the pre_restore path.
        payload["allowed_actions"] = _actions_for_code(
            SNAPSHOT_MEMORY_INSUFFICIENT_CODE,
            SNAPSHOT_SKIPPABLE_GATE_ACTIONS
            if "skip" in declared
            else SNAPSHOT_FAILURE_GATE_ACTIONS,
        )
    if retry_context is not None:
        payload["retry_context"] = sanitize_error_payload(
            dict(retry_context), max_chars=3000
        )
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
            "allowed_actions": _actions_for_code(code, SNAPSHOT_SKIPPABLE_GATE_ACTIONS),
            "snapshot_id": snapshot_id,
        },
        retry_context={
            "producer": "chat",
            "resume_policy": "stable_status",
            "snapshot_request": {"phase": "post_chat"},
            "previous_status": prev,
        },
    )


def upgrade_legacy_snapshot_failure_gate(state: Any) -> bool:
    """Upgrade a persisted pre-producer snapshot gate in place.

    Older versions emitted and persisted the service-level failure before the
    outer producer catch attached ``retry_context``. Those gates expose only
    Retry/Manage/Terminate, and Retry/Skip routes cannot dispatch them after a
    restart. A context-free required-capture gate is a workflow boundary unless
    state proves a dedicated producer. Explicit typed contexts, especially
    ``pre_restore``, remain authoritative and are never widened.
    """
    failure = getattr(state, "snapshot_failure", None)
    if not isinstance(failure, dict) or not failure:
        return False
    retry_context = failure.get("retry_context")
    if isinstance(retry_context, dict) and retry_context.get("producer"):
        return False

    details = failure.get("details")
    details = details if isinstance(details, dict) else {}
    phase = str(details.get("phase") or "").strip()
    if phase == "pre_restore":
        return False

    pending = getattr(state, "pending_required_snapshot", None)
    if isinstance(pending, dict) and pending:
        ensure_post_report_retry_context(
            state,
            previous_status="generating_report",
            snapshot_request=dict(pending),
        )
        return True

    if phase == "post_report":
        ensure_post_report_retry_context(
            state,
            previous_status="generating_report",
            snapshot_request={"phase": "post_report"},
        )
        return True

    error = RequiredSnapshotBlocked(
        str(failure.get("code") or "REQUIRED_SNAPSHOT_BLOCKED"),
        str(failure.get("message") or "Required snapshot capture failed"),
        details=details,
        snapshot_id=(
            str(failure.get("snapshot_id"))
            if failure.get("snapshot_id")
            else None
        ),
        phase=phase or None,
    )

    if phase == "post_chat":
        # A post-chat gate reaches disk in two writes: the service-level bare
        # failure is persisted by `_create_chat_snapshot` before it re-raises,
        # and only the chat endpoint attaches the typed context and persists
        # again. A crash (or a failed second write) between them leaves a bare
        # gate whose phase is unambiguous. Without a branch here it fell through
        # to the unknown-phase refusal below and stayed context-free forever, so
        # both Retry and Skip answered 409 and Terminate — which stops the
        # session and deletes the graph checkpoint — was the only way out of a
        # failed *chat* snapshot. Chat is a first-class skippable producer
        # everywhere else, so the "unknown captures are not automatically made
        # skippable" rationale does not apply to it.
        enter_chat_snapshot_failure_gate(
            state,
            error,
            # The pre-gate status is not recoverable from a bare gate, and the
            # live status is the gate's own WAITING_INPUT — restoring that would
            # leave the session parked. PAUSED is the neutral answer: the UI
            # offers Resume and nothing claims progress that is not happening.
            previous_status=TaskStatus.PAUSED,
        )
        return True

    if phase and phase not in _WORKFLOW_BOUNDARY_PHASES:
        # Unknown/manual captures are not automatically made skippable.
        return False

    ensure_workflow_boundary_retry_context(
        state,
        error,
        previous_status="running",
        phase=phase or None,
    )
    return True


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
    """Clear the HITL gate while a newly queued restore owns the session."""
    state.snapshot_failure = None
    state.wait_input_reason = None
    state.status = TaskStatus.PAUSED


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


def consume_snapshot_skip(state: Any, phase: str) -> bool:
    """One-shot check-and-consume of a "skip snapshot & continue" marker.

    Returns True (and clears the marker) when the user armed a skip for *phase*
    (or the wildcard ``"*"``). A required workflow-boundary capture calls this at
    entry; on True it proceeds without capturing at that boundary.
    """
    marker = getattr(state, "skip_required_snapshot_phase", None)
    if not marker:
        return False
    if marker == "*" or marker == phase:
        state.skip_required_snapshot_phase = None
        return True
    return False


#: Session ids that currently hold a live "capture anyway" grant, i.e. a
#: Proceed the user pressed in *this* process whose capture has not run yet.
#:
#: The persisted ``force_snapshot_memory_override`` flag on AnalysisState records
#: only *that* the risk was accepted; this set records that the capture it was
#: accepted for is still pending, and it is deliberately not persisted. The flag
#: alone is force-written to session.json by the Proceed handler, and every path
#: on which the re-dispatched capture never runs — a 409 for a missing retry
#: context, SAFE_POINT_BUSY, the disk gate, or a workflow_boundary hand-off to a
#: WebSocket reconnect the user never makes — used to leave it armed on disk
#: forever. The next capture to come along, on a different and possibly much
#: larger AnnData, then skipped the preflight on a risk nobody had accepted for
#: it. An approval is for one capture in one running process; a flag found
#: without a matching grant is stale and is cleared rather than honoured.
_MEMORY_OVERRIDE_GRANTS: set[str] = set()


def _override_session_id(state: Any) -> str:
    return str(getattr(state, "id", "") or "")


def arm_memory_preflight_override(state: Any) -> None:
    """Accept the OOM risk so the next capture bypasses the memory preflight.

    Armed by the Proceed action on a memory gate; consumed by the next runtime
    capture (``runtime_parts.capture._consume_memory_preflight_override``). The
    caller is expected to follow this with the ordinary retry dispatch — this
    only lowers the preflight, it does not itself re-run the capture — and to
    call :func:`clear_memory_preflight_override` on every path where that
    dispatch does not reach a capture.
    """
    state.force_snapshot_memory_override = True
    session_id = _override_session_id(state)
    if session_id:
        _MEMORY_OVERRIDE_GRANTS.add(session_id)


def has_memory_preflight_grant(session_id: str) -> bool:
    """True while *session_id* holds an unconsumed "capture anyway" grant."""
    return bool(session_id) and session_id in _MEMORY_OVERRIDE_GRANTS


def clear_memory_preflight_override(state: Any) -> bool:
    """Disarm the "capture anyway" approval; True when something was armed.

    Called when the approved capture is consumed, when it demonstrably will not
    run, and when the user resolves the gate some other way. The caller persists
    *state* — an override left on disk is exactly the failure this closes.
    """
    session_id = _override_session_id(state)
    had_grant = session_id in _MEMORY_OVERRIDE_GRANTS
    _MEMORY_OVERRIDE_GRANTS.discard(session_id)
    was_armed = bool(getattr(state, "force_snapshot_memory_override", False))
    if was_armed:
        state.force_snapshot_memory_override = False
    return had_grant or was_armed


def skip_workflow_boundary(state: Any, phase: str | None) -> None:
    """Skip a workflow-boundary required snapshot and resume RUNNING.

    Arms a one-shot skip marker for *phase* so the graph resume re-hits the same
    boundary and proceeds without capturing, then continues the analysis.
    """
    state.snapshot_failure = None
    state.wait_input_reason = None
    state.skip_required_snapshot_phase = (phase or "*")
    state.status = TaskStatus.RUNNING


def skip_post_report(state: Any) -> None:
    """Skip the post-report snapshot: reports are already generated, mark done."""
    state.snapshot_failure = None
    state.wait_input_reason = None
    if hasattr(state, "pending_required_snapshot"):
        state.pending_required_snapshot = None
    state.status = TaskStatus.COMPLETED


def skip_chat(state: Any, previous_status: TaskStatus | str | None = None) -> None:
    """Skip the post-chat snapshot and restore *previous_status*."""
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


def skip_snapshot_producer(state: Any, context: SnapshotRetryContext) -> RetryOutcome:
    """Dispatch "Skip snapshot & continue" to the matching producer.

    workflow_boundary / post_report / chat are skippable. pre_restore is not:
    skipping there would proceed with a restore that overwrites live state
    without a safety point, so it returns an unsuccessful outcome and leaves the
    gate intact.
    """
    producer = context.producer
    if producer == "workflow_boundary":
        phase = (context.snapshot_request or {}).get("phase")
        skip_workflow_boundary(state, phase)
        return RetryOutcome(
            success=True,
            still_blocked=False,
            message="Snapshot skipped; workflow continuing without this checkpoint",
            status=_status_value(state),
            wait_input_reason=None,
            snapshot_failure=None,
        )
    if producer == "post_report":
        skip_post_report(state)
        return RetryOutcome(
            success=True,
            still_blocked=False,
            message="Post-report snapshot skipped; analysis marked complete",
            status=_status_value(state),
            wait_input_reason=None,
            snapshot_failure=None,
        )
    if producer == "chat":
        skip_chat(state, context.previous_status)
        return RetryOutcome(
            success=True,
            still_blocked=False,
            message="Post-chat snapshot skipped; conversation state retained in memory",
            status=_status_value(state),
            wait_input_reason=None,
            snapshot_failure=None,
        )
    # pre_restore (or any unknown producer): skip is unsafe. Leave gate intact.
    return RetryOutcome(
        success=False,
        still_blocked=True,
        message=(
            f"Skip is not supported for producer '{producer}'; "
            "the snapshot failure gate is left intact"
        ),
        status=_status_value(state),
        wait_input_reason=getattr(state, "wait_input_reason", None),
        snapshot_failure=getattr(state, "snapshot_failure", None),
    )


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
    # A Skip armed before the user changed their mind and terminated must not
    # outlive the run it was meant for. The marker is only consumed at a
    # *matching* boundary, so an unconsumed one lingers on the session and would
    # silently suppress the first required capture of some later run.
    if hasattr(state, "skip_required_snapshot_phase"):
        state.skip_required_snapshot_phase = None
    # Same rule for a "capture anyway" approval: the capture it was granted for
    # is being abandoned, so the approval goes with it.
    clear_memory_preflight_override(state)

SnapshotRetryDependencies.__module__ = "backend.services.snapshot_v2.failure_gate"

__all__ = [name for name in globals() if not name.startswith("__")]
