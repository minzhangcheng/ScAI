"""Central operator feature gate for Snapshot V2 HTTP operations."""

from fastapi import HTTPException

from backend.config import is_snapshot_v2_enabled
from backend.models.schemas import TaskStatus


def require_snapshot_v2_enabled() -> None:
    """Reject Snapshot V2 HTTP operations when the subsystem is disabled."""
    if not is_snapshot_v2_enabled():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "SNAPSHOT_V2_DISABLED",
                "message": "Snapshot V2 is disabled",
            },
        )


def _stable_status(value: object, *, workflow_active: bool) -> TaskStatus:
    try:
        status = value if isinstance(value, TaskStatus) else TaskStatus(str(value))
    except (TypeError, ValueError):
        return TaskStatus.PAUSED
    if status == TaskStatus.RUNNING and not workflow_active:
        return TaskStatus.PAUSED
    if status == TaskStatus.WAITING_INPUT:
        return TaskStatus.PAUSED
    return status


def normalize_disabled_snapshot_state(
    state: object,
    *,
    workflow_active: bool = False,
) -> bool:
    """Remove stale Snapshot V2-only blockers without starting snapshot work.

    Restore transaction metadata is intentionally preserved for audit. A status
    that claimed to be running is downgraded to paused when no live workflow
    owns the session, so a persisted failure cannot become a phantom run.
    """
    failure = getattr(state, "snapshot_failure", None)
    pending = getattr(state, "pending_required_snapshot", None)
    wait_reason = getattr(state, "wait_input_reason", None)
    if not failure and not pending and wait_reason != "snapshot_failure":
        return False

    retry_context = failure.get("retry_context", {}) if isinstance(failure, dict) else {}
    if not isinstance(retry_context, dict):
        retry_context = {}
    producer = str(retry_context.get("producer") or "")
    if not producer and isinstance(pending, dict):
        if pending.get("phase") == "post_report":
            producer = "post_report"

    previous_status = retry_context.get("previous_status")
    if producer == "post_report":
        next_status = TaskStatus.COMPLETED
    elif producer == "pre_restore":
        next_status = TaskStatus.PAUSED
    else:
        next_status = _stable_status(
            previous_status,
            workflow_active=workflow_active,
        )

    changed = False
    if failure is not None:
        setattr(state, "snapshot_failure", None)
        changed = True
    if wait_reason == "snapshot_failure":
        setattr(state, "wait_input_reason", None)
        changed = True
    if pending is not None:
        setattr(state, "pending_required_snapshot", None)
        changed = True
    if getattr(state, "skip_required_snapshot_phase", None) is not None:
        setattr(state, "skip_required_snapshot_phase", None)
        changed = True
    if getattr(state, "status", None) != next_status:
        setattr(state, "status", next_status)
        changed = True
    return changed
