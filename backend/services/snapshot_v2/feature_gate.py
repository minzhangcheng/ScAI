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


def _has_unfinished_restore_journal(session_id: str) -> bool:
    """True while a STAGED/STAGING/APPLYING journal describes a half-swapped tree.

    Mirrors the startup scan in ``backend/main.py``. Only a restore that never
    reached a terminal state leaves the session's files mixing pre- and
    post-restore content; COMMITTED and FAILED journals are merely waiting to be
    reaped. Anything unreadable counts as unfinished, because a truncated write
    is itself evidence of an interrupted one.
    """
    import json

    from backend.services.snapshot_v2.paths import SnapshotV2Paths

    if not session_id:
        return False
    unfinished = {"STAGED", "STAGING", "APPLYING"}
    try:
        journals = SnapshotV2Paths.for_session(session_id).journals
        if not journals.is_dir():
            return False
        for entry in journals.iterdir():
            if not entry.is_file():
                continue
            try:
                doc = json.loads(entry.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return True
            if str(doc.get("status", "")).upper() in unfinished:
                return True
    except OSError:
        # Unknown means fail closed: the flag exists to stop a half-applied
        # restore from being driven forward.
        return True
    return False


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

    Restore transaction metadata is preserved for audit, with one exception:
    the ``recovery_error`` key is a blocker rather than a record, so it goes
    with the flag it belongs to (and takes the whole ``snapshot_transaction``
    with it when it was the only key left). A status that claimed to be running
    is downgraded to paused when no live workflow owns the session, so a
    persisted failure cannot become a phantom run.

    ``recovery_blocked`` is cleared here too, and independently of the failure
    gate above: it is set by journal recovery, which only ever runs while the
    subsystem is enabled, and ``assert_recovery_clear`` refuses every start,
    resume, WebSocket attach and kernel launch while it is set. With V2 off
    nothing can ever clear it again and no snapshot endpoint is reachable to try
    — so a session that was merely recoverable before the operator turned the
    subsystem off would become permanently unstartable, as would any session
    imported from an archive that carried the flag.

    That unblocking stops at evidence of an actual half-applied restore: when an
    unfinished journal is still on disk the flag stays, because clearing it would
    let the graph run against a session whose state, notebook, checkpoint and
    workspace describe different points in the run. Such a session needs
    ``SCAI_ENABLE_SNAPSHOT_V2`` back on so recovery can finish or roll back —
    being unstartable is the correct outcome until then.
    """
    failure = getattr(state, "snapshot_failure", None)
    pending = getattr(state, "pending_required_snapshot", None)
    wait_reason = getattr(state, "wait_input_reason", None)
    transaction = getattr(state, "snapshot_transaction", None)

    failure_gate = bool(failure) or bool(pending) or wait_reason == "snapshot_failure"
    recovery_gate = bool(getattr(state, "recovery_blocked", False)) or (
        isinstance(transaction, dict) and "recovery_error" in transaction
    )
    if not failure_gate and not recovery_gate:
        return False

    recovery_cleared = False
    if recovery_gate and not _has_unfinished_restore_journal(
        str(getattr(state, "id", "") or "")
    ):
        from backend.services.session_store import clear_recovery_blocked

        recovery_cleared = clear_recovery_blocked(
            str(getattr(state, "id", "") or ""), state
        )

    if not failure_gate:
        # Nothing else to reconcile — in particular the status must be left
        # alone, since a recovery block says nothing about where the run was.
        return recovery_cleared

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

    changed = recovery_cleared
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
