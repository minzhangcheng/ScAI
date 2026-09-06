"""Clear the blockers Snapshot V2 left behind, and refuse the one that is real.

Spec 4.6. Deleting Snapshot V2 removes the dialogs, routes and recovery that
answered its blockers, so any session persisted with ``snapshot_failure``,
``pending_required_snapshot``, ``skip_required_snapshot_phase``, a snapshot
``wait_input_reason`` or ``recovery_blocked`` would be permanently unstartable
with nothing able to clear it. This runs on session load, on import and on
re-run, and clears them.

It stops at one thing. A restore that never reached a terminal state leaves the
session's own files -- state, notebook, checkpoint, workspace -- describing
different points in the run. Clearing ``recovery_blocked`` there would let the
graph run against that mixture. Such a session must be repaired or rolled back
by the previous version before upgrading, and staying unstartable is the
correct outcome until then. The admin manuals say so; this refuses regardless.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.config import get_session_dir
from backend.models.schemas import TaskStatus

#: Journal kinds whose unfinished transaction really does leave the session's
#: own files at different points in the run. ``journals/`` was a single
#: directory shared by every kind, so status alone cannot answer this: a delete
#: or an import touched only the object store, and an export protection journal
#: was not a mutation at all. Only a restore swapped the four session files
#: against each other. Legacy flat journals carry no ``kind``, and one whose
#: kind cannot be read counts as a restore for the same fail-closed reason an
#: unparseable journal does.
_HALF_SWAP_JOURNAL_KINDS = frozenset({"", "restore", "restore_apply", "restore_job"})

_UNFINISHED = frozenset({"STAGED", "STAGING", "APPLYING"})

#: Where Snapshot V2 kept its journals, spelled here rather than imported: the
#: package is gone, and this predicate exists only to recognise what it left.
_LEGACY_JOURNAL_DIR = ("journals",)


def _legacy_journals(session_id: str) -> Path | None:
    if not session_id:
        return None
    return get_session_dir(session_id).joinpath(*_LEGACY_JOURNAL_DIR)


def has_unfinished_restore_journal(session_id: str) -> bool:
    """True while a STAGED/STAGING/APPLYING restore journal is still on disk.

    Anything unreadable counts as unfinished: a truncated write is itself
    evidence of an interrupted one.
    """
    journals = _legacy_journals(session_id)
    if journals is None:
        return False
    try:
        if not journals.is_dir():
            return False
        for entry in journals.iterdir():
            if not entry.is_file():
                continue
            try:
                doc = json.loads(entry.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return True
            if not isinstance(doc, dict):
                return True
            if str(doc.get("status", "")).upper() not in _UNFINISHED:
                continue
            kind = doc.get("kind")
            kind = "" if kind is None else str(kind).strip().lower()
            if kind in _HALF_SWAP_JOURNAL_KINDS:
                return True
    except OSError:
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


def sanitize_legacy_state(state: Any, *, workflow_active: bool = False) -> bool:
    """Clear retired snapshot blockers. Returns whether anything changed."""
    failure = getattr(state, "snapshot_failure", None)
    pending = getattr(state, "pending_required_snapshot", None)
    wait_reason = getattr(state, "wait_input_reason", None)
    transaction = getattr(state, "snapshot_transaction", None)

    failure_gate = (
        bool(failure)
        or bool(pending)
        or wait_reason in ("snapshot_failure", "runtime_persistence_failure")
    )
    recovery_gate = bool(getattr(state, "recovery_blocked", False)) or (
        isinstance(transaction, dict) and "recovery_error" in transaction
    )
    if not failure_gate and not recovery_gate:
        return False

    session_id = str(getattr(state, "id", "") or "")
    recovery_cleared = False
    if recovery_gate and not has_unfinished_restore_journal(session_id):
        from backend.services.session_store import clear_recovery_blocked

        recovery_cleared = clear_recovery_blocked(session_id, state)

    if not failure_gate:
        # A recovery block says nothing about where the run was, so the status
        # is left alone.
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
        next_status = _stable_status(previous_status, workflow_active=workflow_active)

    changed = recovery_cleared
    if failure is not None:
        setattr(state, "snapshot_failure", None)
        changed = True
    if wait_reason in ("snapshot_failure", "runtime_persistence_failure"):
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
