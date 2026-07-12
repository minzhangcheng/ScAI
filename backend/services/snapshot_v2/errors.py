"""Snapshot V2 typed errors and failure-gate contracts."""

from __future__ import annotations

from typing import Any


# Snapshot V2 failure HITL: never Skip. Retry re-enters capture only;
# manage opens snapshot management; terminate stops the workflow.
SNAPSHOT_FAILURE_ACTIONS: tuple[str, ...] = ("retry", "manage", "terminate")


def is_snapshot_failure_blocked(state: Any) -> bool:
    """True when forward resume/start must refuse until dedicated HITL action.

    Blocks when ``wait_input_reason == "snapshot_failure"`` or a structured
    ``snapshot_failure`` payload is present. Only retry/manage/terminate may
    clear the gate — generic run/resume/stop-summarize must not.
    """
    if state is None:
        return False
    reason = getattr(state, "wait_input_reason", None)
    if reason == "snapshot_failure":
        return True
    if getattr(state, "snapshot_failure", None):
        return True
    return False


class SnapshotV2Error(Exception):
    def __init__(self, code: str, message: str, *, details: dict | None = None):
        self.code = code
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if self.code and self.code not in base:
            return f"{self.code}: {base}"
        return base


class SnapshotCaptureError(SnapshotV2Error):
    pass


class SnapshotStorageError(SnapshotV2Error):
    pass


class SnapshotIntegrityError(SnapshotV2Error):
    pass


class SnapshotCompatibilityError(SnapshotV2Error):
    pass


class SnapshotTransactionError(SnapshotV2Error):
    pass


class SnapshotDeleteConflict(SnapshotV2Error):
    pass


class RequiredSnapshotBlocked(SnapshotV2Error):
    """Fail-closed signal: a required snapshot capture failed.

    Producers raise this after persisting WAITING_INPUT / snapshot_failure
    so the workflow stops forward mutation. Caught once at the workflow
    run/resume boundary — never converted into ordinary node retries.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict | None = None,
        snapshot_id: str | None = None,
    ):
        super().__init__(code, message, details=details)
        self.snapshot_id = snapshot_id
