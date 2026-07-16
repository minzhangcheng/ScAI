"""Snapshot V2 typed errors and failure-gate contracts."""

from __future__ import annotations

from typing import Any


# Snapshot V2 failure HITL base actions. Retry re-enters capture only; manage
# opens snapshot management; terminate stops the workflow. Used for producers
# where skipping the capture is unsafe (e.g. pre_restore, whose snapshot is the
# only safety point before a restore overwrites live state).
SNAPSHOT_FAILURE_ACTIONS: tuple[str, ...] = ("retry", "manage", "terminate")

# Actions for producers where "Skip snapshot & continue" is safe — the analysis
# proceeds with reduced restorability at that one boundary
# (workflow_boundary / post_report / chat).
SNAPSHOT_FAILURE_ACTIONS_SKIPPABLE: tuple[str, ...] = (
    "retry", "skip", "manage", "terminate",
)


def is_snapshot_failure_blocked(state: Any) -> bool:
    """True when forward resume/start must refuse until dedicated HITL action.

    Blocks when ``wait_input_reason == "snapshot_failure"`` or a structured
    ``snapshot_failure`` payload is present. Only dedicated producer-aware
    retry/skip/manage/terminate handling may clear the gate — generic
    run/resume/stop-summarize must not.
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

    When available, ``details["phase"]`` carries the real capture phase so
    workflow-boundary retry can re-enter the same producer path.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict | None = None,
        snapshot_id: str | None = None,
        phase: str | None = None,
    ):
        merged = dict(details or {})
        if phase and not merged.get("phase"):
            merged["phase"] = phase
        super().__init__(code, message, details=merged)
        self.snapshot_id = snapshot_id
        self.phase = merged.get("phase")


def _snapshot_error_status(exc: SnapshotV2Error) -> int:
    """Map a SnapshotV2Error type/code to an HTTP status code.

    The canonical mapping is:

    ==================================  ====
    Exception type / code               Code
    ==================================  ====
    MANIFEST_NOT_FOUND /               404
      SNAPSHOT_NOT_FOUND
    V1 restore (no SnapshotV2Error)    410  (caller must handle separately)
    SnapshotCompatibilityError         422
    SnapshotIntegrityError (other)     422
    SnapshotDeleteConflict             409
    SnapshotTransactionError           503
    SnapshotStorageError               507
    ExportIntegrityError               422  (bad request / unknown snapshots)
    Any other SnapshotV2Error          500
    ==================================  ====

    The order of checks matters: more specific (subclass) checks must come
    before the generic ``SnapshotV2Error`` fallback.
    """
    if isinstance(exc, SnapshotIntegrityError):
        if exc.code in ("MANIFEST_NOT_FOUND", "SNAPSHOT_NOT_FOUND"):
            return 404
        return 422
    if isinstance(exc, SnapshotCompatibilityError):
        return 422
    if isinstance(exc, SnapshotStorageError):
        return 507
    if isinstance(exc, SnapshotDeleteConflict):
        return 409
    if isinstance(exc, SnapshotTransactionError):
        if exc.code == "SAFE_POINT_BUSY":
            return 409
        return 503
    # ExportIntegrityError lives in exporter.py but inherits SnapshotV2Error
    if exc.__class__.__name__ == "ExportIntegrityError":
        return 422
    return 500


def snapshot_error_detail(exc: SnapshotV2Error, context: dict | None = None) -> dict:
    status_code = _snapshot_error_status(exc)
    result = {
        "status_code": status_code,
        "code": getattr(exc, "code", "UNKNOWN"),
        "message": str(exc),
    }
    if getattr(exc, "details", None):
        result["details"] = dict(exc.details)
    for key in ("runtime_status",):
        if context and context.get(key) is not None:
            result[key] = context[key]
    return result
