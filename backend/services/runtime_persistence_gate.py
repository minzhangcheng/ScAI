"""Runtime-persistence failure and degraded-recovery HITL contracts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from backend.constants import WaitInputReason
from backend.models.schemas import TaskStatus


RUNTIME_PERSISTENCE_FAILURE_CODE = "runtime_persistence_failure"
RUNTIME_PERSISTENCE_DEGRADED_CODE = "runtime_persistence_degraded_recovery"
RUNTIME_PERSISTENCE_ACTIONS = ("retry", "manage", "terminate")


class RuntimePersistenceBlocked(RuntimeError):
    """Stop forward mutation while the runtime persistence gate is armed."""


class RuntimePersistenceShutdownBlocked(RuntimePersistenceBlocked):
    """Keep a live kernel when its final persistence attempt did not commit."""


class RuntimePersistenceActionOutcome(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE_OR_CONFLICT = "duplicate_or_conflict"
    STALE = "stale"
    INACTIVE = "inactive"


@dataclass(frozen=True)
class RuntimePersistenceActionResult:
    outcome: RuntimePersistenceActionOutcome
    action: str
    gate_token: str = ""
    generation: int = 0

    @property
    def accepted(self) -> bool:
        return self.outcome is RuntimePersistenceActionOutcome.ACCEPTED


def _safe_text(value: Any, limit: int = 128) -> str:
    """Redact credentials/private paths before applying a consumer limit."""
    from backend.utils.error_parsing import compact_traceback

    rendered = compact_traceback(str(value or ""), max_chars=max(512, limit * 4))
    # Keep the shared redaction but discard its traceback-layout marker for
    # ordinary one-line payload fields such as resource names and gate tokens.
    rendered = rendered.replace("[... traceback frames omitted ...]", "")
    return " ".join(rendered.split())[:limit]


def _safe_name(value: Any) -> str:
    return _safe_text(value, 128)


def _safe_number(value: Any) -> int | float | str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return _safe_text(value, 128)
    return None


def _safe_budget_violations(raw: Any) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return findings
    for item in raw[:20]:
        if not isinstance(item, dict):
            continue
        resource = _safe_name(item.get("resource"))
        observed = _safe_number(item.get("observed"))
        limit = _safe_number(item.get("limit"))
        if not resource:
            continue
        findings.append({
            "resource": resource,
            "observed": observed,
            "limit": limit,
        })
    return findings


def _safe_top_offenders(raw: Any) -> list[dict[str, Any]]:
    offenders: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return offenders
    for item in raw[:12]:
        if not isinstance(item, dict):
            continue
        name = _safe_name(item.get("name"))
        type_name = _safe_name(item.get("type") or item.get("type_name"))
        size = _safe_number(
            item.get("bytes")
            if item.get("bytes") is not None
            else item.get("estimated_bytes")
        )
        if not name:
            continue
        safe = {"name": name, "type": type_name}
        if isinstance(size, (int, float)):
            safe["bytes"] = max(0, size)
        offenders.append(safe)
    return offenders


def is_runtime_persistence_blocked(state: Any) -> bool:
    if state is None:
        return False
    if getattr(state, "wait_input_reason", None) == WaitInputReason.RUNTIME_PERSISTENCE_FAILURE.value:
        return True
    failure = getattr(state, "snapshot_failure", None)
    return bool(
        isinstance(failure, dict)
        and str(failure.get("code") or "").startswith("runtime_persistence_")
    )


def build_runtime_persistence_failure(
    session_id: str,
    persistence: dict[str, Any] | None,
    *,
    degraded_recovery: bool | None = None,
) -> dict[str, Any]:
    """Build a reconnect-safe payload without exposing values or tracebacks."""
    data = dict(persistence or {})
    if degraded_recovery is None:
        degraded_recovery = bool(data.get("degraded_recovery"))
    code = (
        RUNTIME_PERSISTENCE_DEGRADED_CODE
        if degraded_recovery
        else RUNTIME_PERSISTENCE_FAILURE_CODE
    )
    committed_revision = int(data.get("committed_revision") or 0)
    attempted_revision = int(
        data.get("last_attempt_revision")
        or data.get("current_revision")
        or committed_revision
    )
    if degraded_recovery:
        message = (
            "The last verified runtime generation was restored, but a newer "
            "persistence attempt failed. Work after the committed revision may "
            "need to be recomputed before analysis can continue."
        )
    else:
        message = (
            "The live analysis runtime could not be persisted safely. Forward "
            "execution is paused so the last verified generation is not replaced "
            "by incomplete state."
        )
    generation = max(1, int(data.get("gate_generation") or 1))
    gate_token = _safe_text(data.get("gate_token"), 160)
    if not gate_token:
        gate_token = f"runtime-{generation}-{uuid4().hex}"
    details = {
        "degraded_recovery": bool(degraded_recovery),
        "committed_revision": committed_revision,
        "attempted_revision": attempted_revision,
        "budget_violations": _safe_budget_violations(
            data.get("budget_violations")
        ),
        "top_offenders": _safe_top_offenders(data.get("top_offenders")),
        "failed_variables": [
            _safe_name(name) for name in list(data.get("failed_variables") or [])[:20]
            if _safe_name(name)
        ],
        "recompute_variables": [
            _safe_name(name) for name in list(data.get("recompute_variables") or [])[:20]
            if _safe_name(name)
        ],
        "recommended_action": _safe_name(
            data.get("recommended_action") or "recompute_or_reduce_runtime"
        ),
        "gate_token": gate_token,
        "gate_generation": generation,
        "action_pending": bool(data.get("gate_action_pending")),
    }
    error_summary = _safe_text(data.get("error_summary"), 500)
    if error_summary:
        details["error_summary"] = error_summary
    return {
        "snapshot_id": "",
        "session_id": _safe_name(session_id),
        "failed_at": datetime.now().isoformat(),
        "code": code,
        "message": message,
        "details": details,
        "allowed_actions": list(RUNTIME_PERSISTENCE_ACTIONS),
        "retry_context": {
            "producer": "runtime_persistence",
            "gate_token": gate_token,
            "generation": generation,
        },
    }


def enter_runtime_persistence_gate(
    state: Any,
    persistence: dict[str, Any] | None = None,
    *,
    degraded_recovery: bool | None = None,
    rotate_generation: bool = False,
) -> dict[str, Any]:
    data = dict(persistence or getattr(state, "runtime_persistence", None) or {})
    existing_failure = getattr(state, "snapshot_failure", None)
    existing_details = (
        existing_failure.get("details")
        if isinstance(existing_failure, dict)
        and str(existing_failure.get("code") or "").startswith("runtime_persistence_")
        else None
    )
    existing_details = existing_details if isinstance(existing_details, dict) else {}
    current_generation = int(
        data.get("gate_generation")
        or existing_details.get("gate_generation")
        or 0
    )
    current_token = str(
        data.get("gate_token")
        or existing_details.get("gate_token")
        or ""
    )
    if rotate_generation or current_generation <= 0 or not current_token:
        current_generation = max(0, current_generation) + 1
        current_token = f"runtime-{current_generation}-{uuid4().hex}"
        data["gate_action_pending"] = False
        data["gate_action"] = None
    data["gate_generation"] = current_generation
    data["gate_token"] = current_token
    state.runtime_persistence = data
    failure = build_runtime_persistence_failure(
        getattr(state, "id", ""),
        data,
        degraded_recovery=degraded_recovery,
    )
    state.snapshot_failure = failure
    state.status = TaskStatus.WAITING_INPUT
    state.wait_input_reason = WaitInputReason.RUNTIME_PERSISTENCE_FAILURE.value
    return failure


def runtime_persistence_gate_identity(state: Any) -> tuple[str, int]:
    failure = getattr(state, "snapshot_failure", None)
    details = failure.get("details") if isinstance(failure, dict) else None
    details = details if isinstance(details, dict) else {}
    persistence = dict(getattr(state, "runtime_persistence", None) or {})
    return (
        str(details.get("gate_token") or persistence.get("gate_token") or ""),
        int(details.get("gate_generation") or persistence.get("gate_generation") or 0),
    )


def validate_runtime_persistence_action(
    state: Any,
    action: str,
    *,
    gate_token: str | None,
    generation: int | None,
    active: bool,
) -> RuntimePersistenceActionResult:
    """Atomically classifies an action against the durable gate identity.

    The caller must hold the per-session mutation lock. This same classifier is
    used before both live delivery and no-workflow fallback, so neither path can
    reinterpret a stale UI request as a fresh action.
    """
    token, current_generation = runtime_persistence_gate_identity(state)
    if not is_runtime_persistence_blocked(state):
        return RuntimePersistenceActionResult(
            RuntimePersistenceActionOutcome.INACTIVE,
            action,
            token,
            current_generation,
        )
    if (
        not gate_token
        or generation is None
        or str(gate_token) != token
        or int(generation) != current_generation
    ):
        return RuntimePersistenceActionResult(
            RuntimePersistenceActionOutcome.STALE,
            action,
            token,
            current_generation,
        )
    persistence = dict(getattr(state, "runtime_persistence", None) or {})
    if persistence.get("gate_action_pending"):
        return RuntimePersistenceActionResult(
            RuntimePersistenceActionOutcome.DUPLICATE_OR_CONFLICT,
            action,
            token,
            current_generation,
        )
    if not active:
        return RuntimePersistenceActionResult(
            RuntimePersistenceActionOutcome.INACTIVE,
            action,
            token,
            current_generation,
        )
    persistence["gate_action_pending"] = True
    persistence["gate_action"] = action
    state.runtime_persistence = persistence
    failure = getattr(state, "snapshot_failure", None)
    if isinstance(failure, dict):
        details = dict(failure.get("details") or {})
        details["action_pending"] = True
        failure = dict(failure)
        failure["details"] = details
        state.snapshot_failure = failure
    return RuntimePersistenceActionResult(
        RuntimePersistenceActionOutcome.ACCEPTED,
        action,
        token,
        current_generation,
    )


def clear_runtime_persistence_gate(state: Any) -> None:
    if is_runtime_persistence_blocked(state):
        state.snapshot_failure = None
        state.wait_input_reason = None
    persistence = dict(getattr(state, "runtime_persistence", None) or {})
    persistence["gate_action_pending"] = False
    persistence["gate_action"] = None
    state.runtime_persistence = persistence


def prepare_runtime_recompute(state: Any, sandbox: Any | None = None) -> None:
    """Release the gate for an explicit recompute/reduce attempt."""
    persistence = dict(getattr(state, "runtime_persistence", None) or {})
    persistence.update({
        "requires_recompute": False,
        "recommended_action": "recompute_or_reduce_runtime",
        "recompute_authorized": True,
    })
    notebook = getattr(sandbox, "notebook_state", None)
    if notebook is not None:
        notebook.runtime_persistence.update(persistence)
        notebook.save_metadata()
        persistence = dict(notebook.runtime_persistence)
    state.runtime_persistence = persistence
    state.runtime_recompute_pending = True
    clear_runtime_persistence_gate(state)
    state.status = TaskStatus.RUNNING
