"""Snapshot-failure serialization helpers."""

from __future__ import annotations

import re
from typing import Any


_RUNTIME_GATE_TOKEN_RE = re.compile(r"^runtime-(\d+)-[0-9a-f]{32}$")


def sanitize_snapshot_failure(payload: dict[str, Any]) -> Any:
    """Sanitize diagnostics without destroying a runtime-gate nonce.

    ``gate_token`` is normally a credential-shaped key, but the runtime
    persistence failure payload uses it as a public, single-generation nonce.
    Keep only the strictly generated nonce already present in the independent
    ``runtime_persistence`` record.  This also repairs states written by
    0.38.10 where the diagnostic copy was persisted as ``<redacted-secret>``.
    """
    from backend.utils.error_parsing import sanitize_error_payload

    failure = sanitize_error_payload(payload.get("snapshot_failure"), max_chars=3000)
    if not isinstance(failure, dict):
        return failure
    if not str(failure.get("code") or "").startswith("runtime_persistence_"):
        return failure

    persistence = payload.get("runtime_persistence")
    if not isinstance(persistence, dict):
        return failure
    token = str(persistence.get("gate_token") or "")
    match = _RUNTIME_GATE_TOKEN_RE.fullmatch(token)
    generation = int(persistence.get("gate_generation") or 0)
    if match is None or int(match.group(1)) != generation:
        return failure

    details = dict(failure.get("details") or {})
    details["gate_token"] = token
    details["gate_generation"] = generation
    retry_context = dict(failure.get("retry_context") or {})
    retry_context["gate_token"] = token
    retry_context["generation"] = generation
    failure = dict(failure)
    failure["details"] = details
    failure["retry_context"] = retry_context
    return failure
