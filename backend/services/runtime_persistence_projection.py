"""Project the authoritative notebook lifecycle into session-facing state."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from backend.models.schemas import AnalysisState

logger = logging.getLogger(__name__)


def project_runtime_persistence(
    state_path: Path,
    session_id: str,
    state: AnalysisState,
) -> Optional[tuple[Dict[str, Any], Dict[str, Any]]]:
    """Read and validate one authoritative NotebookState projection."""
    if not state_path.is_file():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read runtime persistence authority for %s", session_id, exc_info=True)
        return None
    if str(payload.get("session_id") or "") != session_id:
        logger.error("Runtime persistence authority identity mismatch for %s", session_id)
        return None
    raw = payload.get("runtime_persistence")
    if not isinstance(raw, dict):
        return None

    authoritative = dict(raw)
    sidecar_generation = int(authoritative.get("lifecycle_generation") or 0)
    state_generation = int(getattr(state, "lifecycle_generation", 0) or 0)
    if sidecar_generation and state_generation and sidecar_generation != state_generation:
        projected = dict(getattr(state, "runtime_persistence", None) or {})
        error = (
            f"NotebookState lifecycle generation {sidecar_generation} does "
            f"not match AnalysisState generation {state_generation}."
        )
        projected.update({
            "status": "failed",
            "last_attempt_status": "failed",
            "requires_recompute": True,
            "recommended_action": "restore_matching_lifecycle",
            "error_summary": error,
            "projection_status": "lifecycle_mismatch",
            "projection_error": error,
            "observed_at": datetime.now().isoformat(),
        })
        sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
        sandbox_state["runtime_persistence"] = projected
        return projected, sandbox_state

    try:
        current_revision = max(0, int(authoritative.get("current_revision") or 0))
        saved_revision = max(0, int(authoritative.get("saved_revision") or 0))
    except (TypeError, ValueError):
        logger.error("Invalid runtime persistence revisions for %s", session_id)
        return None
    if saved_revision > current_revision:
        logger.error(
            "Runtime persistence authority for %s has saved revision %d after current revision %d",
            session_id, saved_revision, current_revision,
        )
        return None

    authoritative.update({
        "source": "notebook_state",
        "current_revision": current_revision,
        "saved_revision": saved_revision,
        "observed_at": datetime.now().isoformat(),
        "projection_status": "current",
        "projection_error": None,
    })
    sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
    sandbox_state.update({
        "runtime_revision": current_revision,
        "saved_runtime_revision": saved_revision,
        "runtime_persistence": authoritative,
    })
    return authoritative, sandbox_state


def apply_runtime_persistence_projection(
    state_path: Path,
    session_id: str,
    state: AnalysisState,
) -> Dict[str, Any]:
    """Refresh the mutable projection without creating another authority."""
    projection = project_runtime_persistence(state_path, session_id, state)
    if projection is None:
        return dict(getattr(state, "runtime_persistence", None) or {})
    state.runtime_persistence, state.sandbox_state = projection
    if state.runtime_persistence.get("projection_status") == "lifecycle_mismatch":
        # Recorded and disclosed, not held. The record travels to the UI as a
        # persistent non-blocking warning (spec 5.1); a mismatch belongs to
        # another session incarnation and there is nothing a dialog could do
        # about it.
        sandbox_state = dict(state.sandbox_state or {})
        sandbox_state["runtime_persistence"] = dict(state.runtime_persistence)
        state.sandbox_state = sandbox_state
    return dict(state.runtime_persistence)
