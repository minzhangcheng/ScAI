"""Read the analysis progress belonging to a captured graph checkpoint."""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from contextlib import closing

from backend.models.schemas import AnalysisState
from backend.utils.checkpoint import DurableJsonPlusSerializer

logger = logging.getLogger(__name__)

# These are session controls, not analysis progress. A rollback must not restore
# old credentials or undo settings the user changed while the run was stopped.
_CURRENT_CONTROLS = (
    "name", "lifecycle_generation", "state_version", "last_sync_timestamp",
    "max_rounds", "original_max_rounds", "round_wait_seconds",
    "max_directions_per_gate", "plan_review_wait_seconds", "auto_approve_plans",
    "max_integration_cycles", "custom_prompts", "custom_models",
    "custom_temperatures", "agent_server_configs", "agent_vision_settings",
    "literature_config",
)


def _read_pair_state(current: AnalysisState, revision: int) -> AnalysisState | None:
    from backend.config import get_session_states
    from backend.utils.sandbox_parts.regular_save import GENERATION_PAIR_DIRNAME

    pair = get_session_states(current.id) / GENERATION_PAIR_DIRNAME / str(revision)
    path = pair / "checkpoints.sqlite"
    if not (path.is_file() and (pair / "state.json").is_file()):
        return None
    try:
        # Read the same root checkpoint selected by workflow.resume(), without
        # opening a workflow or allowing SQLite setup to modify the saved pair.
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            row = conn.execute(
                "SELECT type, checkpoint FROM checkpoints WHERE thread_id=? "
                "AND checkpoint_ns='' ORDER BY checkpoint_id DESC LIMIT 1",
                (current.id,),
            ).fetchone()
        if row is None:
            return None
        serde = DurableJsonPlusSerializer(allowed_json_modules=[
            ("backend.models.schemas", name)
            for name in ("AgentType", "AnalysisState", "AnalysisPhase",
                         "RoundResult", "TaskStatus", "MessageRole")
        ])
        checkpoint = serde.loads_typed(row)
        restored = checkpoint["channel_values"]["analysis_state"]
        if not isinstance(restored, AnalysisState):
            restored = AnalysisState.model_validate(restored)
        if restored.id != current.id or restored.user_id != current.user_id:
            raise ValueError("paired analysis identity does not match the session")
        if restored.lifecycle_generation not in (0, current.lifecycle_generation):
            raise ValueError("paired analysis belongs to a different lifecycle")
        restored = restored.model_copy(deep=True)
        controls = current.model_copy(deep=True)
        for name in _CURRENT_CONTROLS:
            setattr(restored, name, getattr(controls, name))
        # Until the pair's final declaration is published, retain the stale
        # projection so interruption before publication still requires recovery.
        restored.runtime_persistence = dict(current.runtime_persistence)
        return restored
    except Exception:
        logger.warning("[%s] Unusable paired analysis state", current.id, exc_info=True)
        return None


async def read_generation_pair_state(
    current: AnalysisState, *, revision: int,
) -> AnalysisState | None:
    return await asyncio.to_thread(_read_pair_state, current, revision)
