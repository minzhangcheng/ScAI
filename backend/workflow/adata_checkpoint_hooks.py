"""Workflow-side ordering for immutable AnnData checkpoint settlement.

The capture service owns serialization, blob publication and catalog records.
This module owns the lock/state/runtime-save ordering that must not interleave
with another kernel execution.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from backend.services.adata_checkpoints import CaptureOutcome, capture_boundary_locked

logger = logging.getLogger(__name__)


def clear_settlement_scope_callback(analysis_state: Any, scope_id: Optional[str]):
    """Return a bounded callback that clears one completed attempt scope.

    The callback only removes the exact scope named by the capture event; it is
    invoked by the shared helper after capture and before the state save, so the
    terminal checkpoint and the removed accumulator become durable together.
    """
    target = str(scope_id or "").strip()

    def _clear(outcome: CaptureOutcome) -> bool:
        if not target:
            return False
        # Terminal settlement is independent of capture availability: an
        # available, failed, unavailable, or no-canonical-adata outcome all
        # close the same boundary attempt scope.
        try:
            mapping = dict(
                getattr(analysis_state, "bioinformatics_settlement_attempt_ids", None)
                or {}
            )
        except Exception:
            return False
        if target not in mapping:
            return False
        mapping.pop(target, None)
        analysis_state.bioinformatics_settlement_attempt_ids = mapping
        return True

    return _clear


async def capture_and_persist_boundary(
    workflow: Any,
    analysis_state: Any,
    *,
    boundary: str,
    event_scope: str = "",
    round_number: int = 0,
    planning_stage: str = "full",
    phase: str = "",
    source_plan_id: Optional[str] = None,
    round_result: Any = None,
    direction: str = "",
    research_label: str = "",
    summary_at_capture: str = "",
    outcome_at_capture: str = "",
    content_hash: str = "",
    source_preparation_ids_at_capture: Any = (),
    active_source_preparation_id: Optional[str] = None,
    source_scope_id_at_capture: Optional[str] = None,
    executor_settlement_scope_id_at_capture: Optional[str] = None,
    supplementary_request_id: Optional[str] = None,
    execution_ids_at_capture: Any = (),
    attempt_execution_ids_at_capture: Any = (),
    before_state_save: Optional[Callable[[CaptureOutcome], bool]] = None,
    save_runtime: bool = True,
    executor_checkpoint_name: Optional[str] = None,
    executor_checkpoint_names: Any = (),
) -> CaptureOutcome:
    sandbox = getattr(workflow, "sandbox", None)
    lock = getattr(sandbox, "_exec_lock", None)
    if lock is None:
        outcome = await capture_boundary_locked(
            sandbox,
            analysis_state,
            boundary=boundary,
            event_scope=event_scope,
            round_number=round_number,
            planning_stage=planning_stage,
            phase=phase,
            source_plan_id=source_plan_id,
            round_result=round_result,
            direction=direction,
            research_label=research_label,
            summary_at_capture=summary_at_capture,
            outcome_at_capture=outcome_at_capture,
            content_hash=content_hash,
            source_preparation_ids_at_capture=source_preparation_ids_at_capture,
            active_source_preparation_id=active_source_preparation_id,
            source_scope_id_at_capture=source_scope_id_at_capture,
            executor_settlement_scope_id_at_capture=(
                executor_settlement_scope_id_at_capture
            ),
            supplementary_request_id=supplementary_request_id,
            execution_ids_at_capture=execution_ids_at_capture,
            attempt_execution_ids_at_capture=attempt_execution_ids_at_capture,
        )
    else:
        async with lock:
            outcome = await capture_boundary_locked(
                sandbox,
                analysis_state,
                boundary=boundary,
                event_scope=event_scope,
                round_number=round_number,
                planning_stage=planning_stage,
                phase=phase,
                source_plan_id=source_plan_id,
                round_result=round_result,
                direction=direction,
                research_label=research_label,
                summary_at_capture=summary_at_capture,
                outcome_at_capture=outcome_at_capture,
                content_hash=content_hash,
                source_preparation_ids_at_capture=source_preparation_ids_at_capture,
                active_source_preparation_id=active_source_preparation_id,
                source_scope_id_at_capture=source_scope_id_at_capture,
                executor_settlement_scope_id_at_capture=(
                    executor_settlement_scope_id_at_capture
                ),
                supplementary_request_id=supplementary_request_id,
                execution_ids_at_capture=execution_ids_at_capture,
                attempt_execution_ids_at_capture=attempt_execution_ids_at_capture,
            )

    # Capture above is the only kernel operation before Host-state settlement.
    # The remaining mutations apply equally to sandboxes without an explicit
    # execution lock (test/fallback implementations).
    callback_changed = False
    if before_state_save is not None:
        try:
            callback_changed = bool(before_state_save(outcome))
        except Exception:
            logger.exception("checkpoint before_state_save callback failed")

    if (
        outcome.record is not None
        and round_result is not None
        and outcome.record.status == "available"
    ):
        try:
            round_result.canonical_adata_output_checkpoint_id = (
                outcome.record.checkpoint_id
            )
        except Exception:
            logger.warning("could not attach canonical checkpoint id to round result")

    state_changed = outcome.record is not None or callback_changed
    state_saved = False
    if state_changed:
        try:
            await analysis_state.increment_version()
            from backend.services.session_store import async_save_session

            await async_save_session(
                str(getattr(analysis_state, "id", "")),
                analysis_state,
                force=True,
            )
            state_saved = True
        except Exception:
            logger.exception(
                "AnnData checkpoint state persistence failed; keeping the "
                "in-memory record and previous durable state"
            )

    if (
        state_saved
        and save_runtime
        and sandbox is not None
        and getattr(sandbox, "km", None) is not None
    ):
        try:
            if lock is None:
                await sandbox._save_session_locked()
            else:
                async with lock:
                    await sandbox._save_session_locked()
            sandbox.sync_runtime_persistence_state(analysis_state)
        except Exception:
            logger.exception(
                "AnnData checkpoint paired runtime save failed; keeping the "
                "previous committed runtime generation"
            )
    checkpoint_names = list(executor_checkpoint_names or [])
    if executor_checkpoint_name:
        checkpoint_names.append(executor_checkpoint_name)
    if state_saved and checkpoint_names:
        try:
            from backend.agents.bioinformatics.executor_checkpoint import (
                is_executor_checkpoint_name,
            )
            from pathlib import Path

            state_dir = Path(getattr(sandbox, "state_dir"))
            for raw_name in dict.fromkeys(str(name) for name in checkpoint_names):
                checkpoint_name = Path(raw_name).name
                if checkpoint_name == raw_name and is_executor_checkpoint_name(
                    checkpoint_name,
                    session_id=str(getattr(analysis_state, "id", "")),
                ):
                    (state_dir / checkpoint_name).unlink(missing_ok=True)
        except Exception:
            logger.warning(
                "Could not retire settled executor checkpoints %s",
                checkpoint_names,
                exc_info=True,
            )
    return outcome
