"""Executor checkpoint validation and payload construction."""
from __future__ import annotations

from typing import Any


def validate_checkpoint(
    checkpoint: dict,
    *,
    current_round: int,
    planning_stage: str,
) -> list[str]:
    errors: list[str] = []
    step_number = checkpoint.get("step_number")
    if not isinstance(step_number, int) or step_number < 0:
        errors.append(f"step_number invalid: {step_number}")
    messages = checkpoint.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append("messages invalid: expected list with >= 2 items")
    if not isinstance(checkpoint.get("findings"), list):
        errors.append("findings invalid: expected list")
    if not isinstance(checkpoint.get("figures"), list):
        errors.append("figures invalid: expected list")
    checkpoint_round = checkpoint.get("round_number")
    if checkpoint_round is not None and checkpoint_round != current_round:
        errors.append(
            f"round_number mismatch: checkpoint={checkpoint_round}, "
            f"current={current_round}"
        )
    checkpoint_stage = checkpoint.get("planning_stage")
    if checkpoint_stage is not None and checkpoint_stage != planning_stage:
        errors.append(
            f"planning_stage mismatch: checkpoint={checkpoint_stage}, "
            f"current={planning_stage}"
        )
    return errors


def build_checkpoint_payload(
    *,
    step_name: str,
    state: Any,
    planning_stage: str,
    step_number: int,
    total_steps: int,
    attempt_start_step: int,
    active_plan_id: str | None,
    messages: list,
    findings: list,
    figures: list,
    code_history: list,
    analysis_log: list,
    analysis_context: dict,
    budget_exhausted: bool,
) -> dict:
    return {
        "step_name": step_name,
        "session_id": state.id,
        "round_number": state.current_round,
        "planning_stage": planning_stage,
        "step_number": step_number,
        "total_steps": total_steps,
        "attempt_start_step": attempt_start_step,
        "active_plan_id": active_plan_id,
        "messages": messages,
        "findings": findings,
        "figures": figures,
        "code_history": code_history,
        "analysis_log": analysis_log,
        "analysis_context": analysis_context,
        "budget_exhausted": budget_exhausted,
        "completed": False,
    }



async def load_checkpoint(
    agent,
    checkpoint_manager,
    state,
    *,
    resume_from_checkpoint: bool,
    planning_stage: str,
    bindings,
) -> tuple[dict | None, bool]:
    logger = getattr(bindings, "logger")
    # Try to resume from checkpoint if enabled
    checkpoint = None
    checkpoint_valid = False
    if resume_from_checkpoint:
        try:
            checkpoint = await checkpoint_manager.load()
            if checkpoint and not checkpoint.get('completed', False):
                validation_errors = validate_checkpoint(
                    checkpoint,
                    current_round=state.current_round,
                    planning_stage=planning_stage,
                )

                if not validation_errors:
                    checkpoint_valid = True
                    logger.info(f"[Bioinformatics] Resuming from checkpoint at step {checkpoint['step_number']}")
                    await agent._notify("status", f"Resuming analysis from step {checkpoint['step_number']}...")
                else:
                    logger.warning(f"[Bioinformatics] Checkpoint validation failed: {validation_errors}, starting fresh")
                    checkpoint_manager.clear()
                    checkpoint = None
        except Exception as e:
            logger.exception("[Bioinformatics] Failed to load checkpoint")
            # Only clear if it's a data corruption issue, not a transient error
            try:
                checkpoint_manager.clear()
                logger.warning("[Bioinformatics] Cleared corrupted checkpoint data")
            except Exception as clear_err:
                logger.error(f"[Bioinformatics] Failed to clear checkpoint: {clear_err}")
            checkpoint = None
            await agent._notify("status", "Previous checkpoint corrupted, starting fresh analysis...")


    return checkpoint, checkpoint_valid
