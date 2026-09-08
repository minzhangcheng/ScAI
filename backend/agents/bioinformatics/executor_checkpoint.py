"""Executor checkpoint validation and payload construction."""
from __future__ import annotations

from typing import Any


def validate_checkpoint(
    checkpoint: dict,
    *,
    current_round: int,
    planning_stage: str,
    committed_revision: int | None = None,
    exact_revision: bool = False,
) -> list[str]:
    errors: list[str] = []
    recorded_revision = checkpoint.get("runtime_revision")
    behind_or_mismatched = (
        committed_revision is not None
        and recorded_revision is not None
        and (
            int(recorded_revision) < int(committed_revision)
            or (
                # The namespace was just moved back onto this generation
                # (B2-2), so "ahead" no longer means a live kernel that has
                # carried on -- it means a position from work the restore
                # discarded, and resuming there would skip the steps that
                # rebuilt it.
                exact_revision
                and int(recorded_revision) != int(committed_revision)
            )
        )
    )
    if behind_or_mismatched:
        # Behind the committed generation means the namespace moved past this
        # position: something restored or replaced the runtime after the
        # checkpoint was written, and resuming would replay against variables
        # that have since been rebuilt.
        #
        # Ahead is not an error and must not become one. The executor
        # checkpoints every five steps while the dump waits out the save floor,
        # so a live kernel is normally ahead of its newest generation -- that is
        # what §3.1 describes, and refusing it would break every ordinary
        # resume. Absent is not an error either: every checkpoint written before
        # this field existed has none, and absence is not evidence of mismatch.
        errors.append(
            f"runtime_revision does not belong to the committed generation: "
            f"checkpoint={recorded_revision}, committed={committed_revision}"
        )
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
    runtime_revision: int | None = None,
) -> dict:
    return {
        # Which runtime generation this position belongs to. Spec §3.11: the
        # two fields below cannot see a namespace, so without this a checkpoint
        # written against one kernel validated cleanly against another.
        "runtime_revision": runtime_revision,
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
                    committed_revision=(
                        getattr(state, "runtime_persistence", None) or {}
                    ).get("committed_revision"),
                    # The notebook projection carries the restored generation
                    # marker across process restarts and onto the paired
                    # AnalysisState selected by the run route.
                    exact_revision=bool(
                        (getattr(state, "runtime_persistence", None) or {})
                        .get("pair_restore")
                    ),
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
