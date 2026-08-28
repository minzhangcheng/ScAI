"""Rare-exit helpers shared by autonomous executor controllers."""
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Any

from .executor_report import terminal_status


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )


class LoopExit(Enum):
    CONTINUE = "continue"
    PLAN_REVISED = "plan_revised"
    TERMINAL = "terminal"


@dataclass
class TurnStart:
    response: Any = None
    exit: LoopExit = LoopExit.CONTINUE


@dataclass
class TurnEffects:
    response: Any
    step_figures: list
    has_substantive_work: bool
    has_figure_work: bool
    has_register_score: bool


def classify_loop_exit(*, plan_revision_needed: bool, terminal: bool) -> LoopExit:
    if plan_revision_needed:
        return LoopExit.PLAN_REVISED
    if terminal:
        return LoopExit.TERMINAL
    return LoopExit.CONTINUE


async def finalize_run(context) -> dict:
    status = terminal_status(
        llm_error=context.llm_error,
        plan_revision_needed=context.plan_revision_needed,
        completed=context.completed,
        evidence_gate_degraded=context.evidence_gate_degraded,
    )
    if status in ("max_steps_reached", "completed_with_fallback"):
        try:
            await context.save_checkpoint(
                f"step_{context.step_count}_budget_exhausted",
                budget_exhausted=True,
            )
        except Exception as exc:
            logger.warning(
                "[Bioinformatics] Could not persist exhausted executor budget "
                "at step %s: %s",
                context.step_count,
                exc,
            )
    from ...services.figure_contract import audit_figure_contract

    figure_contract = audit_figure_contract(
        context.analysis_state,
        plan=context.analysis_state.current_plan,
        planning_stage=context.planning_stage,
        include_stage_baseline=False,
        defer_platform_baseline=True,
    )
    saved_figure_count = count_saved_figures(context.analysis_log)
    steps_executed = context.step_count - context.attempt_start_step
    logger.info(
        "[Bioinformatics] Analysis %s: %s steps executed in this attempt, "
        "%s findings, %s inline figures, %s saved figures",
        status,
        steps_executed,
        len(context.findings),
        len(context.figures),
        saved_figure_count,
    )
    if context.completed:
        context.checkpoint_manager.clear()
    return {
        "status": status,
        "findings": context.findings,
        "finding_records": context.accepted_finding_records,
        "figures": context.figures,
        "steps_executed": steps_executed,
        "code_history": context.agent.code_history,
        "analysis_log": context.analysis_log,
        "success": status != "error",
        "mechanism_programs": context.mechanism_programs,
        # Producer-supplied execution IDs are candidates only. The workflow's
        # Supervisor reconciliation promotes semantically valid links later.
        "evidence_verified": False,
        "evidence_gate_degraded": context.evidence_gate_degraded,
        "evidence_execution_ids": [],
        "candidate_evidence_execution_ids": context.reported_evidence_ids,
        "evidence_gate_findings": context.evidence_gate_findings,
        "figure_contract": figure_contract.summary(),
        "cohort_identity": dict(
            (context.analysis_state.sandbox_state or {}).get(
                "cohort_identity_v1", {}
            ) or {}
        ),
        "cohort_transition": dict(
            (context.analysis_state.sandbox_state or {}).get(
                "cohort_transition_notice", {}
            ) or {}
        ),
        "forced_stop": bool(context.force_break),
        **({"error": context.llm_error} if context.llm_error else {}),
        "plan_revision_needed": context.plan_revision_needed,
        "plan_revision_context": (
            context.plan_revision_context if context.plan_revision_needed else {}
        ),
    }
