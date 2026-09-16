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
    _run_attempt_ids = list(getattr(context, "attempt_execution_ids", None) or [])
    _run_kind = str(getattr(context, "executor_run_kind", "ordinary") or "ordinary")
    _run_scope_id = str(getattr(context, "executor_run_scope_id", "") or "")
    _settlement_scope_id = getattr(context, "executor_settlement_scope_id", None)
    _scope_id = _settlement_scope_id
    _figure_attempt_ids: list[str] = []
    _settlement_attempt_ids: list[str] = []
    if _run_kind == "figure_followup":
        _scope_id = None
        _figure_map = dict(
            getattr(context.analysis_state, "figure_attempt_execution_ids", None)
            or {}
        )
        _persisted_figure = list(_figure_map.get(_run_scope_id, []) or [])
        _figure_attempt_ids = sorted(set(_persisted_figure) | set(_run_attempt_ids))[-256:]
        if _run_scope_id and _figure_attempt_ids:
            _figure_map[_run_scope_id] = _figure_attempt_ids
            # Bound by scope count without converting these audit IDs into
            # analytical settlement evidence.
            context.analysis_state.figure_attempt_execution_ids = dict(
                list(_figure_map.items())[-128:]
            )
            try:
                await context.analysis_state.increment_version()
                from backend.services.session_store import async_save_session

                await async_save_session(
                    str(context.analysis_state.id),
                    context.analysis_state,
                    force=True,
                )
            except Exception as exc:
                logger.warning(
                    "Could not force-save figure-attempt IDs for %s: %s",
                    _run_scope_id,
                    exc,
                )
    else:
        try:
            _attempt_map = dict(
                context.analysis_state.bioinformatics_settlement_attempt_ids or {}
            )
        except Exception:
            _attempt_map = {}
        _persisted = list(_attempt_map.get(_scope_id, []) or [])
        _settlement_attempt_ids = sorted(set(_persisted) | set(_run_attempt_ids))
        if _scope_id:
            _attempt_map[str(_scope_id)] = _settlement_attempt_ids
            context.analysis_state.bioinformatics_settlement_attempt_ids = _attempt_map
            # Persist the reached-kernel accumulator before the workflow can route
            # to a plan revision or terminal settlement.  The scoped executor
            # checkpoint remains the fallback if this Host-state write fails.
            try:
                await context.analysis_state.increment_version()
                from backend.services.session_store import async_save_session

                await async_save_session(
                    str(context.analysis_state.id),
                    context.analysis_state,
                    force=True,
                )
            except Exception as exc:
                logger.warning(
                    "Could not force-save executor settlement attempts for %s: %s",
                    _scope_id,
                    exc,
                )
    if context.completed:
        # Keep the exact transcript until the workflow has durably settled the
        # corresponding AnnData boundary.  `completed=True` prevents a retry
        # from replaying it during that crash window.
        try:
            await context.save_checkpoint(
                f"step_{context.step_count}_pending_settlement",
                completed=True,
            )
        except Exception as exc:
            logger.warning(
                "Could not mark completed executor checkpoint pending settlement: %s",
                exc,
            )
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
        # Producer-supplied execution IDs are candidates only. The workflow
        # validates them against current successful execution receipts later.
        "evidence_verified": False,
        "evidence_gate_degraded": context.evidence_gate_degraded,
        "evidence_execution_ids": [],
        "candidate_evidence_execution_ids": context.reported_evidence_ids,
        "evidence_gate_findings": context.evidence_gate_findings,
        "figure_contract": figure_contract.summary(),
        "executor_run_kind": _run_kind,
        "executor_run_scope_id": _run_scope_id or None,
        "executor_settlement_scope_id": _scope_id,
        "attempt_execution_ids": _run_attempt_ids,
        "settlement_attempt_execution_ids": _settlement_attempt_ids,
        "figure_attempt_execution_ids": _figure_attempt_ids,
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
        "supplementary_request_id": getattr(
            context, "supplementary_request_id", None
        ),
        "resolved_adata_source_context": dict(
            getattr(context, "resolved_adata_source_context", None) or {}
        ),
        "adata_source": (
            context.adata_source.model_dump(mode="json")
            if hasattr(context.adata_source, "model_dump")
            else context.adata_source
        ),
        "executor_checkpoint_name": getattr(
            getattr(context, "checkpoint_manager", None),
            "checkpoint_path",
            None,
        ).name if getattr(
            getattr(context, "checkpoint_manager", None),
            "checkpoint_path",
            None,
        ) is not None else None,
    }
