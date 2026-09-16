"""Public autonomous-executor controller."""
from __future__ import annotations

from . import (
    executor_collaboration,
    executor_auxiliary,
    executor_execute,
    executor_gene_sets,
    executor_loop,
    executor_registration,
    executor_report,
    executor_report_gates,
)


async def run_autonomous_analysis(
    agent,
    state,
    initial_instruction: str,
    max_steps: int = 200,
    figure_review_callback=None,
    supervisor_callback=None,
    resume_from_checkpoint: bool = True,
    planning_stage="full",
    figure_followup: bool = False,
    adata_source=None,
    run_scope_id=None,
    settlement_scope_id=None,
    run_kind=None,
    supplementary_request_id=None,
    *,
    bindings,
):
    requested_run_kind = str(run_kind or "").strip()
    if figure_followup and requested_run_kind not in {"", "figure_followup"}:
        raise ValueError(
            "figure_followup conflicts with the requested executor run kind."
        )
    effective_figure_followup = bool(
        figure_followup or requested_run_kind == "figure_followup"
    )
    effective_run_kind = (
        "figure_followup" if effective_figure_followup else run_kind
    )

    executor_loop.bind_facade(bindings.facade)
    executor_collaboration.bind_facade(bindings.facade)
    executor_auxiliary.bind_facade(bindings.facade)
    executor_execute.bind_facade(bindings.facade)
    executor_gene_sets.bind_facade(bindings.facade)
    executor_registration.bind_facade(bindings.facade)
    executor_report.bind_facade(bindings.facade)
    executor_report_gates.bind_facade(bindings.facade)
    parent = {
        name: getattr(agent, name, None)
        for name in ("code_history", "analysis_context", "last_result")
    } if effective_figure_followup else {}
    main_sandbox = (
        getattr(agent, "sandbox", None) if effective_figure_followup else None
    )
    disposable_runtime = None
    context = None
    try:
        if effective_figure_followup:
            from backend.services.figure_followup_runtime import FigureFollowupRuntime

            disposable_runtime = FigureFollowupRuntime(main_sandbox, state)
            agent.sandbox = await disposable_runtime.__aenter__()
            agent.code_history = list(getattr(state, "code_history", None) or [])
        context = await executor_loop.prepare_loop_context(
            agent, state, initial_instruction,
            max_steps=max_steps,
            figure_review_callback=figure_review_callback,
            supervisor_callback=supervisor_callback,
            resume_from_checkpoint=resume_from_checkpoint,
            planning_stage=planning_stage,
            figure_followup=effective_figure_followup,
            adata_source=adata_source,
            run_scope_id=run_scope_id,
            settlement_scope_id=settlement_scope_id,
            run_kind=effective_run_kind,
            supplementary_request_id=supplementary_request_id,
        )
        return await executor_loop.run_turn_loop(context)
    except Exception as exc:
        # Source preparation is operation-local: surface it as a bounded
        # plan-revision signal, never as a global execute refusal.
        from backend.services.adata_checkpoints.errors import (
            AdataSourcePreparationError,
        )

        if not isinstance(exc, AdataSourcePreparationError):
            raise
        effective_kind = (
            "figure_followup"
            if effective_figure_followup
            else str(run_kind or "ordinary")
        )
        request_record = {}
        plan = None if effective_kind == "supplementary" else getattr(
            state, "current_plan", None
        )
        if effective_kind == "supplementary":
            request_record = dict(
                ((getattr(state, "sandbox_state", None) or {}).get(
                    "supplementary_requests_v1"
                ) or {}).get(str(supplementary_request_id or ""))
                or {}
            )
        resolved_run_scope = str(
            request_record.get("executor_run_scope_id")
            or request_record.get("request_scope_id")
            or getattr(plan, "bioinformatics_run_scope_id", None)
            or run_scope_id
            or ""
        ) or None
        resolved_settlement_scope = str(
            request_record.get("executor_settlement_scope_id")
            or request_record.get("request_scope_id")
            or getattr(plan, "bioinformatics_settlement_scope_id", None)
            or settlement_scope_id
            or ""
        ) or None
        source_context = dict(
            (getattr(state, "sandbox_state", None) or {}).get(
                "resolved_adata_source_context_v1"
            )
            or {}
        )
        checkpoint_name = getattr(
            getattr(agent, "sandbox", None),
            "active_executor_checkpoint_name",
            None,
        )
        return {
            "status": "adata_source_error",
            "error": str(exc),
            "findings": [],
            "figures": [],
            "mechanism_programs": [],
            "code_history": [],
            "analysis_log": [],
            "steps_executed": 0,
            "evidence_execution_ids": [],
            "evidence_verified": None,
            "evidence_gate_degraded": False,
            "evidence_gate_findings": [],
            "executor_run_kind": effective_kind,
            "executor_run_scope_id": resolved_run_scope,
            "executor_settlement_scope_id": resolved_settlement_scope,
            "supplementary_request_id": supplementary_request_id,
            "attempt_execution_ids": [],
            "settlement_attempt_execution_ids": list(
                (getattr(state, "bioinformatics_settlement_attempt_ids", None) or {})
                .get(resolved_settlement_scope, [])
                if resolved_settlement_scope else []
            ),
            "resolved_adata_source_context": source_context,
            "executor_checkpoint_name": checkpoint_name,
            "plan_revision_needed": True,
            "plan_revision_context": {
                "technical_invariant": "adata_source_preparation",
                "adata_source_error": exc.to_context(),
            },
        }
    finally:
        if effective_figure_followup:
            # Figure rendering is an isolated capability. Restore the parent's
            # live handle before discarding the disposable kernel; never merge
            # its namespace, executor transcript, or analysis log.
            agent.sandbox = main_sandbox
            if disposable_runtime is not None:
                await disposable_runtime.__aexit__(None, None, None)
            for name, value in parent.items():
                setattr(agent, name, value)
