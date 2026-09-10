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
    *,
    bindings,
):
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
    } if figure_followup else {}
    context = None
    try:
        context = await executor_loop.prepare_loop_context(
            agent, state, initial_instruction,
            max_steps=max_steps,
            figure_review_callback=figure_review_callback,
            supervisor_callback=supervisor_callback,
            resume_from_checkpoint=resume_from_checkpoint,
            planning_stage=planning_stage,
            figure_followup=figure_followup,
        )
        return await executor_loop.run_turn_loop(context)
    finally:
        if figure_followup:
            # Preserve receipts even on cancellation, before restoring the parent
            # executor's in-memory context. Kernel objects remain shared.
            known = {item.id for item in state.code_history}
            for item in agent.code_history:
                if item.id not in known:
                    state.code_history.append(item)
                    known.add(item.id)
            if context is not None:
                known = {item.id for item in state.analysis_log}
                for item in context.analysis_log:
                    if item.id not in known:
                        item.round_number = state.current_round
                        state.analysis_log.append(item)
                        known.add(item.id)
            for name, value in parent.items():
                setattr(agent, name, value)
