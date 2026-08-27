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
    context = await executor_loop.prepare_loop_context(
        agent,
        state,
        initial_instruction,
        max_steps=max_steps,
        figure_review_callback=figure_review_callback,
        supervisor_callback=supervisor_callback,
        resume_from_checkpoint=resume_from_checkpoint,
        planning_stage=planning_stage,
    )
    return await executor_loop.run_turn_loop(context)
