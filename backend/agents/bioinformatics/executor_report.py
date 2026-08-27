"""Terminal result helpers for autonomous analysis reporting."""
from __future__ import annotations


def terminal_status(*, llm_error: bool, plan_revision_needed: bool, completed: bool,
                    evidence_gate_degraded: bool) -> str:
    if llm_error:
        return "error"
    if plan_revision_needed:
        return "paused_for_revision"
    if completed:
        return "partial" if evidence_gate_degraded else "completed"
    return "max_steps_reached"
