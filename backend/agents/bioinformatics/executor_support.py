"""Small shared helpers for autonomous executor setup."""
from __future__ import annotations


def active_plan_id(state) -> str | None:
    plan = state.current_plan
    if plan and plan.status == "approved":
        return str(plan.id)
    return None
