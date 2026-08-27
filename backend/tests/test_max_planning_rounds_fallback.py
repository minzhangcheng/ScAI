"""P1-10: When MAX_PLANNING_ROUNDS is exhausted with no agreed plan, PI must
unilaterally finalize the plan from the dialogue history instead of returning
an empty state that leads to infinite "revise" loops.
"""
import inspect
import re

import backend.workflow.planning as planning_mod
from backend.tests.split_facade import split_source


def test_pi_generate_final_plan_method_exists():
    """P1-10: PI agent must expose method to unilaterally finalize plan."""
    src = split_source("backend.agents.pi_agent")
    assert (
        "def generate_final_plan_from_dialogue" in src
        or "async def generate_final_plan_from_dialogue" in src
    ), "P1-10: pi_agent.generate_final_plan_from_dialogue must exist"


def test_max_rounds_branch_falls_back_to_pi_decision():
    """P1-10: when MAX_PLANNING_ROUNDS exhausted with NO existing plan, PI must
    produce a final plan and approve it."""
    from backend.workflow.planning_parts import dialogue

    body = inspect.getsource(dialogue.finalize_planning_dialogue)
    assert "generate_final_plan_from_dialogue" in body, (
        "P1-10: max-rounds branch must call generate_final_plan_from_dialogue"
    )
    assert "plan_approved" in body and "True" in body, (
        "P1-10: max-rounds final plan must set plan_approved=True"
    )
