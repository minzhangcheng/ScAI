"""P1-10: When MAX_PLANNING_ROUNDS is exhausted with no agreed plan, PI must
unilaterally finalize the plan from the dialogue history instead of returning
an empty state that leads to infinite "revise" loops.
"""
import inspect
import logging
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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


@pytest.mark.asyncio
async def test_unilateral_workflow_finalization_does_not_approve_invalid_innovation_scope(
    monkeypatch,
):
    from backend.models.schemas import (
        AcceptedFoundationInventory,
        AgentType,
        AnalysisPhase,
        AnalysisPlan,
        AnalysisState,
        PlanStep,
        TaskStatus,
    )
    import backend.workflow.planning_parts.dialogue as dialogue_module
    from backend.workflow.planning_parts.dialogue import finalize_planning_dialogue

    # The production facade injects these shared names through bind_facade;
    # this unit calls the split implementation directly.
    monkeypatch.setattr(dialogue_module, "AgentType", AgentType, raising=False)
    monkeypatch.setattr(dialogue_module, "TaskStatus", TaskStatus, raising=False)
    monkeypatch.setattr(
        dialogue_module,
        "logger",
        logging.getLogger("test_planning_dialogue"),
        raising=False,
    )

    analysis_state = AnalysisState(
        user_id="scope-test",
        current_round=2,
        analysis_phase=AnalysisPhase.INNOVATION,
        accepted_foundation_inventories=[AcceptedFoundationInventory(
            inventory_id="foundation-v1-r1",
            round_number=1,
            plan_id="round1-plan",
            planning_stage="stage1",
        )],
        status=TaskStatus.WAITING_PLAN_REVIEW,
    )
    seed_plan = AnalysisPlan(
        phase="innovation",
        stage="full",
        status="draft",
        steps=[PlanStep(id=1, title="Seed", method="scanpy")],
    )
    invalid_final_plan = AnalysisPlan(
        phase="innovation",
        stage="full",
        status="pi_reviewing",
        scope_contract_version=1,
        foundation_inventory_id="foundation-v1-r1",
        steps=[PlanStep(id=1, title="Missing scope", method="scanpy")],
    )
    workflow = SimpleNamespace(
        session_id="scope-test",
        pi_agent=SimpleNamespace(
            generate_final_plan_from_dialogue=AsyncMock(
                return_value=invalid_final_plan
            )
        ),
        _send_message=AsyncMock(),
        _send_state_sync=AsyncMock(),
        _report_direction_coverage_gaps=AsyncMock(),
    )
    workflow._select_fallback_plan_from_history = lambda _history: None
    state = {"plan_approved": False, "should_continue": True}
    ctx = {
        "analysis_state": analysis_state,
        "stage": "full",
        "dialogue_history": [],
        "existing_plan": seed_plan,
        "round_num": 4,
        "finalization_reason": "max_rounds",
    }

    result = await finalize_planning_dialogue(workflow, ctx, state)

    assert result["plan_approved"] is False
    assert result["failed_node"] == "planning_dialogue"
    assert analysis_state.current_plan is seed_plan
    assert analysis_state.status == TaskStatus.FAILED
    workflow.pi_agent.generate_final_plan_from_dialogue.assert_awaited_once()
