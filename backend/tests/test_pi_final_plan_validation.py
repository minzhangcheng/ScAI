"""Focused A3 tests for final-plan schema repair and validation boundaries."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.pi_agent import PIAgent
from backend.models.schemas import (
    AcceptedFoundationInventory,
    AcceptedFoundationStep,
    AnalysisPhase,
    AnalysisPlan,
    AnalysisState,
    PlanAnalysisScope,
    PlanStep,
)


def _make_pi_agent() -> PIAgent:
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    provider.get_llm_client.return_value = MagicMock()
    agent = PIAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


def _seed_plan() -> AnalysisPlan:
    return AnalysisPlan(
        phase="exploration",
        stage="stage1",
        overview="Last valid plan",
        steps=[PlanStep(id=1, title="Seed", method="scanpy")],
    )


def _plan_payload(platform_baseline: str = "") -> str:
    return json.dumps({
        "overview": "Final Stage 1 plan",
        "steps": [{
            "id": 1,
            "title": "Cluster nuclei",
            "method": "scanpy.tl.leiden",
            "input": "Final neighborhood graph",
            "expected_output": "Stable cluster labels",
            "figure_requirements": [{
                "figure_type": "cluster_overview",
                "description": "Primary cluster overview",
                "minimum_count": 1,
                "platform_baseline": platform_baseline,
            }],
            "allow_text_only": False,
            "text_only_reason": "",
            "rationale": "Resolve major populations",
            "depends_on": [],
        }],
        "key_decision_points": ["Select the stable resolution"],
        "expected_runtime": "~1 step",
    })


def _innovation_state(*, current_plan=None) -> AnalysisState:
    inventory = AcceptedFoundationInventory(
        inventory_id="foundation-v1-r1",
        round_number=1,
        plan_id="round1-plan",
        planning_stage="stage1",
        adata_checkpoint_id="checkpoint-r1",
        steps=[AcceptedFoundationStep(
            plan_step_id=4,
            title="Accepted clustering baseline",
            method="scanpy.tl.leiden",
            evidence_execution_ids=["receipt-r1-step4"],
        )],
    )
    return AnalysisState(
        user_id="test-user",
        current_round=2,
        analysis_phase=AnalysisPhase.INNOVATION,
        current_plan=current_plan,
        accepted_foundation_inventories=[inventory],
    )


def _innovation_plan_payload(*, with_scope: bool) -> str:
    step = {
        "id": 1,
        "title": "Targeted subset analysis",
        "method": "scanpy.tl.leiden",
        "input": "Accepted measurements for the selected subset.",
        "expected_output": "A subset sensitivity result.",
        "figure_requirements": [],
        "allow_text_only": True,
        "text_only_reason": "This is a planning validation fixture.",
        "rationale": "Use prior results for a bounded sensitivity analysis.",
        "depends_on": [],
    }
    if with_scope:
        step["analysis_scope"] = {
            "relationship": "reuse",
            "cohort_scope": "subset",
            "foundation_refs": [{
                "inventory_id": "foundation-v1-r1",
                "round_number": 1,
                "plan_id": "round1-plan",
                "plan_step_ids": [4],
                "execution_ids": ["receipt-r1-step4"],
                "checkpoint_id": "checkpoint-r1",
            }],
            "reason": "Reuse the accepted measurements for a targeted subset.",
        }
    return json.dumps({
        "overview": "Targeted innovation analysis",
        "steps": [step],
        "scope_contract_version": 1,
        "foundation_inventory_id": "foundation-v1-r1",
        "key_decision_points": [],
        "expected_runtime": "1 step",
    })


def _valid_innovation_seed() -> AnalysisPlan:
    return AnalysisPlan(
        phase="innovation",
        stage="full",
        status="approved",
        scope_contract_version=1,
        foundation_inventory_id="foundation-v1-r1",
        steps=[PlanStep(
            id=1,
            title="Existing subset analysis",
            method="scanpy.tl.leiden",
            analysis_scope=PlanAnalysisScope(
                relationship="reuse",
                cohort_scope="subset",
                foundation_refs=[{
                    "inventory_id": "foundation-v1-r1",
                    "round_number": 1,
                    "plan_id": "round1-plan",
                    "plan_step_ids": [4],
                    "execution_ids": ["receipt-r1-step4"],
                    "checkpoint_id": "checkpoint-r1",
                }],
                reason="Reuse the accepted measurements.",
            ),
        )],
    )


@pytest.mark.asyncio
async def test_final_plan_twice_invalid_preserves_seed_without_approving_it():
    agent = _make_pi_agent()
    state = AnalysisState(user_id="test-user")
    seed_plan = _seed_plan()
    state.current_plan = seed_plan
    agent.chat = AsyncMock(side_effect=[
        _plan_payload("cluster_summary"),
        _plan_payload("cluster_summary"),
    ])

    original = seed_plan.model_dump()
    with pytest.raises(Exception) as caught:
        await agent.generate_final_plan_from_dialogue(
            state,
            [{"pi_decision": "revise"}],
            current_plan=seed_plan,
        )

    assert agent.chat.await_count == 2
    assert type(caught.value).__name__ == "PlanGenerationFailure"
    assert seed_plan.model_dump() == original

    no_seed_agent = _make_pi_agent()
    no_seed_agent.chat = AsyncMock(side_effect=[
        _plan_payload("cluster_summary"),
        _plan_payload("cluster_summary"),
    ])
    with pytest.raises(Exception) as exc_info:
        await no_seed_agent.generate_final_plan_from_dialogue(
            AnalysisState(user_id="test-user"),
            [],
            current_plan=None,
        )
    assert type(exc_info.value).__name__ == "PlanGenerationFailure"
    assert no_seed_agent.chat.await_count == 2


@pytest.mark.asyncio
async def test_final_plan_does_not_accept_a_truncated_step_as_a_complete_plan():
    agent = _make_pi_agent()
    valid = _plan_payload()
    truncated = valid[:valid.index('"expected_output"')] + '"expected_output": "unfinished'
    agent.chat = AsyncMock(side_effect=[truncated, valid])

    plan = await agent.generate_final_plan_from_dialogue(AnalysisState(user_id="test"), [])

    assert agent.chat.await_count == 2
    assert "Incomplete or invalid plan JSON" in agent.chat.call_args_list[1].args[0]
    assert "model signal" not in agent.chat.call_args_list[1].args[0]
    assert plan.steps[0].expected_output == "Stable cluster labels"


@pytest.mark.asyncio
async def test_invalid_innovation_finalization_does_not_fall_back_to_seed_plan():
    agent = _make_pi_agent()
    seed = _valid_innovation_seed()
    state = _innovation_state(current_plan=seed)
    agent.chat = AsyncMock(side_effect=[
        _innovation_plan_payload(with_scope=False),
        _innovation_plan_payload(with_scope=False),
    ])

    with pytest.raises(Exception) as exc_info:
        await agent.generate_final_plan_from_dialogue(
            state,
            [],
            current_plan=seed,
        )

    assert type(exc_info.value).__name__ == "PlanGenerationFailure"
    assert agent.chat.await_count == 2
