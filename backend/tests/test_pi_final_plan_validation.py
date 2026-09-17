"""Focused A3 tests for final-plan schema repair and validation boundaries."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from backend.agents.pi_agent import PIAgent
from backend.models.schemas import (
    AcceptedFoundationInventory,
    AcceptedFoundationStep,
    AnalysisPhase,
    AnalysisPlan,
    AnalysisState,
    FigureRequirement,
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


@pytest.mark.parametrize(("alias", "canonical"), [
    ("annotation_overview", "annotation"),
    ("batch_overview", "batch"),
    ("cluster_overview", "cluster"),
    ("marker_overview", "markers"),
    (" Markers ", "markers"),
])
def test_figure_requirement_canonicalizes_only_explicit_platform_aliases(alias, canonical):
    requirement = FigureRequirement(platform_baseline=alias)
    assert requirement.platform_baseline == canonical


def test_unknown_platform_alias_remains_invalid():
    with pytest.raises(ValidationError, match="literal_error"):
        FigureRequirement(platform_baseline="cluster_summary")


@pytest.mark.asyncio
async def test_malformed_then_repaired_plan_uses_one_retry():
    agent = _make_pi_agent()
    state = AnalysisState(user_id="test-user")
    state.current_plan = _seed_plan()
    agent.chat = AsyncMock(side_effect=[
        _plan_payload("cluster_summary"),
        _plan_payload("cluster"),
    ])

    plan = await agent.generate_final_plan_from_dialogue(
        state,
        [],
        current_plan=state.current_plan,
    )

    assert agent.chat.await_count == 2
    assert plan.steps[0].figure_requirements[0].platform_baseline == "cluster"
    repair_prompt = agent.chat.call_args_list[1].args[0]
    assert "cluster_summary" in repair_prompt
    assert "schema validation" in repair_prompt


@pytest.mark.asyncio
async def test_final_plan_repairs_once_then_accepts():
    agent = _make_pi_agent()
    state = AnalysisState(user_id="test-user")
    state.current_plan = _seed_plan()
    agent.chat = AsyncMock(side_effect=[
        _plan_payload("cluster_summary"),
        _plan_payload("cluster"),
    ])

    plan = await agent.generate_final_plan_from_dialogue(
        state,
        [{"round": 1, "pi_decision": "revise"}],
        current_plan=state.current_plan,
    )

    assert agent.chat.await_count == 2
    assert len(plan.steps) == 1
    assert plan.status == "pi_reviewing"


@pytest.mark.asyncio
async def test_final_plan_twice_invalid_uses_valid_seed_or_raises_without_one():
    agent = _make_pi_agent()
    state = AnalysisState(user_id="test-user")
    seed_plan = _seed_plan()
    state.current_plan = seed_plan
    agent.chat = AsyncMock(side_effect=[
        _plan_payload("cluster_summary"),
        _plan_payload("cluster_summary"),
    ])

    retained = await agent.generate_final_plan_from_dialogue(
        state,
        [],
        current_plan=seed_plan,
    )

    assert agent.chat.await_count == 2
    assert retained is not seed_plan
    assert retained.steps
    assert retained.status == "pi_reviewing"
    assert "fallback retained" in retained.revision_reason

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
async def test_innovation_finalization_receives_contract_and_repairs_invalid_scope():
    agent = _make_pi_agent()
    state = _innovation_state()
    agent.chat = AsyncMock(side_effect=[
        _innovation_plan_payload(with_scope=False),
        _innovation_plan_payload(with_scope=True),
    ])

    plan = await agent.generate_final_plan_from_dialogue(
        state,
        [{"round": 1, "pi_decision": "revise"}],
    )

    assert agent.chat.await_count == 2
    first_prompt = agent.chat.call_args_list[0].args[0]
    repair_prompt = agent.chat.call_args_list[1].args[0]
    assert "Innovation Scope Contract v1" in first_prompt
    assert "foundation-v1-r1" in first_prompt
    assert "missing analysis_scope" in repair_prompt
    assert plan.scope_contract_version == 1
    assert plan.steps[0].analysis_scope.foundation_refs[0].execution_ids == [
        "receipt-r1-step4"
    ]


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


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_succeeds", [True, False])
async def test_truncated_foundation_id_repair_preserves_identity_and_reports_cause(repair_succeeds):
    agent = _make_pi_agent()
    state = _innovation_state()
    valid = _innovation_plan_payload(with_scope=True)
    invalid = valid.replace("foundation-v1-r1", "foundation-v1-r")
    agent.chat = AsyncMock(side_effect=[invalid, valid if repair_succeeds else invalid])

    if repair_succeeds:
        plan = await agent.generate_final_plan_from_dialogue(state, [])
        assert plan.steps[0].analysis_scope.foundation_refs[0].inventory_id == "foundation-v1-r1"
    else:
        from backend.agents.bioinformatics.planning import PlanGenerationFailure

        with pytest.raises(PlanGenerationFailure) as exc_info:
            await agent.generate_final_plan_from_dialogue(state, [])
        message = str(exc_info.value)
        assert "unknown inventory_id" in message
        assert "no executable steps" not in message
        assert "expected 'foundation-v1-r1'" in message

    assert agent.chat.await_count == 2
    repair = agent.chat.call_args_list[1].args[0]
    assert "Innovation step 1 analysis_scope.foundation_refs[0]" in repair
    assert "received 'foundation-v1-r'" in repair
    assert "expected 'foundation-v1-r1'" in repair


@pytest.mark.asyncio
async def test_final_plan_preserves_aliased_output_figures_and_step_dependencies():
    agent = _make_pi_agent()
    payload = json.loads(_plan_payload())
    step = payload["steps"][0]
    step["step_id"] = 7
    del step["id"]
    step["output"] = step.pop("expected_output")
    step["required_figures"] = step.pop("figure_requirements")
    payload["steps"].append({**step, "step_id": 9, "depends_on": [7]})
    agent.chat = AsyncMock(return_value=json.dumps(payload))

    plan = await agent.generate_final_plan_from_dialogue(AnalysisState(user_id="test"), [])

    assert agent.chat.await_count == 1
    assert [s.id for s in plan.steps] == [7, 9]
    assert plan.steps[1].depends_on == [7]
    assert plan.steps[0].expected_output == "Stable cluster labels"
    assert plan.steps[0].figure_requirements[0].description == "Primary cluster overview"
