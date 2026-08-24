"""Focused A3 tests for final-plan schema repair and validation boundaries."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from backend.agents.pi_agent import PIAgent
from backend.models.schemas import AnalysisPlan, AnalysisState, FigureRequirement, PlanStep


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
