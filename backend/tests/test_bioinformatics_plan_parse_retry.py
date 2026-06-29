import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.models.schemas import AgentType, AnalysisPlan, AnalysisState, PlanStep


def _make_agent() -> BioinformaticsAgent:
    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_system_prompt.return_value = "system"
    provider.get_temperature.return_value = 0.3
    provider.get_vision_setting.return_value = False
    provider.get_llm_client.return_value = MagicMock()
    agent = BioinformaticsAgent(settings_provider=provider, sandbox=None)
    agent.session_id = "test-session"
    return agent


def _make_state() -> AnalysisState:
    return AnalysisState(user_id="test-user")


@pytest.mark.asyncio
async def test_layer2_retry_prompt_is_unique():
    agent = _make_agent()
    state = _make_state()
    agent.chat = AsyncMock(side_effect=[
        "not json",
        '{"overview":"ok","steps":[{"id":1,"title":"QC","method":"scanpy.pp.filter_cells","depends_on":[]}]}',
    ])

    plan = await agent.generate_analysis_plan(
        state=state,
        pi_tasks=["QC"],
        overview="overview",
        phase="exploration",
    )

    assert len(plan.steps) == 1
    assert agent.chat.call_count == 2
    first_prompt = agent.chat.call_args_list[0].args[0]
    retry_prompt = agent.chat.call_args_list[1].args[0]
    assert first_prompt != retry_prompt
    assert re.search(r"\[retry:[0-9a-f]{8}\]", retry_prompt)


@pytest.mark.asyncio
async def test_generate_plan_retry_on_empty_steps():
    agent = _make_agent()
    state = _make_state()
    agent.chat = AsyncMock(side_effect=[
        '{"overview":"empty","steps":[]}',
        '{"overview":"ok","steps":[{"id":1,"title":"Load","method":"scanpy.read_h5ad","depends_on":[]}]}',
    ])

    plan = await agent.generate_analysis_plan(
        state=state,
        pi_tasks=["load"],
        overview="overview",
        phase="exploration",
    )

    assert agent.chat.call_count == 2
    assert len(plan.steps) == 1
    assert plan.steps[0].title == "Load"


@pytest.mark.asyncio
async def test_generate_plan_retry_fallback(caplog):
    agent = _make_agent()
    state = _make_state()
    agent.chat = AsyncMock(side_effect=["not json", "also not json"])

    plan = await agent.generate_analysis_plan(
        state=state,
        pi_tasks=["QC"],
        overview="fallback overview",
        phase="exploration",
    )

    assert agent.chat.call_count == 2
    assert plan.steps == []
    assert plan.overview == "fallback overview"
    assert "retry also failed" in caplog.text


@pytest.mark.asyncio
async def test_revise_parse_error_preserves_current_plan():
    agent = _make_agent()
    state = _make_state()
    agent.chat = AsyncMock(side_effect=["not json", "also not json"])
    current_plan = AnalysisPlan(
        phase="exploration",
        stage="full",
        overview="current overview",
        steps=[
            PlanStep(
                id=1,
                title="Existing QC",
                method="scanpy.pp.filter_cells",
                depends_on=[2],
            )
        ],
        key_decision_points=["decision"],
        expected_runtime="~1 step",
    )

    revised = await agent.revise_plan_incrementally(
        state=state,
        current_plan=current_plan,
        pi_feedback="keep existing",
    )

    assert agent.chat.call_count == 2
    assert len(revised.steps) == 1
    assert revised.steps[0].title == "Existing QC"
    assert revised.steps[0].depends_on == [2]
    assert revised.steps[0].depends_on is not current_plan.steps[0].depends_on
    assert revised.revision_of == current_plan.id
    assert revised.key_decision_points == ["decision"]


@pytest.mark.asyncio
async def test_revise_empty_steps_retries_then_preserves_current_plan():
    agent = _make_agent()
    state = _make_state()
    agent.chat = AsyncMock(side_effect=[
        '{"overview":"empty","steps":[]}',
        '{"overview":"still empty","steps":[]}',
    ])
    current_plan = AnalysisPlan(
        phase="exploration",
        stage="stage1",
        overview="stage1 plan",
        steps=[PlanStep(id=1, title="Load", method="scanpy.read_h5ad")],
        expected_runtime="~1 step",
    )

    revised = await agent.revise_plan_incrementally(
        state=state,
        current_plan=current_plan,
        pi_feedback="revise but preserve if empty",
    )

    assert agent.chat.call_count == 2
    assert len(revised.steps) == 1
    assert revised.steps[0].title == "Load"
    assert revised.overview == "stage1 plan"
    assert revised.stage == "stage1"
