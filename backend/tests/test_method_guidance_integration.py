"""Switch-controlled Bioinformatics prompt and method-lookup integration."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import backend.agents.bioinformatics_agent as bio_mod
from backend.agents.bioinformatics import executor_collaboration
from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.models.schemas import AnalysisPlan, AnalysisState, PlanStep


PLAN_JSON = (
    '{"overview":"bounded","steps":[{"id":1,"title":"DE",'
    '"method":"pydeseq2","input":"counts","expected_output":"effects",'
    '"rationale":"sample-level comparison","depends_on":[]}],'
    '"key_decision_points":[],"expected_runtime":"1 step"}'
)


def _agent(enabled: bool) -> BioinformaticsAgent:
    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_system_prompt.return_value = "system"
    provider.get_temperature.return_value = 0.3
    provider.get_vision_setting.return_value = False
    provider.get_llm_client.return_value = MagicMock()
    provider.get_method_decision_guidance_enabled.return_value = enabled
    agent = BioinformaticsAgent(settings_provider=provider, sandbox=None)
    agent.session_id = "method-guidance-test"
    agent.chat_without_history = AsyncMock(return_value=PLAN_JSON)
    return agent


def _state() -> AnalysisState:
    return AnalysisState(user_id="test-user", current_round=1)


@pytest.mark.asyncio
async def test_switch_off_keeps_initial_planning_prompt_free_of_cards():
    agent = _agent(False)
    await agent.generate_analysis_plan(
        state=_state(),
        pi_tasks=["Compare expression between groups"],
        overview="Choose a defensible differential expression method",
        phase="exploration",
    )
    prompt = agent.chat_without_history.await_args.args[0]
    assert "Non-binding Method Decision Guidance" not in prompt
    assert "[decision card:" not in prompt


@pytest.mark.asyncio
async def test_switch_on_injects_bounded_initial_planning_cards():
    agent = _agent(True)
    await agent.generate_analysis_plan(
        state=_state(),
        pi_tasks=["Compare expression between groups using independent samples"],
        overview="Choose marker or pseudobulk analysis",
        phase="exploration",
    )
    prompt = agent.chat_without_history.await_args.args[0]
    assert "Non-binding Method Decision Guidance" in prompt
    assert "[decision card: differential-expression-method-choice]" in prompt
    assert prompt.count("[decision card:") <= 3
    assert "unresolved material alternatives" in prompt
    assert "key_decision_points" in prompt


@pytest.mark.asyncio
async def test_switch_on_injects_revision_cards_from_existing_methods():
    agent = _agent(True)
    current = AnalysisPlan(
        phase="exploration",
        stage="full",
        overview="Resolve lineage direction",
        steps=[
            PlanStep(
                id=1,
                title="Trajectory",
                method="scanpy.tl.paga and scanpy.tl.dpt",
                input="neighbor graph",
                expected_output="pseudotime",
                rationale="Compare PAGA, DPT, and RNA velocity assumptions",
            )
        ],
    )
    await agent.revise_plan_incrementally(
        state=_state(),
        current_plan=current,
        pi_feedback="Do not treat RNA velocity as an interchangeable fallback.",
    )
    prompt = agent.chat_without_history.await_args.args[0]
    assert "Non-binding Method Decision Guidance" in prompt
    assert "[decision card: trajectory-connectivity-and-velocity]" in prompt


def _lookup_context(enabled: bool):
    executor_collaboration.bind_facade(bio_mod)
    provider = MagicMock()
    provider.get_method_decision_guidance_enabled.return_value = enabled
    technique_agent = SimpleNamespace(lookup_docs=AsyncMock())
    agent = SimpleNamespace(technique_agent=technique_agent, _provider=provider)
    return SimpleNamespace(
        agent=agent,
        messages=[],
        analysis_log=[],
        step_count=1,
        agent_reasoning="compare methods",
    ), technique_agent


@pytest.mark.asyncio
async def test_switch_off_method_lookup_preserves_rule_only_result():
    context, technique = _lookup_context(False)
    call = SimpleNamespace(
        id="off", arguments={"method_question": "pseudobulk biological replicate"}
    )
    await executor_collaboration.handle_lookup_documentation(context, call)
    text = context.messages[-1]["content"]
    assert "[method rule:" in text
    assert "Method decision guidance" not in text
    assert "[decision card:" not in text
    technique.lookup_docs.assert_not_awaited()


@pytest.mark.asyncio
async def test_switch_on_method_lookup_labels_cards_and_rules_separately():
    context, technique = _lookup_context(True)
    call = SimpleNamespace(
        id="on", arguments={"method_question": "pseudobulk biological replicate"}
    )
    await executor_collaboration.handle_lookup_documentation(context, call)
    text = context.messages[-1]["content"]
    assert text.startswith("Method decision guidance\n")
    assert "[decision card: differential-expression-method-choice]" in text
    assert "\n\nRelated statistical caveats\n" in text
    assert "[method rule:" in text
    technique.lookup_docs.assert_not_awaited()
