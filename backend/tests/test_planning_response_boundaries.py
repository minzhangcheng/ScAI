"""Focused A3 tests for shared planner representation normalization."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.pi_agent import PIAgent
from backend.models.schemas import AnalysisPlan, AnalysisState, PlanStep


def _make_pi_agent() -> PIAgent:
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    provider.get_llm_client.return_value = MagicMock()
    agent = PIAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


def _plan() -> AnalysisPlan:
    return AnalysisPlan(
        phase="exploration",
        stage="stage1",
        overview="Bounded plan",
        steps=[PlanStep(id=1, title="QC", method="scanpy")],
    )


@pytest.mark.asyncio
async def test_truncated_user_feedback_cannot_approve_execution():
    from backend.utils.llm_client import UpstreamExhaustedError

    agent = _make_pi_agent()
    agent.chat = AsyncMock(return_value='{"action":"proceed_to_execution","guidance_for_bioinfo":"')

    with pytest.raises(UpstreamExhaustedError) as caught:
        await agent.process_user_plan_feedback("Add a comparison before execution", _plan())

    assert caught.value.failure_kind == "parse_error"


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ['{}', '{"intent":null,"action":null}'])
async def test_empty_feedback_decision_does_not_approve_user_modification(response):
    agent = _make_pi_agent()
    agent.chat = AsyncMock(return_value=response)
    feedback = "Add a sensitivity analysis before execution"

    result = await agent.process_user_plan_feedback(feedback, _plan())

    assert result["action"] == "revise_with_bioinfo"
    assert result["guidance_for_bioinfo"] == feedback


@pytest.mark.asyncio
@pytest.mark.parametrize("lookup", [False, True])
async def test_truncated_pi_review_cannot_approve_plan(lookup):
    from backend.utils.llm_client import UpstreamExhaustedError

    agent = _make_pi_agent()
    replies = ['{"decision":"approved","feedback":"unfinished']
    if lookup:
        replies.insert(0, '{"checkpoint_lookup":{"page":1}}')
    agent.chat = AsyncMock(side_effect=replies)

    with pytest.raises(UpstreamExhaustedError) as caught:
        await agent.review_bioinfo_plan(AnalysisState(user_id="test"), _plan())

    assert caught.value.failure_kind == "parse_error"
    assert agent.chat.await_count == len(replies)


@pytest.mark.asyncio
async def test_null_user_feedback_preserves_requested_change():
    agent = _make_pi_agent()
    agent.chat = AsyncMock(return_value=json.dumps({
        "intent": None,
        "action": None,
        "guidance_for_bioinfo": None,
        "clarification_question": None,
    }))
    user_input = "Add one sensitivity analysis before execution."

    result = await agent.process_user_plan_feedback(user_input, _plan())

    assert result["intent"] == "modify"
    assert result["action"] == "revise_with_bioinfo"
    assert result["guidance_for_bioinfo"] == user_input
