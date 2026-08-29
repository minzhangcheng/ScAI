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
async def test_scalar_mapping_list_and_null_cross_shared_planning_boundary():
    agent = _make_pi_agent()
    agent.chat = AsyncMock(return_value=json.dumps({
        "overview": {"summary": "Run a broad exploration"},
        "bioinformatics_tasks": "Run QC",
        "pathology_tasks": None,
        "literature_topics": {"topic": "Method limitations"},
    }))

    result = await agent.initial_instruction(
        AnalysisState(user_id="test-user"),
        "Characterize the available data",
    )

    assert result["overview"] == json.dumps(
        {"summary": "Run a broad exploration"},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert result["bioinformatics_tasks"] == ["Run QC"]
    assert result["pathology_tasks"] == []
    assert result["literature_topics"] == [{"topic": "Method limitations"}]


@pytest.mark.asyncio
async def test_structured_text_is_lossless_and_deterministic():
    agent = _make_pi_agent()
    agent.chat = AsyncMock(return_value=json.dumps({
        "decision": "approved",
        "feedback": {"z": 1, "a": "ok"},
        "modifications": {"step_id": 1, "suggestion": "Keep"},
        "modification_dispositions": None,
    }))

    result = await agent.review_bioinfo_plan(
        AnalysisState(user_id="test-user"),
        _plan(),
    )

    assert result["feedback"] == json.dumps(
        {"a": "ok", "z": 1},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert result["modifications"] == [
        {"step_id": 1, "suggestion": "Keep"}
    ]
    assert result["modification_dispositions"] == []


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


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ["[]", "null", '"stop"'])
async def test_nonmapping_user_action_response_uses_original_text(response):
    agent = _make_pi_agent()
    agent.chat = AsyncMock(return_value=response)

    result = await agent.process_user_input(
        AnalysisState(user_id="test-user"),
        "Continue endothelial analysis.",
        [],
    )

    assert result["action"] == "continue"


@pytest.mark.asyncio
async def test_unsupported_structured_action_falls_back_to_original_stop_text():
    agent = _make_pi_agent()
    agent.chat = AsyncMock(return_value=json.dumps({"action": "stop_later"}))

    result = await agent.process_user_input(
        AnalysisState(user_id="test-user"),
        "Please stop now.",
        [],
    )

    assert result["action"] == "stop"


@pytest.mark.asyncio
async def test_missing_action_with_upload_falls_back_to_analyze_new_data():
    agent = _make_pi_agent()
    agent.chat = AsyncMock(return_value=json.dumps({"response": "Received"}))

    result = await agent.process_user_input(
        AnalysisState(user_id="test-user"),
        "Please inspect this image.",
        [{"type": "image", "filename": "figure.png"}],
    )

    assert result["action"] == "analyze_new_data"
    assert result["file_assignments"]["pathology"][0]["filename"] == "figure.png"


@pytest.mark.asyncio
async def test_supported_structured_action_is_casefolded():
    agent = _make_pi_agent()
    agent.chat = AsyncMock(return_value=json.dumps({"action": "STOP"}))

    result = await agent.process_user_input(
        AnalysisState(user_id="test-user"),
        "continue",
        [],
    )

    assert result["action"] == "stop"
