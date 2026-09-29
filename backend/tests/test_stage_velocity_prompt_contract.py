"""Prompt contracts for carrying deferred Loom inputs beyond Stage 1."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.bioinformatics.shared import (
    _build_swallowed_error_feedback,
    _stage_guidance_text,
)
from backend.agents.bioinformatics.tool_schemas import TOOLS
from backend.agents.pi_agent import PIAgent
from backend.models.schemas import AnalysisPlan, AnalysisState, PlanStep
from backend.workflow.agent_nodes.shared import build_phase_instruction


def _make_pi_agent() -> PIAgent:
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    provider.get_llm_client.return_value = MagicMock()
    agent = PIAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


def test_stage_guidance_defers_loom_without_forcing_residency_or_method():
    stage1 = _stage_guidance_text("stage1")
    stage2 = _stage_guidance_text("stage2")
    later_round = _stage_guidance_text("full")
    normalized_stage1 = " ".join(stage1.split())
    normalized_stage2 = " ".join(stage2.split())
    normalized_later_round = " ".join(later_round.split())

    assert "whether or not a profiling object remains live in memory" in normalized_stage1
    assert "Deferral is a scope decision" in stage1
    assert "original registered Loom inputs be opened read-only" in stage2
    assert "accepted atlas alone do not establish" in stage2
    assert "must not require scVelo or any other specific algorithm" in normalized_stage2
    assert "different analysis" in stage2
    assert "not host pass/fail thresholds" in normalized_stage2
    assert "register_method_decision" in stage2
    assert "velocity or diagnostic figure" in stage2
    assert "original registered Loom inputs be opened read-only" in later_round
    assert "must not require scVelo or any other specific algorithm" in normalized_later_round


@pytest.mark.parametrize(("planning_stage", "analysis_phase"), [
    ("stage2", "exploration"),
    ("full", "innovation"),
])
def test_post_stage1_executor_inspects_original_loom_without_mandating_scvelo(
    planning_stage, analysis_phase,
):
    instruction = build_phase_instruction(
        planning_stage, analysis_phase, has_bulk=False, has_scrna=True, direction="",
    )
    normalized_instruction = " ".join(instruction.split())

    assert "inspect the original registered files read-only" in instruction
    assert "absence of layers on the current atlas alone is not evidence" in instruction
    assert "scVelo is an option to evaluate, not a required algorithm" in instruction
    assert "not an automatic replacement for RNA velocity" in normalized_instruction
    assert "phase portraits" in instruction
    assert "it is not by itself proof" in normalized_instruction
    assert "register_method_decision" in instruction
    assert "backend='threading'" in instruction
    assert "Do not apply a host-prescribed threshold" in instruction


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["stage2", "full"])
async def test_pi_post_stage1_review_checks_original_loom_without_requiring_algorithm(stage):
    agent = _make_pi_agent()
    agent.chat = AsyncMock(return_value=json.dumps({
        "decision": "approved",
        "feedback": "The input inspection and method choice are explicit.",
        "modifications": [],
        "adata_source": None,
        "checkpoint_lookup": None,
    }))
    plan = AnalysisPlan(
        phase="exploration" if stage == "stage2" else "innovation",
        stage=stage,
        overview="Evaluate a directional splicing question",
        steps=[PlanStep(id=1, title="Inspect inputs", method="read-only inspection")],
    )

    await agent.review_bioinfo_plan(AnalysisState(user_id="test"), plan)

    prompt = agent.chat.await_args.args[0]
    normalized_prompt = " ".join(prompt.split())
    assert "inspects the original registered Loom inputs" in prompt
    assert "accepted atlas lacks spliced / unspliced layers" in prompt
    assert "does not mandate scVelo or any other algorithm" in prompt
    assert "selected, not-selected, or inconclusive methods" in normalized_prompt
    assert "Do not prescribe a velocity method" in prompt


def test_velocity_error_feedback_does_not_call_dpt_an_equivalent_fallback():
    feedback = _build_swallowed_error_feedback("PermissionError", "")

    assert "consult the PI" in feedback
    assert "record the observed diagnostics and method decision" in feedback
    assert "must never be presented as equivalent fallbacks" in feedback
    assert "fall back to an equivalent method" not in feedback


def test_method_decision_tool_is_generic_advisory_provenance():
    tool = next(
        item["function"]
        for item in TOOLS
        if item["function"]["name"] == "register_method_decision"
    )
    required = tool["parameters"]["required"]

    assert "never controls workflow progress" in tool["description"]
    assert "modality" in required
    assert "decision_status" in required
    assert "input_profile_id" not in required
    assert "diagnostics" not in required
