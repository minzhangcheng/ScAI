from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.pi_agent import PIAgent
from backend.models.schemas import (
    AnalysisPlan,
    AnalysisState,
    FigureRequirement,
    PIDecisionRecord,
    PlanStep,
    PublicationFigure,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["test-model", "openai/gpt-5.6-sol"])
async def test_figure_review_receives_the_linked_plan_deliverable(model):
    provider = MagicMock()
    provider.get_model.return_value = model
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.2
    provider.get_vision_setting.return_value = True
    vision = AsyncMock(return_value=SimpleNamespace(content=(
        '{"approved":true,"quality_score":8,"feedback":"ok",'
        '"strengths":[],"issues":[],"suggestions":[],"must_fix":[]}'
    )))
    provider.get_llm_client.return_value = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1",
        chat_with_vision=vision,
        set_context=lambda *_args: None,
    )
    agent = PIAgent(settings_provider=provider)
    agent.set_session("review-plan-context")
    agent._set_llm_context = lambda: None

    state = AnalysisState(
        user_id="u", name="Project", initial_instruction="Analyze data",
        current_round=4,
    )
    state.current_plan = AnalysisPlan(
        id="plan-current",
        phase="innovation",
        steps=[PlanStep(
            id=7,
            title="Differential expression summary",
            method="Fit the declared differential-expression model",
            figure_requirements=[FigureRequirement(
                figure_type="volcano",
                description="Effect size versus adjusted significance",
            )],
        )],
    )
    figure = PublicationFigure(
        name="de_result",
        description="DE result",
        figure_type="volcano",
        png_path="/tmp/de_result.jpg",
        pdf_path="/tmp/de_result.pdf",
        plan_step_id=7,
        plan_id="plan-current",
        round_number=4,
    )
    state.pi_decision_records = [
        PIDecisionRecord(
            question="Which contrast should the volcano plot show?",
            response="Use the responder-versus-control contrast and label CXCL10.",
            plan_id="plan-current",
            round_number=4,
        ),
        PIDecisionRecord(
            question="Unrelated later-round choice",
            response="Use an unrelated panel.",
            plan_id="plan-other",
            round_number=4,
        ),
    ]

    result = await agent.review_figure(state, figure, "base64")

    assert result["approved"] is True
    assert vision.await_args.kwargs["prefer_stream"] is True
    prompt = vision.await_args.kwargs["text_prompt"]
    assert "Linked plan: plan-current" in prompt
    assert "Linked step 7: Differential expression summary" in prompt
    assert "volcano x1: Effect size versus adjusted significance" in prompt
    assert "Later PI decisions applicable to this plan and round" in prompt
    assert "Use the responder-versus-control contrast and label CXCL10" in prompt
    assert "Use an unrelated panel" not in prompt
    assert "merely attractive plot of the wrong result" in prompt


    # A new render and later PI decision must not invalidate the review rules
    # or the reusable project background at the beginning of the request.
    figure.name = "de_result_revised"
    figure.description = "Updated labels"
    figure.revision_count = 2
    state.pi_decision_records[0].response = "Label the revised contrast clearly."
    await agent.review_figure(state, figure, "revised-base64")
    revised_prompt = vision.await_args.kwargs["text_prompt"]
    prefix, first_tail = prompt.split("**Figure Information:**", 1)
    revised_prefix, revised_tail = revised_prompt.split("**Figure Information:**", 1)
    assert prefix == revised_prefix
    assert "**Review Criteria:**" in prefix
    assert "**Respond in JSON format:**" in prefix
    assert "**Critical review instructions:**" in prefix
    assert "Project: Project" in prefix
    assert prefix.index("**Respond in JSON format:**") < prefix.index("**Research Context:**")
    assert "Name: de_result\n" in first_tail
    assert "Name: de_result_revised\n" in revised_tail
    assert "Updated labels" in revised_tail
    assert "Label the revised contrast clearly." in revised_tail
    assert "REVISION LENIENCY POLICY" in revised_tail
    assert "REVISION LENIENCY POLICY" not in prompt
    assert vision.await_args.kwargs["images"] == ["revised-base64"]
    for call in vision.await_args_list:
        if model == "test-model":
            assert "prompt_cache_plan" not in call.kwargs
        else:
            plan = call.kwargs["prompt_cache_plan"]
            assert plan.responses_prefix == prefix
            assert plan.prefer_anthropic_messages is False
            assert "**Figure Information:**" not in plan.responses_prefix
