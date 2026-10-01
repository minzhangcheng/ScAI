import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.models.schemas import AnalysisPlan, AnalysisState, PlanStep
from backend.utils.llm_client import UpstreamExhaustedError


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
@pytest.mark.parametrize("revise", [False, True])
async def test_truncated_plan_requires_complete_retry_before_acceptance(revise):
    agent = _make_agent()
    truncated = '{"steps":[{"id":1,"title":"Partial","method":"unfinished'
    complete = {"steps": [
        {"id": 1, "title": "First", "method": "m"},
        {"id": 2, "title": "Second", "method": "m", "depends_on": [1]},
    ]}
    agent.chat_without_history = AsyncMock(side_effect=[
        truncated, "```json\n" + json.dumps(complete) + "\n```",
    ])
    if revise:
        plan = await agent.revise_plan_incrementally(
            _make_state(),
            AnalysisPlan(phase="exploration", steps=[
                PlanStep(id=1, title="Original", method="m"),
            ]),
            "Revise the plan",
        )
    else:
        plan = await agent.generate_analysis_plan(
            _make_state(), ["Analyse measurements"], "Strategy", "exploration",
        )

    assert agent.chat_without_history.await_count == 2
    assert [step.title for step in plan.steps] == ["First", "Second"]
    assert "missing a closed JSON object" in agent.chat_without_history.call_args.args[0]
    assert "model signal" not in agent.chat_without_history.call_args.args[0]


@pytest.mark.asyncio
async def test_truncated_revision_and_retry_preserve_complete_previous_plan():
    agent = _make_agent()
    current = AnalysisPlan(phase="exploration", steps=[
        PlanStep(id=1, title="Original first", method="m"),
        PlanStep(id=2, title="Original second", method="m", depends_on=[1]),
    ])
    original = current.model_dump()
    agent.chat_without_history = AsyncMock(return_value=(
        '{"steps":[{"id":1,"title":"Partial","method":"m"},'
        '{"id":2,"title":"Unfinished'
    ))

    with pytest.raises(UpstreamExhaustedError) as caught:
        await agent.revise_plan_incrementally(_make_state(), current, "Revise")

    assert agent.chat_without_history.await_count == 2
    assert caught.value.failure_kind == "parse_error"
    assert current.model_dump() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("response, detail", [
    ("I'll inspect the data before revising the plan.", "missing a closed JSON object"),
    ('{"steps": [???]}', "Failed to extract valid JSON"),
    ('{"steps": null}', "steps must be a JSON array"),
    ('{"parse_error": "model error", "_incomplete_plan_json": true}', "model signal: model error"),
])
async def test_twice_malformed_plan_raises_typed_failure_not_empty_plan(caplog, monkeypatch, response, detail):
    from backend.utils import formatting

    monkeypatch.setattr(formatting, "_JSON_REPAIR_AVAILABLE", False)
    agent = _make_agent()
    state = _make_state()
    agent.chat_without_history = AsyncMock(return_value=response)

    with pytest.raises(Exception) as exc_info:
        await agent.generate_analysis_plan(
            state=state,
            pi_tasks=["QC"],
            overview="fallback overview",
            phase="exploration",
        )

    assert agent.chat_without_history.call_count == 2
    assert type(exc_info.value).__name__ == "PlanGenerationFailure"
    assert "retry also failed" in caplog.text
    assert detail in str(exc_info.value)
    assert detail in agent.chat_without_history.call_args.args[0]
    if "model signal" not in detail:
        assert "model signal" not in str(exc_info.value)
        assert "model signal" not in agent.chat_without_history.call_args.args[0]


def test_host_plan_diagnosis_survives_repeated_normalization():
    agent = _make_agent()
    parsed = agent._parse_plan_response("I'll inspect the data first.")
    prepared = agent._prepare_plan_data(parsed)

    assert agent._prepare_plan_data(prepared) == prepared
    assert "model signal" not in prepared["parse_error"]
    assert json.loads(json.dumps(prepared))["parse_error"] == str(prepared["parse_error"])


@pytest.mark.asyncio
async def test_revise_preserves_completed_step_content():
    agent = _make_agent()
    state = _make_state()
    agent.chat_without_history = AsyncMock(return_value='''{
        "steps": [
            {"id": 1, "title": "Mutated", "method": "wrong", "rationale": "wrong", "depends_on": ["new_step"]},
            {"id": "new_step", "title": "New prerequisite", "method": "correct", "depends_on": []}
        ]
    }''')
    current_plan = AnalysisPlan(
        phase="exploration",
        stage="stage2",
        steps=[PlanStep(
            id=1,
            title="Completed original",
            method="scanpy.pp.normalize_total",
            rationale="locked",
            status="completed",
        )],
    )

    revised = await agent.revise_plan_incrementally(
        state=state,
        current_plan=current_plan,
        pi_feedback="insert prerequisite",
    )

    completed = next(step for step in revised.steps if step.id == 1)
    assert completed.title == "Completed original"
    assert completed.method == "scanpy.pp.normalize_total"
    assert completed.rationale == "locked"
    assert completed.status == "completed"
    assert completed.depends_on == [2]
