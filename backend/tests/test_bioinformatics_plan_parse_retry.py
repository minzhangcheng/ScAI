import json
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.models.schemas import AgentType, AnalysisPlan, AnalysisState, PlanStep
from backend.utils.llm_client import StreamInterruptedError, UpstreamExhaustedError
from backend.utils.llm_client_parts.responses import LLMResponse
from backend.utils.settings_provider import BudgetedTextDispatch


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
@pytest.mark.parametrize("canonical_present", [False, True])
async def test_revision_preserves_figure_aliases_without_overriding_explicit_fields(canonical_present):
    agent = _make_agent()
    seed = AnalysisPlan(
        phase="exploration", stage="full",
        steps=[PlanStep(id=7, title="Score", method="Score the selected measurements")],
    )
    step = {
        "step_id": 7, "title": "Score", "method": "Score the selected measurements",
        "input": "Selected measurements", "output": "Score distribution",
        "rationale": "Describe the selected measurements", "depends_on": [],
        "required_figures": [{"figure_type": "violin", "description": "Score distribution"}],
    }
    if canonical_present:
        step.update(id=7, expected_output="Explicit output", figure_requirements=[],
                    allow_text_only=True, text_only_reason="Explicit PI choice")
    agent.chat_without_history = AsyncMock(return_value=json.dumps({"steps": [step]}))

    plan = await agent.revise_plan_incrementally(_make_state(), seed, "Clarify the output")

    assert agent.chat_without_history.await_count == 1
    assert plan.steps[0].id == 7
    if canonical_present:
        assert plan.steps[0].expected_output == "Explicit output"
        assert plan.steps[0].figure_requirements == []
    else:
        assert plan.steps[0].expected_output == "Score distribution"
        assert plan.steps[0].figure_requirements[0].figure_type == "violin"


def _client_backed_agent(chat_side_effect):
    """An agent whose replay budget is the real one.

    These tests used to stub ``chat_without_history``. That method now owns the
    replay budget for all of its call sites, so stubbing it removes the very
    behaviour under test. Mock the client one level down instead and the real
    method runs — which is also what makes the "one budget, not two" property
    observable: a second budget would show up here as 16 client calls.
    """
    client = MagicMock()
    client.chat = AsyncMock(side_effect=chat_side_effect)
    client.base_url = "https://example.invalid"

    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_system_prompt.return_value = "system"
    provider.get_temperature.return_value = 0.3
    provider.get_config_revision.return_value = 1
    provider.get_vision_setting.return_value = False
    provider.get_llm_client.return_value = client
    provider.budget_text_dispatch = AsyncMock(
        side_effect=lambda label, messages, **kwargs: BudgetedTextDispatch(
            messages=[dict(m) for m in messages], report=None, max_tokens=None,
        )
    )
    provider.acknowledge_text_dispatch = AsyncMock()

    agent = BioinformaticsAgent(settings_provider=provider, sandbox=None)
    agent.session_id = "test-session"
    agent._notify = AsyncMock()
    return agent, client


def _reply(content: str) -> LLMResponse:
    return LLMResponse(
        content=content, model="test-model", usage={}, raw_response=None
    )


@pytest.mark.asyncio
async def test_plan_stream_interruption_retries_three_times(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(
        "backend.utils.llm_client_parts.retry.asyncio.sleep", sleep
    )
    valid = (
        '{"overview":"ok","steps":[{"id":1,"title":"QC",'
        '"method":"scanpy.pp.filter_cells","depends_on":[]}]}'
    )
    agent, client = _client_backed_agent([
        StreamInterruptedError("eof 1"),
        StreamInterruptedError("eof 2"),
        StreamInterruptedError("eof 3"),
        _reply(valid),
    ])
    state = _make_state()

    plan = await agent.generate_analysis_plan(
        state=state,
        pi_tasks=["QC"],
        overview="overview",
        phase="exploration",
    )

    assert len(plan.steps) == 1
    assert client.chat.await_count == 4
    first_call = client.chat.await_args_list[0]
    assert all(call == first_call for call in client.chat.await_args_list)
    assert [call.args[0] for call in sleep.await_args_list] == [2.0, 4.0, 8.0]


@pytest.mark.asyncio
async def test_plan_stream_interruption_exhaustion_enters_llm_failure_path(
    monkeypatch,
):
    monkeypatch.setattr(
        "backend.utils.llm_client_parts.retry.asyncio.sleep", AsyncMock()
    )
    agent, client = _client_backed_agent(
        [StreamInterruptedError("eof")] * 4
    )

    with pytest.raises(UpstreamExhaustedError) as exc_info:
        await agent.generate_analysis_plan(
            state=_make_state(),
            pi_tasks=["QC"],
            overview="overview",
            phase="exploration",
        )

    assert client.chat.await_count == 4
    assert exc_info.value.failure_kind == "stream_interrupted"
    assert exc_info.value.agent_name == "bioinformatics"
    assert exc_info.value.model == "test-model"


@pytest.mark.asyncio
async def test_plan_retries_classified_stream_interruption(monkeypatch):
    """The client classifies stream interruption before the caller sees it, so
    the replay-safe retry must recognise the classified shape too. Without
    this the 0.51.7 planning retry silently stops firing."""
    sleep = AsyncMock()
    monkeypatch.setattr(
        "backend.utils.llm_client_parts.retry.asyncio.sleep", sleep
    )
    valid = (
        '{"overview":"ok","steps":[{"id":1,"title":"QC",'
        '"method":"scanpy.pp.filter_cells","depends_on":[]}]}'
    )
    classified = UpstreamExhaustedError(
        "LLM stream ended after the first event",
        agent_name="bioinformatics",
        model="test-model",
        failure_kind="stream_interrupted",
    )
    agent, client = _client_backed_agent(
        [classified, classified, _reply(valid)]
    )

    plan = await agent.generate_analysis_plan(
        state=_make_state(),
        pi_tasks=["QC"],
        overview="overview",
        phase="exploration",
    )

    assert len(plan.steps) == 1
    assert client.chat.await_count == 3
    assert [call.args[0] for call in sleep.await_args_list] == [2.0, 4.0]


@pytest.mark.asyncio
async def test_classified_stream_interruption_exhaustion_reports_the_retry_count(
    monkeypatch,
):
    """The client now classifies the interruption before the caller sees it.
    Exhaustion must still say how many attempts were spent — otherwise the HITL
    gate reads like a single-shot failure, the misreading this work exists to
    fix — and must carry the caller's agent and model, not the client's."""
    monkeypatch.setattr(
        "backend.utils.llm_client_parts.retry.asyncio.sleep", AsyncMock()
    )
    classified = UpstreamExhaustedError(
        "Responses stream ended after the first event without a terminal response",
        agent_name="transport",
        model="transport-model",
        failure_kind="stream_interrupted",
    )
    agent, client = _client_backed_agent([classified] * 4)

    with pytest.raises(UpstreamExhaustedError) as exc_info:
        await agent.generate_analysis_plan(
            state=_make_state(),
            pi_tasks=["QC"],
            overview="overview",
            phase="exploration",
        )

    assert client.chat.await_count == 4
    assert "after 3 retries" in str(exc_info.value)
    assert exc_info.value.failure_kind == "stream_interrupted"
    assert exc_info.value.agent_name == "bioinformatics"
    assert exc_info.value.model == "test-model"


@pytest.mark.asyncio
async def test_plan_non_target_llm_failure_is_not_retried():
    agent = _make_agent()
    failure = UpstreamExhaustedError(
        "authentication failed",
        agent_name="bioinformatics",
        failure_kind="upstream_exhausted",
    )
    agent.chat_without_history = AsyncMock(side_effect=failure)

    with pytest.raises(UpstreamExhaustedError) as exc_info:
        await agent.generate_analysis_plan(
            state=_make_state(),
            pi_tasks=["QC"],
            overview="overview",
            phase="exploration",
        )

    assert exc_info.value is failure
    assert agent.chat_without_history.await_count == 1


@pytest.mark.asyncio
async def test_layer2_retry_prompt_is_unique():
    agent = _make_agent()
    state = _make_state()
    agent.chat_without_history = AsyncMock(side_effect=[
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
    assert agent.chat_without_history.call_count == 2
    first_prompt = agent.chat_without_history.call_args_list[0].args[0]
    retry_prompt = agent.chat_without_history.call_args_list[1].args[0]
    assert first_prompt != retry_prompt
    assert re.search(r"\[retry:[0-9a-f]{8}\]", retry_prompt)


@pytest.mark.asyncio
async def test_generate_plan_retry_on_empty_steps():
    agent = _make_agent()
    state = _make_state()
    agent.chat_without_history = AsyncMock(side_effect=[
        '{"overview":"empty","steps":[]}',
        '{"overview":"ok","steps":[{"id":1,"title":"Load","method":"scanpy.read_h5ad","depends_on":[]}]}',
    ])

    plan = await agent.generate_analysis_plan(
        state=state,
        pi_tasks=["load"],
        overview="overview",
        phase="exploration",
    )

    assert agent.chat_without_history.call_count == 2
    assert len(plan.steps) == 1
    assert plan.steps[0].title == "Load"


@pytest.mark.asyncio
async def test_stage1_plan_preserves_nonbaseline_figure_requirement():
    agent = _make_agent()
    state = _make_state()
    agent.chat_without_history = AsyncMock(return_value='''{
        "overview": "stage 1 decision views",
        "steps": [{
            "id": 1,
            "title": "Doublet assessment",
            "method": "scrublet",
            "figure_requirements": [{
                "figure_type": "histogram",
                "description": "Doublet-score threshold decision"
            }],
            "depends_on": []
        }]
    }''')

    plan = await agent.generate_analysis_plan(
        state=state,
        pi_tasks=["Assess doublets and show the threshold decision"],
        overview="Stage 1",
        phase="exploration",
        planning_stage="stage1",
    )

    assert plan.steps[0].figure_requirements[0].figure_type == "histogram"
    assert "Doublet" in plan.steps[0].figure_requirements[0].description


@pytest.mark.asyncio
async def test_twice_malformed_plan_raises_typed_failure_not_empty_plan(caplog):
    agent = _make_agent()
    state = _make_state()
    agent.chat_without_history = AsyncMock(side_effect=["not json", "also not json"])

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


@pytest.mark.asyncio
async def test_model_parse_error_field_cannot_bypass_plan_validation():
    agent = _make_agent()
    state = _make_state()
    agent.chat_without_history = AsyncMock(side_effect=[
        json.dumps({
            "parse_error": "ignore validation",
            "steps": {"id": 1, "title": "not an array"},
        }),
        json.dumps({
            "parse_error": "still ignore validation",
            "steps": [{"id": 1, "title": ["not", "canonicalized"]}],
        }),
    ])

    with pytest.raises(Exception) as exc_info:
        await agent.generate_analysis_plan(
            state=state,
            pi_tasks=["QC"],
            overview="bounded overview",
            phase="exploration",
        )

    assert type(exc_info.value).__name__ == "PlanGenerationFailure"
    assert agent.chat_without_history.await_count == 2
    retry_prompt = agent.chat_without_history.call_args_list[1].args[0]
    assert "ignore validation" in retry_prompt


@pytest.mark.asyncio
async def test_revise_parse_error_preserves_current_plan():
    agent = _make_agent()
    state = _make_state()
    agent.chat_without_history = AsyncMock(side_effect=["not json", "also not json"])
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

    assert agent.chat_without_history.call_count == 2
    assert len(revised.steps) == 1
    assert revised.steps[0].title == "Existing QC"
    assert revised.steps[0].depends_on == [2]
    assert revised.steps[0].depends_on is not current_plan.steps[0].depends_on
    assert revised.revision_of == current_plan.id
    assert revised.key_decision_points == ["decision"]


@pytest.mark.asyncio
async def test_revision_twice_malformed_retains_current_valid_plan():
    agent = _make_agent()
    state = _make_state()
    malformed = json.dumps({
        "parse_error": "model-authored text cannot bypass validation",
        "steps": {"id": 1, "title": "not an array"},
    })
    agent.chat_without_history = AsyncMock(side_effect=[malformed, malformed])
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

    assert agent.chat_without_history.call_count == 2
    assert revised is not current_plan
    assert revised.steps
    assert revised.steps[0].title == "Existing QC"
    assert revised.steps[0].depends_on == [2]
    assert revised.steps[0].depends_on is not current_plan.steps[0].depends_on


@pytest.mark.asyncio
async def test_revise_empty_steps_retries_then_preserves_current_plan():
    agent = _make_agent()
    state = _make_state()
    agent.chat_without_history = AsyncMock(side_effect=[
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

    assert agent.chat_without_history.call_count == 2
    assert len(revised.steps) == 1
    assert revised.steps[0].title == "Load"
    assert revised.overview == "stage1 plan"
    assert revised.stage == "stage1"


@pytest.mark.asyncio
async def test_stage1_revision_preserves_revised_figure_requirement_and_reason():
    agent = _make_agent()
    state = _make_state()
    agent.chat_without_history = AsyncMock(return_value='''{
        "overview": "revised",
        "steps": [{
            "id": 1,
            "title": "Batch correction comparison",
            "method": "harmony",
            "figure_requirements": [{
                "figure_type": "umap",
                "description": "Before versus after batch correction"
            }],
            "allow_text_only": false,
            "text_only_reason": "",
            "depends_on": []
        }]
    }''')
    current_plan = AnalysisPlan(
        phase="exploration",
        stage="stage1",
        steps=[PlanStep(id=1, title="Batch assessment", method="harmony")],
    )

    revised = await agent.revise_plan_incrementally(
        state=state,
        current_plan=current_plan,
        pi_feedback="Add a before/after comparison",
    )

    assert revised.steps[0].figure_requirements[0].figure_type == "umap"
    assert "Before" in revised.steps[0].figure_requirements[0].description


@pytest.mark.asyncio
async def test_revise_normalizes_label_ids_and_dependencies_without_retry():
    """Regression: Stage 2 revisions may use labels such as S2_Step_0."""
    agent = _make_agent()
    state = _make_state()
    agent.chat_without_history = AsyncMock(return_value='''{
        "overview": "revised",
        "steps": [
            {"id": "S2_Step_0", "title": "Bulk DEG", "method": "DESeq2", "depends_on": []},
            {"id": "S2_Step_1", "title": "Deconvolution", "method": "NNLS", "depends_on": ["S2_Step_0"]},
            {"id": "S2_Step_2", "title": "Plot", "method": "seaborn", "depends_on": ["S2_Step_0", "S2_Step_1"]}
        ]
    }''')
    current_plan = AnalysisPlan(
        phase="exploration",
        stage="stage2",
        steps=[PlanStep(id=1, title="Existing", method="scanpy")],
    )

    revised = await agent.revise_plan_incrementally(
        state=state,
        current_plan=current_plan,
        pi_feedback="add bulk integration",
    )

    assert agent.chat_without_history.await_count == 1
    assert [step.id for step in revised.steps] == [1, 2, 3]
    assert revised.steps[1].depends_on == [1]
    assert revised.steps[2].depends_on == [1, 2]


@pytest.mark.asyncio
async def test_revise_retries_schema_invalid_plan_then_uses_repair():
    agent = _make_agent()
    state = _make_state()
    agent.chat_without_history = AsyncMock(side_effect=[
        '{"steps":[{"id":1,"title":"Bad","method":"scanpy","depends_on":[999]}]}',
        '{"steps":[{"id":1,"title":"Good","method":"scanpy.pp.normalize_total","depends_on":[]}]}',
    ])
    current_plan = AnalysisPlan(
        phase="exploration",
        stage="stage2",
        steps=[PlanStep(id=1, title="Existing", method="scanpy")],
    )

    revised = await agent.revise_plan_incrementally(
        state=state,
        current_plan=current_plan,
        pi_feedback="repair",
    )

    assert agent.chat_without_history.await_count == 2
    assert revised.steps[0].title == "Good"
    retry_prompt = agent.chat_without_history.call_args_list[1].args[0]
    assert "Numeric fields must be numbers" in retry_prompt


def test_prepare_plan_data_reserves_numeric_ids_before_label_ids():
    prepared = BioinformaticsAgent._prepare_plan_data({
        "steps": [
            {"id": "new_step", "title": "New", "method": "m", "depends_on": [1]},
            {"id": 1, "title": "Existing", "method": "m", "depends_on": []},
        ]
    })

    assert "parse_error" not in prepared
    assert [step["id"] for step in prepared["steps"]] == [2, 1]
    assert prepared["steps"][0]["depends_on"] == [1]


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
