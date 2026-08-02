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
async def test_generate_plan_retry_fallback(caplog):
    agent = _make_agent()
    state = _make_state()
    agent.chat_without_history = AsyncMock(side_effect=["not json", "also not json"])

    plan = await agent.generate_analysis_plan(
        state=state,
        pi_tasks=["QC"],
        overview="fallback overview",
        phase="exploration",
    )

    assert agent.chat_without_history.call_count == 2
    assert plan.steps == []
    assert plan.overview == "fallback overview"
    assert "retry also failed" in caplog.text


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
        '{"steps":[{"id":1,"title":"Bad","method":[],"depends_on":[]}]}',
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
