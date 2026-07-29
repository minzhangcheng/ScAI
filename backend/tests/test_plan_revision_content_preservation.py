"""A plan revision must not quietly strip the content it was asked to keep.

The revision prompt used to render each step as ``title (method)`` and then ask
for a FULL plan back, so input/expected_output/rationale were regenerated from a
name alone. PI feedback accumulated every round while the plan was re-derived
every round, so caution compounded and specifics decayed: after several review
rounds a plan retained its methodology and none of the biology that motivated
it. These tests pin the pieces that stop that.
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.bioinformatics.planning import (
    render_pi_modifications,
    render_plan_steps,
)
from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.models.schema_parts.outputs import ExplorationReport
from backend.models.schemas import AnalysisPlan, AnalysisState, PlanStep
from backend.utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY

DIRECTION = "Determine whether perturbation X drives program Y"

CANDIDATE = {
    "direction": DIRECTION,
    "main_hypothesis": "Perturbation X activates program Y",
    "key_uncertainties": ["Whether the state corresponds to modeA or modeB."],
    "validation_plan": {"next_analysis": ["Score nuclei for modeA and modeB separately."]},
    "key_genes": ["Geneaa1"],
}


def _agent() -> BioinformaticsAgent:
    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_system_prompt.return_value = "system"
    provider.get_temperature.return_value = 0.3
    provider.get_vision_setting.return_value = False
    provider.get_llm_client.return_value = MagicMock()
    agent = BioinformaticsAgent(settings_provider=provider, sandbox=None)
    agent.session_id = "test-session"
    return agent


def _state() -> AnalysisState:
    state = AnalysisState(user_id="test-user")
    state.initial_instruction = (
        "Objective: establish whether perturbation X promotes program Y via modeA or modeB."
    )
    state.current_round_direction = DIRECTION
    state.exploration_report = ExplorationReport(research_direction_candidates=[CANDIDATE])
    return state


def _rich_plan() -> AnalysisPlan:
    return AnalysisPlan(
        phase="innovation",
        stage="full",
        overview="Score prespecified modeA, modeB, and modeC modules across states.",
        steps=[
            PlanStep(
                id=1,
                title="Load Object",
                method="scanpy.read_h5ad",
                input="Integrated object",
                expected_output="Loaded AnnData",
                rationale="Downstream steps need the integrated object.",
                depends_on=[],
            ),
            PlanStep(
                id=2,
                title="Score Stress Modules",
                method="scanpy.tl.score_genes",
                input="Curated gene sets for modeA, modeB, modeC, and factorQ activation",
                expected_output="Per-nucleus modeA, modeB, and modeC scores in obs",
                rationale="Distinguishes modeA from modeB in the remodelled compartment.",
                depends_on=[1],
            ),
        ],
        key_decision_points=["Approve the modeA and modeB gene sets before scoring."],
        expected_runtime="~2 steps",
    )


def _plan_json(*, generic: bool) -> str:
    if generic:
        steps = [
            {
                "id": 1, "title": "Load Object", "method": "scanpy.read_h5ad",
                "input": "Integrated object", "expected_output": "Loaded AnnData",
                "rationale": "Downstream steps need the integrated object.",
                "depends_on": [],
            },
            {
                "id": 2, "title": "Score Stress Modules", "method": "scanpy.tl.score_genes",
                "input": "Curated stress and death-associated gene modules",
                "expected_output": "Per-nucleus module scores in obs",
                "rationale": "Summarises programs without overclaiming execution.",
                "depends_on": [1],
            },
        ]
        overview = "Descriptive module scoring with cautious interpretation."
        decisions = ["Approve the gene modules before scoring."]
    else:
        steps = [
            {
                "id": 1, "title": "Load Object", "method": "scanpy.read_h5ad",
                "input": "Integrated object", "expected_output": "Loaded AnnData",
                "rationale": "Downstream steps need the integrated object.",
                "depends_on": [],
            },
            {
                "id": 2, "title": "Score Stress Modules", "method": "scanpy.tl.score_genes",
                "input": "Curated gene sets for modeA, modeB, modeC, and factorQ activation",
                "expected_output": "Per-nucleus modeA, modeB, and modeC scores in obs",
                "rationale": (
                    "Distinguishes modeA from modeB; scores report susceptibility, "
                    "not execution. Geneaa1 is reported explicitly."
                ),
                "depends_on": [1],
            },
        ]
        overview = "Score prespecified modeA, modeB, and modeC modules across states."
        decisions = ["Approve the modeA and modeB gene sets before scoring."]
    return json.dumps({
        "overview": overview,
        "steps": steps,
        "key_decision_points": decisions,
        "expected_runtime": "~2 steps",
    })


# --- rendering -------------------------------------------------------------

def test_rendered_steps_carry_every_field():
    block = render_plan_steps(_rich_plan().steps, "TO REVISE", budget=100_000)
    assert "Curated gene sets for modeA, modeB, modeC" in block
    assert "Per-nucleus modeA, modeB, and modeC scores" in block
    assert "Distinguishes modeA from modeB" in block
    assert "Depends on: [1]" in block


def test_step_tag_may_vary_per_step():
    """A mixed list must still show which steps are locked."""
    plan = _rich_plan()
    plan.steps[0].status = "completed"
    block = render_plan_steps(
        plan.steps, lambda s: getattr(s, "status", "pending"), budget=100_000
    )
    assert "[completed] Step 1:" in block
    assert "[pending] Step 2:" in block


def test_rendering_shrinks_fields_rather_than_dropping_steps():
    steps = [
        PlanStep(
            id=i,
            title=f"Step {i}",
            method="scanpy.tl.score_genes",
            input="x" * 900,
            expected_output="y" * 900,
            rationale="z" * 900,
            depends_on=[],
        )
        for i in range(1, 31)
    ]
    block = render_plan_steps(steps, "TO REVISE", budget=4_000)
    for i in range(1, 31):
        assert f"Step {i}:" in block
    assert render_plan_steps([], "TO REVISE", budget=100) == ""


def test_pi_modifications_render_from_both_shapes():
    flat = render_pi_modifications([{"step_id": 14, "suggestion": "Keep modeA scoring."}])
    assert "Step 14" in flat and "Keep modeA scoring." in flat

    wrapped = render_pi_modifications([
        {"round": 1, "modifications": [{"step_id": 14, "suggestion": "Keep modeA scoring."}]},
        {"round": 2, "modifications": []},
    ])
    assert "[Round 1]" in wrapped and "Keep modeA scoring." in wrapped
    assert render_pi_modifications([]) == ""
    assert render_pi_modifications(None) == ""


# --- revision prompt -------------------------------------------------------

@pytest.mark.asyncio
async def test_revision_prompt_carries_everything_the_revision_must_preserve():
    agent = _agent()
    agent.chat_without_history = AsyncMock(return_value=_plan_json(generic=False))

    await agent.revise_plan_incrementally(
        state=_state(),
        current_plan=_rich_plan(),
        pi_feedback="Treat cell-level tests as descriptive.",
        pi_tasks=["Score modules including modeA, modeB, and modeC programs."],
        overview="Round 2 strategy",
        pi_modifications=[
            {"round": 1, "modifications": [
                {"step_id": 2, "suggestion": "Clarify that modeA scores are susceptibility."},
            ]},
        ],
    )

    prompt = agent.chat_without_history.call_args_list[0].args[0]
    # Research context: objective, direction, and the direction's own payload.
    assert "perturbation X promotes program Y" in prompt
    assert DIRECTION in prompt
    assert "Whether the state corresponds to modeA or modeB." in prompt
    assert "Score nuclei for modeA and modeB separately." in prompt
    assert "Geneaa1" in prompt
    # The PI's tasks stay binding on a revision.
    assert "Score modules including modeA, modeB, and modeC programs." in prompt
    assert "Round 2 strategy" in prompt
    # Structured step-level asks, not just the prose feedback.
    assert "Clarify that modeA scores are susceptibility." in prompt
    assert "Treat cell-level tests as descriptive." in prompt
    # Full step payload, not title + method.
    assert "Curated gene sets for modeA, modeB, modeC" in prompt
    assert MECHANISM_EVIDENCE_POLICY in prompt


@pytest.mark.asyncio
async def test_revision_keeps_working_without_the_new_context():
    """Callers that pass only the original arguments must still succeed."""
    agent = _agent()
    agent.chat_without_history = AsyncMock(return_value=_plan_json(generic=False))
    revised = await agent.revise_plan_incrementally(
        state=_state(),
        current_plan=_rich_plan(),
        pi_feedback="tighten it",
    )
    assert len(revised.steps) == 2


# --- retention guard -------------------------------------------------------

@pytest.mark.asyncio
async def test_genericised_revision_triggers_one_repair_and_the_repair_wins(caplog):
    agent = _agent()
    agent.chat_without_history = AsyncMock(side_effect=[
        _plan_json(generic=True),
        _plan_json(generic=False),
    ])

    revised = await agent.revise_plan_incrementally(
        state=_state(),
        current_plan=_rich_plan(),
        pi_feedback="Separate activity from execution.",
        pi_tasks=["Score modules including modeA, modeB, and modeC programs."],
    )

    assert agent.chat_without_history.call_count == 2
    repair_prompt = agent.chat_without_history.call_args_list[1].args[0]
    assert "Retention Check FAILED" in repair_prompt
    assert "modea" in repair_prompt.lower()
    scoring_step = next(s for s in revised.steps if s.id == 2)
    assert "modeA" in scoring_step.input
    assert "modeC" in scoring_step.input


@pytest.mark.asyncio
async def test_faithful_revision_costs_no_extra_call():
    agent = _agent()
    agent.chat_without_history = AsyncMock(return_value=_plan_json(generic=False))

    await agent.revise_plan_incrementally(
        state=_state(),
        current_plan=_rich_plan(),
        pi_feedback="Separate activity from execution.",
        pi_tasks=["Score modules including modeA, modeB, and modeC programs."],
    )

    assert agent.chat_without_history.call_count == 1


@pytest.mark.asyncio
async def test_repair_that_recovers_nothing_is_discarded():
    agent = _agent()
    agent.chat_without_history = AsyncMock(side_effect=[
        _plan_json(generic=True),
        _plan_json(generic=True),
    ])

    revised = await agent.revise_plan_incrementally(
        state=_state(),
        current_plan=_rich_plan(),
        pi_feedback="Separate activity from execution.",
        pi_tasks=["Score modules including modeA, modeB, and modeC programs."],
    )

    assert agent.chat_without_history.call_count == 2
    assert len(revised.steps) == 2
    assert revised.overview == "Descriptive module scoring with cautious interpretation."


@pytest.mark.asyncio
async def test_unparseable_repair_falls_back_to_the_original_revision():
    agent = _agent()
    agent.chat_without_history = AsyncMock(side_effect=[
        _plan_json(generic=True),
        "not json at all",
    ])

    revised = await agent.revise_plan_incrementally(
        state=_state(),
        current_plan=_rich_plan(),
        pi_feedback="Separate activity from execution.",
        pi_tasks=["Score modules including modeA, modeB, and modeC programs."],
    )

    assert len(revised.steps) == 2
    assert revised.overview == "Descriptive module scoring with cautious interpretation."


# --- workflow wiring -------------------------------------------------------

@pytest.mark.asyncio
async def test_planning_dialogue_forwards_tasks_and_step_level_modifications():
    from backend.workflow import SingleCellAnalysisWorkflow

    workflow = object.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "test-session"
    workflow._send_message = AsyncMock()
    workflow._build_planning_context = MagicMock(return_value=None)
    workflow._require_core_agent = AsyncMock(return_value=None)

    first_revision = _rich_plan()
    first_revision.overview = "first revision"
    second_revision = _rich_plan()
    second_revision.overview = "second revision"
    second_revision.steps[1].title = "Score Stress Modules (revised)"

    bioinfo = MagicMock()
    bioinfo.reset_history = MagicMock()
    bioinfo.revise_plan_incrementally = AsyncMock(
        side_effect=[first_revision, second_revision]
    )
    workflow.bioinformatics_agent = bioinfo
    workflow.pi_agent = MagicMock()
    workflow.pi_agent.review_bioinfo_plan = AsyncMock(side_effect=[
        {
            "decision": "revise",
            "feedback": "Separate activity from execution.",
            "modifications": [{"step_id": 2, "suggestion": "Keep modeA and modeB scoring."}],
        },
        {"decision": "approved", "feedback": "ok", "modifications": []},
    ])

    analysis_state = _state()
    analysis_state.id = "test-session"
    analysis_state.current_plan = _rich_plan()
    state = {
        "analysis_state": analysis_state,
        "pi_instructions": {
            "bioinformatics_tasks": ["Score modeA, modeB, and modeC programs."],
            "overview": "Round 2 strategy",
        },
        "planning_stage": "full",
    }

    await workflow._planning_dialogue(state)

    assert bioinfo.revise_plan_incrementally.await_count == 2
    kwargs = bioinfo.revise_plan_incrementally.await_args.kwargs
    assert kwargs["pi_tasks"] == ["Score modeA, modeB, and modeC programs."]
    assert kwargs["overview"] == "Round 2 strategy"
    forwarded = json.dumps(kwargs["pi_modifications"])
    assert "Keep modeA and modeB scoring." in forwarded
