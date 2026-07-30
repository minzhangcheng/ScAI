"""Direction coverage must also answer what a plan actually addresses.

Recording that a round ran for a direction says nothing about whether the plan
tested what the direction was selected to resolve, so a genericised plan can be
booked as covering a direction it never examines.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.models.schema_parts.outputs import ExplorationReport
from backend.models.schemas import AnalysisPlan, AnalysisState, PlanStep
from backend.workflow.direction_coverage import (
    plan_text_for_coverage,
    uncovered_direction_requirements,
)

DIRECTION = "Determine whether perturbation X drives program Y"

CANDIDATE = {
    "direction": DIRECTION,
    "key_uncertainties": [
        "Whether the remodelled state corresponds to modeA, modeB, or reversible change.",
    ],
    "validation_plan": {
        "next_analysis": ["Compare rankings against public gammareference datasets."],
    },
}


def _state() -> AnalysisState:
    state = AnalysisState(user_id="u")
    state.current_round_direction = DIRECTION
    state.exploration_report = ExplorationReport(research_direction_candidates=[CANDIDATE])
    return state


def _plan(scoring_input: str) -> AnalysisPlan:
    return AnalysisPlan(
        phase="innovation",
        stage="full",
        overview="Hepatocyte-centred analysis",
        steps=[
            PlanStep(
                id=1,
                title="Score Modules",
                method="scanpy.tl.score_genes",
                input=scoring_input,
                expected_output="Per-nucleus scores",
                rationale="Summarises programs.",
                depends_on=[],
            )
        ],
        key_decision_points=[],
        expected_runtime="~1 step",
    )


def test_coverage_text_reads_every_step_field():
    text = plan_text_for_coverage(_plan("modeA and modeB gene sets"))
    assert "modeA and modeB gene sets" in text
    assert "Per-nucleus scores" in text
    assert "Summarises programs." in text
    assert plan_text_for_coverage(None) == ""


def test_generic_plan_is_reported_as_not_addressing_the_direction():
    gaps = uncovered_direction_requirements(
        _state(), _plan("curated stress and death-associated gene modules")
    )
    assert len(gaps) == 2
    assert any("modeA" in g for g in gaps)
    assert any("gammareference" in g for g in gaps)


def test_specific_plan_closes_the_gap():
    covered = _plan(
        "Gene sets for modeA and modeB to test whether the remodelled state "
        "corresponds to reversible change; compared against public gammareference "
        "rankings datasets"
    )
    assert uncovered_direction_requirements(_state(), covered) == []


def test_no_candidate_means_no_claim():
    bare = AnalysisState(user_id="u")
    bare.current_round_direction = DIRECTION
    assert uncovered_direction_requirements(bare, _plan("anything")) == []


@pytest.mark.asyncio
async def test_gaps_are_reported_to_the_user_at_the_plan_gate():
    from backend.workflow import SingleCellAnalysisWorkflow

    workflow = object.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "test-session"
    workflow._send_message = AsyncMock()

    analysis_state = _state()
    analysis_state.current_plan = _plan("curated stress and death-associated gene modules")
    await workflow._report_direction_coverage_gaps(analysis_state)

    workflow._send_message.assert_awaited_once()
    body = workflow._send_message.await_args.args[1]
    assert "not addressed by the plan" in body
    assert "modeA" in body


@pytest.mark.asyncio
async def test_no_message_when_the_plan_covers_the_direction():
    from backend.workflow import SingleCellAnalysisWorkflow

    workflow = object.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "test-session"
    workflow._send_message = AsyncMock()

    analysis_state = _state()
    analysis_state.current_plan = None
    await workflow._report_direction_coverage_gaps(analysis_state)
    workflow._send_message.assert_not_awaited()

    analysis_state.current_plan = _plan(
        "Gene sets for modeA and modeB to test whether the remodelled state "
        "corresponds to reversible change; compared against public gammareference "
        "rankings datasets"
    )
    await workflow._report_direction_coverage_gaps(analysis_state)
    workflow._send_message.assert_not_awaited()
