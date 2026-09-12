"""Round/plan preservation is controlled only by the one-shot explicit-start flag.

Durable restore origin (``last_restored_snapshot_id``) identifies parent/child
lineage. It must never suppress the round increment or preserve the previous
plan/cache when a restored semantic action continues into the next round.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.models.schemas import (
    AgentRoundCache,
    AnalysisPhase,
    AnalysisPlan,
    AnalysisState,
    PlanStep,
)


def _make_workflow():
    from backend.workflow import SingleCellAnalysisWorkflow

    wf = object.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "test-session"
    wf._send_message = AsyncMock()
    wf._send_state_sync = AsyncMock()
    wf._set_status = MagicMock()
    wf.pi_agent = MagicMock()
    return wf


def _state_with_approved_plan() -> AnalysisState:
    state = AnalysisState(
        user_id="test",
        current_round=1,
        analysis_phase=AnalysisPhase.EXPLORATION,
        current_plan=AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="Test",
            steps=[PlanStep(id=1, title="S1", method="m")],
            status="approved",
        ),
        current_round_cache=AgentRoundCache(round_number=1, results={"x": {}}),
    )
    return state


@pytest.mark.asyncio
async def test_durable_origin_alone_does_not_preserve_round_or_plan():
    """A restored semantic action must initialize the next round."""
    wf = _make_workflow()
    state = _state_with_approved_plan()
    state.last_restored_snapshot_id = "source/r001-direction-g1"

    result = await wf._initialize_round({"analysis_state": state})

    assert result["analysis_state"].current_round == 2
    assert result["analysis_state"].current_plan is None
    assert result["analysis_state"].current_round_cache is None
    assert result["plan_approved"] is False
    assert result["_preserve_round_on_initialize"] is False


@pytest.mark.asyncio
async def test_explicit_start_one_shot_preserves_round_plan_and_cache():
    wf = _make_workflow()
    state = _state_with_approved_plan()
    state.last_restored_snapshot_id = "source/r001-direction-g1"

    result = await wf._initialize_round({
        "analysis_state": state,
        "_preserve_round_on_initialize": True,
    })

    assert result["analysis_state"].current_round == 1
    assert result["analysis_state"].current_plan is state.current_plan
    assert result["analysis_state"].current_round_cache is state.current_round_cache
    assert result["plan_approved"] is True
    # Consumed through the returned graph channel, not a local pop.
    assert result["_preserve_round_on_initialize"] is False

    # The next ordinary initialization increments and resets normally.
    second = await wf._initialize_round(result)
    assert second["analysis_state"].current_round == 2
    assert second["analysis_state"].current_plan is None
    assert second["analysis_state"].current_round_cache is None


@pytest.mark.asyncio
async def test_stale_preserve_flag_does_not_leak_across_calls():
    """An explicit-start value cannot survive the node that consumed it."""
    wf = _make_workflow()
    state = _state_with_approved_plan()
    result = await wf._initialize_round({
        "analysis_state": state,
        "_preserve_round_on_initialize": True,
    })
    assert result["_preserve_round_on_initialize"] is False

    state.current_plan = None
    third = await wf._initialize_round({
        "analysis_state": state,
        "_preserve_round_on_initialize": result["_preserve_round_on_initialize"],
    })
    assert third["analysis_state"].current_round == 2
