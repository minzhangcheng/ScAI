"""Tests that _approve_plan transitions TaskStatus from WAITING_PLAN_REVIEW to RUNNING."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import AsyncMock

from backend.models.schemas import AnalysisState, AnalysisPlan, PlanStep, TaskStatus
from backend.services.figure_contract import visual_plan_inheritance_for_plan


def _make_workflow():
    from backend.workflow import SingleCellAnalysisWorkflow
    wf = object.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "test-session"
    wf._send_message = AsyncMock()
    wf._send_state_sync = AsyncMock()
    wf._settings_provider = None
    return wf


@pytest.fixture
def wf():
    return _make_workflow()


@pytest.fixture
def plan():
    return AnalysisPlan(
        phase="exploration",
        stage="full",
        overview="test",
        steps=[PlanStep(id=1, title="QC", method="scanpy")],
        status="user_reviewing",
    )


@pytest.fixture
def analysis_state():
    s = AnalysisState(user_id="test")
    s.status = TaskStatus.WAITING_PLAN_REVIEW
    return s


@pytest.fixture
def state_dict():
    return {"plan_approved": False}


class TestApprovePlanStatus:
    """Verify _approve_plan transitions TaskStatus to RUNNING."""

    @pytest.mark.asyncio
    async def test_approve_plan_sets_status_running(self, wf, plan, analysis_state, state_dict):
        """After approval, analysis_state.status must be RUNNING, not WAITING_PLAN_REVIEW."""
        await wf._approve_plan(state_dict, plan, analysis_state, "approved")
        assert analysis_state.status == TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_approve_plan_sends_state_sync(self, wf, plan, analysis_state, state_dict):
        """After approval, a state_sync must be pushed to the frontend."""
        await wf._approve_plan(state_dict, plan, analysis_state, "approved")
        wf._send_state_sync.assert_called_once_with(analysis_state, "plan_approved")

    @pytest.mark.asyncio
    async def test_skip_plan_sets_status_running(self, wf, plan, analysis_state, state_dict):
        """The 'skip' path also goes through _approve_plan and must set RUNNING."""
        await wf._approve_plan(state_dict, plan, analysis_state, "skipped")
        assert analysis_state.status == TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_revision_flag_set_when_plan_revision_needed(self, wf, plan, analysis_state, state_dict):
        """If state has plan_revision_needed=True, _approve_plan must set
        plan_revision_resume=True and clear plan_revision_needed."""
        state_dict["plan_revision_needed"] = True
        await wf._approve_plan(state_dict, plan, analysis_state, "approved")
        assert state_dict["plan_revision_resume"] is True
        assert state_dict["plan_revision_needed"] is False

    @pytest.mark.asyncio
    async def test_approve_revision_records_unchanged_completed_step_figure_lineage(
        self, wf, analysis_state, state_dict
    ):
        parent = AnalysisPlan(
            id="parent-plan",
            phase="innovation",
            steps=[PlanStep(
                id=3,
                title="Differential expression",
                method="pydeseq2",
                figure_requirements=[{"figure_type": "volcano"}],
                status="completed",
            )],
        )
        await wf._approve_plan(state_dict, parent, analysis_state, "approved")

        revised = AnalysisPlan(
            id="revised-plan",
            revision_of=parent.id,
            phase="innovation",
            steps=[parent.steps[0].model_copy(deep=True)],
        )
        await wf._approve_plan(state_dict, revised, analysis_state, "approved")

        assert visual_plan_inheritance_for_plan(analysis_state, revised) == {
            3: (parent.id,),
        }

    @pytest.mark.asyncio
    async def test_approve_plan_state_sync_before_plan_approved_message(self, wf, plan, analysis_state, state_dict):
        """State sync should be sent before plan_approved message so the
        frontend status badge updates before the dialog closes."""
        # Use side_effect + shared list to track call order (wf is a real
        # object, not a MagicMock, so wf.method_calls is unavailable).
        call_order = []

        async def record_state_sync(*args, **kwargs):
            call_order.append("_send_state_sync")

        async def record_send_message(*args, **kwargs):
            call_order.append("_send_message")

        wf._send_state_sync = AsyncMock(side_effect=record_state_sync)
        wf._send_message = AsyncMock(side_effect=record_send_message)

        await wf._approve_plan(state_dict, plan, analysis_state, "approved")

        assert call_order == ["_send_state_sync", "_send_message"], \
            f"Expected state_sync before plan_approved, got {call_order}"
