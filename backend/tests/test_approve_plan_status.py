"""Tests that _approve_plan transitions TaskStatus from WAITING_PLAN_REVIEW to RUNNING."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import AsyncMock

from backend.models.schemas import (
    AcceptedFoundationInventory,
    AcceptedFoundationStep,
    AnalysisPhase,
    AnalysisState,
    AnalysisPlan,
    PlanStep,
    TaskStatus,
)


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

    @pytest.mark.parametrize("action", ["approved", "skipped"])
    @pytest.mark.asyncio
    async def test_invalid_innovation_scope_cannot_transition_through_approval(
        self, wf, state_dict, action
    ):
        analysis_state = AnalysisState(
            user_id="test",
            current_round=2,
            analysis_phase=AnalysisPhase.INNOVATION,
            status=TaskStatus.WAITING_PLAN_REVIEW,
            accepted_foundation_inventories=[AcceptedFoundationInventory(
                inventory_id="foundation-v1-r1",
                round_number=1,
                plan_id="round1-plan",
                planning_stage="stage1",
                steps=[AcceptedFoundationStep(
                    plan_step_id=4,
                    title="Accepted baseline",
                    method="scanpy.tl.leiden",
                    evidence_execution_ids=["receipt-r1-step4"],
                )],
            )],
        )
        plan = AnalysisPlan(
            phase="innovation",
            stage="full",
            status="user_reviewing",
            scope_contract_version=1,
            foundation_inventory_id="foundation-v1-r1",
            steps=[PlanStep(id=1, title="Missing scope", method="scanpy")],
        )

        approved = await wf._approve_plan(
            state_dict, plan, analysis_state, action
        )

        assert approved is False
        assert state_dict["plan_approved"] is False
        assert "corrected Innovation scope" in state_dict["user_plan_feedback"]
        assert plan.status == "user_reviewing"
        assert analysis_state.current_plan is None
        assert analysis_state.bioinformatics_settlement_attempt_ids == {}
        wf._send_state_sync.assert_awaited_once_with(
            analysis_state, "plan_scope_validation_failed"
        )

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
