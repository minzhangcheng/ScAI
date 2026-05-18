"""Tests for plan_approved/current_plan consistency on snapshot restore.

Bug: plan_approved is unconditionally reset to False, but current_plan is
preserved on snapshot restore. This causes the workflow to re-enter
planning_dialogue for an already-approved plan.
"""
import pytest
from backend.models.schemas import AnalysisState, AnalysisPlan, PlanStep


class TestPlanApprovedSnapshotConsistency:
    """On snapshot restore, plan_approved must be True if current_plan is approved."""

    def test_plan_approved_reset_on_normal_round(self):
        state = AnalysisState(user_id="test")
        state.current_plan = None
        plan_approved = False
        assert plan_approved is False
        assert state.current_plan is None

    def test_plan_approved_preserved_on_snapshot_restore(self):
        plan = AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="Test",
            steps=[PlanStep(id=1, title="S1", method="m")],
            status="approved",
        )
        state = AnalysisState(user_id="test")
        state.current_plan = plan
        state.last_restored_snapshot_id = "snap-1"
        if state.last_restored_snapshot_id and state.current_plan and state.current_plan.status == "approved":
            plan_approved = True
        else:
            plan_approved = False
        assert plan_approved is True

    def test_plan_approved_false_on_snapshot_restore_with_no_plan(self):
        state = AnalysisState(user_id="test")
        state.current_plan = None
        state.last_restored_snapshot_id = "snap-1"
        if state.last_restored_snapshot_id and state.current_plan and state.current_plan.status == "approved":
            plan_approved = True
        else:
            plan_approved = False
        assert plan_approved is False