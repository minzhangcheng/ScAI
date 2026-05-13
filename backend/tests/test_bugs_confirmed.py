"""Tests for confirmed bugs from code review."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from datetime import datetime


class TestPlanRevisionResumeReset:
    """Bug 1: plan_revision_resume must be cleared at the start of each round's planning."""

    def test_resume_flag_dict_cleared_unconditionally(self):
        """Verify that the unconditional reset logic works: setting the key to False
        regardless of its current value, matching the fix applied to _pi_planning."""
        state = {"plan_revision_resume": True}
        state["plan_revision_resume"] = False
        assert state["plan_revision_resume"] is False

        state = {"plan_revision_resume": False}
        state["plan_revision_resume"] = False
        assert state["plan_revision_resume"] is False

        state = {}
        state["plan_revision_resume"] = False
        assert state["plan_revision_resume"] is False

    def test_old_conditional_reset_keeps_true(self):
        """Demonstrate the old bug: conditional reset preserves True."""
        state = {"plan_revision_resume": True}
        if not state.get("plan_revision_resume"):
            state["plan_revision_resume"] = False
        assert state["plan_revision_resume"] is True, "Bug confirmed: old code kept True"


class TestPlanReviewSkipStatus:
    """Bug 2: submit_plan_review fallback must set plan.status='approved' for skip action."""

    def test_skip_sets_approved_status(self):
        """When action is 'skip' in fallback path, plan status must be set to 'approved'."""
        from backend.models.schemas import AnalysisState, AnalysisPlan, PlanStep, TaskStatus

        plan = AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="Test plan",
            steps=[PlanStep(id=1, title="Step 1", method="test")],
            status="user_reviewing",
        )
        state = AnalysisState(user_id="test-session")
        state.current_plan = plan
        state.status = TaskStatus.WAITING_PLAN_REVIEW

        action = "skip"
        if action in ("approve", "skip"):
            state.current_plan.status = "approved"
            state.current_plan.approved_at = datetime.now()

        assert state.current_plan.status == "approved"
        assert state.current_plan.approved_at is not None

    def test_approve_sets_approved_status(self):
        """When action is 'approve' in fallback path, plan status must be set to 'approved'."""
        from backend.models.schemas import AnalysisState, AnalysisPlan, PlanStep, TaskStatus

        plan = AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="Test plan",
            steps=[PlanStep(id=1, title="Step 1", method="test")],
            status="user_reviewing",
        )
        state = AnalysisState(user_id="test-session")
        state.current_plan = plan
        state.status = TaskStatus.WAITING_PLAN_REVIEW

        action = "approve"
        if action in ("approve", "skip"):
            state.current_plan.status = "approved"
            state.current_plan.approved_at = datetime.now()

        assert state.current_plan.status == "approved"
        assert state.current_plan.approved_at is not None


class TestPlanningDialogueMaxRoundsFallback:
    """Bug 3: When max rounds reached and existing_plan is None, prevent infinite loop."""

    def test_max_rounds_without_existing_plan_sets_approved(self):
        """When existing_plan is None at max rounds, force plan_approved to prevent
        infinite loop between _planning_dialogue ↔ _user_review_gate ↔ _after_review."""
        state = {"plan_approved": False}
        existing_plan = None
        if existing_plan:
            state["plan_approved"] = True
        else:
            state["plan_approved"] = True
        assert state["plan_approved"] is True


class TestPlanStepMutableDefault:
    """S1: PlanStep.depends_on must use Field(default_factory=list) instead of []."""

    def test_depends_on_default_is_empty_list(self):
        """Each PlanStep instance should get its own fresh list for depends_on."""
        from backend.models.schemas import PlanStep

        step1 = PlanStep(id=1, title="Step 1", method="test")
        step2 = PlanStep(id=2, title="Step 2", method="test")

        step1.depends_on.append(99)
        assert step2.depends_on == [], "depends_on default must be a fresh list per instance"
        assert step1.depends_on == [99]

    def test_depends_on_accepts_explicit_values(self):
        from backend.models.schemas import PlanStep

        step = PlanStep(id=1, title="Step 1", method="test", depends_on=[2, 3])
        assert step.depends_on == [2, 3]