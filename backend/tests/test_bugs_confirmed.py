"""Tests for confirmed bugs from code review.

These tests exercise real production code paths, not mirrored logic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestApprovedPlanPreservedOnResume:
    """Bug fix: when a plan is already approved (from no-workflow fallback resume),
    _pi_planning must preserve the approval instead of resetting current_plan to None.

    This ensures the user's approve/skip decision survives a workflow restart.
    """

    def _make_workflow(self):
        """Create a minimal workflow instance with _pi_planning dependencies mocked."""
        from backend.workflow import SingleCellAnalysisWorkflow
        wf = object.__new__(SingleCellAnalysisWorkflow)
        wf.session_id = "test-session"
        wf._send_message = AsyncMock()
        wf._send_state_sync = AsyncMock()
        wf._set_status = MagicMock()
        wf.pi_agent = MagicMock()
        return wf

    @pytest.mark.asyncio
    async def test_approved_plan_preserved_on_resume(self):
        """An already-approved plan is preserved by _pi_planning, not reset to None."""
        from backend.models.schemas import AnalysisState, AnalysisPlan, PlanStep

        wf = self._make_workflow()
        approved_plan = AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="User-approved plan",
            steps=[PlanStep(id=1, title="Step 1", method="test")],
            status="approved",
        )
        analysis_state = AnalysisState(user_id="test")
        analysis_state.current_plan = approved_plan

        state = {"analysis_state": analysis_state}

        result = await wf._pi_planning(state)

        # The approved plan must NOT be reset
        assert result["analysis_state"].current_plan is not None
        assert result["analysis_state"].current_plan.status == "approved"
        assert result["plan_approved"] is True
        assert result["planning_result"]["plan"]["id"] == approved_plan.id

    @pytest.mark.asyncio
    async def test_unapproved_plan_reset_on_new_round(self):
        """When no plan is approved, _pi_planning resets current_plan and proceeds normally."""
        from backend.models.schemas import AnalysisState, AnalysisPlan, PlanStep

        wf = self._make_workflow()
        draft_plan = AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="Draft plan",
            steps=[PlanStep(id=1, title="Step 1", method="test")],
            status="draft",
        )
        analysis_state = AnalysisState(user_id="test")
        analysis_state.current_plan = draft_plan

        state = {"analysis_state": analysis_state}

        result = await wf._pi_planning(state)

        # Draft plan should be reset to None for re-planning
        assert result["analysis_state"].current_plan is None
        assert result["plan_approved"] is False

    @pytest.mark.asyncio
    async def test_no_plan_reset_on_new_round(self):
        """When no plan exists, _pi_planning resets and proceeds normally."""
        from backend.models.schemas import AnalysisState

        wf = self._make_workflow()
        analysis_state = AnalysisState(user_id="test")
        analysis_state.current_plan = None

        state = {"analysis_state": analysis_state}

        result = await wf._pi_planning(state)

        assert result["analysis_state"].current_plan is None
        assert result["plan_approved"] is False


class TestPlanRevisionResumeReset:
    """Verify _pi_planning's plan_revision_resume handling.

    The code intentionally preserves True values set by _user_review_gate
    (so _after_bioinformatics can read them). When absent or False, it defaults to False.
    """

    def _make_workflow(self):
        from backend.workflow import SingleCellAnalysisWorkflow
        wf = object.__new__(SingleCellAnalysisWorkflow)
        wf.session_id = "test-session"
        wf._send_message = AsyncMock()
        wf._set_status = MagicMock()
        wf.pi_agent = MagicMock()
        return wf

    @pytest.mark.asyncio
    async def test_resume_flag_set_false_when_absent(self):
        """When plan_revision_resume is absent from state, _pi_planning sets it to False."""
        from backend.models.schemas import AnalysisState

        wf = self._make_workflow()
        analysis_state = AnalysisState(user_id="test")
        state = {"analysis_state": analysis_state}

        result = await wf._pi_planning(state)
        assert result["plan_revision_resume"] is False

    @pytest.mark.asyncio
    async def test_resume_flag_set_false_when_false(self):
        """When plan_revision_resume is False, _pi_planning keeps it False."""
        from backend.models.schemas import AnalysisState

        wf = self._make_workflow()
        analysis_state = AnalysisState(user_id="test")
        state = {"analysis_state": analysis_state, "plan_revision_resume": False}

        result = await wf._pi_planning(state)
        assert result["plan_revision_resume"] is False

    @pytest.mark.asyncio
    async def test_resume_flag_preserved_when_true(self):
        """When plan_revision_resume is True (set by _user_review_gate),
        _pi_planning preserves it for _after_bioinformatics to read."""
        from backend.models.schemas import AnalysisState

        wf = self._make_workflow()
        analysis_state = AnalysisState(user_id="test")
        state = {"analysis_state": analysis_state, "plan_revision_resume": True}

        result = await wf._pi_planning(state)
        assert result["plan_revision_resume"] is True


class TestPlanReviewRouteStatus:
    """Verify that the no-workflow fallback path correctly transitions to PAUSED.

    _pause_and_restart_ws overwrites status to PAUSED to trigger a WebSocket
    reconnect and workflow resume. The final status after the route handler
    must be PAUSED, not RUNNING.
    """

    @pytest.fixture(autouse=True)
    def _patch_deps(self, monkeypatch):
        self._saved_state = {}

        async def _mock_require_ownership(session_id, user_id):
            from backend.models.schemas import AnalysisState, AnalysisPlan, PlanStep, TaskStatus
            state = AnalysisState(user_id="test-user")
            plan = AnalysisPlan(
                phase="exploration",
                stage="full",
                overview="Test plan",
                steps=[PlanStep(id=1, title="Step 1", method="test")],
                status="user_reviewing",
            )
            state.current_plan = plan
            state.status = TaskStatus.WAITING_PLAN_REVIEW
            self._saved_state["state"] = state
            return state

        async def _mock_save_session(session_id, state):
            self._saved_state["saved"] = True

        async def _mock_pause_restart(session_id, state):
            from backend.models.schemas import TaskStatus
            state.status = TaskStatus.PAUSED
            self._saved_state["paused_restarted"] = True

        monkeypatch.setattr(
            "backend.routes.interaction.require_session_ownership",
            _mock_require_ownership,
        )
        monkeypatch.setattr(
            "backend.routes.interaction.async_save_session",
            _mock_save_session,
        )
        monkeypatch.setattr(
            "backend.routes.interaction._pause_and_restart_ws",
            _mock_pause_restart,
        )
        monkeypatch.setattr("backend.routes.interaction.workflows", {})
        import asyncio
        monkeypatch.setattr(
            "backend.routes.interaction._state_lock", asyncio.Lock()
        )

    @pytest.mark.asyncio
    async def test_approve_transitions_to_paused(self):
        """Approve action ends with PAUSED status (not RUNNING) after _pause_and_restart_ws."""
        from backend.routes.interaction import submit_plan_review
        from backend.models.schemas import TaskStatus
        from backend.models.api_requests import PlanReviewRequest
        from unittest.mock import MagicMock

        request = MagicMock()
        request.state.user_id = "test-user"
        body = PlanReviewRequest(action="approve")

        await submit_plan_review("test-session", request, body)

        state = self._saved_state["state"]
        assert state.current_plan.status == "approved"
        assert state.status == TaskStatus.PAUSED

    @pytest.mark.asyncio
    async def test_skip_transitions_to_paused(self):
        """Skip action ends with PAUSED status after _pause_and_restart_ws."""
        from backend.routes.interaction import submit_plan_review
        from backend.models.schemas import TaskStatus
        from backend.models.api_requests import PlanReviewRequest
        from unittest.mock import MagicMock

        request = MagicMock()
        request.state.user_id = "test-user"
        body = PlanReviewRequest(action="skip")

        await submit_plan_review("test-session", request, body)

        state = self._saved_state["state"]
        assert state.current_plan.status == "approved"
        assert state.status == TaskStatus.PAUSED


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