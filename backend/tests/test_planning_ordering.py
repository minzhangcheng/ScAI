"""Tests for planning dialogue ordering: wet-lab agents run before planning_dialogue."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from backend.models.schemas import AnalysisState, RoundResult


class TestBuildPlanningContext:
    """Test _build_planning_context merges wet-lab results with planning_context."""

    def _make_workflow(self):
        """Create a minimal workflow instance with _build_planning_context available."""
        from backend.workflow import SingleCellAnalysisWorkflow
        wf = object.__new__(SingleCellAnalysisWorkflow)
        return wf

    def test_current_round_biochem_findings_included(self):
        """Current-round biochemistry findings are included in planning context."""
        wf = self._make_workflow()
        state = {
            "analysis_state": AnalysisState(user_id="test"),
            "planning_context": None,
            "biochemistry_result": {
                "findings": ["protein A elevated", "protein B normal"],
                "status": "completed",
            },
            "pathology_result": None,
        }
        result = wf._build_planning_context(state)
        assert result is not None
        assert "biochemistry_findings" in result
        assert result["biochemistry_findings"] == ["protein A elevated", "protein B normal"]

    def test_current_round_pathology_findings_included(self):
        """Current-round pathology findings are included in planning context."""
        wf = self._make_workflow()
        state = {
            "analysis_state": AnalysisState(user_id="test"),
            "planning_context": None,
            "biochemistry_result": None,
            "pathology_result": {
                "findings": ["tumor infiltration observed"],
                "status": "completed",
            },
        }
        result = wf._build_planning_context(state)
        assert result is not None
        assert "pathology_findings" in result
        assert result["pathology_findings"] == ["tumor infiltration observed"]

    def test_skipped_agents_excluded_fallback_to_previous_round(self):
        """Skipped wet-lab results are excluded; fallback to previous round."""
        wf = self._make_workflow()
        prev_round = RoundResult(
            round_number=1,
            biochemistry_findings=["previous biochem finding"],
            pathology_findings=["previous path finding"],
        )
        analysis_state = AnalysisState(user_id="test")
        analysis_state.round_results = [prev_round]
        state = {
            "analysis_state": analysis_state,
            "planning_context": None,
            "biochemistry_result": {"findings": [], "status": "skipped"},
            "pathology_result": {"findings": [], "status": "skipped"},
        }
        result = wf._build_planning_context(state)
        assert result is not None
        assert result["biochemistry_findings"] == ["previous biochem finding"]
        assert result["pathology_findings"] == ["previous path finding"]

    def test_planning_context_merged_with_wet_lab(self):
        """planning_context (stage1 bioinfo results) is merged with wet-lab results."""
        wf = self._make_workflow()
        state = {
            "analysis_state": AnalysisState(user_id="test"),
            "planning_context": {
                "stage1_summary": "QC passed, clustering done",
                "stage1_findings": ["cluster1", "cluster2"],
            },
            "biochemistry_result": {
                "findings": ["protein A elevated"],
                "status": "completed",
            },
            "pathology_result": None,
        }
        result = wf._build_planning_context(state)
        assert result is not None
        assert result["stage1_summary"] == "QC passed, clustering done"
        assert result["stage1_findings"] == ["cluster1", "cluster2"]
        assert result["biochemistry_findings"] == ["protein A elevated"]

    def test_returns_none_when_empty(self):
        """Returns None when no planning_context and no wet-lab results."""
        wf = self._make_workflow()
        state = {
            "analysis_state": AnalysisState(user_id="test"),
            "planning_context": None,
            "biochemistry_result": None,
            "pathology_result": None,
        }
        result = wf._build_planning_context(state)
        assert result is None

    def test_no_data_status_excluded_fallback_to_previous_round(self):
        """Results with 'no_data' status are excluded, fallback to previous round."""
        wf = self._make_workflow()
        prev_round = RoundResult(
            round_number=1,
            pathology_findings=["previous path finding"],
        )
        analysis_state = AnalysisState(user_id="test")
        analysis_state.round_results = [prev_round]
        state = {
            "analysis_state": analysis_state,
            "planning_context": None,
            "biochemistry_result": None,
            "pathology_result": {"findings": [], "status": "no_data"},
        }
        result = wf._build_planning_context(state)
        assert result is not None
        assert result["pathology_findings"] == ["previous path finding"]

    def test_current_round_takes_priority_over_previous(self):
        """Current-round findings take priority over previous-round findings."""
        wf = self._make_workflow()
        prev_round = RoundResult(
            round_number=1,
            biochemistry_findings=["old biochem finding"],
            pathology_findings=["old path finding"],
        )
        analysis_state = AnalysisState(user_id="test")
        analysis_state.round_results = [prev_round]
        state = {
            "analysis_state": analysis_state,
            "planning_context": None,
            "biochemistry_result": {
                "findings": ["new biochem finding"],
                "status": "completed",
            },
            "pathology_result": None,
        }
        result = wf._build_planning_context(state)
        assert result is not None
        # Current-round takes priority
        assert result["biochemistry_findings"] == ["new biochem finding"]
        # Fallback for pathology (no current-round result)
        assert result["pathology_findings"] == ["old path finding"]


class TestShouldPlan:
    """Test _should_plan returns correct routing after pi_planning resets plan."""

    def _make_workflow(self):
        from backend.workflow import SingleCellAnalysisWorkflow
        return object.__new__(SingleCellAnalysisWorkflow)

    def test_should_plan_returns_do_planning_when_no_plan(self):
        """_should_plan returns 'do_planning' when current_plan is None."""
        wf = self._make_workflow()
        analysis_state = AnalysisState(user_id="test")
        state = {"analysis_state": analysis_state, "plan_approved": False}
        result = wf._should_plan(state)
        assert result == "do_planning", f"Expected 'do_planning', got '{result}'"

    def test_should_plan_returns_skip_planning_when_approved(self):
        """_should_plan returns 'skip_planning' when current_plan is approved."""
        from backend.models.schemas import AnalysisPlan, PlanStep
        wf = self._make_workflow()
        analysis_state = AnalysisState(user_id="test")
        plan = AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="Test",
            steps=[PlanStep(id=1, title="Step 1", method="test")],
            status="approved",
        )
        analysis_state.current_plan = plan
        state = {"analysis_state": analysis_state}
        result = wf._should_plan(state)
        assert result == "skip_planning", f"Expected 'skip_planning', got '{result}'"


class TestAfterReviewRouting:
    """Test _after_review returns correct routing after graph restructuring."""

    def _make_workflow(self):
        from backend.workflow import SingleCellAnalysisWorkflow
        return object.__new__(SingleCellAnalysisWorkflow)

    def test_approved_full_returns_approved(self):
        """Full/stage1 approval routes to 'approved' (→ bioinformatics_analysis)."""
        wf = self._make_workflow()
        state = {
            "analysis_state": AnalysisState(user_id="test"),
            "plan_approved": True,
            "planning_stage": "full",
            "plan_revision_resume": False,
        }
        result = wf._after_review(state)
        assert result == "approved", f"Expected 'approved', got '{result}'"

    def test_approved_stage1_returns_approved(self):
        """Stage1 approval routes to 'approved' (→ bioinformatics_analysis)."""
        wf = self._make_workflow()
        state = {
            "analysis_state": AnalysisState(user_id="test"),
            "plan_approved": True,
            "planning_stage": "stage1",
            "plan_revision_resume": False,
        }
        result = wf._after_review(state)
        assert result == "approved", f"Expected 'approved', got '{result}'"

    def test_approved_stage2_returns_approved_stage2(self):
        """Stage2 approval routes to 'approved_stage2' (→ bioinformatics_analysis)."""
        wf = self._make_workflow()
        state = {
            "analysis_state": AnalysisState(user_id="test"),
            "plan_approved": True,
            "planning_stage": "stage2",
            "plan_revision_resume": False,
        }
        result = wf._after_review(state)
        assert result == "approved_stage2", f"Expected 'approved_stage2', got '{result}'"

    def test_revision_resume_returns_approved_revision(self):
        """Revision resume routes to 'approved_revision' (→ bioinformatics_analysis)."""
        wf = self._make_workflow()
        state = {
            "analysis_state": AnalysisState(user_id="test"),
            "plan_approved": True,
            "planning_stage": "full",
            "plan_revision_resume": True,
        }
        result = wf._after_review(state)
        assert result == "approved_revision", f"Expected 'approved_revision', got '{result}'"

    def test_not_approved_returns_revise(self):
        """Not approved routes to 'revise' (→ planning_dialogue)."""
        wf = self._make_workflow()
        state = {
            "analysis_state": AnalysisState(user_id="test"),
            "plan_approved": False,
        }
        result = wf._after_review(state)
        assert result == "revise", f"Expected 'revise', got '{result}'"


class TestInitializeRoundResetsPlanningContext:
    """Test that _initialize_round resets planning_context to None."""

    def _make_workflow(self):
        """Create a minimal workflow instance with _initialize_round dependencies mocked."""
        from backend.workflow import SingleCellAnalysisWorkflow
        from unittest.mock import AsyncMock, MagicMock

        wf = object.__new__(SingleCellAnalysisWorkflow)
        wf.session_id = "test-session"
        wf._send_message = AsyncMock()
        wf._send_state_sync = AsyncMock()
        wf._set_status = MagicMock()
        wf.pi_agent = MagicMock()
        return wf

    @pytest.mark.asyncio
    async def test_planning_context_cleared_on_new_round(self):
        """_initialize_round resets planning_context to None so stale data doesn't leak."""
        from backend.models.schemas import AnalysisState

        wf = self._make_workflow()
        analysis_state = AnalysisState(user_id="test")

        # State with stale planning_context from a previous round
        state = {
            "analysis_state": analysis_state,
            "planning_context": {"stage1_summary": "old data", "stage1_findings": ["old"]},
            "biochemistry_result": {"findings": ["old"], "status": "completed"},
            "pathology_result": None,
            "bioinformatics_result": None,
            "literature_result": None,
            "round_result": None,
            "node_retry_count": {},
            "node_errors": {},
            "failed_node": "some_node",
            "skip_nodes": ["old_skip"],
            "should_continue": True,
            "human_stop_requested": False,
        }

        result = await wf._initialize_round(state)

        # planning_context must be reset to None
        assert result["planning_context"] is None, \
            f"Expected planning_context=None, got {result['planning_context']}"

    @pytest.mark.asyncio
    async def test_skip_nodes_cleared_on_new_round(self):
        """_initialize_round resets skip_nodes to empty list for the new round."""
        from backend.models.schemas import AnalysisState

        wf = self._make_workflow()
        analysis_state = AnalysisState(user_id="test")

        state = {
            "analysis_state": analysis_state,
            "skip_nodes": ["bioinformatics", "literature"],
        }

        result = await wf._initialize_round(state)

        assert result["skip_nodes"] == [], \
            f"Expected empty skip_nodes, got {result['skip_nodes']}"

    @pytest.mark.asyncio
    async def test_failed_node_cleared_on_new_round(self):
        """_initialize_round resets failed_node to None for the new round."""
        from backend.models.schemas import AnalysisState

        wf = self._make_workflow()
        analysis_state = AnalysisState(user_id="test")

        state = {
            "analysis_state": analysis_state,
            "failed_node": "bioinformatics",
        }

        result = await wf._initialize_round(state)

        assert result["failed_node"] is None, \
            f"Expected failed_node=None, got {result['failed_node']}"


class TestPlanReviewRouteIntegration:
    """Integration tests for the no-workflow fallback path in submit_plan_review.

    These tests mock the route dependencies and call the actual route handler,
    verifying the complete code path including the modify rejection.
    """

    @pytest.fixture(autouse=True)
    def _patch_deps(self, monkeypatch):
        """Patch all external dependencies used by the route handler."""
        self._saved_state = {}

        async def _mock_require_ownership(session_id, user_id):
            """Return a real AnalysisState with a plan awaiting review."""
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
            # Mimic real pause_and_restart_ws: set PAUSED status
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
        # No live workflow in the workflows dict
        monkeypatch.setattr("backend.routes.interaction.workflows", {})
        # Make _state_lock a no-op async context manager
        import asyncio
        monkeypatch.setattr(
            "backend.routes.interaction._state_lock", asyncio.Lock()
        )

    def _make_request_body(self, action, user_input=""):
        """Create a PlanReviewRequest instance."""
        from backend.models.api_requests import PlanReviewRequest
        return PlanReviewRequest(action=action, user_input=user_input)

    @pytest.mark.asyncio
    async def test_approve_in_no_workflow_fallback(self):
        """Approve action approves plan and transitions session to PAUSED for resume."""
        from backend.routes.interaction import submit_plan_review
        from backend.models.schemas import TaskStatus
        from unittest.mock import MagicMock

        request = MagicMock()
        request.state.user_id = "test-user"
        body = self._make_request_body("approve")

        result = await submit_plan_review("test-session", request, body)

        state = self._saved_state["state"]
        assert state.current_plan.status == "approved"
        assert state.status == TaskStatus.PAUSED
        assert self._saved_state.get("saved") is True

    @pytest.mark.asyncio
    async def test_skip_in_no_workflow_fallback(self):
        """Skip action also approves plan and transitions session to PAUSED for resume."""
        from backend.routes.interaction import submit_plan_review
        from backend.models.schemas import TaskStatus
        from unittest.mock import MagicMock

        request = MagicMock()
        request.state.user_id = "test-user"
        body = self._make_request_body("skip")

        result = await submit_plan_review("test-session", request, body)

        state = self._saved_state["state"]
        assert state.current_plan.status == "approved"
        assert state.status == TaskStatus.PAUSED

    @pytest.mark.asyncio
    async def test_modify_in_no_workflow_raises_409(self):
        """Modify action must raise HTTP 409 without workflow — can't revise without PI agent."""
        from backend.routes.interaction import submit_plan_review
        from fastapi import HTTPException
        from unittest.mock import MagicMock

        request = MagicMock()
        request.state.user_id = "test-user"
        body = self._make_request_body("modify", user_input="Please add trajectory analysis")

        with pytest.raises(HTTPException) as exc_info:
            await submit_plan_review("test-session", request, body)

        assert exc_info.value.status_code == 409
        assert "modify" in exc_info.value.detail.lower() or "workflow" in exc_info.value.detail.lower()