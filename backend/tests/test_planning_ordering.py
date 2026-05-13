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