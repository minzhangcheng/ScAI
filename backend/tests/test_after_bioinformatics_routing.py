import pytest
from backend.workflow import SingleCellAnalysisWorkflow


class TestAfterBioinformaticsRouting:
    """Verify _after_bioinformatics routing logic for stage2 + plan_approved."""

    def _make_state(self, **overrides):
        state = {
            "plan_revision_needed": False,
            "planning_stage": "full",
            "plan_approved": False,
        }
        state.update(overrides)
        return state

    def test_stage2_not_approved_routes_to_stage2_planning(self):
        """Normal case: stage1 done, plan not yet approved → stage2 planning."""
        wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
        wf.session_id = "test"
        result = wf._after_bioinformatics(self._make_state(
            planning_stage="stage2", plan_approved=False
        ))
        assert result == "stage2_planning"

    def test_stage2_already_approved_routes_to_synthesis(self):
        """Bug fix: plan already approved in stage2 → skip re-review, go to synthesis."""
        wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
        wf.session_id = "test"
        result = wf._after_bioinformatics(self._make_state(
            planning_stage="stage2", plan_approved=True
        ))
        assert result == "continue_to_synthesis"

    def test_plan_revision_needed_takes_priority(self):
        """plan_revision_needed should route to plan_revision regardless of stage."""
        wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
        wf.session_id = "test"
        result = wf._after_bioinformatics(self._make_state(
            plan_revision_needed=True,
            planning_stage="stage2",
            plan_approved=True,
        ))
        assert result == "plan_revision"

    def test_full_stage_routes_to_synthesis(self):
        """Full stage with no revision → continue to synthesis."""
        wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
        wf.session_id = "test"
        result = wf._after_bioinformatics(self._make_state(
            planning_stage="full", plan_approved=True
        ))
        assert result == "continue_to_synthesis"