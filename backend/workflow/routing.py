"""Pure routing functions for the workflow graph. Stateless except for logging."""
import logging

from ..models.schemas import TaskStatus
from .state import WorkflowState

logger = logging.getLogger(__name__)


class RoutingMixin:
    """Mixin providing graph-routing decision functions. Composed into SingleCellAnalysisWorkflow."""

    session_id: str  # provided by concrete class
    config: object   # provided by concrete class

    def _should_plan(self, state: "WorkflowState") -> str:
        """Determine if planning dialogue is needed."""
        analysis_state = state["analysis_state"]
        plan = analysis_state.current_plan
        # If plan is already approved (from revision), skip to execution
        if plan and plan.status == "approved":
            return "skip_planning"
        return "do_planning"

    def _after_review(self, state: "WorkflowState") -> str:
        """Route after user review gate.

        Outcomes:
        0. abort → generate_final_report
        1. approved + full/stage1 → bioinformatics_analysis
        2. approved + stage2 → bioinformatics_analysis
        3. approved + plan_revision_resume → bioinformatics_analysis
        4. not approved → planning_dialogue

        plan_revision_resume is a one-shot flag: cleared as soon as we route on
        it so a stale True doesn't mis-route subsequent re-entries.
        """
        if state.get("should_continue") is False and not state.get("plan_approved"):
            logger.info(
                f"[{self.session_id}] Planning aborted (should_continue=False, no plan), "
                "routing to final report"
            )
            return "abort"
        if state.get("plan_approved"):
            stage = state.get("planning_stage", "full")
            revision_resume = state.get("plan_revision_resume", False)
            if revision_resume:
                state["plan_revision_resume"] = False
            if stage == "stage2":
                return "approved_stage2"
            if revision_resume:
                return "approved_revision"
            return "approved"
        return "revise"

    def _after_bioinformatics(self, state: "WorkflowState") -> str:
        """Route after bioinformatics execution."""
        # Check for mid-execution plan revision
        if state.get("plan_revision_needed"):
            return "plan_revision"
        # Check for stage transition
        stage = state.get("planning_stage", "full")
        if stage == "stage2":
            if state.get("plan_approved"):
                return "continue_to_synthesis"
            return "stage2_planning"
        return "continue_to_synthesis"

    def _should_continue_after_init(self, state: "WorkflowState") -> str:
        """Route to report generation if user requested stop during round initialization."""
        if state.get("human_stop_requested", False):
            logger.info(f"[{self.session_id}] Stop requested during init, routing to final report")
            return "stop"
        if state.get("should_continue") is False:
            logger.info(f"[{self.session_id}] should_continue=False after init, routing to final report")
            return "stop"
        return "continue"

    def _should_continue(self, state: "WorkflowState") -> str:
        """Determine if analysis should continue"""
        analysis_state = state["analysis_state"]

        # Check if human requested stop (processed by PI Agent)
        if state.get("human_stop_requested", False):
            logger.info(f"[{self.session_id}] Stopping: human stop requested")
            return "stop"

        # Check explicit should_continue flag (set by human intervention)
        if state.get("should_continue") is False:
            logger.info(f"[{self.session_id}] Stopping: should_continue flag is False")
            return "stop"

        # Check max rounds (use state.max_rounds if valid, otherwise config default)
        # max_rounds <= 0 means use config default; max_rounds > 0 is user-specified limit
        max_rounds = analysis_state.max_rounds if analysis_state.max_rounds > 0 else self.config.max_rounds
        # Ensure max_rounds is always at least 1 to prevent infinite loops
        max_rounds = max(1, max_rounds)
        if analysis_state.current_round >= max_rounds:
            logger.info(f"[{self.session_id}] Stopping: reached max rounds ({max_rounds})")
            return "stop"

        # Check if forced to stop
        if analysis_state.status == TaskStatus.PAUSED:
            logger.info(f"[{self.session_id}] Stopping: analysis is paused")
            return "stop"

        logger.info(f"[{self.session_id}] Continuing analysis (round {analysis_state.current_round}/{max_rounds})")
        return "continue"

    def _check_convergence_condition(self, state: "WorkflowState") -> str:
        """Conditional check for convergence"""
        if "should_continue" not in state:
            logger.warning(f"[{self.session_id}] should_continue key missing from state, defaulting to stop")
        if state.get("should_continue", False):
            return "continue"
        return "stop"
