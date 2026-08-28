"""Planning workflow facade retaining graph-facing mixin methods."""
import sys
from .planning_parts import dialogue as planning_dialogue
from .planning_parts import pi as planning_pi
from .planning_parts import review as planning_review
from .planning_parts import shared as _shared
globals().update(
    {name: value for name, value in vars(_shared).items() if not name.startswith("__")}
)

class PlanningMixin:
    """Mixin providing planning-dialogue methods. Composed into SingleCellAnalysisWorkflow."""
    @with_retry("pi_planning")
    async def _pi_planning(self, state: WorkflowState) -> WorkflowState:
        planning_pi.bind_facade(sys.modules[__name__])
        return await planning_pi.run_pi_planning(self, state)
    
    def _select_fallback_plan_from_history(
        self,
        dialogue_history: list,
    ) -> Optional["AnalysisPlan"]:
        """Walk dialogue history in reverse and return the first plan that
        passes model_validate.  Entries with pi_decision=='reject' or empty
        steps are skipped.  Returns None if no valid plan is found.

        Formerly, a manual PlanStep(**s) + AnalysisPlan(**plan_kwargs)
        fallback existed in the except branch.  That fallback was removed
        because it would fail for the same reasons model_validate failed
        (type errors in step fields), and it created a divergence from the
        schema that could silently accept malformed plans.

        Note: phase/stage parameters were removed along with the manual
        construction branch — they only served that deleted code path.
        """
        for entry in reversed(dialogue_history):
            if entry.get("pi_decision") == "reject":
                continue
            plan_dict = entry.get("plan", {})
            if not plan_dict.get("steps"):
                continue
            try:
                fallback_plan = AnalysisPlan.model_validate(plan_dict)
            except Exception:
                logger.warning(
                    "[%s] Fallback: model_validate failed for dialogue entry, skipping",
                    self.session_id,
                )
                continue
            logger.info(
                f"[{self.session_id}] Fallback: using plan from dialogue round "
                f"{entry.get('round', '?')} with {len(fallback_plan.steps)} steps"
            )
            return fallback_plan
        return None

    async def _planning_dialogue(self, state: WorkflowState) -> WorkflowState:
        planning_dialogue.bind_facade(sys.modules[__name__])
        return await planning_dialogue.run_planning_dialogue(self, state)

    async def _report_direction_coverage_gaps(self, analysis_state) -> None:
        planning_review.bind_facade(sys.modules[__name__])
        return await planning_review._report_direction_coverage_gaps(self, analysis_state)

    async def _approve_plan(self, state, plan, analysis_state, action):
        planning_review.bind_facade(sys.modules[__name__])
        return await planning_review._approve_plan(
            self, state, plan, analysis_state, action
        )

    async def _user_review_gate(self, state: WorkflowState) -> WorkflowState:
        planning_review.bind_facade(sys.modules[__name__])
        return await planning_review._user_review_gate(self, state)

    @staticmethod
    def _build_plan_dialogue_html(dialogue_history: List[Dict]) -> str:
        planning_review.bind_facade(sys.modules[__name__])
        return planning_review._build_plan_dialogue_html(dialogue_history)
