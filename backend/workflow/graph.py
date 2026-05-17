"""Graph construction mixin — builds the LangGraph StateGraph and compiles it."""
import logging

from .state import WorkflowState
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


class GraphMixin:
    """Mixin providing _build_graph for LangGraph workflow construction.

    Composed into SingleCellAnalysisWorkflow. Relies on self.checkpointer,
    self.graph, self.compiled_graph, and the routing/node methods from other
    mixins.
    """

    def _build_graph(self):
        """Build the LangGraph workflow with sequential execution for analysis nodes"""

        # Create graph
        self.graph = StateGraph(WorkflowState)

        # Add nodes
        self.graph.add_node("initialize_round", self._initialize_round)
        self.graph.add_node("pi_planning", self._pi_planning)
        self.graph.add_node("planning_dialogue", self._planning_dialogue)
        self.graph.add_node("user_review_gate", self._user_review_gate)
        self.graph.add_node("pathology_analysis", self._pathology_analysis)
        self.graph.add_node("bioinformatics_analysis", self._bioinformatics_analysis)
        self.graph.add_node("biochemistry_analysis", self._biochemistry_analysis)
        self.graph.add_node("literature_research", self._literature_research)
        self.graph.add_node("pi_synthesis", self._pi_synthesis)
        self.graph.add_node("wait_for_user_input", self._wait_for_user_input)
        self.graph.add_node("check_convergence", self._check_convergence)
        self.graph.add_node("generate_final_report", self._generate_final_report)
        self.graph.add_node("generate_graphic_abstract", self._generate_graphic_abstract)
        self.graph.add_node("generate_detailed_report", self._generate_detailed_report)

        # Add edges - sequential execution (with early-stop check after init)
        self.graph.add_conditional_edges(
            "initialize_round",
            self._should_continue_after_init,
            {
                "continue": "pi_planning",
                "stop": "generate_final_report"
            }
        )

        # pi_planning always proceeds to wet-lab analysis first
        self.graph.add_edge("pi_planning", "biochemistry_analysis")
        self.graph.add_edge("planning_dialogue", "user_review_gate")

        self.graph.add_conditional_edges(
            "user_review_gate",
            self._after_review,
            {
                "approved": "bioinformatics_analysis",
                "approved_stage2": "bioinformatics_analysis",
                # approved_revision currently routes to the same node as approved.
                # The distinct route name is preserved for future differentiation
                # (e.g., skipping upstream agents after mid-execution revision).
                "approved_revision": "bioinformatics_analysis",
                "revise": "planning_dialogue",
            }
        )

        # Sequential analysis: biochemistry -> pathology, then planning or direct to bioinformatics
        self.graph.add_edge("biochemistry_analysis", "pathology_analysis")
        self.graph.add_conditional_edges(
            "pathology_analysis",
            self._should_plan,
            {
                "do_planning": "planning_dialogue",
                "skip_planning": "bioinformatics_analysis",
            }
        )

        # After bioinformatics, may go to planning_dialogue for stage2 or revision
        self.graph.add_conditional_edges(
            "bioinformatics_analysis",
            self._after_bioinformatics,
            {
                "continue_to_synthesis": "literature_research",
                "stage2_planning": "planning_dialogue",
                "plan_revision": "planning_dialogue",
            }
        )
        self.graph.add_edge("literature_research", "pi_synthesis")

        # After synthesis, wait for user input (blocks until frontend signals continue)
        self.graph.add_edge("pi_synthesis", "wait_for_user_input")

        # After user input wait, check if we should continue
        self.graph.add_conditional_edges(
            "wait_for_user_input",
            self._should_continue,
            {
                "continue": "check_convergence",
                "stop": "generate_final_report"
            }
        )

        self.graph.add_conditional_edges(
            "check_convergence",
            self._check_convergence_condition,
            {
                "continue": "initialize_round",
                "stop": "generate_final_report"
            }
        )

        # Final report flow: summary -> graphic abstract -> detailed report -> END
        self.graph.add_edge("generate_final_report", "generate_graphic_abstract")
        self.graph.add_edge("generate_graphic_abstract", "generate_detailed_report")
        self.graph.add_edge("generate_detailed_report", END)

        # Set entry point
        self.graph.set_entry_point("initialize_round")

        # Compile
        self.compiled_graph = self.graph.compile(checkpointer=self.checkpointer)
