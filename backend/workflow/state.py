"""Workflow state definition and initialization."""
from typing import Any, Dict, List, Optional, TypedDict, Annotated
import operator

from ..models.schemas import AnalysisState, RoundResult, WebSocketMessage


class WorkflowState(TypedDict, total=False):
    """State for the LangGraph workflow"""
    analysis_state: AnalysisState
    current_round: int
    pi_instructions: Dict[str, Any]
    pathology_result: Optional[Dict[str, Any]]
    bioinformatics_result: Optional[Dict[str, Any]]
    biochemistry_result: Optional[Dict[str, Any]]
    literature_result: Optional[Dict[str, Any]]
    round_result: Optional[RoundResult]
    should_continue: bool
    error: Optional[str]
    messages: Annotated[List[WebSocketMessage], operator.add]
    # Human intervention fields
    human_guidance: Optional[str]
    human_stop_requested: bool
    human_guidance_processed: bool
    # User input processing fields
    user_action: Optional[str]
    new_data_files: List[Dict[str, Any]]
    adjusted_instructions: Optional[Dict[str, Any]]
    data_requests: Optional[List[Dict[str, Any]]]
    # Retry mechanism fields
    failed_node: Optional[str]  # Name of failed node (for retry)
    node_retry_count: Dict[str, int]  # Retry count per node
    node_errors: Dict[str, str]  # Error messages per node
    skip_nodes: List[str]  # Nodes to skip on resume
    # Planning-related fields
    planning_stage: str  # "full", "stage1", "stage2"
    plan_approved: bool  # Whether current plan is approved for execution
    planning_result: Optional[Dict[str, Any]]  # Result from planning_dialogue node
    planning_context: Optional[Dict[str, Any]]  # Context passed to planning (stage1 results, etc.)
    user_plan_feedback: Optional[str]  # User's modification feedback for plan revision
    plan_revision_needed: bool  # True when mid-execution revision is triggered
    plan_revision_context: Optional[Dict[str, Any]]  # Context for mid-execution revision
    plan_revision_resume: bool  # True when mid-execution revision is approved, route back to bioinfo
    # Report validation fields
    report_blocked: bool
    # Integration loop routing (set by _integration_synthesis, read by _after_integration)
    integration_next: Optional[str]  # "gate" | "stop"
    # Stable checkpoint identity.  TaskStatus is lifecycle state and may be
    # RUNNING while a restored semantic wait node is executing.
    semantic_gate: Optional[str]


def make_initial_workflow_state(analysis_state: AnalysisState, current_round: int = 0) -> "WorkflowState":
    """Build a clean WorkflowState for a fresh graph invocation."""
    plan = analysis_state.current_plan
    plan_approved = plan is not None and plan.status == "approved"
    planning_stage = plan.stage if plan and plan.stage else "full"
    result: WorkflowState = {
        "analysis_state": analysis_state,
        "current_round": current_round,
        "pi_instructions": {},
        "pathology_result": None,
        "bioinformatics_result": None,
        "biochemistry_result": None,
        "literature_result": None,
        "round_result": None,
        "should_continue": True,
        "error": None,
        "messages": [],
        "plan_approved": plan_approved,
        "planning_stage": planning_stage,
    }
    # Graph-node restore: preserve the target node in the workflow state so
    # _should_continue_after_init can route directly to it, bypassing the
    # normal initialize_round → pi_planning → … pipeline.
    if analysis_state.restore_resume_kind == "graph_node" and analysis_state.restore_resume_node:
        result["_restore_resume_node"] = analysis_state.restore_resume_node
    return result
