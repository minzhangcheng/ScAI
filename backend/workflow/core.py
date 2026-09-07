"""
LangGraph workflow for multi-agent single-cell analysis
"""
import asyncio
import copy
import logging
import os
import traceback
from collections import deque
from typing import Callable, Dict, Any, List, Optional, Tuple
from datetime import datetime

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite

# Configure logger
logger = logging.getLogger(__name__)

# Fixed base: graph entry, the reporting chain, integration cycles, and slack.
GRAPH_RECURSION_LIMIT = 100
# One analysis round executes roughly a dozen graph nodes, and Stage 2 planning
# or a plan revision each add another pass through the execution nodes. A fixed
# ceiling therefore aborts a long run right before its report: a ten-round run
# reached its graphic abstract at superstep 100 and raised GraphRecursionError
# before the detailed-report node ever ran.
GRAPH_RECURSION_PER_ROUND = 25
CHECKPOINTS_TO_KEEP_PER_NAMESPACE = 5


def graph_recursion_limit(max_rounds: Optional[int]) -> int:
    """Superstep ceiling for one graph invocation of a ``max_rounds`` run."""
    rounds = int(max_rounds or 0)
    return GRAPH_RECURSION_LIMIT + GRAPH_RECURSION_PER_ROUND * max(1, rounds)


from .helpers import ack_restore_safe_point, wait_with_skip_time
from .retry import with_retry, MAX_NODE_RETRIES
from .state import WorkflowState, make_initial_workflow_state
from .routing import RoutingMixin
from .context import ContextMixin
from .messaging import MessagingMixin
from .callbacks import CallbacksMixin
from .annotation import AnnotationMixin
from .planning import PlanningMixin
from .agents import AgentNodesMixin
from .synthesis import SynthesisMixin
from .initialize import InitializeMixin
from .graph import GraphMixin
from .reporting import ReportingMixin
from .integration import IntegrationMixin
from .post_execution import PostExecutionReviewMixin

from ..services.snapshot_v2.errors import RequiredSnapshotBlocked
from ..services.snapshot_v2.restore_entry import build_restore_entry_plan

from ..models.schemas import (
    AnalysisState, AgentType, TaskStatus, RoundResult, AnalysisPhase,
    AnalysisStep, WebSocketMessage, AgentMessage, MessageRole,
    DetailedReportSegment,
)
from ..agents.pi_agent import PIAgent
from ..agents.pathology_agent import PathologyAgent
from ..agents.bioinformatics_agent import BioinformaticsAgent
from ..agents.biochemistry_agent import BiochemistryAgent
from ..agents.literature_agent import LiteratureAgent
from ..agents.painter_agent import PainterAgent
from ..agents.supervisor_agent import SupervisorAgent
from ..agents.technique_agent import TechniqueAgent
from ..utils.llm_client import LLMManager, UpstreamExhaustedError
from ..constants import (
    LlmFailureAction,
    WaitInputReason,
    StateSyncChangeType,
    SEMANTIC_WAIT_GATES,
)
from ..utils.settings_provider import AgentSettingsProvider
from ..utils.sandbox import LocalJupyterSandbox
from ..utils.runtime_apply import is_forward_resume_blocked
from ..utils.asyncio_tasks import consume_current_cancellation_request
from ..utils.checkpoint import (
    DurableJsonPlusSerializer,
    compact_checkpoint_history,
    configure_checkpoint_retention,
)
from ..config import SystemConfig, get_session_workspace, get_session_states

# Keep the historical module-level name for tests and downstream importers.
JsonPlusSerializer = DurableJsonPlusSerializer


# Defined in helpers.py (dependency-free) and re-exported here for backward
# compatibility — existing code imports it via `from .core import WorkflowTerminatedError`.
from .helpers import (  # noqa: E402,F401
    WorkflowSupersededError,
    WorkflowTerminatedError,
)

import sys
from .core_parts import execution as core_execution
from .core_parts import initialization as core_initialization
from .core_parts import interventions as core_interventions
from .core_parts import shutdown as core_shutdown
from .core_parts import supplementary as core_supplementary

class SingleCellAnalysisWorkflow(RoutingMixin, ContextMixin, MessagingMixin, CallbacksMixin, AnnotationMixin, PlanningMixin, AgentNodesMixin, PostExecutionReviewMixin, SynthesisMixin, InitializeMixin, GraphMixin, ReportingMixin, IntegrationMixin):
    """
    Multi-agent workflow for single-cell analysis using LangGraph
    """

    @staticmethod
    def _set_status(state: AnalysisState, status: TaskStatus):
        core_initialization.bind_facade(sys.modules[__name__])
        return core_initialization._set_status(state, status)

    def _consume_restore_routing_metadata(self, analysis_state: AnalysisState) -> None:
        core_initialization.bind_facade(sys.modules[__name__])
        return core_initialization._consume_restore_routing_metadata(self, analysis_state)

    def _consume_restore_resume_metadata(self, analysis_state: AnalysisState) -> None:
        core_initialization.bind_facade(sys.modules[__name__])
        return core_initialization._consume_restore_resume_metadata(self, analysis_state)

    def _restore_entry(self, state: WorkflowState) -> WorkflowState:
        core_initialization.bind_facade(sys.modules[__name__])
        return core_initialization._restore_entry(self, state)

    async def resume_restored_wait_state(self, analysis_state: AnalysisState) -> AnalysisState:
        core_initialization.bind_facade(sys.modules[__name__])
        return await core_initialization.resume_restored_wait_state(self, analysis_state)

    def __init__(self, config: SystemConfig, session_id: str, message_callback: Optional[Callable]=None, state_callback: Optional[Callable]=None):
        core_initialization.bind_facade(sys.modules[__name__])
        return core_initialization.__init__(self, config, session_id, message_callback, state_callback)

    def _next_generation(self, node: str) -> int:
        core_interventions.bind_facade(sys.modules[__name__])
        return core_interventions._next_generation(self, node)

    def _generation_for(self, node: str) -> int:
        core_interventions.bind_facade(sys.modules[__name__])
        return core_interventions._generation_for(self, node)

    async def _drive_hitl_gate(self, node_name: str, analysis_state: AnalysisState, *, reason: str, info_attr: str, info_payload: dict, enter_change: str, exit_change: str, clear_cache_on_retry: bool, wait_seconds: 'int | None'=None, auto_retry: bool=True, require_explicit_resolution: bool=False) -> LlmFailureAction:
        core_interventions.bind_facade(sys.modules[__name__])
        return await core_interventions._drive_hitl_gate(self, node_name, analysis_state, reason=reason, info_attr=info_attr, info_payload=info_payload, enter_change=enter_change, exit_change=exit_change, clear_cache_on_retry=clear_cache_on_retry, wait_seconds=wait_seconds, auto_retry=auto_retry, require_explicit_resolution=require_explicit_resolution)

    async def _wait_for_llm_intervention(self, error: UpstreamExhaustedError, node_name: str, analysis_state: AnalysisState, *, wait_seconds: 'int | None'=None, auto_retry: bool=True, require_explicit_resolution: bool=False) -> LlmFailureAction:
        core_interventions.bind_facade(sys.modules[__name__])
        return await core_interventions._wait_for_llm_intervention(self, error, node_name, analysis_state, wait_seconds=wait_seconds, auto_retry=auto_retry, require_explicit_resolution=require_explicit_resolution)

    async def _publish_runtime_persistence_transition(self, analysis_state: AnalysisState, *, persist_state: bool, abort_on_close_discard: bool=False) -> bool:
        core_interventions.bind_facade(sys.modules[__name__])
        return await core_interventions._publish_runtime_persistence_transition(self, analysis_state, persist_state=persist_state, abort_on_close_discard=abort_on_close_discard)

    async def _wait_for_runtime_persistence_intervention(self, analysis_state: AnalysisState, *, successful_status: Optional[TaskStatus]=TaskStatus.RUNNING, persist_transitions: bool=False, rotate_generation: bool=False) -> str:
        core_interventions.bind_facade(sys.modules[__name__])
        return await core_interventions._wait_for_runtime_persistence_intervention(self, analysis_state, successful_status=successful_status, persist_transitions=persist_transitions, rotate_generation=rotate_generation)

    async def _pathology_llm_intervention_callback(self, error: UpstreamExhaustedError, image_label: str) -> LlmFailureAction:
        core_interventions.bind_facade(sys.modules[__name__])
        return await core_interventions._pathology_llm_intervention_callback(self, error, image_label)

    async def _wait_for_disabled_agent_intervention(self, agent_name: str, node_name: str, analysis_state: AnalysisState) -> LlmFailureAction:
        core_interventions.bind_facade(sys.modules[__name__])
        return await core_interventions._wait_for_disabled_agent_intervention(self, agent_name, node_name, analysis_state)

    async def _wait_for_marker_db_intervention(self, node_name: str, analysis_state: AnalysisState) -> LlmFailureAction:
        core_interventions.bind_facade(sys.modules[__name__])
        return await core_interventions._wait_for_marker_db_intervention(self, node_name, analysis_state)

    async def _marker_db_intervention_callback(self) -> LlmFailureAction:
        core_interventions.bind_facade(sys.modules[__name__])
        return await core_interventions._marker_db_intervention_callback(self)

    def signal_continue(self):
        core_interventions.bind_facade(sys.modules[__name__])
        return core_interventions.signal_continue(self)

    def signal_resume(self):
        core_interventions.bind_facade(sys.modules[__name__])
        return core_interventions.signal_resume(self)

    def get_snapshot_workflow_state(self) -> dict[str, Any]:
        core_interventions.bind_facade(sys.modules[__name__])
        return core_interventions.get_snapshot_workflow_state(self)

    async def acknowledge_at_boundary(self) -> bool:
        core_interventions.bind_facade(sys.modules[__name__])
        return await core_interventions.acknowledge_at_boundary(self)

    async def await_restore_safe_point(self) -> None:
        core_interventions.bind_facade(sys.modules[__name__])
        return await core_interventions.await_restore_safe_point(self)

    def signal_annotation_confirmed(self):
        core_interventions.bind_facade(sys.modules[__name__])
        return core_interventions.signal_annotation_confirmed(self)

    def signal_plan_review(self, action: str, user_input: Optional[str]=None) -> bool:
        core_interventions.bind_facade(sys.modules[__name__])
        return core_interventions.signal_plan_review(self, action, user_input)

    def signal_species_input(self, user_input: str) -> bool:
        core_interventions.bind_facade(sys.modules[__name__])
        return core_interventions.signal_species_input(self, user_input)

    def signal_llm_failure_action(self, node_name: str, action: str, generation: int) -> bool:
        core_interventions.bind_facade(sys.modules[__name__])
        return core_interventions.signal_llm_failure_action(self, node_name, action, generation)

    def signal_runtime_persistence_action(self, action: str, gate_token: str | None=None, generation: int | None=None):
        core_interventions.bind_facade(sys.modules[__name__])
        return core_interventions.signal_runtime_persistence_action(self, action, gate_token, generation)

    def signal_supplementary_analysis(self, request_text: str, target: str):
        core_supplementary.bind_facade(sys.modules[__name__])
        return core_supplementary.signal_supplementary_analysis(self, request_text, target)

    @property
    def _vision_settings(self) -> Dict[str, bool]:
        core_initialization.bind_facade(sys.modules[__name__])
        return core_initialization._vision_settings(self)

    def _create_optional_agent(self, agent_name: str):
        core_initialization.bind_facade(sys.modules[__name__])
        return core_initialization._create_optional_agent(self, agent_name)

    def ensure_agent_ready(self, agent_name: str) -> bool:
        core_initialization.bind_facade(sys.modules[__name__])
        return core_initialization.ensure_agent_ready(self, agent_name)

    async def initialize(self):
        core_initialization.bind_facade(sys.modules[__name__])
        return await core_initialization.initialize(self)

    def _install_retry_notifiers(self):
        core_initialization.bind_facade(sys.modules[__name__])
        return core_initialization._install_retry_notifiers(self)

    async def _build_and_save_detailed_segment(self, state: AnalysisState, phase: str, title: str, graph_state=None, round_result=None) -> DetailedReportSegment:
        core_initialization.bind_facade(sys.modules[__name__])
        return await core_initialization._build_and_save_detailed_segment(self, state, phase, title, graph_state, round_result)

    @staticmethod
    def _make_initial_workflow_state(analysis_state, current_round=0):
        core_initialization.bind_facade(sys.modules[__name__])
        return core_initialization._make_initial_workflow_state(analysis_state, current_round)

    async def _create_auto_snapshot(self, state: AnalysisState, phase: str, *, workflow_state: Optional[dict[str, Any]]=None):
        core_initialization.bind_facade(sys.modules[__name__])
        return await core_initialization._create_auto_snapshot(self, state, phase, workflow_state=workflow_state)

    def _merge_supplementary_findings(self, agent_key: str, new_findings: List[str]):
        core_supplementary.bind_facade(sys.modules[__name__])
        return core_supplementary._merge_supplementary_findings(self, agent_key, new_findings)

    def _absorb_supplementary_executions(self, state: AnalysisState, supp_result: Dict[str, Any]):
        core_supplementary.bind_facade(sys.modules[__name__])
        return core_supplementary._absorb_supplementary_executions(self, state, supp_result)

    async def _run_supplementary_analysis(self, state: AnalysisState, request_text: str, context: str):
        core_supplementary.bind_facade(sys.modules[__name__])
        return await core_supplementary._run_supplementary_analysis(self, state, request_text, context)

    async def run(self, initial_state: AnalysisState, thread_id: str) -> AnalysisState:
        core_execution.bind_facade(sys.modules[__name__])
        return await core_execution.run(self, initial_state, thread_id)

    async def generate_report(self, initial_state: AnalysisState, thread_id: str) -> AnalysisState:
        core_execution.bind_facade(sys.modules[__name__])
        return await core_execution.generate_report(self, initial_state, thread_id)

    def _superseded_state(self, *fallbacks: Optional[AnalysisState]) -> Optional[AnalysisState]:
        core_execution.bind_facade(sys.modules[__name__])
        return core_execution._superseded_state(self, *fallbacks)

    def _superseded_report_result(self, initial_state: AnalysisState, exc: BaseException) -> AnalysisState:
        core_execution.bind_facade(sys.modules[__name__])
        return core_execution._superseded_report_result(self, initial_state, exc)

    async def _invoke_restore_entry(self, analysis_state: AnalysisState, thread_id: str) -> AnalysisState:
        core_execution.bind_facade(sys.modules[__name__])
        return await core_execution._invoke_restore_entry(self, analysis_state, thread_id)

    async def resume(self, thread_id: str, analysis_state: Optional[AnalysisState]=None) -> AnalysisState:
        core_execution.bind_facade(sys.modules[__name__])
        return await core_execution.resume(self, thread_id, analysis_state)

    async def _recover_failed_pause_persistence(self, thread_id: str, analysis_state: AnalysisState) -> None:
        core_execution.bind_facade(sys.modules[__name__])
        return await core_execution._recover_failed_pause_persistence(self, thread_id, analysis_state)

    async def pause(self, thread_id: str, analysis_state: Optional[AnalysisState]=None) -> Dict[str, Any]:
        core_execution.bind_facade(sys.modules[__name__])
        return await core_execution.pause(self, thread_id, analysis_state)

    async def _register_close_persistence_gate(self, analysis_state: AnalysisState) -> tuple[bool, bool]:
        core_shutdown.bind_facade(sys.modules[__name__])
        return await core_shutdown._register_close_persistence_gate(self, analysis_state)

    async def _release_close_persistence_gate_registration(self) -> None:
        core_shutdown.bind_facade(sys.modules[__name__])
        return await core_shutdown._release_close_persistence_gate_registration(self)

    def _ensure_close_retry_task(self) -> None:
        core_shutdown.bind_facade(sys.modules[__name__])
        return core_shutdown._ensure_close_retry_task(self)

    async def close(self, *, save_runtime: bool=True) -> bool:
        core_shutdown.bind_facade(sys.modules[__name__])
        return await core_shutdown.close(self, save_runtime=save_runtime)

    async def _close_owned_resources(self) -> bool:
        core_shutdown.bind_facade(sys.modules[__name__])
        return await core_shutdown._close_owned_resources(self)

    async def _compact_checkpoints(self, keep: int=CHECKPOINTS_TO_KEEP_PER_NAMESPACE, *, vacuum: bool=True) -> None:
        core_shutdown.bind_facade(sys.modules[__name__])
        return await core_shutdown._compact_checkpoints(self, keep, vacuum=vacuum)

    def get_notebook_path(self) -> Optional[str]:
        core_shutdown.bind_facade(sys.modules[__name__])
        return core_shutdown.get_notebook_path(self)
