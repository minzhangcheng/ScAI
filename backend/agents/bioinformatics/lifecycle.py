# Entry points and sandbox state persistence.
# Moved verbatim out of backend/agents/bioinformatics_agent.py; composed into BioinformaticsAgent.
"""
Bioinformatics Agent - Handles snRNA-seq analysis with iterative code execution
Supports autonomous technical steps and PI consultation for biological decisions
With step-level checkpointing for recovery from interruptions
"""
from typing import Dict, Any, List, Optional, Callable, Awaitable, Tuple
import asyncio
import functools
import json
import logging
import base64
import io
import uuid
import re
import ast
import textwrap
import time
from pathlib import Path
from datetime import datetime

import aiofiles

from ..base import BaseAgent, AgentConfig
from ...utils.settings_provider import AgentSettingsProvider
from ...utils.sandbox import (
    validate_code_safety, DANGEROUS_PATTERNS, extract_code_from_markdown,
    LocalJupyterSandbox, CodeExecutionResult, _TPL_NOOP_INSTALL,
)
from ...utils.checkpoint import CheckpointManager
import inspect
import json as _json
from backend.services import ora_kernel
from backend.services.marker_priors import MarkerDB, summarize_database_priors

from ...utils.formatting import format_revision_history, truncate_for_llm
from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY

# Configure logger
logger = logging.getLogger(__name__)


from .shared import (  # noqa: F401
    Dict,
    Any,
    List,
    Optional,
    Callable,
    Awaitable,
    Tuple,
    asyncio,
    functools,
    json,
    logging,
    base64,
    io,
    uuid,
    re,
    ast,
    textwrap,
    time,
    Path,
    datetime,
    aiofiles,
    BaseAgent,
    AgentConfig,
    AgentSettingsProvider,
    validate_code_safety,
    DANGEROUS_PATTERNS,
    extract_code_from_markdown,
    LocalJupyterSandbox,
    CodeExecutionResult,
    _TPL_NOOP_INSTALL,
    CheckpointManager,
    inspect,
    _json,
    ora_kernel,
    MarkerDB,
    summarize_database_priors,
    _ORA_KERNEL_SRC,
    format_revision_history,
    truncate_for_llm,
    MECHANISM_EVIDENCE_POLICY,
    MAX_WORKSPACE_H5AD_FILES,
    _TPL_SAFE_CLOSE_ALL,
    _TPL_PLT_RESTORE_BASELINE,
    _build_exec_autosave_wrapper,
    _build_publication_figure_wrapper,
    _static_h5ad_path,
    workspace_h5ad_write_guard,
    _OVERCLAIM_WORDS,
    _NEGATION_WORDS,
    _MAX_PROGRAMS,
    _MAX_GENES,
    _MAX_OBS_LEN,
    _DEFAULT_MISSING,
    _REQUIRED_FIELDS,
    _ensure_list,
    _overclaim_unnegated,
    validate_mechanism_program_payload,
    TOOLS,
    KNOWN_TOOL_NAMES,
    build_gate_skip_note,
    count_saved_figures,
    AgentType,
    AnalysisPhase,
    AnalysisState,
    MessageRole,
    CodeExecution,
    DataDrivenMechanismProgram,
    PublicationFigure,
    FigureReviewStatus,
    AnalysisStep,
    PlanStage,
    DATA_TYPE_SCRNASEQ,
    DATA_TYPE_BULK_RNASEQ,
    OpenAICompatibleClient,
    ToolCall,
    UpstreamExhaustedError,
    EMPTY_CONTENT_PLACEHOLDER,
    tool_calls_are_paired,
    append_missing_tool_results,
    PLAN_REVISION_NEEDED_TAG,
    strip_ansi,
    extract_error_type,
    TECHNICAL_ERROR_TYPES,
    detect_swallowed_error,
    build_expression_preflight_code,
    extract_sensitive_expression_inputs,
    mutates_expression_representation,
    parse_expression_contracts,
    prepares_expression_before_sensitive_call,
    record_expression_contract,
    record_invalidated_file_stems,
    record_representation_change,
    unchanged_artifact_stems,
    workspace_artifact_versions,
    SupervisorEvalResult,
    _DYNAMIC_CONTEXT_PREFIX,
    _APPROVED_PLAN_PREFIX,
    _CELL_TYPE_ASSIGN_RE,
    _FAILED_EVIDENCE_OUTPUT_RE,
    build_approved_plan_context,
    inject_revised_plan_context,
    execution_supports_evidence,
    successful_evidence_executions,
    _STAGE1_GUIDANCE,
    _STAGE2_GUIDANCE,
    _stage_guidance_text,
    _build_swallowed_error_feedback,
    _MARKER_DB,
    _get_marker_db,
    marker_priors_write,
)


class LifecycleMixin:
    """Entry points and sandbox state persistence."""

    async def execute_step(self, state: AnalysisState, instruction: str) -> Dict[str, Any]:
        """Execute a single analysis step (legacy interface)"""
        return await self.run_autonomous_analysis(state, instruction, max_steps=5)

    async def process(self, state: AnalysisState, instruction: str, **kwargs) -> Dict[str, Any]:
        """Main entry point for workflow"""
        self.refresh_runtime_config_if_needed()
        # Reset per-run state to avoid pollution from previous rounds
        self.code_history = []
        self.analysis_context = {}
        self._recent_sup_feedback: set = set()  # Reset supervisor dedup set per round
        self._force_skip_counter = 0
        self._force_skip_error_type = None
        self.last_result = None

        # For round 2+, inject previous round findings into instruction so the agent
        # knows what was already completed (Jupyter kernel retains variables but the
        # agent's code_history is cleared)
        if state.current_round > 1 and state.round_results:
            prev_findings = []
            for prev_round in state.round_results:
                for f in prev_round.bioinformatics_findings[:5]:
                    prev_findings.append(str(f) if not isinstance(f, str) else f)
            if prev_findings:
                prev_context = "; ".join(prev_findings)
                instruction = (
                    f"[IMPORTANT — Previous round completed these analyses: {prev_context}. "
                    f"The Jupyter kernel retains all variables from previous rounds. "
                    f"Do NOT repeat QC, clustering, or annotation. Build upon existing results.]\n\n"
                    + instruction
                )

        if not state.has_sequencing_data():
            return {
                "status": "waiting_input",
                "message": "Please upload snRNA-seq data file",
                "findings": [],
                "success": False
            }
        
        result = await self.run_autonomous_analysis(
            state, 
            instruction,
            max_steps=kwargs.get("max_steps", 30)
        )
        
        return {
            "status": result["status"],
            "findings": result["findings"],
            "figures": result["figures"],
            "code_execution": self.code_history[-1] if self.code_history else None,
            "code_history": self.code_history,
            "analysis_log": result.get("analysis_log", []),
            "success": result["success"],
            "steps_executed": result["steps_executed"],
            "mechanism_programs": result.get("mechanism_programs", []),
            "evidence_verified": result.get("evidence_verified", False),
            "evidence_execution_ids": result.get("evidence_execution_ids", []),
        }

    def get_all_code(self) -> str:
        """Get all executed code"""
        return "\n\n".join([
            f"# Cell {i+1}\n{ce.code}" 
            for i, ce in enumerate(self.code_history)
        ])

    def get_execution_summary(self) -> str:
        """Get summary of executions"""
        if not self.code_history:
            return "No code executed yet."
        
        errors = sum(1 for ce in self.code_history if ce.error)
        figures = sum(len(ce.figures) for ce in self.code_history)
        
        return f"Cells: {len(self.code_history)}, Success: {len(self.code_history) - errors}, Errors: {errors}, Figures: {figures}"

    async def save_state(self):
        """Save current state"""
        if self.sandbox:
            await self.sandbox.save_session()

    async def restore_state(self) -> dict:
        """Restore state

        Returns:
            dict with sandbox start status including session_restored and restore_failed flags
        """
        return await self.sandbox.start()
