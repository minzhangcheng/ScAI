# Notebook/analysis context strings and history compaction.
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
    _TPL_SAFE_CLOSE_ALL,
    _TPL_PLT_RESTORE_BASELINE,
    _build_exec_autosave_wrapper,
    _build_publication_figure_wrapper,
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


class ContextMixin:
    """Notebook/analysis context strings and history compaction."""

    def _build_notebook_context(self) -> str:
        """Build context string from notebook history.

        Shows last 5 cells to keep dynamic context compact.
        Older cells get shorter output truncation to save tokens.
        """
        if not self.code_history:
            return "Notebook is empty. This is the first code cell."

        # Prevent unbounded memory growth across rounds
        MAX_CODE_HISTORY = 100
        if len(self.code_history) > MAX_CODE_HISTORY:
            self.code_history = self.code_history[-MAX_CODE_HISTORY:]

        context_parts = ["=== Notebook History (last 5 cells) ===\n"]

        recent_history = self.code_history[-5:]
        start_idx = len(self.code_history) - len(recent_history)

        for i, execution in enumerate(recent_history):
            context_parts.append(f"--- Cell {start_idx + i + 1} ---")
            context_parts.append(f"```python\n{execution.code}\n```")

            if execution.output:
                # Recent 2 cells get full output (2000 chars), older cells get 500
                max_output = 2000 if i >= len(recent_history) - 2 else 500
                if len(execution.output) > max_output:
                    output = f"[... truncated to last {max_output} chars]\n{execution.output[-max_output:]}"
                else:
                    output = execution.output
                context_parts.append(f"Output:\n{output}")

            if execution.error:
                context_parts.append(f"Error: {execution.error}")

            if execution.figures:
                context_parts.append(f"[Generated {len(execution.figures)} figure(s)]")

            context_parts.append("")

        return "\n".join(context_parts)

    def _build_analysis_context(self) -> str:
        """Build context about current analysis state"""
        if not self.analysis_context:
            return ""
        
        parts = ["=== Current Analysis State ==="]
        for key, value in self.analysis_context.items():
            if isinstance(value, list):
                parts.append(f"{key}: {', '.join(str(v) for v in value[:10])}")
            else:
                parts.append(f"{key}: {value}")
        return "\n".join(parts)

    def _build_dynamic_context(self, state: AnalysisState) -> str:
        """Build the dynamic context portion (data files, notebook history, analysis state).

        Frozen — NOT updated during the loop. The LLM sees updates through the
        append-only conversation history (assistant + tool messages), not through
        refreshing this context.
        """
        parts = [
            self._build_data_context(state),
            self._build_notebook_context(),
            self._build_analysis_context(),
        ]
        sandbox_state = state.sandbox_state or {}
        cohort = sandbox_state.get("cohort_identity_v1") or {}
        if cohort:
            parts.append(
                "## Active analysis cohort\n"
                f"generation={int(cohort.get('cohort_generation', 0) or 0)}; "
                f"n_obs={int(cohort.get('n_obs', 0) or 0)}; "
                f"n_vars={int(cohort.get('n_vars', 0) or 0)}; "
                f"digest={str(cohort.get('cohort_digest') or '')[:16] or 'none'}; "
                f"lineage={cohort.get('lineage_id') or 'main'}. "
                "Only evidence stamped with this cohort can support current "
                "findings and report artifacts."
            )
        transition = sandbox_state.get("cohort_transition_notice") or {}
        if transition.get("changed"):
            transition_reason = str(
                transition.get("reason") or "unspecified"
            )[:200]
            parts.append(
                "## Cohort transition notice\n"
                f"The canonical analysis cohort changed ({transition_reason}). "
                "Prior-cohort outputs remain audit history; "
                "recompute or explicitly compare them before making current-cohort "
                "claims."
            )
        if state.species:
            parts.append(
                f"## Species\nThe species for this analysis is: {state.species}\n"
                f"Use this species for all organism-dependent operations "
                f"(gene naming conventions, pathway databases, etc.).\n"
                f"After loading data, call validate_species_genes with ~50 randomly "
                f"sampled gene names to verify species consistency."
            )
        if state.bulk_input_profiles:
            profile_lines = []
            for profile in state.bulk_input_profiles[:12]:
                profile_lines.append(
                    f"- profile_id={profile.id}; source_file_id={profile.source_file_id}; "
                    f"shape={profile.shape}; annotation_columns={profile.annotation_columns}; "
                    f"candidate_sample_columns={profile.candidate_sample_columns}; "
                    f"integer_like_fraction={profile.integer_like_fraction}; "
                    f"negative_values={profile.negative_value_count}; "
                    f"declared_scale={profile.declared_scale}; "
                    f"profile_status={profile.profile_status}; "
                    f"rows_scanned={profile.rows_scanned}"
                )
            parts.append(
                "## Host-observed bulk input profiles\n"
                "These are input facts, not a method choice. PI determines claim "
                "level and Bioinformatics selects and justifies a compatible method. "
                "When bulk findings depend on a method choice, record that choice with "
                "register_method_decision. Register CSV outputs intended for reports "
                "with register_publication_table. Missing registrations remain visible "
                "for PI review and do not block the workflow.\n"
                + "\n".join(profile_lines)
            )
        pending_repairs = [
            task for task in (state.figure_revision_tasks or [])
            if task.status in {"pending", "in_progress"}
        ]
        if pending_repairs:
            repair_lines = []
            for task in pending_repairs[:12]:
                repair_lines.append(
                    f"- task {task.id}: logical figure {task.logical_figure_id}; "
                    f"requirement {task.plan_requirement_key or 'supplementary'}; "
                    f"reason {task.reason}; must fix: "
                    f"{'; '.join(task.must_fix[:5]) or 'see prior PI feedback'}"
                )
            parts.append(
                "## Pending figure repair tasks\n"
                "Resolve these durable tasks before claiming the corresponding "
                "plan requirement complete. Reuse the same logical_figure_id and "
                "cite a plan-bound repair execution when reanalysis is required.\n"
                + "\n".join(repair_lines)
            )
        return "\n\n".join(p for p in parts if p)
