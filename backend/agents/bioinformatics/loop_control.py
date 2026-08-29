# Tool-choice, empty-response backoff, and gate-keyword checks for the agent loop.
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
    ToolProtocolError,
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


class LoopControlMixin:
    """Tool-choice, empty-response backoff, and gate-keyword checks for the agent loop."""

    @staticmethod
    def _compute_tool_choice(text_response_count: int, max_text_responses: int) -> str:
        """Decide the tool_choice for the next chat_with_tools call.

        When the agent has already burned its text-only budget down to its
        last attempt (text_response_count >= max - 1), force a tool call by
        returning "required". Otherwise return "auto" and let the model
        decide naturally. This is the Gemini-3.x-Pro safety net: that family
        frequently replies with "I will now do X..." prose instead of
        actually calling execute_code/report_findings.
        """
        if text_response_count >= max_text_responses - 1:
            return "required"
        return "auto"

    @staticmethod
    def _empty_retry_backoff(count: int) -> float:
        """Incremental backoff (seconds) between empty-response retries.

        Capped so a transient truncation streak doesn't hammer the upstream,
        but kept small enough that real work isn't meaningfully delayed.
        """
        return min(float(count), 5.0)

    @staticmethod
    def _is_empty_response(response) -> bool:
        """True when the LLM returned nothing usable: no tool call AND
        (empty/whitespace content, the extraction-error placeholder, or zero
        completion tokens with empty content). This signals transient upstream
        truncation, which should be retried rather than escalated as a 'text-only'
        model behavior."""
        if response.has_tool_calls:
            return False
        content = (response.content or "").strip()
        if content == "" or content == EMPTY_CONTENT_PLACEHOLDER:
            return True
        # Non-empty content with real text is a genuine text-only response
        # (the model chose to talk), even if usage is missing or completion_tokens
        # is zero. Only treat zero completion_tokens as empty when content is
        # also empty (checked above).
        return False

    def _raise_text_only_exhausted(self, text_response_count: int) -> None:
        """Raise UpstreamExhaustedError when consecutive text-only LLM
        responses have exhausted the budget.

        The autonomous loop calls this instead of silently `break`-ing so
        that the workflow's @with_retry decorator can catch
        UpstreamExhaustedError and route the user to the LLM-failure HITL
        dialog (Retry / Skip). Skipping marks the node as failed instead
        of fabricating a "[Fallback extraction]" success that would push
        the workflow into Stage 2 with no marker genes or annotation.
        """
        base_url = getattr(self.llm_client, "base_url", "") or ""
        raise ToolProtocolError(
            message=(
                f"Bioinformatics agent returned {text_response_count} consecutive "
                "text-only responses without calling any tool. The model is refusing "
                "to use tools; aborting autonomous loop so the user can intervene."
            ),
            agent_name="bioinformatics",
            model=self.model,
            llm_server=base_url,
        )

    def _validate_code(self, code: str) -> Tuple[bool, str]:
        """
        Validate code for security before execution.
        Delegates to the shared validate_code_safety() in sandbox.py.

        Args:
            code: The Python code to validate

        Returns:
            Tuple of (is_safe, error_message)
        """
        return validate_code_safety(code)

    async def _notify(self, message_type: str, content: str):
        """Send notification via callback"""
        if self.message_callback:
            await self.message_callback(message_type, content)
