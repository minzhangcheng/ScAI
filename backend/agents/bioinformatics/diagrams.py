# Text-diagram generation, revision, and review.
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

from ...utils.formatting import (
    format_revision_history,
    normalize_text_list,
    truncate_for_llm,
)
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


class DiagramsMixin:
    """Text-diagram generation, revision, and review."""

    async def generate_text_diagram(
        self,
        state,
        pi_prompt: str
    ) -> str:
        """
        Draw an ASCII/Unicode pathway diagram following PI's instruction.

        Raises ValueError if the LLM returns empty content.
        """
        logger.info(f"[{getattr(self, 'session_id', '')}][Bio] Generating text-mode diagram")

        user_message = f"""## Draw a text-mode graphic abstract

**PI instruction:**
{pi_prompt}

{self._TEXT_DIAGRAM_RULES}

Draw the diagram now.
"""
        response = await self.chat_without_history(
            user_message, temperature=0.3,
            system_prompt=self._TEXT_DIAGRAM_SYSTEM_PROMPT,
            timeout=600.0,
            max_retries=3,
        )
        diagram = extract_code_from_markdown(response).strip()
        if not diagram:
            raise ValueError("Bioinformatics agent returned empty text diagram")
        return diagram

    async def revise_text_diagram(
        self,
        state,
        current_diagram: str,
        pi_prompt: str,
        review: Dict[str, Any],  # Consistently pass the review dict
        revision_history: List[Dict[str, Any]]
    ) -> str:
        """
        Revise a text-mode diagram based on reviewer feedback.

        Raises on empty response.
        """
        feedback = review.get("feedback", "")
        suggestions = review.get("suggestions", [])
        must_fix = normalize_text_list(review.get("must_fix"), limit=20)

        logger.info(f"[{getattr(self, 'session_id', '')}][Bio] Revising text diagram (round {len(revision_history)})")

        history_text = format_revision_history(revision_history)

        suggestions_text = "\n".join(f"  - {s}" for s in (suggestions or [])) or "  (none)"
        must_fix_text = "\n".join(f"  - {m}" for m in (must_fix or [])) or "  (none)"

        user_message = f"""## Revise the text-mode graphic abstract

**PI instruction (unchanged):**
{pi_prompt}

**Current diagram:**
{current_diagram}

**New reviewer feedback:**
{feedback}

**Specific suggestions:**
{suggestions_text}

**Must-fix:**
{must_fix_text}

{history_text}

{self._TEXT_DIAGRAM_RULES}

Return the revised diagram only — no prose, no markdown fences.
"""
        response = await self.chat_without_history(
            user_message, temperature=0.3,
            system_prompt=self._TEXT_DIAGRAM_SYSTEM_PROMPT,
            timeout=600.0,
            max_retries=3,
        )
        diagram = extract_code_from_markdown(response).strip()
        if not diagram:
            raise ValueError("Bioinformatics agent returned empty revised diagram")
        return diagram

    async def review_text_diagram(
        self,
        state,
        text_diagram: str,
        revision_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Fallback reviewer when the PI is unavailable.

        Raises on failure so the workflow descends to the Supervisor tier.
        """
        logger.info(f"[{getattr(self, 'session_id', '')}][Bio] Reviewing text diagram (fallback reviewer)")

        history_text = format_revision_history(revision_history)

        system_prompt = (
            "You are a bioinformatics analyst reviewing a text-mode (ASCII) pathway "
            "diagram. You are the fallback reviewer — the PI is unavailable. Read the "
            "diagram character by character and give concrete feedback."
        )

        user_message = f"""## Review the text diagram

{history_text}

**Diagram:**
```
{text_diagram}
```

Evaluate scientific accuracy, completeness, clarity of directional flow, and label
correctness. Respond in JSON:

```json
{{
    "quality_score": 1-10,
    "feedback": "Plain-English assessment",
    "strengths": ["..."],
    "issues": ["..."],
    "suggestions": ["Concrete text edits, e.g. 'Add the missing intermediate node between the upstream signal and the observed response'"],
    "must_fix": ["Critical omissions"]
}}
```
"""
        response_text = await self.chat_without_history(
            user_message, temperature=0.3, system_prompt=system_prompt
        )
        if not response_text or not response_text.strip():
            raise ValueError("Bio text diagram review returned empty response")

        data = self._safe_parse_json(response_text, default={})
        if not data or "feedback" not in data:
            raise ValueError("Bio text diagram review did not return parseable JSON")

        return {
            "quality_score": int(data.get("quality_score", 5)),
            "feedback": data.get("feedback", ""),
            "strengths": data.get("strengths", []),
            "issues": data.get("issues", []),
            "suggestions": data.get("suggestions", []),
            "must_fix": normalize_text_list(data.get("must_fix"), limit=20)
        }
