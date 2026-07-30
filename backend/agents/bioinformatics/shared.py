"""
Bioinformatics Agent - Handles snRNA-seq analysis with iterative code execution
Supports autonomous technical steps and PI consultation for biological decisions
With step-level checkpointing for recovery from interruptions
"""
from typing import Dict, Any, List, Optional, Callable, Awaitable, Sequence, Tuple
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

_ORA_KERNEL_SRC = inspect.getsource(ora_kernel)
from ...utils.formatting import format_revision_history, truncate_for_llm
from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY

# Configure logger
logger = logging.getLogger(__name__)

MAX_WORKSPACE_H5AD_FILES = 2

# Implementations live in ./bioinformatics/; re-exported here so existing
# importers and tests keep resolving them from this module.
from ..bioinformatics.kernel_templates import (  # noqa: F401
    _TPL_SAFE_CLOSE_ALL,
    _TPL_PLT_RESTORE_BASELINE,
    _build_exec_autosave_wrapper,
    _build_publication_figure_wrapper,
)
from ..bioinformatics.workspace_guards import (  # noqa: F401
    _static_h5ad_path,
    workspace_h5ad_write_guard,
)
from ..bioinformatics.mechanism_validation import (  # noqa: F401
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
)
from ..bioinformatics.tool_schemas import (  # noqa: F401
    TOOLS,
    KNOWN_TOOL_NAMES,
)


def build_gate_skip_note(gate_label: str) -> str:
    """Visible note appended to findings when a mandatory gate is released after
    max rejections without the analysis actually being performed."""
    return (f"[LIMITATION] The mandatory '{gate_label}' analysis was NOT completed "
            f"after repeated attempts and was released by the gate. Downstream "
            f"conclusions should not assume this validation was performed.")


def count_saved_figures(analysis_log) -> int:
    """Count figures actually written to disk during a run.

    Each AnalysisStep records the filenames it saved in saved_figure_filenames.
    Accepts AnalysisStep objects or plain dicts (node-retry serialization form).
    Used for completion logging instead of the misleading inline-capture count.
    """
    if not analysis_log:
        return 0
    saved_names = set()
    for step in analysis_log:
        if step is None:
            continue
        if isinstance(step, dict):
            names = step.get("saved_figure_filenames") or []
        else:
            names = getattr(step, "saved_figure_filenames", None) or []
        saved_names.update(Path(str(name)).stem for name in names if name)
        pub_name = (
            step.get("pub_figure_name")
            if isinstance(step, dict)
            else getattr(step, "pub_figure_name", None)
        )
        if pub_name:
            saved_names.add(Path(str(pub_name)).stem)
    return len(saved_names)

from ...models.schemas import (
    AgentType, AnalysisPhase, AnalysisState, MessageRole, CodeExecution,
    DataDrivenMechanismProgram, PublicationFigure, FigureReviewStatus,
    AnalysisStep, PlanStage,
    DATA_TYPE_SCRNASEQ, DATA_TYPE_BULK_RNASEQ,
)
from ...utils.llm_client import OpenAICompatibleClient, ToolCall, UpstreamExhaustedError, EMPTY_CONTENT_PLACEHOLDER
from ...utils.tool_messages import tool_calls_are_paired, append_missing_tool_results
from ...constants import PLAN_REVISION_NEEDED_TAG
from ...utils.error_parsing import strip_ansi, extract_error_type, TECHNICAL_ERROR_TYPES, detect_swallowed_error
from ...utils.expression_contract import (
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
)
from ..supervisor_agent import SupervisorEvalResult

# Pre-compiled pattern for detecting manual cell_type assignment in code
_DYNAMIC_CONTEXT_PREFIX = "[Analysis Context]"
_APPROVED_PLAN_PREFIX = "[Approved Analysis Plan]"
_CELL_TYPE_ASSIGN_RE = re.compile(
    r"""\.obs\[\s*['"](?:cell_?type|cluster_identity|cell_identity|cell_annotation|annotation)['"]\s*\]\s*=(?!=)"""
)


_FAILED_EVIDENCE_OUTPUT_RE = re.compile(
    r"(?:\bvalidation\s+failed\b|\bcritical\s*:\s*[^\n]*(?:failed|no\s+\w+\s+found)|"
    r"\bmatched\s+in\s+bulk\s*:\s*\[\s*\]|\bexecution\s+failed\b)",
    re.IGNORECASE,
)


def build_approved_plan_context(
    plan: Any,
    *,
    resumed_from_plan_id: Optional[str] = None,
) -> str:
    """Build the bounded plan message used by the autonomous executor.

    Planning conversations deliberately live outside the executor transcript.
    On a mid-execution revision, the executor receives the approved replacement
    plan as one explicit message while its checkpointed tool history remains
    unchanged.
    """
    steps = "\n".join(
        f"  [{step.status}] Step {step.id}: "
        f"{truncate_for_llm(str(step.title), max_chars=300)} — "
        f"{truncate_for_llm(str(step.method), max_chars=700)}"
        for step in plan.steps
    ) or "  (No executable steps were provided.)"
    steps = truncate_for_llm(steps, max_chars=24_000)
    decisions = truncate_for_llm(
        ", ".join(str(item) for item in plan.key_decision_points),
        max_chars=4_000,
    ) or "None specified"
    overview = truncate_for_llm(str(plan.overview), max_chars=4_000)

    if resumed_from_plan_id is None:
        heading = "APPROVED ANALYSIS PLAN (follow this plan)"
        transition = "Execute the steps in order."
    else:
        heading = "APPROVED PLAN REVISION (apply to remaining work)"
        replaced = plan.revision_of or resumed_from_plan_id or "unknown"
        reason = truncate_for_llm(
            plan.revision_reason or "The plan was revised after execution feedback.",
            max_chars=2_000,
        )
        transition = (
            f"This plan replaces plan {replaced}. Revision reason: {reason}\n"
            "Resume from the existing checkpoint. Preserve completed work and do "
            "not repeat successful steps; apply this revision to remaining work only."
        )

    return f"""{_APPROVED_PLAN_PREFIX}
## {heading}
Plan ID: {plan.id}
Overview: {overview}

Steps:
{steps}

Key Decision Points: {decisions}

{transition} For key decision points, use consult_pi.
If results significantly deviate from plan expectations, report via consult_pi immediately."""


def inject_revised_plan_context(
    messages: List[Dict[str, Any]],
    plan: Any,
    checkpoint_plan_id: Optional[str],
) -> bool:
    """Append an approved plan only when it differs from the checkpoint plan.

    Returns True when a message was appended. Old checkpoints have no plan ID;
    they intentionally receive the current approved plan once during migration.
    """
    if not plan or getattr(plan, "status", None) != "approved":
        return False
    if checkpoint_plan_id == str(plan.id):
        return False
    messages.append({
        "role": "user",
        "content": build_approved_plan_context(
            plan,
            resumed_from_plan_id=checkpoint_plan_id or "legacy-checkpoint",
        ),
    })
    return True


def code_is_statically_unsafe(
    code: str,
    main_var: str,
    denylist: Sequence[str],
) -> Optional[str]:
    """Return a reason string if code is statically unsafe, else None.

    The denied operations differ per caller — the annotation validator bans
    analysis, the supplementary executor allows it — but the assignment guard
    does not: '<main_var> =' but not '<main_var>_val ='. Rebinding the main
    variable would strand every execution id already cited as evidence.
    """
    low = code.lower()
    for pat in denylist:
        if pat in low:
            return f"disallowed operation '{pat}'"
    if re.search(rf"(^|\n)\s*{re.escape(main_var)}\s*=(?!=)", code):
        return f"assignment to main object '{main_var}'"
    return None


def execution_supports_evidence(execution: CodeExecution) -> bool:
    """Return whether a code cell may be cited as scientific evidence."""
    if execution.error or not execution.evidence_eligible:
        return False
    return not _FAILED_EVIDENCE_OUTPUT_RE.search(execution.output or "")


def successful_evidence_executions(
    executions: List[CodeExecution],
    *,
    round_number: Optional[int] = None,
) -> List[CodeExecution]:
    """Filter code history to successful, non-contradictory executions."""
    return [
        execution
        for execution in executions
        if execution_supports_evidence(execution)
        and (round_number is None or execution.round_number == round_number)
    ]


_STAGE1_GUIDANCE = """
This is STAGE 1 of exploration. Plan ONLY these steps:
1. Data loading and initial inspection (count matrices only — see exclusions below)
2. Quality control (filter cells/genes)
3. Doublet detection/assessment
4. Batch effect assessment and correction
5. Normalization, HVG selection
6. Dimensionality reduction (PCA, UMAP)
7. Clustering (leiden/louvain)
8. Cell type annotation (via marker genes + annotate_clusters tool)

Do NOT include downstream analysis (DEG, GO/KEGG, cell communication, RNA velocity, trajectory/pseudotime, etc.) — those will be planned in Stage 2 after seeing clustering results.

Do NOT load, read, merge, or preprocess any data whose ONLY purpose is to feed a Stage 2 analysis. In particular:
- Do NOT load Loom files or assign spliced / unspliced layers (these belong to Stage 2 RNA velocity).
- Do NOT compute or store inputs for trajectory / pseudotime, cell-cell communication (CellChat / liana / NicheNet ligand-receptor tables), or DEG contrasts.
- Do NOT pre-stage gene sets, signatures, or pathway databases for Stage 2 enrichment.

If a step's rationale references "for downstream", "to support Stage 2", "for velocity", "for trajectory", "for communication", or any Stage 2 analysis by name, the step does NOT belong in Stage 1. Defer it to Stage 2 where the loading can happen alongside the analysis."""

_STAGE2_GUIDANCE = """
This is STAGE 2 of exploration. Based on Stage 1 results (clustering + annotation above),
plan downstream analysis including but not limited to:
1. Condition-level differential expression for relevant clusters
2. GO/KEGG pathway enrichment for DEG results
3. Cell-cell communication analysis (liana)
4. Visualization of key findings
5. Any additional analyses suggested by the clustering/annotation results"""


def _stage_guidance_text(planning_stage: str) -> str:
    if planning_stage == "stage1":
        return _STAGE1_GUIDANCE
    if planning_stage == "stage2":
        return _STAGE2_GUIDANCE
    return ""


def _build_swallowed_error_feedback(error_type: str, doc_hint: str) -> str:
    """Build the steering message injected when agent code caught a technical
    error and tried to skip the step. Pure function (no I/O) for testability."""
    msg = (
        f"\n[System] Your code caught a {error_type} and recorded a skip/abandon "
        f"decision. A skip is ONLY acceptable when the DATA is genuinely "
        f"incompatible (e.g. missing spliced/unspliced layers, no overlapping "
        f"barcodes). If this was an API or version error, do NOT skip: it is "
        f"fixable. Correct the call and retry, or fall back to an equivalent "
        f"method (e.g. DPT/PAGA for trajectory) instead of dropping the analysis."
    )
    if doc_hint:
        msg += f"\n[API Documentation]\n{doc_hint}"
    else:
        msg += (
            "\nCall lookup_documentation(function_path=...) for the failing "
            "function before retrying."
        )
    return msg


_MARKER_DB = None


def _get_marker_db():
    """Lazily construct the shared MarkerDB; sessions that never need DB priors
    don't pay for it at import time."""
    global _MARKER_DB
    if _MARKER_DB is None:
        _MARKER_DB = MarkerDB()
    return _MARKER_DB


def marker_priors_write(organism: str, workspace_dir):
    """模块级间接层，便于测试 monkeypatch。"""
    return _get_marker_db().write_workspace_input(organism, workspace_dir)


