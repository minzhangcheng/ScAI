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
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import aiofiles

from .base import BaseAgent, AgentConfig
from ..utils.settings_provider import AgentSettingsProvider
from ..utils.sandbox import (
    validate_code_safety, DANGEROUS_PATTERNS, extract_code_from_markdown,
    LocalJupyterSandbox, CodeExecutionResult, _TPL_NOOP_INSTALL,
)
from ..utils.checkpoint import CheckpointManager
import json as _json
from backend.services.marker_priors import MarkerDB, summarize_database_priors
from ..utils.formatting import format_revision_history, truncate_for_llm
from ..utils.context_budget import (
    ContextBudget,
    FRAMING_RESERVE_TOKENS,
    compact_messages,
    estimate_tokens,
    message_text,
    resolve_ceiling,
    summarize_dropped_messages,
)
from ..utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from ..services.statistical_gates import run_gates as run_statistical_gates
from ..services.sample_design import (
    BULK_RNASEQ_MODALITY,
    SCRNA_MODALITY,
    finding_evidence_errors,
    has_replicated_condition_design,
    infer_execution_modalities,
    samples_per_group_by_modality,
)
from ..services.cohort_identity import (
    active_cohort_identity,
    probe_canonical_cohort,
    record_cohort_transition,
)
from .bioinformatics.executor_collaboration import (
    handle_annotate_clusters,
    handle_consult_pi,
    handle_lookup_documentation,
    handle_publication_figure,
    handle_search_literature,
    handle_validate_species,
    method_guidance,
    species_validation_response,
)
from .bioinformatics.executor_dispatch import (
    DispatchStatus,
    ToolAction,
    ToolDispatchContext,
    ToolRuntime,
    dispatch_tool,
    tool_error_message,
)
from .bioinformatics.executor_execute import (
    build_execution_tool_result,
    handle_execute_code,
)
from .bioinformatics.executor_gene_sets import (
    fetch_failure,
    handle_fetch_gene_set,
    partition_failure,
    resolve_fetch_organism,
)
from .bioinformatics.executor_registration import (
    eligible_execution_ids,
    handle_register_analysis_plan,
    handle_register_method_decision,
    handle_register_mechanisms,
    handle_register_publication_table,
    handle_register_score,
    mechanism_execution_id_error,
)
from .bioinformatics.executor_report import handle_report_findings, terminal_status
from .bioinformatics.executor_report_gates import (
    evidence_gate_record as _evidence_gate_record,
    gate_samples_per_group_by_modality as _gate_samples_per_group_by_modality,
    gate_score_definitions,
)
from .bioinformatics import executor_controller
from .bioinformatics.executor_context import ExecutorBindings


def _method_rule_text(rule_id: str) -> str:
    """Governing rule text, or '' when the corpus is unavailable.

    Retrieved guidance must never be able to break an analysis, so every lookup
    failure degrades to silence rather than raising.
    """
    try:
        from ..services.method_rules import rule_text

        return rule_text(rule_id)
    except Exception:  # pragma: no cover - corpus is optional by design
        return ""


def _gate_score_definitions(state) -> List[Any]:
    """Compatibility facade for the report-gate owner."""
    return gate_score_definitions(state, run_gates=run_statistical_gates)


def _gate_samples_per_group(state) -> Dict[str, int]:
    """Compatibility facade for legacy callers requiring one count map."""
    designs = samples_per_group_by_modality(state)
    merged: Dict[str, int] = {}
    for counts in designs.values():
        for group, count in counts.items():
            merged[group] = min(merged.get(group, count), count)
    return merged


# Configure logger
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DatabasePriorResult:
    """Outcome of the optional offline marker-database evidence path."""

    status: str
    priors: Dict[str, List[Dict[str, Any]]]
    detail: str = ""
    marker_source: str = ""

    def summary(self, total_clusters: int) -> str:
        if self.status == "used":
            matched = summarize_database_priors(self.priors, total_clusters)
            source = f"; marker source: {self.marker_source}" if self.marker_source else ""
            return f"used: {matched or 'database priors available'}{source}"
        return f"{self.status}: {self.detail or 'no additional detail'}"

    def realtime_message(self, total_clusters: int) -> str:
        if self.status == "used":
            return "Marker database priors used: " + self.summary(total_clusters)[len("used: "):]
        if self.status == "empty":
            return (
                "Marker database priors computed but no eligible matches were found; "
                "continuing with literature evidence."
            )
        if self.status == "failed":
            return (
                f"Marker database prior computation failed: {self.detail}; "
                "continuing with literature evidence."
            )
        return (
            f"Marker database priors skipped: {self.detail}; "
            "continuing with literature evidence."
        )


def _compact_database_prior_error(value: Any, limit: int = 500) -> str:
    text = " ".join(str(value or "unknown error").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _resume_budget_after_plan_revision(
    checkpoint: Optional[Dict[str, Any]],
    *,
    checkpoint_valid: bool,
    step_count: int,
    current_max_steps: int,
    per_attempt_max_steps: int,
    active_plan_id: Optional[str],
) -> tuple[int, int, bool]:
    """Reset one executor attempt when the PI approves a replacement plan."""
    attempt_start_step = 0
    if checkpoint:
        raw_attempt_start = checkpoint.get("attempt_start_step", 0)
        if isinstance(raw_attempt_start, int):
            attempt_start_step = min(max(0, raw_attempt_start), step_count)
    if not checkpoint or not checkpoint_valid:
        return current_max_steps, attempt_start_step, False
    if not active_plan_id or active_plan_id == checkpoint.get("active_plan_id"):
        return current_max_steps, attempt_start_step, False
    return step_count + per_attempt_max_steps, step_count, True

# Re-exported so `backend.agents.bioinformatics_agent.<name>` keeps resolving
# for existing importers and for tests that patch these by name.
from .bioinformatics.shared import (  # noqa: F401
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
    compact_traceback,
    extract_error_type,
    TECHNICAL_ERROR_TYPES,
    detect_swallowed_error,
    build_expression_runtime_code,
    build_expression_preflight_code,
    extract_sensitive_expression_inputs,
    has_direct_expression_contract_call,
    instrument_expression_contract_calls,
    mutates_expression_representation,
    parse_expression_contracts,
    prepares_expression_before_sensitive_call,
    record_expression_contract,
    record_executed_representation_change,
    workspace_artifact_versions,
    strip_expression_contract_markers,
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
from .bioinformatics import (  # noqa: F401
    ContextMixin,
    LoopControlMixin,
    IntrospectionMixin,
    ValidationMixin,
    SupplementaryMixin,
    PlanningMixin,
    FiguresMixin,
    DiagramsMixin,
    LifecycleMixin,
    ProfilingMixin,
)
from .bioinformatics.executor_prompt import build_static_executor_prompt
from .bioinformatics import (
    executor_checkpoint,
    executor_context,
    executor_priors,
    executor_restore,
    executor_runtime,
    executor_support,
)

class BioinformaticsAgent(
    ContextMixin,
    LoopControlMixin,
    IntrospectionMixin,
    ValidationMixin,
    SupplementaryMixin,
    PlanningMixin,
    FiguresMixin,
    DiagramsMixin,
    LifecycleMixin,
    BaseAgent,
    ProfilingMixin,
):
    """
    Bioinformatics Agent with Autonomous Mode

    Can autonomously execute technical steps:
    - Data loading, QC, filtering
    - Normalization, HVG selection
    - Dimensionality reduction, clustering
    - Basic visualization

    Requires PI consultation for biological decisions:
    - Cell type annotation
    - Result interpretation
    - Analysis direction choices
    - Gene/pathway selection
    - Trajectory analysis decisions
    """

    # Dangerous code patterns — now defined centrally in sandbox.py and enforced
    # at the sandbox.execute() level. This class-level alias is kept for backward
    # compatibility (e.g. tests or external callers that reference it).
    DANGEROUS_PATTERNS = DANGEROUS_PATTERNS
    
    def __init__(
        self,
        settings_provider: 'AgentSettingsProvider',
        sandbox: LocalJupyterSandbox,
        literature_callback: Optional[Callable[[str], Awaitable[str]]] = None,
        pi_callback: Optional[Callable[[str, str], Awaitable[str]]] = None,
        annotate_clusters_callback: Optional[Callable[..., Awaitable[Dict[str, Any]]]] = None,
        message_callback: Optional[Callable[[str, str], Awaitable[None]]] = None,
        figure_review_callback: Optional[Callable[[PublicationFigure], Awaitable[Dict[str, Any]]]] = None,
        supervisor_callback: Optional[Callable[['AnalysisStep', List[str]], Awaitable[SupervisorEvalResult]]] = None,
        figure_plan_binding_callback: Optional[Callable[[AnalysisState, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
        technique_agent=None,
        species_validation_callback: Optional[Callable[[List[str], str, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
        marker_db_intervention_callback: Optional[Callable[[], Awaitable["LlmFailureAction"]]] = None,
    ):
        super().__init__(
            agent_type=AgentType.BIOINFORMATICS,
            settings_provider=settings_provider,
        )
        self.sandbox = sandbox
        self.literature_callback = literature_callback
        self.pi_callback = pi_callback
        self.annotate_clusters_callback = annotate_clusters_callback
        self.message_callback = message_callback
        self.figure_review_callback = figure_review_callback
        self.supervisor_callback = supervisor_callback
        self.figure_plan_binding_callback = figure_plan_binding_callback
        self.technique_agent = technique_agent
        self.species_validation_callback = species_validation_callback
        self.marker_db_intervention_callback = marker_db_intervention_callback

        # Track execution history
        self.code_history: List[CodeExecution] = []
        self.last_result: Optional[CodeExecutionResult] = None
        self.analysis_context: Dict[str, Any] = {}
        self._recent_sup_feedback: set = set()
        self._force_skip_counter: int = 0
        self._force_skip_error_type: Optional[str] = None
        # Current AnalysisState reference for modality inference in system prompt.
        # Set at the start of each run; used by _infer_modality / _static_system_prompt.
        self._current_analysis_state: Optional[AnalysisState] = None
        self._last_findings: List[str] = []

        # Cooperative pause gate, populated by the workflow before each run.
        # When CLEAR, the autonomous loop blocks between steps until SET.
        self._pause_event: Optional[asyncio.Event] = None







    
    
    @staticmethod
    def _render_registered_data_files(state: AnalysisState) -> str:
        """Render the host-owned upload registry as the executor's authority."""
        lines = ["REGISTERED UPLOAD INVENTORY (AnalysisState.data_files):"]
        for data_file in getattr(state, "data_files", []) or []:
            path = str(getattr(data_file, "file_path", "") or "").strip()
            if not path:
                continue
            exists = Path(path).is_file()
            lines.append(
                "- id={id}; section={section}; filename={filename}; "
                "data_type={data_type}; group_id={group_id}; sample_number={sample}; "
                "host_file_exists={exists}; path={path}".format(
                    id=str(getattr(data_file, "id", "") or ""),
                    section=str(getattr(data_file, "section", "") or ""),
                    filename=str(getattr(data_file, "filename", "") or ""),
                    data_type=str(getattr(data_file, "data_type", "") or ""),
                    group_id=str(getattr(data_file, "group_id", "") or ""),
                    sample=str(getattr(data_file, "sample_number", "") or ""),
                    exists="true" if exists else "false",
                    path=path,
                )
            )
        if len(lines) == 1:
            lines.append("- (none)")
        return "\n".join(lines)

    def _build_data_context(self, state: AnalysisState) -> str:
        """Build context about available data"""
        data_parts = ["=== Available Data ===\n"]
        # The record, not the authorization. A restore that fell back to a
        # verified older generation records degraded_recovery on its own; before
        # spec 5.1 removed the gate, its /recompute route was what set
        # runtime_recompute_pending, so gating the warning on that flag now
        # would let a degraded restore proceed in silence.
        _persistence = dict(getattr(state, "runtime_persistence", None) or {})
        _degraded = bool(
            _persistence.get("degraded_recovery")
            or _persistence.get("recommended_action")
            in {"reduce_runtime_footprint", "recompute_or_reduce_runtime"}
        )
        if getattr(state, "runtime_recompute_pending", False) or _degraded:
            persistence = _persistence
            if _degraded:
                data_parts.append(
                    "**RUNTIME RECOVERY REQUIRED.** The last verified runtime may be "
                    "older than the latest failed persistence attempt, or the live "
                    "namespace exceeded its persistence budget. Recompute any missing "
                    "continuation state from the input files and reduce disposable or "
                    "duplicated large objects before the next checkpoint. Do not treat "
                    "the restored namespace as proof that the newest work survived.\n"
                )
            else:
                data_parts.append(
                    "**KERNEL STATE IS EMPTY.** This session was restored from an archive "
                    "that carried no kernel runtime, so no variables from the earlier run "
                    "exist -- `adata` and every other in-memory object are gone. Load the "
                    "input files listed below from disk before doing anything else, and "
                    "recompute whatever downstream objects you need. Do not assume any "
                    "variable is already defined.\n"
                )

        # Experimental design overview
        design = state.build_experimental_design_summary()
        if design:
            data_parts.append(design)
            data_parts.append("")

        if state.data_files:
            data_parts.append("REGISTERED UPLOAD INVENTORY IS AUTHORITATIVE")
            data_parts.append(self._render_registered_data_files(state))

        # List sequencing files with group/sample context
        seq_files = [f for f in state.data_files if f.section == "sequencing"]
        var_name_counts: dict = {}  # Track usage to disambiguate duplicates
        bulk_var_name_counts: dict = {}  # Separate counter for bulk files
        pandas_imported = False
        if seq_files:
            data_parts.append("Sequencing Data Files:")
            for f in seq_files:
                group = state.get_group_by_id(f.group_id)
                group_label = group.name if group else "Ungrouped"
                sample_label = f"Sample {f.sample_number}" if f.sample_number else ""
                label = f"{group_label} {sample_label}".strip()

                data_parts.append(f"  {label} ({f.data_type}): {f.file_path}")
                if f.data_type == DATA_TYPE_SCRNASEQ:
                    # Generate a safe variable name for loading (deduplicate)
                    base_name = (group_label.lower().replace(" ", "_") +
                                (f"_s{f.sample_number}" if f.sample_number else ""))
                    var_name_counts[base_name] = var_name_counts.get(base_name, 0) + 1
                    var_name = f"{base_name}_{var_name_counts[base_name]}" if var_name_counts[base_name] > 1 else base_name
                    ext = Path(f.file_path).suffix.lower()
                    loaders = {
                        '.h5ad':     f"adata_{var_name} = sc.read_h5ad('{f.file_path}')",
                        '.h5':       f"adata_{var_name} = sc.read_10x_h5('{f.file_path}')",
                        '.loom':     f"adata_{var_name} = sc.read_loom('{f.file_path}')",
                        '.csv':      f"adata_{var_name} = sc.read_csv('{f.file_path}').T  # transpose if cells in columns; verify orientation",
                        '.tsv':      f"adata_{var_name} = sc.read_text('{f.file_path}', delimiter='\\t').T  # verify orientation",
                        '.mtx':      f"adata_{var_name} = sc.read_10x_mtx('{Path(f.file_path).parent}')  # expects matrix.mtx + barcodes.tsv + features.tsv in the same directory",
                        '.zarr':     f"adata_{var_name} = sc.read_zarr('{f.file_path}')",
                    }
                    if ext in loaders:
                        data_parts.append(f"    Load: {loaders[ext]}")
                        if ext in ('.h5ad', '.h5', '.loom', '.zarr'):
                            data_parts.append(f"    adata_{var_name}.obs_names_make_unique()")
                            data_parts.append(f"    adata_{var_name}.var_names_make_unique()")
                    elif ext in ('.rds', '.h5seurat'):
                        data_parts.append(
                            f"    # {ext} Seurat file detected at '{f.file_path}'. "
                            f"Convert to h5ad first using SeuratDisk (R) or anndata2ri, then sc.read_h5ad() the converted file."
                        )
                    else:
                        data_parts.append(
                            f"    # Unknown scRNA-seq format '{ext}' at '{f.file_path}'. "
                            f"Inspect the file and consult PI before loading."
                        )
                elif f.data_type == "loom":
                    data_parts.append(f"    Loom file for RNA velocity analysis")
                elif f.data_type == DATA_TYPE_BULK_RNASEQ:
                    # Generate a safe variable name for loading (deduplicate)
                    base_name = (group_label.lower().replace(" ", "_") +
                                (f"_s{f.sample_number}" if f.sample_number else ""))
                    bulk_var_name_counts[base_name] = bulk_var_name_counts.get(base_name, 0) + 1
                    bulk_var_name = f"{base_name}_{bulk_var_name_counts[base_name]}" if bulk_var_name_counts[base_name] > 1 else base_name
                    if not pandas_imported:
                        data_parts.append(f"    Load: import pandas as pd")
                        pandas_imported = True
                    ext = f.file_path.rsplit('.', 1)[-1].lower()
                    if ext == 'tsv':
                        sep_hint = "sep='\\t'"
                    elif ext == 'csv':
                        sep_hint = "sep=','"
                    else:
                        sep_hint = "sep='\\t'  # .txt files are usually tab-separated in bioinformatics; change to ',' if needed"
                    data_parts.append(f"    bulk_{bulk_var_name} = pd.read_csv('{f.file_path}', {sep_hint})")
                    data_parts.append(
                        f"    # Use the host profile for table facts. Verify sample/group "
                        f"mapping from uploaded metadata and the explicit project design "
                        f"before selecting a method."
                    )
        else:
            data_parts.append("⚠️ No sequencing data uploaded")

        # Add explicit data constraint to prevent hallucinating non-existent groups
        if state.treatment_groups:
            group_names = ', '.join(g.name for g in state.treatment_groups)
            data_parts.append(f"\n**CRITICAL DATA CONSTRAINT:** All analysis and figures MUST use only the actual experimental groups listed above. Do NOT fabricate data for groups that don't exist. If a figure requires group comparisons, use ONLY: {group_names}.")

        return "\n".join(data_parts)
    

    def _infer_modality(self) -> str:
        """Infer the analysis modality from available data files.

        Used to construct the agent's identity in the system prompt so the LLM
        knows what kind of analysis it's actually conducting (single-cell vs
        bulk vs RNA velocity etc.), rather than relying on a hardcoded label.
        """
        state = self._current_analysis_state
        if not state or not getattr(state, "data_files", None):
            return "transcriptomics"
        try:
            files = state.get_files_by_section("sequencing")
        except Exception:
            files = []
        types = {getattr(f, "data_type", None) for f in files}
        has_sc = DATA_TYPE_SCRNASEQ in types
        has_bulk = DATA_TYPE_BULK_RNASEQ in types
        has_loom = "loom" in types
        if has_sc and has_bulk:
            return "single-cell and bulk RNA-seq"
        if has_sc:
            return "single-cell RNA-seq"
        if has_bulk:
            return "bulk RNA-seq"
        if has_loom:
            return "RNA velocity (loom)"
        return "transcriptomics"

    @property
    def _static_system_prompt(self) -> str:
        """Build the cached static portion of the executor system prompt."""
        sp = self.system_prompt
        modality = self._infer_modality()
        cache_key = (sp, modality)
        if getattr(self, "_cached_static_sp_key", None) != cache_key:
            result = build_static_executor_prompt(sp, modality)
            self._cached_static_sp_key, self._cached_static_sp = cache_key, result
        return self._cached_static_sp

    # ---- Context compaction (volume-based) ----

    _EXECUTOR_COMPLETION_RESERVE = 32_768
    _COMPACT_THRESHOLD_FACTOR = 0.90
    _COMPACT_TARGET_RATIO = 0.70
    RECENT_PROTECT = 6
    _CHARS_PER_TOKEN = 3.5
    _COMPACT_TRUNCATE_BUDGETS = [64 * 1024, 32 * 1024, 16 * 1024]

    @property
    def COMPACT_THRESHOLD_TOKENS(self) -> int:
        return resolve_ceiling(
            self._resolve_context_window(),
            completion_reserve=self._EXECUTOR_COMPLETION_RESERVE,
            budget_factor=self._COMPACT_THRESHOLD_FACTOR,
        )

    @property
    def COMPACT_TARGET_TOKENS(self) -> int:
        return int(self.COMPACT_THRESHOLD_TOKENS * self._COMPACT_TARGET_RATIO)

    @staticmethod
    def _context_anchor_count(messages: List[Dict[str, Any]]) -> int:
        if len(messages) <= 1:
            return min(1, len(messages))
        content = messages[1].get("content", "")
        if isinstance(content, str):
            dynamic = content.startswith(_DYNAMIC_CONTEXT_PREFIX)
        elif isinstance(content, list):
            dynamic = any(
                part.get("text", "").startswith(_DYNAMIC_CONTEXT_PREFIX)
                for part in content
                if isinstance(part, dict)
            )
        else:
            dynamic = False
        return 2 if dynamic else 1

    @staticmethod
    def _estimate_context_tokens(messages: List[Dict[str, Any]]) -> int:
        return estimate_tokens(messages)

    @staticmethod
    def _compact_context(messages: List[Dict[str, Any]], target_tokens: int) -> Tuple[List[Dict[str, Any]], bool]:
        if estimate_tokens(messages) <= target_tokens:
            return messages, False
        result, report = compact_messages(
            messages,
            ContextBudget(
                context_window=target_tokens + 1 + FRAMING_RESERVE_TOKENS,
                completion_reserve=1,
            ),
            anchor_count=BioinformaticsAgent._context_anchor_count(messages),
            summarize=BioinformaticsAgent._summarize_dropped_messages,
        )
        return result, bool(report.actions)

    @staticmethod
    def _summarize_dropped_messages(dropped: List[Dict[str, Any]]) -> str:
        """Summarize dropped messages for the trimming summary.

        Pure local extraction — no LLM call. Keeps output under ~500 tokens.
        Handles non-string content (e.g. multimodal list, None) gracefully.
        """
        code_snippets: List[str] = []

        for msg in dropped:
            text = message_text(msg)
            if msg.get("role") == "assistant" and text and "execute_code" in text[:500]:
                # Extract first line of reasoning as a brief description
                first_line = text.strip().split("\n")[0][:120]
                if first_line:
                    code_snippets.append(first_line)

        return summarize_dropped_messages(
            dropped,
            extra_sections=(
                (lambda _messages: [
                    "Recent steps: " + "; ".join(code_snippets[-3:])
                ])
                if code_snippets
                else None
            ),
        )











    # Patterns that must never appear in validation cells (extra safeguard; the
    # copy + per-iteration invariant check are the primary mechanisms).
    _VALIDATION_DENYLIST = (
        "sc.pp.filter_", "sc.tl.leiden", "sc.tl.louvain", "sc.tl.umap", "rank_genes_groups",
        "sc.tl.dpt", "scvelo", "velocity", "sc.pp.normalize_total", "sc.pp.log1p",
        "sc.pp.scale", ".write(", ".write_h5ad",
    )





    async def _compute_database_priors(
        self, adata_var, cluster_key, rank_key, organism, workspace_dir,
        tissue="", cluster_markers=None,
    ) -> DatabasePriorResult:
        return await executor_priors._compute_database_priors(
            self, adata_var, cluster_key, rank_key, organism, workspace_dir,
            tissue=tissue, cluster_markers=cluster_markers,
            bindings=sys.modules[__name__],
        )

    async def _bind_gene_set(
        self, set_id: str, genes: List[str], label: str,
    ) -> Optional[str]:
        return await executor_priors._bind_gene_set(
            self, set_id, genes, label, bindings=sys.modules[__name__],
        )

    async def _bind_gene_set_collection(
        self, collection: str, path: Path,
    ) -> Optional[str]:
        return await executor_priors._bind_gene_set_collection(
            self, collection, path, bindings=sys.modules[__name__],
        )


    async def run_autonomous_analysis(
        self,
        state: AnalysisState,
        initial_instruction: str,
        max_steps: int = 200,
        figure_review_callback: Optional[
            Callable[[PublicationFigure], Awaitable[Dict[str, Any]]]
        ] = None,
        supervisor_callback: Optional[Callable] = None,
        resume_from_checkpoint: bool = True,
        planning_stage: PlanStage = "full",
        figure_followup: bool = False,
    ) -> Dict[str, Any]:
        return await executor_controller.run_autonomous_analysis(
            self,
            state,
            initial_instruction,
            max_steps=max_steps,
            figure_review_callback=figure_review_callback,
            supervisor_callback=supervisor_callback,
            resume_from_checkpoint=resume_from_checkpoint,
            planning_stage=planning_stage,
            figure_followup=figure_followup,
            bindings=ExecutorBindings.from_facade(),
        )

    
    
    
    


    # ============================================
    # Publication Figure Generation
    # ============================================





    # ============================================
    # Text-Mode Graphic Abstract (ASCII diagram)
    # ============================================

    _TEXT_DIAGRAM_SYSTEM_PROMPT = (
        "You are a scientific diagram designer. You draw ASCII/Unicode flow diagrams "
        "of molecular signaling pathways and cellular processes. You do NOT run code, "
        "perform analyses, or produce prose reports. Your entire output is a plain-text "
        "diagram that renders correctly inside an HTML <pre> block."
    )

    _TEXT_DIAGRAM_RULES = textwrap.dedent("""
        Rules for the diagram:
        - Use ONLY English labels. Non-ASCII (Chinese, Japanese, emoji) breaks alignment.
        - Allowed drawing characters:  | │ ▼ ▲ ─ ──► ◄── → ┌ ┐ └ ┘ ├ ┤ ┬ ┴ ┼ ● ○ ■ □ [ ] { } and the double arrow ══►
        - Each line must be at most 80 columns wide.
        - Use real newline characters to separate lines.
        - Wrap node labels in square brackets, e.g. [upstream receptor]. Protein/gene tokens
          may additionally be wrapped in curly braces, e.g. {signal mediator}.
        - Show the nodes the evidence supports and make the flow obvious with arrows.
          Do not pad the diagram to reach a node count; use directional arrows only
          where the supplied evidence establishes direction.
        - Include cellular processes only when they are present in the supplied evidence.
        - Do NOT wrap the output in markdown code fences (no ``` lines).
        - Output ONLY the diagram — no prose explanation before or after.
    """).strip()
