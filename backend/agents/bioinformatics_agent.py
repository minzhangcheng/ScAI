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
from ..utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from ..services.statistical_gates import run_gates as run_statistical_gates


def _method_rule_text(rule_id: str) -> str:
    """Governing rule for a refusal, or '' when the corpus is unavailable.

    Retrieved guidance must never be able to break an analysis, so every lookup
    failure degrades to silence rather than raising.
    """
    try:
        from ..services.method_rules import rule_text

        return rule_text(rule_id)
    except Exception:  # pragma: no cover - corpus is optional by design
        return ""

# Configure logger
logger = logging.getLogger(__name__)

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
    MAX_WORKSPACE_H5AD_FILES,
    _HIGH_MEMORY_ANALYSIS_RE,
    _high_memory_referenced_names,
    _build_resource_guard_preamble,
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







    
    
    def _build_data_context(self, state: AnalysisState) -> str:
        """Build context about available data"""
        data_parts = ["=== Available Data ===\n"]

        # Experimental design overview
        design = state.build_experimental_design_summary()
        if design:
            data_parts.append(design)
            data_parts.append("")

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
                    data_parts.append(f"    bulk_{bulk_var_name} = pd.read_csv('{f.file_path}', index_col=0, {sep_hint})")
                    data_parts.append(f"    # Bulk matrix (genes x samples). Print bulk_{bulk_var_name}.columns to infer sample grouping for pydeseq2 metadata.")
                    data_parts.append(f"    # CRITICAL: pydeseq2 requires (samples x genes) - transpose with .T before use.")
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
        """Build the static portion of the system prompt.

        Cached by (system_prompt, modality) — invalidated when either changes,
        preserving OpenAI prefix caching benefits while supporting hot reload
        and data-driven identity.
        """
        sp = self.system_prompt
        modality = self._infer_modality()
        cache_key = (sp, modality)
        if getattr(self, "_cached_static_sp_key", None) != cache_key:
            result = f"""{sp}

You are conducting {modality} analysis. You have access to tools:
1. execute_code - Run Python code (scanpy, anndata, numpy, pandas, matplotlib available)
   For plotting code: use `plt.savefig('name.jpg')` (or `sc.pl.*(save='_name.jpg')`) to save figures — the system intercepts this and auto-generates a companion high-res PDF, so do NOT save a separate PDF yourself. Do NOT call plt.close() or plt.show() — the system handles figure cleanup automatically. Prefer generate_publication_figure for figures intended for the final report (it adds PI review and correct figure typing); reserve execute_code savefig for quick exploratory plots. When you do use execute_code savefig, save with a bare filename — do NOT save figures into nested results/<dir>/figures/ paths.
2. consult_pi - Ask PI for biological decisions (analysis direction, pathway selection, interpretation)
3. search_literature - Search for marker genes, pathways, biological knowledge
4. annotate_clusters - Annotate ALL cell clusters at once with literature evidence (use INSTEAD of per-cluster consult_pi)
5. report_findings - Report conclusions ONLY after ALL assigned tasks are complete (including annotate_clusters in exploration)
6. generate_publication_figure - Generate publication-quality figures with PI review
7. fetch_gene_set - Fetch a named gene set from a versioned resource (any organism)
8. register_analysis_plan - Freeze a gene panel BEFORE scoring it; returns a plan_id
9. register_score - Register a computed score's methodology for the final report
10. register_mechanism_programs - Register data-driven mechanism programs with cited evidence
11. validate_species_genes - Check gene-name convention against the declared organism
12. lookup_documentation - Look up API docs for an installed Python function, OR
    ask a methodological question via method_question (which test is defensible,
    what a statistic assumes, whether a claim is supportable by this design).
    The method corpus records where the usual shortcuts fail in THIS stack --
    consult it before choosing a statistical approach, not after being refused.
   Returns concise summary (signature, key params, example).
   Use when unsure about API usage, especially for: decoupler, gseapy, scvelo, cellrank.
   Example: lookup_documentation(function_path="decoupler.get_collectri")
   IMPORTANT: Figure code MUST use actual analysis data from the sandbox (e.g., adata object).
   NEVER generate dummy/fabricated data. Only plot groups that exist in the experimental design.
   Inside the `code` parameter of THIS tool (generate_publication_figure): do NOT include plt.close(), plt.show(), or plt.savefig() — this tool wraps saving internally with its own filename. (Note: the separate execute_code tool DOES allow plt.savefig — the two tools have different conventions.)

   **PUBLICATION FIGURE MANDATORY CHECKLIST:**
   - When using sc.pl.umap/tsne, ALWAYS set show=False and capture the axes: `ax = sc.pl.umap(adata, ..., show=False, return_fig=True)`
   - ALWAYS manually add ax.set_xlabel('UMAP1') and ax.set_ylabel('UMAP2') to ALL embedding plots
   - If UMAP legends overlap data, use legend_loc='right margin' — NEVER manually set legend coordinates
   - All text elements must have fontsize >= 12pt (title >= 14pt)
   - Legend must NOT overlap data — place outside plot or use bbox_to_anchor
   - Use consistent naming across ALL panels (e.g., pick one canonical label per cell type and reuse it everywhere — never abbreviate it in some panels and spell it out in others)
   - No grid lines unless explicitly needed for the figure type
   - Always call ax.set_xticks() BEFORE ax.set_xticklabels() to ensure label count matches tick count

   **PLOT RULES & MULTI-MODAL FALLBACK (CRITICAL):**
   - Figures must have precise titles, readable (rotated when needed) labels, and colorblind-friendly palettes (e.g. 'viridis').
   - Downstream review nodes cannot read image pixels. AFTER saving any plot, you MUST `print()` the raw data backing the plot's core conclusion so that subsequent agents can interpret it from text.
     * Volcano plot → `print(top_5_up_genes, top_5_down_genes)` (with logFC and adj-p columns).
     * UMAP / clustering → `print(adata.obs['leiden'].value_counts())`.
     * Heatmap → `print(matrix.head())` or top differentially scored rows.
   - All biological interpretations and report claims MUST cite values from this printed text, not from the figure alone.

   **CODE-LEVEL SELF-REPAIR (CRITICAL):**
   - Wrap analysis steps in `try / except` and use adaptive thresholds before giving up.
     * If `sc.tl.rank_genes_groups`-style DEG returns < 5 hits, retry once within the same cell with a relaxed logFC/p-value threshold and `print()` both the original and relaxed results.
     * If cell-cell communication output is overwhelmingly large, automatically filter (e.g. p < 0.01 plus a relevance constraint) and `print()` the filter applied.
   - Always log the recovery action so the PI agent can see what was retried and why.

   **COMPOSITION / PROPORTION BARPLOT RULES:**
   - When plotting cell type proportions across conditions (e.g., Treatment vs Control, or any two experimental groups):
     1. The cell type categories on the y-axis must be CONDITION-INDEPENDENT (e.g., 'CellType_A', not 'CellType_A_Treatment')
     2. If cell types in adata.obs contain condition suffixes (e.g., 'CellType_A_Treatment', 'CellType_A_Control'), you MUST strip/merge them first
     3. Compute proportions as: (cells of type X in condition A) / (total cells in condition A)
     4. A correct composition plot should show MIXED colors in each bar — if any bar is 100% one color, check for tautological grouping
   - For pandas Categorical columns: call .cat.add_categories('NewCat') BEFORE assigning a new category value
   - Example pattern for merging:
     ct = adata.obs['cell_type'].astype(str)
     for cond in conditions:
         ct = ct.str.replace(f'_{{cond}}', '', regex=False)

SECURITY RESTRICTIONS:
- Do NOT use globals(), exec(), eval(), __import__(), or importlib — these are blocked by the sandbox. subprocess usage is restricted — do not use it for package installation.
- ALWAYS use standard `import xxx` or `from xxx import yyy` syntax. NEVER use `__import__('xxx')` — it will be blocked.
- If a package import fails (ModuleNotFoundError), report the missing dependency, explain which analysis step is limited, and use an already-installed alternative only when the substitution is scientifically valid. Do NOT use subprocess to install packages directly.
- Do NOT write files to disk using open(..., 'w') or similar — direct file writes are blocked by the sandbox. Saving via library methods (e.g. pandas df.to_csv(), scanpy sc.pl.*, matplotlib savefig) to the workspace directory IS allowed. Store intermediate data in adata.uns, adata.obs, adata.var, or Python variables.
- Do NOT call annotate_clusters in the same response as execute_code that generates markers.
  You must FIRST execute code and see the printed output, THEN call annotate_clusters in a separate step.

WORKFLOW GUIDELINES:
- Technical steps can be done autonomously: data loading, QC, normalization, PCA, clustering, UMAP, basic DE
- **Serendipity Check (Variance Partitioning)**: Before executing a highly targeted task, run a quick global variance check (e.g. correlate top PCs with each metadata column). If an unmentioned covariate (batch, technical artifact, unexpected cell state) drives variance far exceeding the chosen hypothesis, `print()` it and flag it as a "Counter-intuitive Finding" in the same cell.
- **MEMORY MANAGEMENT (CRITICAL to avoid OOM crashes)**:
  The adata object persists in the Jupyter kernel between code cells. Do NOT reload adata from .h5ad files unless absolutely necessary (e.g., after a kernel restart). Reuse the in-memory adata object. When you no longer need a large intermediate object (e.g., a subset), delete it with `del variable_name` and call `import gc; gc.collect()`.
  When saving checkpoints with adata.write(), keep at most 1-2 checkpoint files — do NOT accumulate many large .h5ad files.
- **REUSE EXISTING RESULTS (CRITICAL for chat follow-up analyses)**:
  Before running any analysis step, CHECK if the result already exists in the current session.
  The notebook environment may already contain results from previous analyses.
  * Cell type annotation: `'cell_type' in adata.obs.columns` — if True, SKIP rank_genes_groups + annotate_clusters
  * Differential expression: `'rank_genes_groups' in adata.uns` — if True, SKIP unless a DIFFERENT groupby is needed
  * Clustering: `'leiden' in adata.obs.columns` — if True, SKIP re-clustering
  * PCA: `'X_pca' in adata.obsm` — if True, SKIP sc.tl.pca
  * UMAP: `'X_umap' in adata.obsm` — if True, SKIP sc.tl.umap
  * Neighbors graph: `'neighbors' in adata.uns` — if True, SKIP sc.pp.neighbors
  * Batch correction: `'X_pca_harmony' in adata.obsm` (or similar) — if True, SKIP
  * Trajectory/Velocity: `'velocity' in adata.layers` — if True, SKIP scvelo steps
  * At the START of any analysis, run a quick diagnostic:
    print(f"obs columns: {{list(adata.obs.columns)}}")
    print(f"obsm keys: {{list(adata.obsm.keys())}}")
    print(f"uns keys: {{list(adata.uns.keys())}}")
    to see what's already available, then SKIP to the new analysis directly.
  * **OBS_NAMES / VAR_NAMES UNIQUENESS (CRITICAL):**
    After ANY of these operations, you MUST call adata.obs_names_make_unique() and adata.var_names_make_unique():
    - ad.concat() or anndata.concat()
    - Subsetting: adata[mask], adata[adata.obs['col'] == val]
    - Merging or joining AnnData objects
    Failure causes InvalidIndexError and corrupts all downstream analysis.
    Pattern: adata = ad.concat([a, b]); adata.obs_names_make_unique(); adata.var_names_make_unique()
  * Only REDO a step if the user EXPLICITLY requests it or if a prerequisite has changed
- **UMAP EXECUTION RULES (CRITICAL)**:
  * ALWAYS call sc.pp.neighbors() before sc.tl.umap() — verify with `assert adata.obsp.get('connectivities') is not None`
  * For datasets with n_obs > 5000, ALWAYS use `init_pos='random'` (e.g., `sc.tl.umap(adata, init_pos='random', random_state=42)`)
    This avoids spectral initialization which triggers slow Numba JIT compilation and can time out on large datasets.
    For single-cell data, 'random' and 'spectral' produce visually equivalent results (scanpy best practice).
  * Only use `init_pos='spectral'` for n_obs <= 5000
- **BATCH EFFECT CHECK**: When loading multiple samples/batches/conditions, assess batch effects BEFORE clustering:
  * After initial PCA/UMAP, visualize cells colored by batch/sample/condition to check for batch effects
  * If batch effects are apparent (samples cluster by batch rather than cell type), consult PI about correction strategy (harmonypy, bbknn, scvi, scanpy.pp.combat, etc.)
  * Do NOT skip this step when multiple samples exist — batch effects can dominate clustering results
- For CELL TYPE ANNOTATION, use annotate_clusters (NOT consult_pi):
  * After clustering and sc.tl.rank_genes_groups(), collect ALL cluster top markers into a dict
  * Call annotate_clusters ONCE with all cluster markers, tissue type, and organism
  * Apply the returned mapping to adata.obs['cell_type']
  * This is far more efficient than asking about each cluster separately
  * NEVER manually assign cell types via execute_code (e.g., adata.obs['cell_type'] = ...).
    Cell type assignment MUST go through annotate_clusters, which triggers user review.
    If annotate_clusters fails, retry it or consult_pi — do NOT work around it with code.
- For OTHER BIOLOGICAL DECISIONS, use consult_pi:
  * Whether batch correction is needed and which method to use
  * Which clusters show interesting changes?
  * What genes or pathways to analyze?
  * Whether to do trajectory/velocity analysis?
  * Interpretation of results
  * Which comparisons matter biologically?
  * Score selection: Which scoring method to use, which gene lists, which databases

SCORE COMPUTATION WORKFLOW (MANDATORY):
When you need to compute ANY score (gene signature, pathway enrichment, TF activity, custom score):
1. FIRST call consult_pi to confirm the scoring approach:
   - Question: "I plan to compute [score name] using [method]. Gene list source: [source]. Is this appropriate?"
   - Context: Include why this score is relevant to the current analysis
   - Options: List alternative methods if applicable
2. Call register_analysis_plan to FREEZE the panel before you compute it. Give the
   exact gene list, its source, the hypothesis, and what result would count against
   the hypothesis. It returns a plan_id.
3. After PI confirms/modifies, execute the computation via execute_code
4. IMMEDIATELY after, call register_score with that plan_id to record the methodology
This applies to: sc.tl.score_genes, gseapy.ssgsea, gseapy.prerank, gseapy.enrichr, decoupler, custom scoring functions.

WHY THE PLAN COMES FIRST:
  Curating a panel is a real judgement -- an ontology term routinely mixes genes that
  drive a process with genes that suppress it, and scoring them as one module answers
  nothing. Splitting such a set is exactly what you should do, and the partition edit
  operation exists for it. But the panel has to be fixed BEFORE the result is visible.
  Revising a panel after seeing which genes carry the signal is selection on the
  outcome, and it is indistinguishable in the final report from having got it right
  the first time -- unless the plan is on record. If you genuinely need to change the
  panel afterwards, do it and say why; the change is recorded either way.

PREFER A VERSIONED SET OVER A REMEMBERED ONE:
  Call fetch_gene_set rather than writing a gene panel from memory. It returns the
  genes plus a provenance string to pass straight into panel_source /
  gene_list_source, and it binds the genes as a sandbox variable so you never retype
  them -- retyping is how a fetched panel quietly becomes a remembered one again.
  Use list_sets=true with a partial identifier to discover what a collection holds
  instead of guessing an identifier.

  A panel assembled from memory is recorded as bespoke and carries less weight,
  because nobody -- including you in the next round -- can reconstruct it.

  But a curated set is NOT automatically the right panel. Ontology sets in
  particular are often broad and mixed-direction: the same set can contain genes
  that drive a process and genes that suppress it, so a high module score is
  equally consistent with the process running and with it being held in check.

  You do not have to fix that by hand. Call fetch_gene_set with
  partition_by_direction=true and the ontology's own regulation terms are used to
  split the set into direction-coherent panels -- positive, negative, ambiguous
  (annotated to both, so unable to support a directional claim) and core (the
  genes that carry out the process rather than tuning it). Each part comes back
  as a panel with its own provenance. Score them SEPARATELY; scoring them
  together is the failure the split exists to prevent.

  Only the biological-process branch of the ontology publishes the regulation
  terms this needs, so other collections refuse the split and say so. When the
  split is unavailable and the set is still direction-mixed, say that in the
  finding rather than reporting one number as if it were directional.

  For an organism the resource does not publish natively, the set is projected
  through orthology and the mapping rate comes back with it. A low rate means most
  of the original set did not survive and the survivors are not a random sample;
  treat such a set as exploratory and say so in any finding that rests on it.

DECLARE THE TESTING FAMILY:
  A GSEA q-value is computed across every gene set in the same call. The identical
  panel with the identical enrichment score comes out significant or not depending
  only on what else you passed alongside it -- padding a call with uninformative sets
  inflates the q-values of the informative ones. List every set going into the same
  call in testing_family so the result can be read.

GENE MODULE SCORING — USE FULL GENE SET (CRITICAL):
  sc.tl.score_genes() on HVG-filtered data will miss most genes. Always:
  1. Preserve raw before HVG filtering: adata.raw = adata.copy()  (do this ONCE, right after normalization)
  2. Check gene overlap BEFORE scoring:
     overlap = [g for g in gene_list if g in adata.raw.var_names]
     print(f"Gene overlap: {{len(overlap)}}/{{len(gene_list)}}")
  3. Score with use_raw=True: sc.tl.score_genes(adata, gene_list, score_name='...', use_raw=True)
  4. If overlap < 50% of gene_list, investigate: wrong organism? wrong gene naming convention?

SPECIES-AWARE GENE NAMES:
  Mouse genes: Title case (Actb, Rplp0, Gapdh)
  Human genes: UPPER CASE (ACTB, RPLP0, GAPDH)
  Check convention: print(adata.var_names[:10])
  Convert human→mouse: [g[0].upper() + g[1:].lower() for g in gene_list]

PACKAGE API LOOKUP:
  If unsure about the correct API for any package (decoupler, gseapy, scvelo, cellrank, etc.),
  call lookup_documentation BEFORE writing code.
  Example: lookup_documentation(function_path="decoupler.get_collectri")
  The system automatically consults the Technique Agent for documentation when it detects a
  repeated OR caught-and-skipped API error, but you must NOT rely on that as an excuse to skip.
  NEVER swallow an API error (AttributeError / TypeError / ImportError / ModuleNotFoundError) with
  try/except and then record a skip decision. An API or version mismatch (e.g. a function that does
  not exist in the installed version) is FIXABLE: call lookup_documentation for the failing function,
  correct the call, and retry. Only skip an analysis when the DATA itself is incompatible (e.g. no
  spliced/unspliced layers, no overlapping barcodes), and even then fall back to an equivalent method
  (e.g. DPT/PAGA for trajectory) instead of dropping the analysis entirely.

DEFENSIVE CODING (CRITICAL — prevents KeyError / empty DataFrame crashes):
  When iterating over clusters or cell types for DE/enrichment analysis, ALWAYS check for empty results:
    df = sc.get.rank_genes_groups_df(adata, group=cluster)
    if df.empty or 'names' not in df.columns:
        print(f"WARNING: No DE genes for {{cluster}}, skipping")
        continue
    genes = df.head(15)['names'].tolist()
  This applies to: rank_genes_groups_df, gseapy.enrichr, gseapy.prerank, pathway analysis, any per-group iteration.
  NEVER assume a DataFrame slice will be non-empty — always guard with `.empty` check or try/except.

CELL TYPE ANNOTATION WORKFLOW:
1. Run sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
2. Extract top 10-20 marker genes per cluster into a dict AND PRINT IT:
   markers = {{}}
   for cluster in adata.obs['leiden'].cat.categories:
       df = sc.get.rank_genes_groups_df(adata, group=cluster)
       if df.empty:
           continue
       genes = df.head(15)['names'].tolist()
       markers[cluster] = genes
   import json
   print(json.dumps(markers, indent=2))
3. IMPORTANT: You MUST see the printed markers output BEFORE calling annotate_clusters.
   Do NOT call annotate_clusters in the same step as the code that generates markers.
4. Determine tissue and organism from the research context or data metadata
5. Call annotate_clusters(cluster_markers=markers, tissue="skin", organism="human")
   where markers is the LITERAL dict content you saw in the print output.
6. Apply the returned mapping: adata.obs['cell_type'] = adata.obs['leiden'].astype(str).map(cluster_to_celltype).fillna('Unknown')
7. Generate UMAP colored by cell_type

=== Working Directory ===
Your Jupyter kernel is running in: {self.sandbox.working_dir}
A variable `_WORKSPACE_DIR` (pathlib.Path) is pre-defined in the kernel and points to this directory.
All output files MUST be saved using `_WORKSPACE_DIR` as the root. NEVER use relative paths like './' or '../'.
- Figures: in execute_code, call `plt.savefig('name.jpg')` or `sc.pl.*(save='_name.jpg')` with a bare filename — files land under `_WORKSPACE_DIR/figures` automatically and a companion PDF is generated. Do NOT manually save a separate PDF, and do NOT save figures into nested results/<dir>/figures/ paths. For report-quality figures, use generate_publication_figure instead. (Inside generate_publication_figure's `code`, the tool wraps saving — see that tool's notes.)
- Tables/CSV results: save to `_WORKSPACE_DIR / 'tables'`
- H5AD files: save to `_WORKSPACE_DIR`
When writing files, ALWAYS use the `_WORKSPACE_DIR` variable for the path:
  (_WORKSPACE_DIR / 'tables').mkdir(exist_ok=True)
  df.to_csv(_WORKSPACE_DIR / 'tables' / 'filename.csv')
  adata.write(_WORKSPACE_DIR / 'output.h5ad')
Do NOT use string literals for paths (e.g., './output.csv', '/workspace/results/').
Do NOT call open('filename', 'w') with bare filenames — use open(_WORKSPACE_DIR / 'tables' / 'filename', 'w').

**MISSING DEPENDENCIES**:
If an optional package is missing, do not install packages directly from notebook code. Report the missing dependency, explain which analysis step is limited, and use an already-installed alternative only when the substitution is scientifically valid.

**CRITICAL: DATA PREPROCESSING (MUST DO BEFORE PCA)**
Before running sc.tl.pca() or any dimensionality reduction:
1. Check for NaN: print(f"NaN in X: {{np.isnan(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X).sum()}}")
2. If NaN exists, fix it:
   - sc.pp.filter_genes(adata, min_cells=3)
   - adata.X = np.nan_to_num(adata.X, nan=0.0) if dense
   - adata.X.data = np.nan_to_num(adata.X.data, nan=0.0) if sparse
3. NEVER skip this step - PCA will fail with NaN values!

**EXPRESSION-SCALE CONTRACT (ENFORCED AT RUNTIME)**:
- Differential expression, gene scoring, and NMF MUST use a normalized continuous
  expression representation, never raw integer counts.
- Preserve raw counts in a layer or `.raw`, create an explicit normalized copy
  (for example library-size normalization followed by log1p), and use that copy.
- If you change normalization or replace `.X`, regenerate every downstream table
  and figure from that representation. The runtime invalidates earlier same-round
  publication artifacts when it detects such a representation change.

**HARMONY BATCH CORRECTION (CRITICAL — dimension mismatch prevention)**:
When using harmonypy for batch correction, ALWAYS verify dimensions match:
  import harmonypy as hm
  # Use the SAME adata that was used for PCA — do NOT subset/filter between PCA and Harmony
  ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, 'batch_column')
  # Verify shape BEFORE assignment
  print(f"Harmony output shape: {{ho.Z_corr.T.shape}}, adata n_obs: {{adata.n_obs}}")
  assert ho.Z_corr.T.shape[0] == adata.n_obs, "Shape mismatch! Do not assign."
  adata.obsm['X_pca_harmony'] = ho.Z_corr.T
  # Then recompute neighbors/UMAP using the corrected embedding
  sc.pp.neighbors(adata, use_rep='X_pca_harmony')
  sc.tl.umap(adata)
NEVER filter cells between sc.tl.pca() and hm.run_harmony() — this causes the dimension mismatch.

**CRITICAL: REPRODUCIBILITY IS MANDATORY**
- ALWAYS use random_state=42 or seed=42 in EVERY function that accepts it
- Examples: sc.tl.umap(..., random_state=42), sc.tl.leiden(..., random_state=42),
  sc.tl.pca(..., random_state=42), train_test_split(..., random_state=42)
- NEVER omit the random_state parameter in stochastic functions
- Use n_jobs=1 when determinism is critical
- Print key metrics and results
- Generate informative visualizations
- Use report_findings when completing an analysis phase
"""
            self._cached_static_sp_key, self._cached_static_sp = cache_key, result
        return self._cached_static_sp


    # ---- Context compaction (volume-based) ----

    COMPACT_THRESHOLD_TOKENS = 150_000
    COMPACT_TARGET_TOKENS = 105_000
    RECENT_PROTECT = 6
    _CHARS_PER_TOKEN = 3.5
    _COMPACT_TRUNCATE_BUDGETS = [64 * 1024, 32 * 1024, 16 * 1024]

    @staticmethod
    def _estimate_context_tokens(messages: List[Dict[str, Any]]) -> int:
        total_chars = 0
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                total_chars += len(json.dumps(content))
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict):
                    args = tc.get("function", {}).get("arguments", "")
                    if isinstance(args, str):
                        total_chars += len(args)
            total_chars += len(m.get("name", "") or "")
            total_chars += len(m.get("tool_call_id", "") or "")
        return int(total_chars / BioinformaticsAgent._CHARS_PER_TOKEN)

    @staticmethod
    def _compact_context(messages: List[Dict[str, Any]], target_tokens: int) -> Tuple[List[Dict[str, Any]], bool]:
        estimated = BioinformaticsAgent._estimate_context_tokens(messages)
        if estimated <= target_tokens:
            return messages, False

        cls = BioinformaticsAgent
        budget_list = cls._COMPACT_TRUNCATE_BUDGETS
        RECENT_PROTECT = cls.RECENT_PROTECT

        result = [dict(m) for m in messages]

        # --- Phase 1: truncate large tool/user content (never assistant tool_calls.arguments) ---
        anchor_count = 2
        if len(result) > 1:
            c1 = result[1].get("content", "")
            is_dynamic = False
            if isinstance(c1, str):
                is_dynamic = c1.startswith(_DYNAMIC_CONTEXT_PREFIX)
            elif isinstance(c1, list):
                is_dynamic = any(
                    p.get("text", "").startswith(_DYNAMIC_CONTEXT_PREFIX)
                    for p in c1 if isinstance(p, dict)
                )
            if not is_dynamic:
                anchor_count = 1

        protect_end = max(anchor_count, len(result) - RECENT_PROTECT)

        for budget in budget_list:
            if cls._estimate_context_tokens(result) <= target_tokens:
                break
            candidates = []
            for idx in range(anchor_count, protect_end):
                m = result[idx]
                role = m.get("role", "")
                content = m.get("content")
                if role not in ("tool", "user"):
                    continue
                if not isinstance(content, str) or len(content) <= budget:
                    continue
                candidates.append((idx, len(content)))
            candidates.sort(key=lambda x: -x[1])
            for idx, _ in candidates:
                if cls._estimate_context_tokens(result) <= target_tokens:
                    break
                old_content = result[idx]["content"]
                new_content = truncate_for_llm(old_content, max_chars=budget)
                if len(new_content) < len(old_content):
                    result[idx] = {**result[idx], "content": new_content}

        # --- Phase 2: drop oldest mid-section with summary ---
        if cls._estimate_context_tokens(result) > target_tokens:
            system_msg = result[0]
            dynamic_msg = result[1] if anchor_count >= 2 else None
            tail_start = max(anchor_count, len(result) - RECENT_PROTECT)

            # Walk backward from tail boundary to avoid orphaning tool_result
            cut = tail_start
            while cut > anchor_count and result[cut].get("role") == "tool":
                cut -= 1

            if cut > anchor_count:
                dropped = result[anchor_count:cut]
                summary = BioinformaticsAgent._summarize_dropped_messages(dropped)
                summary_msg = {
                    "role": "user",
                    "content": f"[Previous analysis summary (compacted)]\n{summary}"
                }
                if dynamic_msg:
                    result = [system_msg, dynamic_msg, summary_msg] + result[cut:]
                else:
                    result = [system_msg, summary_msg] + result[cut:]

        # --- Phase 3: fallback — truncate protected window if still over ---
        if cls._estimate_context_tokens(result) > target_tokens:
            fallback_start = max(0, len(result) - RECENT_PROTECT)
            for idx in range(fallback_start, len(result)):
                m = result[idx]
                role = m.get("role", "")
                content = m.get("content")
                if role not in ("tool", "user"):
                    continue
                if not isinstance(content, str) or len(content) <= 16 * 1024:
                    continue
                new_content = truncate_for_llm(content, max_chars=16 * 1024)
                if len(new_content) < len(content):
                    result[idx] = {**result[idx], "content": new_content}
                if cls._estimate_context_tokens(result) <= target_tokens:
                    break

        return result, cls._estimate_context_tokens(result) < estimated

    @staticmethod
    def _summarize_dropped_messages(dropped: List[Dict[str, Any]]) -> str:
        """Summarize dropped messages for the trimming summary.

        Pure local extraction — no LLM call. Keeps output under ~500 tokens.
        Handles non-string content (e.g. multimodal list, None) gracefully.
        """
        tool_counts: Dict[str, int] = {}
        error_count = 0
        code_snippets: List[str] = []

        for msg in dropped:
            role = msg.get("role", "")
            content = msg.get("content")

            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # Multimodal content — extract text parts
                text = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            else:
                text = ""

            # Count tool calls from assistant messages
            if role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls and isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            name = tc.get("function", {}).get("name", "unknown")
                        else:
                            func = getattr(tc, "function", None)
                            name = getattr(func, "name", "unknown") if func else "unknown"
                        tool_counts[name] = tool_counts.get(name, 0) + 1

            # Count errors in tool results
            if role == "tool" and text and "error" in text[:300].lower():
                error_count += 1

            # Collect code execution descriptions from assistant reasoning
            if role == "assistant" and text and "execute_code" in text[:500]:
                # Extract first line of reasoning as a brief description
                first_line = text.strip().split("\n")[0][:120]
                if first_line:
                    code_snippets.append(first_line)

        parts = []
        if tool_counts:
            parts.append("Tools used: " + ", ".join(f"{k} x{v}" for k, v in tool_counts.items()))
        if error_count:
            parts.append(f"Errors encountered: {error_count}")
        if code_snippets:
            # Show last few code descriptions
            recent = code_snippets[-3:]
            parts.append("Recent steps: " + "; ".join(recent))

        return "\n".join(parts) if parts else "Previous analysis steps completed."











    # Patterns that must never appear in validation cells (extra safeguard; the
    # copy + per-iteration invariant check are the primary mechanisms).
    _VALIDATION_DENYLIST = (
        "sc.pp.filter_", "sc.tl.leiden", "sc.tl.louvain", "sc.tl.umap", "rank_genes_groups",
        "sc.tl.dpt", "scvelo", "velocity", "sc.pp.normalize_total", "sc.pp.log1p",
        "sc.pp.scale", ".write(", ".write_h5ad",
    )





    async def _compute_database_priors(self, adata_var, cluster_key, rank_key,
                                       organism, workspace_dir) -> dict:
        """sandbox 内算 ORA 先验。任何失败返回 {}（绝不抛、绝不阻断注释）。"""
        try:
            if not marker_priors_write(organism, workspace_dir):
                return {}
            src = _ORA_KERNEL_SRC
            call = (
                f"\n_priors = compute_ora_priors({adata_var}, "
                f"'tables/marker_priors_input.tsv', {cluster_key!r}, {rank_key!r})\n"
                f"import json as __j; print(__j.dumps(_priors))\n"
            )
            result = await self.sandbox.execute(src + call, save_to_history=False,
                                                skip_safety_check=True)
            out = getattr(result, "output", "") or ""
            line = next((ln for ln in out.strip().splitlines()[::-1]
                         if ln.strip().startswith("{")), None)
            return _json.loads(line) if line else {}
        except Exception as e:
            logger.warning(f"[{self.session_id}] DB priors computation skipped: {e}")
            return {}










    async def run_autonomous_analysis(
        self,
        state: AnalysisState,
        initial_instruction: str,
        max_steps: int = 200,
        figure_review_callback: Optional[Callable[[PublicationFigure], Awaitable[Dict[str, Any]]]] = None,
        supervisor_callback: Optional[Callable] = None,
        resume_from_checkpoint: bool = True,
        planning_stage: PlanStage = "full",  # "stage1", "stage2", or "full"
    ) -> Dict[str, Any]:
        """
        Run analysis with autonomous technical steps, PI consultation, and step-level checkpointing.

        Args:
            state: Current analysis state
            initial_instruction: The instruction to execute
            max_steps: Maximum number of steps (default 200 to accommodate full analysis + figure review loops)
            figure_review_callback: Optional callback for PI to review generated figures
            resume_from_checkpoint: Whether to attempt resuming from checkpoint (default True)

        Returns:
            Dict with analysis results
        """
        self.refresh_runtime_config_if_needed()
        # Use provided callback or fall back to instance callback
        review_callback = figure_review_callback or self.figure_review_callback
        _supervisor_callback = supervisor_callback or self.supervisor_callback

        # Initialize checkpoint manager for step-level recovery
        if hasattr(self.sandbox, 'state_dir') and self.sandbox.state_dir:
            state_dir = str(self.sandbox.state_dir)
        else:
            # Use sandbox working directory as base for states
            state_dir = str(Path(self.sandbox.working_dir) / "states")
            Path(state_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_mgr = CheckpointManager(session_id=state.id, state_dir=state_dir)

        # Try to resume from checkpoint if enabled
        checkpoint = None
        checkpoint_valid = False
        if resume_from_checkpoint:
            try:
                checkpoint = await checkpoint_mgr.load()
                if checkpoint and not checkpoint.get('completed', False):
                    # Validate checkpoint has required fields with correct types
                    validation_errors = []

                    # Check step_number is a valid integer
                    step_num = checkpoint.get('step_number')
                    if not isinstance(step_num, int) or step_num < 0:
                        validation_errors.append(f"step_number invalid: {step_num}")

                    # Check messages is a non-empty list
                    messages = checkpoint.get('messages')
                    if not isinstance(messages, list) or len(messages) < 2:
                        validation_errors.append(f"messages invalid: expected list with >= 2 items")

                    # Check findings is a list
                    findings = checkpoint.get('findings')
                    if not isinstance(findings, list):
                        validation_errors.append(f"findings invalid: expected list")

                    # Check figures is a list
                    figures = checkpoint.get('figures')
                    if not isinstance(figures, list):
                        validation_errors.append(f"figures invalid: expected list")

                    # Check round_number matches current round
                    cp_round = checkpoint.get('round_number')
                    if cp_round is not None and cp_round != state.current_round:
                        validation_errors.append(
                            f"round_number mismatch: checkpoint={cp_round}, current={state.current_round}"
                        )

                    # Check planning_stage matches. stage1 and stage2 share the same
                    # round_number, so without this a stage1 checkpoint that wasn't
                    # cleared (e.g. stage1 ended max_steps_reached/completed_with_fallback,
                    # which keep completed=False) would be wrongly resumed when stage2
                    # starts — feeding the stage2 run the stale stage1 conversation
                    # instead of the stage2 instruction. Only reject on an explicit
                    # mismatch so pre-fix checkpoints (no planning_stage) still resume.
                    cp_stage = checkpoint.get('planning_stage')
                    if cp_stage is not None and cp_stage != planning_stage:
                        validation_errors.append(
                            f"planning_stage mismatch: checkpoint={cp_stage}, current={planning_stage}"
                        )

                    if not validation_errors:
                        checkpoint_valid = True
                        logger.info(f"[Bioinformatics] Resuming from checkpoint at step {checkpoint['step_number']}")
                        await self._notify("status", f"Resuming analysis from step {checkpoint['step_number']}...")
                    else:
                        logger.warning(f"[Bioinformatics] Checkpoint validation failed: {validation_errors}, starting fresh")
                        checkpoint_mgr.clear()
                        checkpoint = None
            except Exception as e:
                logger.exception("[Bioinformatics] Failed to load checkpoint")
                # Only clear if it's a data corruption issue, not a transient error
                try:
                    checkpoint_mgr.clear()
                    logger.warning("[Bioinformatics] Cleared corrupted checkpoint data")
                except Exception as clear_err:
                    logger.error(f"[Bioinformatics] Failed to clear checkpoint: {clear_err}")
                checkpoint = None
                await self._notify("status", "Previous checkpoint corrupted, starting fresh analysis...")

        logger.info(f"[Bioinformatics] Starting autonomous analysis with max_steps={max_steps}")
        # Start sandbox if needed
        if self.sandbox.km is None:
            await self._notify("status", "Starting analysis environment...")
            start_result = await self.sandbox.start()
            logger.info("[Bioinformatics] Sandbox started")
            if start_result.get("restore_failed"):
                logger.warning("[Bioinformatics] Previous session restore failed, starting with fresh environment")
                await self._notify("status", "Note: Previous session could not be restored, starting fresh...")
            elif start_result.get("session_restored") and not self.code_history:
                # Populate code_history from restored notebook cells so _build_notebook_context()
                # shows previous analysis, preventing the agent from restarting from scratch
                restored_cells = self.sandbox.get_code_history()
                for cell in restored_cells:
                    outputs = cell.get('outputs', [])
                    output_text = '\n'.join(
                        o.get('text', '') for o in outputs if o.get('type') == 'stream'
                    )
                    error_text = '\n'.join(
                        o.get('evalue', '') for o in outputs if o.get('type') == 'error'
                    )
                    figures = [
                        o.get('data', {}).get('image/png', '')
                        for o in outputs
                        if o.get('type') == 'display_data' and 'image/png' in o.get('data', {})
                    ]
                    self.code_history.append(CodeExecution(
                        code=cell.get('code', ''),
                        output=output_text if output_text else None,
                        error=error_text if error_text else None,
                        figures=figures,
                        execution_time=None,
                    ))
                logger.info(f"[Bioinformatics] Restored {len(self.code_history)} cells into code_history from notebook state")

        # Build initial prompt — split into static prefix (cacheable) and dynamic context
        self._current_analysis_state = state
        static_system_prompt = self._static_system_prompt
        dynamic_context = self._build_dynamic_context(state)
        dynamic_ctx_msg = {"role": "user", "content": f"{_DYNAMIC_CONTEXT_PREFIX}\n{dynamic_context}"}

        # Initialize or restore state from checkpoint
        restored_messages_rebuilt = False
        if checkpoint and checkpoint_valid:
            messages = checkpoint.get('messages', [])
            # Validate message structure: each must have role and content
            valid_messages = (
                all(
                    isinstance(m, dict) and "role" in m and ("content" in m or "tool_calls" in m)
                    for m in messages
                )
                and tool_calls_are_paired(messages)
            ) if messages else False
            if len(messages) < 2 or not valid_messages:
                logger.warning("[Bioinformatics] Checkpoint messages invalid (len=%d, valid=%s), using fresh messages",
                               len(messages), valid_messages)
                messages = [
                    {"role": "system", "content": static_system_prompt},
                    dynamic_ctx_msg,
                    {"role": "user", "content": f"Analysis instruction: {initial_instruction}"}
                ]
                restored_messages_rebuilt = True
            else:
                # Migrate old checkpoint format: replace monolithic system prompt
                # with split static + dynamic structure for better prefix caching.
                # Old format: [system(full), user(instruction), ...]
                # New format: [system(static), user(context), user(instruction), ...]
                if (len(messages) >= 2
                        and messages[0].get("role") == "system"
                        and messages[1].get("role") == "user"
                        and isinstance(messages[1].get("content"), str)
                        and messages[1]["content"].startswith("Analysis instruction:")):
                    logger.info("[Bioinformatics] Migrating checkpoint to split prompt format for better caching")
                    messages[0] = {"role": "system", "content": static_system_prompt}
                    messages.insert(1, dynamic_ctx_msg)
            all_findings = checkpoint.get('findings', [])
            all_figures = checkpoint.get('figures', [])
            step_count = checkpoint.get('step_number', 0)
            # Give resumed agent its remaining budget based on original total_steps cap.
            # Use the larger of (caller's max_steps, checkpoint's total_steps) to avoid shrinking budget.
            # max_steps stays at original_total so the agent has (original_total - step_count) steps left.
            original_total = max(max_steps, checkpoint.get('total_steps', max_steps))
            max_steps = original_total
            self.analysis_context = checkpoint.get('analysis_context', {})
            analysis_log = [AnalysisStep(**s) for s in checkpoint.get('analysis_log', [])]
            restored_code_history = checkpoint.get('code_history', [])
            if restored_code_history:
                self.code_history = [CodeExecution(**ce) for ce in restored_code_history]
                logger.info(f"[Bioinformatics] Restored {len(self.code_history)} code_history entries from checkpoint")
            logger.info(f"[Bioinformatics] Restored state: step={step_count}, findings={len(all_findings)}, figures={len(all_figures)}, new max_steps={max_steps}")
        else:
            messages = [
                {"role": "system", "content": static_system_prompt},
                dynamic_ctx_msg,
                {"role": "user", "content": f"Analysis instruction: {initial_instruction}"}
            ]
            # Inject approved analysis plan if available
            if state.current_plan and state.current_plan.status == "approved":
                messages.append({
                    "role": "user",
                    "content": build_approved_plan_context(state.current_plan),
                })
            all_findings = []
            all_figures = []
            step_count = 0
            analysis_log = []
            # Reset per-run state to avoid pollution from previous rounds
            # (process() does this but workflow calls run_autonomous_analysis directly)
            self.code_history = []
            self.analysis_context = {}
            self.last_result = None

        def _active_plan_id() -> Optional[str]:
            plan = state.current_plan
            if plan and plan.status == "approved":
                return str(plan.id)
            return None

        async def _save_executor_checkpoint(
            step_name: str,
            *,
            save_sandbox: bool = False,
        ) -> None:
            """Persist the executor transcript independently of planning history."""
            checkpoint_data = {
                'step_name': step_name,
                'session_id': state.id,
                'round_number': state.current_round,
                'planning_stage': planning_stage,
                'step_number': step_count,
                'total_steps': max_steps,
                'active_plan_id': _active_plan_id(),
                'messages': messages,
                'findings': all_findings,
                'figures': all_figures,
                'code_history': [ce.model_dump(mode="json") for ce in self.code_history],
                'analysis_log': [step.model_dump(mode="json") for step in analysis_log],
                'analysis_context': self.analysis_context,
                'completed': False,
            }
            await checkpoint_mgr.save(checkpoint_data)
            if save_sandbox:
                await self.sandbox.save_session()

        # A resumed executor must keep its original tool transcript, but an
        # approved replacement plan has to be made visible before the next LLM
        # call. A plan ID in the checkpoint prevents duplicate injection on
        # ordinary retries. Legacy/rebuilt checkpoints receive the plan once.
        if checkpoint and checkpoint_valid and state.current_plan:
            checkpoint_plan_id = checkpoint.get('active_plan_id')
            if restored_messages_rebuilt:
                checkpoint_plan_id = None
            if inject_revised_plan_context(
                messages,
                state.current_plan,
                checkpoint_plan_id,
            ):
                try:
                    await _save_executor_checkpoint(
                        f"step_{step_count}_plan_{state.current_plan.id}",
                    )
                    logger.info(
                        "[Bioinformatics] Injected approved plan %s into resumed "
                        "executor checkpoint (previous plan=%s)",
                        state.current_plan.id,
                        checkpoint_plan_id,
                    )
                except Exception as e:
                    logger.warning(
                        "[Bioinformatics] Failed to persist resumed plan context: %s",
                        e,
                    )

        completed = False
        annotation_completed = False  # Mandatory step tracker for exploration phase
        # Restore annotation_completed from analysis_log to avoid Gate 0
        # falsely rejecting report_findings after resume. Failed annotation
        # attempts also write an annotate_clusters step (with summary prefixed
        # "[ANNOTATION FAILED]"), so exclude those.
        if analysis_log and any(
            s.step_type == "annotate_clusters"
            and not (s.annotation_result_summary or "").startswith("[ANNOTATION FAILED]")
            for s in analysis_log
        ):
            annotation_completed = True
        annotation_retry_required = False
        annotation_retry_attempts = 0
        mechanism_programs: List[DataDrivenMechanismProgram] = []
        mechanism_programs_registered = bool(analysis_log and any(
            s.step_type == "register_mechanism_programs"
            and not (s.findings_summary or "").startswith("REJECTED")
            for s in analysis_log
        ))
        mechanism_gate_reject_count = 0
        stats_gate_reject_count = 0  # statistical-validity checks
        score_gate_reject_count = 0  # register_score provenance / plan checks
        # register_score rejections are recoverable in one retry for provenance
        # but need a prior register_analysis_plan call for plan_id, so give the
        # agent room to make that connection before accepting an unverified score.
        MAX_SCORE_GATE_REJECTIONS = 4
        MAX_MECHANISM_REJECTIONS = 2
        # Bounded like the other report_findings gates: a check the agent
        # cannot satisfy must not trap the loop, and an escape is recorded.
        MAX_STATS_GATE_REJECTIONS = 3
        integration_reject_count = 0  # Track rejections for bulk/scRNA-seq integration gate
        MAX_INTEGRATION_REJECTIONS = 2  # Escape hatch: allow through after N rejections
        deconv_reject_count = 0  # Track rejections for cell-type deconvolution gate
        MAX_DECONV_REJECTIONS = 2  # Escape hatch: allow through after N rejections
        innovation_bulk_reject_count = 0
        MAX_INNOVATION_BULK_REJECTIONS = 2
        evidence_gate_degraded = False
        reported_evidence_ids: List[str] = []
        llm_error: Optional[str] = None
        text_response_count = 0
        MAX_TEXT_RESPONSES = 2
        empty_response_count = 0
        MAX_EMPTY_RESPONSES = 3  # transient upstream truncation → bounded retry
        figure_iteration_count = 0
        MAX_FIGURE_ITERATIONS = 5  # Max figure-only iterations excluded from step count
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 5  # Force report_findings after this many consecutive code errors
        total_errors = 0
        MAX_TOTAL_ERRORS = 50  # Hard limit: force report_findings after this many total errors
        # Initialize before loop to prevent NameError when max_steps <= 0
        _plan_revision_needed = False
        _plan_revision_context: Dict[str, Any] = {}
        force_break = False  # Set by supervisor force-skip to hard-exit the loop

        while step_count < max_steps and not completed:
            # Cooperative pause: block here between steps when the workflow has
            # been paused. The Jupyter kernel naturally serializes against
            # sandbox.save_session(), so any in-flight execute() finishes first.
            if self._pause_event is not None and not self._pause_event.is_set():
                logger.info(
                    f"[Bioinformatics] Pause requested at step {step_count}, "
                    f"waiting for resume"
                )
                try:
                    await self._notify("status", "Analysis paused, awaiting resume...")
                except Exception:
                    pass
                await self._pause_event.wait()
                logger.info(f"[Bioinformatics] Resumed at step {step_count}")
                try:
                    await self._notify("status", "Analysis resumed")
                except Exception:
                    pass

            # Restore safe-point between sandbox steps (shared coordinator).
            session_id = getattr(self, "session_id", None) or getattr(state, "id", None)
            if session_id:
                from backend.services.snapshot_v2.safe_point import (
                    get_snapshot_safe_point_manager,
                )

                sp = get_snapshot_safe_point_manager()
                if sp.has_pending_request(session_id):
                    gen = sp.get_pending_generation(session_id)
                    if gen is not None:
                        await sp.acknowledge_safe_point_request(
                            session_id, gen, role="workflow",
                        )
                if sp.is_paused(session_id):
                    logger.info(
                        f"[Bioinformatics] Restore safe-point active at step "
                        f"{step_count}; freezing until release"
                    )
                    await sp.wait_for(session_id)
                    if sp.is_superseded(session_id):
                        # A restore committed while we were frozen. This run's
                        # graph position and checkpoint connection both predate
                        # it, so continuing would write pre-restore state
                        # forward over what the restore installed.
                        from backend.workflow.helpers import (
                            WorkflowSupersededError,
                        )

                        logger.info(
                            f"[Bioinformatics] Run superseded by a snapshot "
                            f"restore at step {step_count}; unwinding"
                        )
                        raise WorkflowSupersededError(
                            "snapshot restore replaced this run's checkpoint"
                        )

            step_count += 1
            force_break = False  # Reset per-iteration

            # Save checkpoint periodically
            if checkpoint_mgr.should_checkpoint(step_count):
                try:
                    await _save_executor_checkpoint(
                        f'step_{step_count}',
                        save_sandbox=True,
                    )
                except Exception as e:
                    logger.warning(f"[Bioinformatics] Checkpoint save failed at step {step_count}: {e}")
            
            self._set_llm_context()

            # Trim message history to prevent unbounded growth.
            # Strategy: trim aggressively (80 → 50) to reduce trim frequency,
            # preserving messages[0] (static system prompt) and messages[1]
            # (frozen dynamic context) for OpenAI prefix caching.
            MAX_MESSAGES = 80
            TRIM_TARGET = 50  # Large trim gap avoids frequent cache-breaking trims
            if len(messages) > MAX_MESSAGES:
                system_msg = messages[0]
                content_val = messages[1].get("content", "") if len(messages) > 1 else ""
                is_dynamic = False
                if isinstance(content_val, str):
                    is_dynamic = content_val.startswith(_DYNAMIC_CONTEXT_PREFIX)
                elif isinstance(content_val, list):
                    is_dynamic = any(p.get("text", "").startswith(_DYNAMIC_CONTEXT_PREFIX) for p in content_val if isinstance(p, dict))
                dynamic_msg = messages[1] if len(messages) > 1 and is_dynamic else None
                anchor_count = 2 if dynamic_msg else 1

                # Keep anchor messages + summary + last (TRIM_TARGET - anchor_count - 1) messages
                keep_tail = TRIM_TARGET - anchor_count - 1  # -1 for the summary message
                cut = len(messages) - keep_tail
                # Walk backward to avoid orphaning tool_result messages
                while cut < len(messages) and messages[cut].get("role") == "tool":
                    cut -= 1
                if cut < anchor_count:
                    cut = anchor_count

                dropped = messages[anchor_count:cut]
                summary = self._summarize_dropped_messages(dropped)
                summary_msg = {
                    "role": "user",
                    "content": f"[Previous analysis summary]\n{summary}"
                }

                if dynamic_msg:
                    messages = [system_msg, dynamic_msg, summary_msg] + messages[cut:]
                else:
                    messages = [system_msg, summary_msg] + messages[cut:]
                logger.info(f"Trimmed message history: dropped {len(dropped)} messages, {len(messages)} remaining")

            # Volume-based compaction: truncate large tool/user content and
            # drop+summarize oldest messages when estimated token count
            # exceeds the threshold. Complements count-based trim above
            # (which only fires at 80 messages; this session failed at 53
            # messages / ~257K tokens due to voluminous execute_code output).
            messages, _compacted = self._compact_context(
                messages, self.COMPACT_THRESHOLD_TOKENS
            )
            if _compacted:
                logger.info(
                    f"[Bioinformatics] Context compacted: "
                    f"estimated {self._estimate_context_tokens(messages)} tokens"
                )

            # If the previous annotate_clusters call failed, prepend a high-priority
            # reminder to force the agent back to the annotation workflow.
            if annotation_retry_required and annotation_retry_attempts <= 3:
                nudge = (
                    f"[System] annotate_clusters has not completed (attempt "
                    f"{annotation_retry_attempts}). Your next action MUST be to either "
                    f"(a) run execute_code that prepares the sandbox (e.g., assign "
                    f"adata.obs['leiden'] from your actual clustering column, ensure "
                    f"sc.tl.rank_genes_groups has populated adata.uns), or (b) call "
                    f"annotate_clusters again with the correct cluster_key/rank_key "
                    f"arguments. Do NOT call report_findings, consult_pi about other "
                    f"topics, or generate_publication_figure until annotation completes."
                )
                nudge_key = f"__annot_retry_nudge_{annotation_retry_attempts}"
                if nudge_key not in self._recent_sup_feedback:
                    messages.append({"role": "user", "content": nudge})
                    self._recent_sup_feedback.add(nudge_key)

            if annotation_retry_required and annotation_retry_attempts > 3:
                logger.warning(
                    f"[Bioinformatics] annotate_clusters retries exhausted "
                    f"({annotation_retry_attempts}); halting loop for human review."
                )
                await self._notify(
                    "status",
                    "Cell type annotation could not be completed automatically. "
                    "Please review the session and resume manually.",
                )
                break

            # When the model has returned consecutive empty responses, force
            # tool_choice="required" so it cannot produce another zero-content
            # turn. A message-level nudge alone is insufficient for models that
            # silently decline to act (content='', tool_calls=None, ~4
            # completion tokens); "required" eliminates the empty-response
            # option entirely.
            effective_tool_choice = self._compute_tool_choice(
                text_response_count, MAX_TEXT_RESPONSES
            )
            if empty_response_count > 0 and effective_tool_choice == "auto":
                effective_tool_choice = "required"
                logger.info(
                    f"[Bioinformatics] Escalating tool_choice to 'required' "
                    f"after {empty_response_count} empty response(s)"
                )
            try:
                response = await self.llm_client.chat_with_tools(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice=effective_tool_choice,
                    temperature=self.config.temperature,
                    max_tokens=32768
                )
            except UpstreamExhaustedError:
                logger.error(f"[Bioinformatics] Upstream exhausted, propagating to @with_retry")
                raise
            except Exception as e:
                llm_error = str(e)
                await self._notify("error", f"LLM call failed: {llm_error}")
                break

            # Capture agent reasoning from response content (before tool calls)
            agent_reasoning = response.content[:2000] if response.content else None

            # Transient empty/truncated upstream response: retry without
            # counting toward the text-only HITL escalation.
            if self._is_empty_response(response):
                empty_response_count += 1
                logger.warning(
                    f"[Bioinformatics] Empty/truncated response "
                    f"({empty_response_count}/{MAX_EMPTY_RESPONSES})"
                )

                # Reactive compaction: empty response is a strong signal of
                # context overflow (usage=0, ~80s duration). Force compact
                # now so the retry has a smaller context.
                messages, _reactive_compacted = self._compact_context(
                    messages, self.COMPACT_TARGET_TOKENS
                )
                if _reactive_compacted:
                    est = self._estimate_context_tokens(messages)
                    logger.info(
                        f"[Bioinformatics] Reactive context compaction after empty "
                        f"response: estimated {est} tokens"
                    )
                    await self._notify(
                        "status",
                        f"Context compacted due to large outputs (≈{est:,} tokens), retrying…"
                    )

                if empty_response_count >= MAX_EMPTY_RESPONSES:
                    # Persist compacted conversation so a Retry from the HITL
                    # gate resumes with reduced context instead of replaying
                    # the same overflow that caused the loop.
                    try:
                        await _save_executor_checkpoint(
                            f'step_{step_count}_empty_exhausted',
                        )
                    except Exception as e:
                        logger.warning(f"[Bioinformatics] Pre-empty-exhaustion checkpoint save failed: {e}")
                    raise UpstreamExhaustedError(
                        f"[Bioinformatics] {MAX_EMPTY_RESPONSES} consecutive empty responses",
                        agent_name="bioinformatics",
                    )
                # Append nudge only if compaction did NOT happen this round
                # (compaction already significantly changed the request;
                #  nudge would add volume without value).
                if not _reactive_compacted:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[System] Your previous reply was empty — no text and no "
                            f"tool call (attempt {empty_response_count}). An empty turn "
                            f"is not a valid step. You MUST respond now with exactly one "
                            f"concrete action: call execute_code to run the next analysis "
                            f"step, call another appropriate tool, or call report_findings "
                            f"if every assigned task is already complete. Do not return an "
                            f"empty message."
                        ),
                    })
                await asyncio.sleep(self._empty_retry_backoff(empty_response_count))
                continue  # retry the LLM call
            empty_response_count = 0

            if not response.has_tool_calls:
                if response.content:
                    await self._notify("response", response.content)
                    # Use raw_assistant_message when available so provider-specific
                    # fields (e.g. DeepSeek v4 thinking reasoning_content) survive into
                    # the next turn; the gateway rejects multi-turn requests otherwise.
                    messages.append(response.raw_assistant_message or {
                        "role": "assistant", "content": response.content
                    })

                # Record text-only response in analysis log
                analysis_log.append(AnalysisStep(
                    step_number=step_count, step_type="text_response",
                    agent_reasoning=agent_reasoning,
                    text_content=(response.content or "")[:2000],
                ))
                agent_reasoning = None

                text_response_count += 1
                if text_response_count >= MAX_TEXT_RESPONSES:
                    logger.error(
                        f"[Bioinformatics] {MAX_TEXT_RESPONSES} consecutive text-only "
                        f"responses; escalating to LLM-failure HITL via UpstreamExhaustedError"
                    )
                    # Save checkpoint so a Retry from the HITL gate resumes
                    # from the current conversation state instead of starting fresh.
                    try:
                        await _save_executor_checkpoint(
                            f'step_{step_count}_text_only_exhausted',
                        )
                    except Exception as e:
                        logger.warning(f"[Bioinformatics] Pre-escalation checkpoint save failed: {e}")
                    self._raise_text_only_exhausted(text_response_count)

                # Ask the LLM to continue with tools instead of text
                messages.append({
                    "role": "user",
                    "content": "Please continue the analysis. If you have completed your analysis, "
                               "call the report_findings tool with your key findings. "
                               "Do NOT respond with text only - you must use a tool."
                })
                continue
            
            # Track what kind of work this iteration contains for step budget decisions
            has_substantive_work = False   # execute_code/consult_pi/search_literature/annotate_clusters/report_findings
            has_figure_work = False        # generate_publication_figure
            has_register_score = False     # register_score
            _plan_revision_needed = False  # Set when PI indicates plan needs revision
            _plan_revision_context = {}

            # Add assistant message ONCE before processing tool calls.
            # Using raw_assistant_message preserves provider-specific fields such as
            # Gemini's thought_signature, which must be echoed back in subsequent requests.
            if response.raw_assistant_message:
                messages.append(response.raw_assistant_message)
            else:
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}
                        }
                        for tc in response.tool_calls
                    ]
                })

            _step_figures: List[str] = []  # Figures generated in this iteration (for supervisor)

            for tool_call in response.tool_calls:
                tool_name = tool_call.name
                args = tool_call.arguments

                # P2-2: surface JSON parse failures back to the LLM so it can retry
                # with valid JSON instead of silently KeyError-ing downstream.
                if tool_call.parse_error:
                    err_msg = (
                        f"ERROR: failed to parse arguments JSON for tool "
                        f"'{tool_name}': {tool_call.parse_error}. "
                        f"Raw arguments: {tool_call.raw_arguments!r}. "
                        f"Please retry the call with a valid JSON arguments object."
                    )
                    logger.warning(f"[Bioinformatics] {err_msg}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": err_msg,
                    })
                    continue

                try:
                    if tool_name == "execute_code":
                        has_substantive_work = True
                        code = args.get("code", "")
                        description = args.get("description", "")
                        step_type = args.get("step_type", "technical")
                        persistent_outputs = args.get("persistent_outputs", [])

                        # Validate code for security before execution
                        is_safe, security_error = self._validate_code(code)
                        if not is_safe:
                            error_msg = f"Code validation failed: {security_error}"
                            logger.warning(f"[Bioinformatics] {error_msg}")
                            await self._notify("error", error_msg)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Execution blocked: {error_msg}. Please rewrite the code without using dangerous operations."
                            })
                            # Record blocked execution in analysis log
                            analysis_log.append(AnalysisStep(
                                step_number=step_count, step_type="execute_code",
                                agent_reasoning=agent_reasoning,
                                code=code, code_error=error_msg,
                                code_description=description, code_step_type=step_type,
                            ))
                            continue

                        h5ad_guard_error = workspace_h5ad_write_guard(
                            code, self.sandbox.working_dir
                        )
                        if h5ad_guard_error:
                            logger.warning("[Bioinformatics] %s", h5ad_guard_error)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Execution blocked: {h5ad_guard_error}",
                            })
                            analysis_log.append(AnalysisStep(
                                step_number=step_count,
                                step_type="execute_code",
                                agent_reasoning=agent_reasoning,
                                code=code,
                                code_error=h5ad_guard_error,
                                code_description=description,
                                code_step_type=step_type,
                            ))
                            continue

                        # Guard: block manual cell_type assignment when annotation
                        # has not been completed through the annotate_clusters workflow
                        if (
                            state.analysis_phase == AnalysisPhase.EXPLORATION
                            and not annotation_completed
                            and _CELL_TYPE_ASSIGN_RE.search(code)
                        ):
                            guard_msg = (
                                "Manual assignment to cell-type-like adata.obs columns "
                                "(cell_type, celltype, cluster_identity, cell_identity, cell_annotation, annotation) "
                                "is not allowed before cell type annotations have been confirmed through "
                                "the annotate_clusters workflow. Please call annotate_clusters first."
                            )
                            logger.warning(f"[Bioinformatics] Blocked manual cell-type-like assignment: annotation not confirmed")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Execution blocked: {guard_msg}"
                            })
                            analysis_log.append(AnalysisStep(
                                step_number=step_count, step_type="execute_code",
                                agent_reasoning=agent_reasoning,
                                code=code, code_error=guard_msg,
                                code_description=description, code_step_type=step_type,
                            ))
                            continue

                        await self._notify("status", f"[{step_type.upper()}] {description}")
                        await self._notify("code", code)

                        # === Wrap code to auto-save figures in PNG + PDF ===
                        figures_dir = Path(self.sandbox.working_dir) / "figures"
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        wrapped_code = _build_exec_autosave_wrapper(
                            code, str(figures_dir), timestamp
                        )

                        expression_specs = extract_sensitive_expression_inputs(code)
                        representation_will_change = mutates_expression_representation(code)
                        preexisting_artifact_versions = (
                            workspace_artifact_versions(self.sandbox.working_dir)
                            if representation_will_change else {}
                        )
                        preflight_result = None
                        if (
                            expression_specs
                            and not prepares_expression_before_sensitive_call(code)
                        ):
                            preflight_result = await self.sandbox.execute(
                                build_expression_preflight_code(expression_specs),
                                save_to_history=False,
                                skip_safety_check=True,
                            )
                            if not preflight_result.error:
                                record_expression_contract(
                                    state,
                                    parse_expression_contracts(preflight_result.output),
                                )

                        execution_started = time.perf_counter()
                        if preflight_result is not None and preflight_result.error:
                            result = preflight_result
                        else:
                            result = await self.sandbox.execute(
                                wrapped_code,
                                declared_outputs=list(persistent_outputs or []),
                                track_globals=True,
                                skip_safety_check=True,
                            )
                        execution_time = time.perf_counter() - execution_started


                        # Auto-fix: ensure obs/var names are unique after every execution
                        await self.sandbox.execute(
                            "import anndata as _ad\n"
                            "for _name, _obj in list(globals().items()):\n"
                            "    if isinstance(_obj, _ad.AnnData):\n"
                            "        _obj.obs_names_make_unique()\n"
                            "        _obj.var_names_make_unique()\n"
                            "del _ad\n",
                            save_to_history=False, skip_safety_check=True
                        )

                        # Strip ANSI escape codes from Jupyter traceback
                        if result.error:
                            result.error = strip_ansi(result.error)
                        else:
                            invalidated_count = record_representation_change(state, code)
                            if representation_will_change:
                                record_invalidated_file_stems(
                                    state,
                                    unchanged_artifact_stems(
                                        self.sandbox.working_dir,
                                        preexisting_artifact_versions,
                                    ),
                                )
                            if invalidated_count:
                                logger.warning(
                                    "[Bioinformatics] Invalidated %d stale artifacts after "
                                    "expression representation change",
                                    invalidated_count,
                                )

                        # Cap output/error at source so all downstream consumers
                        # (code_history, LLM tool message, analysis_log, notify)
                        # share one truncated string — prevents prompt-too-long errors.
                        if result.output:
                            result.output = truncate_for_llm(result.output)
                        if result.error:
                            result.error = truncate_for_llm(result.error)

                        code_exec = CodeExecution(
                            code=code,
                            output=result.output,
                            error=result.error,
                            figures=result.figures,
                            execution_time=execution_time,
                            round_number=state.current_round,
                        )
                        self.code_history.append(code_exec)
                        self.last_result = result

                        # Record execute_code step in analysis log
                        _saved_figs = []
                        if result.output:
                            _saved_figs = [Path(p).name for p in re.findall(r'\[Auto-saved\] (?:JPG|PNG): (.+\.(?:jpe?g|png))', result.output)]
                        analysis_log.append(AnalysisStep(
                            step_number=step_count, step_type="execute_code",
                            agent_reasoning=agent_reasoning,
                            code=code, code_output=(result.output or "")[-3000:],
                            code_error=(result.error or "")[:3000] if result.error else None,
                            code_description=description,
                            code_step_type=step_type, saved_figure_filenames=_saved_figs,
                        ))
                        if result.output:
                            await self._notify("result", result.output[:2000])
                        for fig in result.figures:
                            await self._notify("figure", fig)
                            all_figures.append(fig)
                            _step_figures.append(fig)
                        # Catch technical errors the agent swallowed in-code (try/except)
                        # and then skipped. Computed unconditionally (returns (None, "")
                        # on the hard-error path) so the consumer below always has a value
                        # — no UnboundLocalError and no stale carry-over across iterations.
                        swallowed_type, swallowed_doc_hint = (
                            await self._detect_and_lookup_swallowed_error(result, code)
                        )
                        if swallowed_type:
                            code_exec.evidence_eligible = False
                            code_exec.evidence_rejection_reason = (
                                f"Swallowed technical error: {swallowed_type}"
                            )
                        if result.error:
                            await self._notify("error", result.error)
                            consecutive_errors += 1
                            total_errors += 1
                        else:
                            consecutive_errors = 0
                            self._force_skip_counter = 0
                            self._force_skip_error_type = None

                            # Auto-validate species-gene match after data loading (Round 1 only).
                            # _auto_validate_species_genes is self-guarded: it scans the sandbox
                            # for any object with `var_names` and returns early if none is found,
                            # so we do not need _discover_adata_in_sandbox (which requires a
                            # clustering column that doesn't exist yet at load time).
                            if (self._current_analysis_state
                                    and self._current_analysis_state.species
                                    and not self._current_analysis_state.species_genes_validated
                                    and self._current_analysis_state.current_round == 1):
                                validated = await self._auto_validate_species_genes()
                                if validated:
                                    self._current_analysis_state.species_genes_validated = True

                            # Store required runtime deps in sandbox_state for snapshot tracking.
                            if self._current_analysis_state:
                                sandbox_state = self._current_analysis_state.sandbox_state or {}
                                # The sandbox tracker is the single source of truth: it
                                # already merges explicit declarations + cross-cell
                                # promotions AND prunes names deleted from the namespace
                                # (del). Deriving runtime_dependencies_v2 from it — rather
                                # than append-only accumulation — keeps deleted transient
                                # objects (e.g. a loompy connection) from staying
                                # permanently required and breaking every capture.
                                tracker = getattr(self.sandbox, "runtime_dependency_tracker", None)
                                if tracker is not None:
                                    sandbox_state["runtime_dependencies_v2"] = sorted(
                                        tracker.required_outputs
                                    )
                                    sandbox_state["runtime_dependency_tracker_v2"] = tracker.to_state_dict()
                                else:
                                    # No tracker (should not happen with track_globals): fall
                                    # back to append-only explicit declarations.
                                    deps = sandbox_state.setdefault("runtime_dependencies_v2", [])
                                    for var_name in (persistent_outputs or []):
                                        if var_name and var_name not in deps:
                                            deps.append(var_name)
                                sandbox_state["runtime_revision"] = getattr(
                                    self.sandbox, "runtime_revision", 0
                                )
                                self._current_analysis_state.sandbox_state = sandbox_state

                        tool_result = (
                            f"Execution {'successful' if not result.error else 'failed'}.\n"
                            f"Evidence execution ID: {code_exec.id}\n"
                        )
                        if result.output:
                            tool_result += f"Output:\n{result.output}\n"
                        if result.error:
                            tool_result += f"Error: {result.error}\n"
                            # Hint: if a module is missing, tell the agent to report it
                            if "ModuleNotFoundError" in result.error:
                                tool_result += (
                                    "\nHint: A required package is not installed. "
                                    "Report the missing dependency, explain which analysis step is limited, "
                                    "and use an already-installed alternative only when the substitution is scientifically valid.\n"
                                )
                        if result.figures:
                            tool_result += f"Generated {len(result.figures)} figure(s).\n"

                        if swallowed_type:
                            tool_result += _build_swallowed_error_feedback(
                                swallowed_type, swallowed_doc_hint
                            )

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })

                        # Circuit breaker: too many consecutive errors → force report or redirect
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            if state.analysis_phase == AnalysisPhase.EXPLORATION and not annotation_completed:
                                logger.warning(f"[Bioinformatics] {consecutive_errors} consecutive errors, redirecting to mandatory steps")
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        "[System] Too many consecutive code execution errors. "
                                        "However, mandatory steps are not yet complete. "
                                        "Try a simpler approach: run sc.tl.rank_genes_groups(adata_hvg, 'leiden', method='wilcoxon') "
                                        "then call annotate_clusters."
                                    )
                                })
                            else:
                                logger.warning(f"[Bioinformatics] {consecutive_errors} consecutive code errors, last-chance attempt")
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        "[System] You have had several consecutive code execution errors. "
                                        "This is your last chance to fix the code. Take a completely different "
                                        "approach: simplify the code, use a different library function, or "
                                        "write a minimal version first. If this attempt also fails, you MUST "
                                        "call report_findings with whatever results you have so far."
                                    )
                                })
                            consecutive_errors = 0  # Reset after circuit breaker triggers

                        # Hard limit: too many total errors → force report without giving more chances
                        elif total_errors >= MAX_TOTAL_ERRORS:
                            logger.warning(f"[Bioinformatics] {total_errors} total errors reached hard limit, forcing report")
                            messages.append({
                                "role": "user",
                                "content": (
                                    "[System] Too many code execution errors overall. "
                                    "Stop trying to fix the code. Report your findings so far "
                                    "using the report_findings tool and move on."
                                )
                            })

                    elif tool_name == "consult_pi":
                        has_substantive_work = True
                        question = args.get("question", "")
                        context = args.get("context", "")
                        options = args.get("options", [])
                    
                        await self._notify("status", f"Consulting PI: {question[:100]}...")
                        await self._notify("pi_question", json.dumps({
                            "question": question,
                            "context": context,
                            "options": options
                        }))
                    
                        pi_response = ""
                        if self.pi_callback:
                            pi_response = await self.pi_callback(question, context)
                            await self._notify("pi_response", pi_response)
                        else:
                            pi_response = "PI guidance not available. Proceed with standard practices."
                    
                        # Check for plan revision signal
                        if PLAN_REVISION_NEEDED_TAG in pi_response:
                            clean_response = pi_response.replace(PLAN_REVISION_NEEDED_TAG, "").strip()
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"PI Response: {clean_response}\n\n[System: A plan revision has been triggered. Pause and wait for updated plan.]"
                            })
                            _plan_revision_needed = True
                            _plan_revision_context = {
                                "question": question,
                                "context": context,
                            }
                            break
                        else:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"PI Response: {pi_response}"
                            })

                        # Record consult_pi step in analysis log
                        analysis_log.append(AnalysisStep(
                            step_number=step_count, step_type="consult_pi",
                            agent_reasoning=agent_reasoning,
                            pi_question=question, pi_context=context[:1000],
                            pi_options=options or [], pi_response=(pi_response or "")[:2000],
                        ))

                    elif tool_name == "search_literature":
                        has_substantive_work = True
                        topic = args.get("topic", "")
                    
                        await self._notify("status", f"Searching literature: {topic[:50]}...")
                    
                        lit_result = ""
                        if self.literature_callback:
                            lit_result = await self.literature_callback(topic)
                            await self._notify("literature", lit_result[:1000])
                        else:
                            lit_result = "Literature search not available."
                    
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Literature findings: {lit_result}"
                        })

                        # Record search_literature step in analysis log
                        analysis_log.append(AnalysisStep(
                            step_number=step_count, step_type="search_literature",
                            agent_reasoning=agent_reasoning,
                            literature_topic=topic, literature_result=(lit_result or "")[:2000],
                        ))

                    elif tool_name == "annotate_clusters":
                        has_substantive_work = True
                        cluster_markers = args.get("cluster_markers", {})
                        tissue = args.get("tissue", "")
                        organism = args.get("organism", "mouse")
                        # Override with session species if available
                        if self._current_analysis_state and self._current_analysis_state.species:
                            llm_organism = organism
                            organism = self._current_analysis_state.species
                            if llm_organism != organism:
                                logger.info(f"[Bioinformatics] organism overridden: {llm_organism} -> {organism}")
                        cluster_key_hint = args.get("cluster_key") or None
                        rank_key_hint = args.get("rank_key") or "rank_genes_groups"

                        # Sanitize LLM-provided cluster_markers (handles malformed dict/string formats)
                        cluster_markers = self._sanitize_cluster_markers(cluster_markers)

                        n_clusters = len(cluster_markers)

                        # Discover the active AnnData variable and clustering key in sandbox
                        adata_var, cluster_key = await self._discover_adata_in_sandbox(cluster_key_hint)

                        if adata_var is None:
                            tool_result = (
                                "Error: Could not find an AnnData object with a clustering column "
                                "(leiden/louvain/cluster, or any column starting with 'leiden'/'louvain') "
                                "in the sandbox. Before calling annotate_clusters, "
                                "you must: (1) load data into an AnnData variable, (2) run clustering "
                                "(e.g., sc.tl.leiden(adata)), and (3) run differential expression "
                                "(sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')). "
                                "If using non-default column names (e.g., 'leiden_broad'), pass "
                                "cluster_key explicitly."
                            )
                            logger.warning("[Bioinformatics] annotate_clusters called but no AnnData with clustering found")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result
                            })
                            annotation_retry_required = True
                            annotation_retry_attempts += 1
                            continue

                        # Auto-extract markers from kernel adata if LLM didn't provide them
                        if n_clusters == 0 and self.sandbox:
                            logger.info(f"[Bioinformatics] cluster_markers empty, auto-extracting from {adata_var}.uns[{rank_key_hint!r}]")
                            # P2-11: Read padj cutoff from session-configurable deg_thresholds.
                            _deg_thresholds = getattr(self._current_analysis_state, "deg_thresholds", None) or {}
                            _padj_cutoff = float(_deg_thresholds.get("padj", 0.05))
                            extract_code = (
                                "import json as _json\n"
                                "_markers = {}\n"
                                "try:\n"
                                "    import scanpy as sc\n"
                                f"    for _cl in {adata_var}.obs['{cluster_key}'].cat.categories:\n"
                                f"        _df = sc.get.rank_genes_groups_df({adata_var}, group=str(_cl), key={rank_key_hint!r})\n"
                                f"        _df = _df[_df['pvals_adj'] < {_padj_cutoff}]\n"
                                "        _markers[str(_cl)] = _df.head(15)['names'].tolist()\n"
                                "except Exception as _e:\n"
                                "    _markers = {'__error__': str(_e)}\n"
                                "print(_json.dumps(_markers))\n"
                            )
                            try:
                                extract_result = await self.sandbox.execute(extract_code)
                                if extract_result.success and extract_result.output.strip():
                                    try:
                                        extracted = json.loads(extract_result.output.strip())
                                    except json.JSONDecodeError as je:
                                        logger.warning(f"[Bioinformatics] Auto-extract markers JSON parse failed: {je}")
                                        extracted = {}
                                    if "__error__" not in extracted and len(extracted) > 0:
                                        cluster_markers = extracted
                                        n_clusters = len(cluster_markers)
                                        logger.info(f"[Bioinformatics] Auto-extracted markers for {n_clusters} clusters from adata")
                                    else:
                                        err_msg = extracted.get("__error__", "unknown")
                                        logger.warning(f"[Bioinformatics] Auto-extract failed: {err_msg}")
                            except Exception as e:
                                logger.warning(f"[Bioinformatics] Auto-extract markers exception: {e}")

                        if n_clusters == 0:
                            tool_result = (
                                f"Error: cluster_markers is empty and auto-extraction from "
                                f"{adata_var}.uns[{rank_key_hint!r}] failed. You must first run "
                                f"sc.tl.rank_genes_groups({adata_var}, '{cluster_key}', method='wilcoxon', "
                                f"key={rank_key_hint!r}) and ensure it completes successfully before "
                                f"calling annotate_clusters."
                            )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result
                            })
                            annotation_retry_required = True
                            annotation_retry_attempts += 1
                            continue

                        await self._notify("status", f"Annotating {n_clusters} clusters using literature evidence ({organism} {tissue})...")

                        # 缺文件 HITL 门控（必须在算 priors 之前）
                        if not _get_marker_db().present() and self.marker_db_intervention_callback:
                            from backend.constants import LlmFailureAction as _LFA
                            from backend.workflow.helpers import WorkflowTerminatedError
                            action = await self.marker_db_intervention_callback()
                            if action == _LFA.TERMINATE:
                                raise WorkflowTerminatedError("Analysis terminated by user at marker DB gate")
                            if action == _LFA.RETRY:
                                _get_marker_db().reload()  # clear stale empty cache after fetch
                        workspace_dir = getattr(self.sandbox, "working_dir", None)
                        database_priors = {}
                        if workspace_dir is not None:
                            database_priors = await self._compute_database_priors(
                                adata_var, cluster_key, rank_key_hint, organism, workspace_dir)

                        annotation_result: Dict[str, Any] = {}
                        if self.annotate_clusters_callback:
                            annotation_result = await self.annotate_clusters_callback(
                                cluster_markers, tissue, organism,
                                analysis_log=analysis_log,
                                database_priors=database_priors,
                            )
                        else:
                            annotation_result = {
                                "status": "unavailable",
                                "annotations": {},
                                "error": "Cluster annotation service not available. Use consult_pi instead."
                            }

                        # If annotation service returned an error, inform the agent
                        # and do NOT set annotation_completed (so report_findings gate blocks)
                        if annotation_result.get("status") == "error":
                            error_detail = annotation_result.get("error", "Unknown annotation error")
                            # Check if there's raw_response for debugging
                            raw_response = annotation_result.get("raw_response", "")
                            tool_result = (
                                f"[ANNOTATION FAILED] {error_detail}\n"
                                f"Raw response preview: {raw_response[:500] if raw_response else 'N/A'}\n"
                                "Do NOT manually assign cell types via execute_code. "
                                "Cell type annotation MUST go through annotate_clusters for user review. "
                                "You may retry annotate_clusters or consult_pi for guidance."
                            )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result
                            })
                            analysis_log.append(AnalysisStep(
                                step_number=step_count, step_type="annotate_clusters",
                                agent_reasoning=agent_reasoning,
                                annotation_result_summary=tool_result,
                            ))
                            continue

                        # Format result for the agent to use in code
                        annotations = annotation_result.get("annotations", {})
                        tissue_summary = annotation_result.get("tissue_summary", "")

                        # Build a code-friendly response
                        result_lines = []
                        if annotation_result.get("already_confirmed"):
                            if planning_stage == "stage1":
                                result_lines.append(
                                    "[NOTE: These annotations were already confirmed by the user. "
                                    "No re-annotation needed. Stage 1 is complete — call report_findings now.]\n"
                                )
                            else:
                                result_lines.append(
                                    "[NOTE: These annotations were already confirmed by the user. "
                                    "No re-annotation needed. Proceed to downstream analysis.]\n"
                                )
                        if tissue_summary:
                            result_lines.append(f"Tissue context: {tissue_summary}\n")
                        result_lines.append("Cluster annotations:")
                        for cid, info in annotations.items():
                            if isinstance(info, dict):
                                cell_type = info.get("cell_type", "Unknown")
                                confidence = info.get("confidence", "low")
                                key_markers = info.get("key_markers", [])
                                evidence = info.get("evidence", "")
                                result_lines.append(
                                    f"  Cluster {cid}: {cell_type} (confidence: {confidence})"
                                )
                                if key_markers:
                                    result_lines.append(f"    Key markers: {', '.join(key_markers[:5])}")
                                if evidence:
                                    result_lines.append(f"    Evidence: {evidence[:200]}")

                        result_lines.append(f"\nTotal clusters annotated: {len(annotations)}")

                        # Provide ready-to-use Python code snippet
                        # Use json.dumps for safe serialization (prevents code injection via cell type names)
                        cell_type_dict = {
                            str(cid): (info.get("cell_type", "Unknown") if isinstance(info, dict) else "Unknown")
                            for cid, info in annotations.items()
                        }
                        dict_str = json.dumps(cell_type_dict, ensure_ascii=False)
                        result_lines.append(
                            f"\nYou can now assign these annotations to {adata_var}.obs['cell_type'] using:\n"
                            f"cluster_to_celltype = {dict_str}\n"
                            f"{adata_var}.obs['cell_type'] = {adata_var}.obs['{cluster_key}'].astype(str).map(cluster_to_celltype).fillna('Unknown')"
                        )

                        tool_result = "\n".join(result_lines)

                        # Auto-apply annotations to adata.obs['cell_type'] in sandbox
                        # so the agent doesn't need to manually execute the mapping code
                        if annotations and self.sandbox:
                            cell_type_dict = {
                                str(cid): (info.get("cell_type", "Unknown") if isinstance(info, dict) else "Unknown")
                                for cid, info in annotations.items()
                            }
                            inject_code = (
                                "import json as _json\n"
                                f"_ct_map = _json.loads({json.dumps(json.dumps(cell_type_dict, ensure_ascii=False))})\n"
                                f"{adata_var}.obs['cell_type'] = {adata_var}.obs['{cluster_key}'].astype(str).map(_ct_map).fillna('Unknown')\n"
                                f"print(f'cell_type applied: {{{adata_var}.obs[\"cell_type\"].value_counts().to_dict()}}')\n"
                            )
                            try:
                                inject_result = await self.sandbox.execute(inject_code)
                                if inject_result.success:
                                    logger.info(f"[Bioinformatics] Auto-applied cell_type annotations to adata.obs")
                                else:
                                    logger.warning(f"[Bioinformatics] Failed to auto-apply annotations: {inject_result.error}")
                            except Exception as e:
                                logger.warning(f"[Bioinformatics] Exception auto-applying annotations: {e}")

                        await self._notify("annotation_result", json.dumps(
                            annotation_result, ensure_ascii=False, default=str
                        )[:2000])

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })

                        # Record annotate_clusters step in analysis log
                        markers_summary = ", ".join(
                            f"Cluster {k}: {', '.join(str(x) for x in v[:3]) if isinstance(v, list) else str(v)[:50]}"
                            for k, v in list(cluster_markers.items())[:5]
                        )
                        if len(cluster_markers) > 5:
                            markers_summary += f" ... ({len(cluster_markers)} clusters total)"
                        analysis_log.append(AnalysisStep(
                            step_number=step_count, step_type="annotate_clusters",
                            agent_reasoning=agent_reasoning,
                            cluster_markers_summary=markers_summary, annotation_tissue=tissue,
                            annotation_organism=organism, annotation_result_summary=(tool_result or ""),
                            annotation_database_priors_summary=summarize_database_priors(
                                database_priors, len(cluster_markers)),
                        ))
                        annotation_completed = True
                        annotation_retry_required = False
                        annotation_retry_attempts = 0
                        # In stage1, explicitly instruct agent that stage is complete
                        if planning_stage == "stage1":
                            messages.append({
                                "role": "user",
                                "content": (
                                    "STAGE 1 COMPLETE. Cell annotation is done and confirmed by the user. "
                                    "You MUST now call report_findings immediately. "
                                    "Do NOT run differential expression, pathway enrichment, "
                                    "cell communication, trajectory analysis, or any other downstream analysis. "
                                    "These will be done in Stage 2 after PI reviews your Stage 1 results "
                                    "and creates a new analysis plan."
                                )
                            })

                    elif tool_name == "report_findings":
                        # Enforce mandatory pipeline in exploration phase
                        if state.analysis_phase == AnalysisPhase.EXPLORATION and not annotation_completed:
                            logger.warning(
                                f"[Bioinformatics] report_findings rejected: annotate_clusters not yet called"
                            )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": (
                                    "[REJECTED] Cannot report findings yet. Mandatory step not completed: "
                                    "cell type annotation via annotate_clusters. "
                                    "You MUST first run sc.tl.rank_genes_groups(), then call annotate_clusters, "
                                    "before reporting findings."
                                )
                            })
                            analysis_log.append(AnalysisStep(
                                step_number=step_count, step_type="report_findings",
                                agent_reasoning=agent_reasoning,
                                findings=[], findings_summary="REJECTED: annotate_clusters not completed",
                            ))
                            continue

                        has_scrna_data = state.has_scrnaseq()
                        has_bulk_data = state.has_bulk_rnaseq()
                        evidence_executions = successful_evidence_executions(
                            self.code_history, round_number=state.current_round
                        )
                        lowered_codes = [ce.code.lower() for ce in evidence_executions if ce.code]

                        # Notes appended when a mandatory gate is released after max
                        # rejections. Initialized in the common scope so every branch
                        # (exploration / stage1 / innovation) can read it post-gate.
                        _escaped_notes = []

                        # Gate S (statistical validity). Runs against structured
                        # state -- registered scores and frozen plans -- rather
                        # than the agent's code text, so it observes what the
                        # analysis actually did. Bounded like the other gates:
                        # a check that can never be satisfied must not trap the
                        # loop, and an escape is recorded in the report.
                        _stats_report = run_statistical_gates(
                            args.get("findings"),
                            score_definitions=state.score_definitions,
                            scoring_plans=state.scoring_plans,
                            samples_per_group={
                                g.name: g.sample_count
                                for g in (state.treatment_groups or [])
                                if getattr(g, "sample_count", None) is not None
                            },
                            # cluster_labels deliberately not supplied: measured,
                            # feeding annotation vocabulary to the composition
                            # check costs 7 false positives and gains nothing,
                            # because recall is already 1.000 without it.
                        )
                        # Advisories ride along with an accepted call: they are
                        # notes the model cannot suppress, not refusals.
                        _stats_advice = _stats_report.advisory_message()
                        if _stats_advice:
                            _escaped_notes.append(_stats_advice)

                        # A design defect the model cannot repair mid-run must not
                        # be argued with three times. Every session in the corpus
                        # this was measured on has one cell-resolved library per
                        # condition, so a cell-level-inference rejection has
                        # exactly one satisfiable response: reword the claim as
                        # descriptive. Asking twice more trains the phrasing, not
                        # the analysis, and burns two steps doing it.
                        _unsatisfiable = {"cell-level-inference", "unidentifiable-design"}
                        _cap = (1 if all(f.gate in _unsatisfiable
                                         for f in _stats_report.blocking)
                                else MAX_STATS_GATE_REJECTIONS)

                        if not _stats_report.passed:
                            if stats_gate_reject_count < _cap:
                                stats_gate_reject_count += 1
                                logger.warning(
                                    f"[Bioinformatics] report_findings rejected: "
                                    f"statistical validity "
                                    f"({stats_gate_reject_count}/{_cap})"
                                )
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": _stats_report.message(),
                                })
                                continue
                            logger.warning(
                                "[Bioinformatics] statistical gate: max rejections "
                                "reached, allowing report_findings"
                            )
                            _escaped_notes.append(
                                "Statistical validity checks did not pass after "
                                + str(_cap)
                                + " attempt(s) and were force-released: "
                                + "; ".join(f.detail for f in _stats_report.blocking)
                            )

                        # Gate M (mandatory data-driven mechanism program capture) — Stage 2
                        # downstream exploration only. Flag + bounded counter (NOT keyword gate):
                        # a legitimate empty-programs call with no_program_reason sets the flag.
                        if (state.analysis_phase == AnalysisPhase.EXPLORATION
                                and planning_stage != "stage1"
                                and not mechanism_programs_registered):
                            if mechanism_gate_reject_count < MAX_MECHANISM_REJECTIONS:
                                mechanism_gate_reject_count += 1
                                logger.warning(
                                    f"[Bioinformatics] report_findings rejected: "
                                    f"register_mechanism_programs not called "
                                    f"({mechanism_gate_reject_count}/{MAX_MECHANISM_REJECTIONS})"
                                )
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": (
                                        "[REJECTED] Before reporting findings you must call "
                                        "register_mechanism_programs. After completing downstream "
                                        "analysis, identify up to 8 data-driven mechanism programs that "
                                        "are genuinely supported by observed genes, pathways, scores, "
                                        "enrichment terms, or cross-validated patterns (do NOT invent "
                                        "weak programs to reach a count). "
                                        + MECHANISM_EVIDENCE_POLICY
                                        + " If none are "
                                        "supportable, call register_mechanism_programs with an empty "
                                        "programs list and explain why in no_program_reason."
                                    ),
                                })
                                analysis_log.append(AnalysisStep(
                                    step_number=step_count, step_type="report_findings",
                                    agent_reasoning=agent_reasoning,
                                    findings=[], findings_summary="REJECTED: register_mechanism_programs not called",
                                ))
                                continue
                            else:
                                _escaped_notes.append(build_gate_skip_note("data-driven mechanism program capture"))

                        # Gate 0 (mandatory annotate_clusters) always applies in exploration;
                        # Gates 1-3 are Stage 2 tasks, skip in stage1.
                        if state.analysis_phase == AnalysisPhase.EXPLORATION and planning_stage != "stage1":

                            # Gate 1: Bulk RNA-seq analysis
                            if has_bulk_data:
                                code_mentions_bulk = any(
                                    "bulk" in lc or "merged_counts" in lc
                                    or "pydeseq" in lc or "deseqdataset" in lc
                                    for lc in lowered_codes
                                )
                                if not code_mentions_bulk:
                                    logger.warning(
                                        "[Bioinformatics] report_findings rejected: bulk RNA-seq data exists but was not analyzed"
                                    )
                                    bulk_paths = ", ".join(
                                        f.file_path for f in state.get_files_by_section("sequencing")
                                        if f.data_type == DATA_TYPE_BULK_RNASEQ
                                    )
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": (
                                            "[REJECTED] Cannot report findings yet. Mandatory step not completed: "
                                            "bulk RNA-seq analysis. Bulk RNA-seq data is available and MUST be analyzed. "
                                            f"Bulk data files: {bulk_paths}. "
                                            "You MUST: (1) Load bulk count matrix with pd.read_csv(), "
                                            "(2) Perform DEG analysis using pydeseq2 (if ImportError, fallback to scipy.stats.ttest_ind + statsmodels FDR correction), "
                                            "(3) Compare bulk DEGs with scRNA-seq pseudobulk DEGs (Venn diagram), "
                                            "THEN call report_findings."
                                        )
                                    })
                                    analysis_log.append(AnalysisStep(
                                        step_number=step_count, step_type="report_findings",
                                        agent_reasoning=agent_reasoning,
                                        findings=[], findings_summary="REJECTED: bulk RNA-seq not analyzed",
                                    ))
                                    continue

                            # Gate 2: Bulk/scRNA-seq cross-validation
                            if has_scrna_data and has_bulk_data:
                                integration_reject_count, rejected, escaped = self._check_gate_keyword(
                                    lowered_codes,
                                    ["venn", "pseudobulk", "overlap", "intersect",
                                     "matplotlib_venn", "venn2", "venn3"],
                                    integration_reject_count, MAX_INTEGRATION_REJECTIONS,
                                    "bulk/scRNA-seq cross-validation not performed",
                                )
                                if rejected:
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": (
                                            "[REJECTED] Cannot report findings yet. Mandatory step not completed: "
                                            "bulk/scRNA-seq cross-validation. "
                                            "Both bulk RNA-seq and scRNA-seq data are available — you MUST "
                                            "compare their DEG results before reporting. Specifically:\n"
                                            "(1) Aggregate scRNA-seq to pseudobulk DEGs (or use your existing "
                                            "condition-level DEG results across all cells)\n"
                                            "(2) Find overlapping DEGs between bulk and scRNA-seq using set intersection\n"
                                            "(3) Visualize overlap with a Venn diagram (from matplotlib_venn import venn2)\n"
                                            "(4) For overlapping DEGs, show cell-type-specific expression patterns\n"
                                            "THEN call report_findings."
                                        )
                                    })
                                    analysis_log.append(AnalysisStep(
                                        step_number=step_count, step_type="report_findings",
                                        agent_reasoning=agent_reasoning,
                                        findings=[], findings_summary="REJECTED: bulk/scRNA-seq cross-validation not performed",
                                    ))
                                    continue
                                if escaped:
                                    _escaped_notes.append(build_gate_skip_note("bulk/scRNA-seq cross-validation"))
                                    evidence_gate_degraded = True

                            # Gate 3: Cell-type deconvolution
                            if has_scrna_data and has_bulk_data:
                                deconv_reject_count, rejected, escaped = self._check_gate_keyword(
                                    lowered_codes,
                                    ["deconvolution", "nnls", "scaden", "least_squares"],
                                    deconv_reject_count, MAX_DECONV_REJECTIONS,
                                    "cell-type deconvolution not performed",
                                )
                                if rejected:
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": (
                                            "[REJECTED] Cannot report findings yet. Mandatory step not completed: "
                                            "cell-type deconvolution. "
                                            "Both bulk RNA-seq and scRNA-seq data are available — you MUST "
                                            "perform deconvolution to estimate cell-type proportions in the bulk samples.\n"
                                            "(1) Use scRNA-seq as a reference profile\n"
                                            "(2) Prioritize pure Python methods like scipy.optimize.nnls or sklearn Non-negative Least Squares\n"
                                            "(3) Visualize inferred cell-type proportions across bulk conditions\n"
                                            "THEN call report_findings."
                                        )
                                    })
                                    analysis_log.append(AnalysisStep(
                                        step_number=step_count, step_type="report_findings",
                                        agent_reasoning=agent_reasoning,
                                        findings=[], findings_summary="REJECTED: cell-type deconvolution not performed",
                                    ))
                                    continue
                                if escaped:
                                    _escaped_notes.append(build_gate_skip_note("cell-type deconvolution"))
                                    evidence_gate_degraded = True

                        # Gate 4: INNOVATION-phase bulk direction validation
                        elif state.analysis_phase == AnalysisPhase.INNOVATION and has_bulk_data and has_scrna_data:
                            innovation_bulk_reject_count, rejected, escaped = self._check_gate_keyword(
                                lowered_codes,
                                ["bulk_", "merged_counts"],
                                innovation_bulk_reject_count, MAX_INNOVATION_BULK_REJECTIONS,
                                "innovation-phase bulk direction validation missing",
                            )
                            if rejected:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": (
                                        "[REJECTED] Cannot report findings yet. Mandatory step not completed: "
                                        "bulk RNA-seq direction validation. Both bulk and scRNA-seq data are "
                                        "available — you MUST validate this direction's findings on the bulk data. "
                                        "Reuse the bulk variables already loaded in Round 1 (see the data context "
                                        "section for their exact names); do NOT re-run the bulk DEG pipeline from "
                                        "scratch. Use bulk data as an independent validation layer for your new "
                                        "findings, choosing methods appropriate to your specific results and the "
                                        "bulk sample design. THEN call report_findings."
                                    )
                                })
                                analysis_log.append(AnalysisStep(
                                    step_number=step_count, step_type="report_findings",
                                    agent_reasoning=agent_reasoning,
                                    findings=[],
                                    findings_summary="REJECTED: innovation-phase bulk direction validation missing",
                                ))
                                continue
                            if escaped:
                                _escaped_notes.append(build_gate_skip_note("innovation-phase bulk direction validation"))
                                evidence_gate_degraded = True

                        requested_evidence_ids = {
                            str(value).strip()
                            for value in _ensure_list(args.get("evidence_execution_ids"))
                            if str(value).strip()
                        }
                        eligible_evidence_ids = {execution.id for execution in evidence_executions}
                        invalid_evidence_ids = requested_evidence_ids - eligible_evidence_ids
                        if not requested_evidence_ids or invalid_evidence_ids:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": (
                                    "[REJECTED] Findings must cite at least one successful current-run "
                                    "execute_code result using evidence_execution_ids. Failed executions "
                                    "or outputs that explicitly say validation failed cannot support a "
                                    f"claim. Invalid IDs: {sorted(invalid_evidence_ids)}. "
                                    f"Available IDs: {sorted(eligible_evidence_ids)[-20:]}"
                                ),
                            })
                            analysis_log.append(AnalysisStep(
                                step_number=step_count,
                                step_type="report_findings",
                                agent_reasoning=agent_reasoning,
                                findings=[],
                                findings_summary="REJECTED: missing or invalid execution evidence",
                                evidence_execution_ids=sorted(requested_evidence_ids),
                            ))
                            continue

                        has_substantive_work = True
                        findings = await self._handle_report_findings(args)
                        summary = args.get("summary", "")
                        suggestions = args.get("suggestions", [])

                        for note in _escaped_notes:
                            if note not in findings:
                                findings.append(note)

                        all_findings.extend(findings)
                        reported_evidence_ids = sorted(requested_evidence_ids)

                        await self._notify("findings", json.dumps({
                            "findings": findings,
                            "summary": summary,
                            "suggestions": suggestions
                        }))

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Findings recorded. Continue or wait for instructions."
                        })

                        # Record report_findings step in analysis log
                        analysis_log.append(AnalysisStep(
                            step_number=step_count, step_type="report_findings",
                            agent_reasoning=agent_reasoning,
                            findings=findings, findings_summary=summary,
                            findings_suggestions=suggestions or [],
                            evidence_execution_ids=reported_evidence_ids,
                        ))

                        completed = True
                        break

                    elif tool_name == "fetch_gene_set":
                        from ..services.gene_set_fetch import (
                            list_collections as list_gene_set_collections,
                            load_collection, resolve_gene_set,
                            resolve_partitioned_gene_set,
                        )
                        from ..services.gene_set_resolver import (
                            direction_sibling_ids, normalize_organism, suggest_similar,
                        )

                        set_id = str(args.get("set_id", "") or "").strip()
                        collection = str(args.get("collection", "") or "").strip()
                        version = str(args.get("version", "") or "").strip()
                        organism = normalize_organism(
                            args.get("organism") or state.species or ""
                        )

                        if not organism:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": (
                                    "fetch_gene_set rejected: no organism is known for this "
                                    "session and none was supplied."
                                ),
                            })
                            continue

                        if args.get("list_collections"):
                            # Discovery first: the identifiers are organism-specific
                            # and a guess fails with a 404 that teaches nothing.
                            try:
                                available = await asyncio.to_thread(
                                    list_gene_set_collections, version, organism
                                )
                                content = (
                                    f"Collections available for {organism} in release "
                                    f"{version}: {', '.join(available)}. "
                                    f"Pick one and call again with collection=<id> and "
                                    f"either list_sets=true to browse or set_id=<id> to "
                                    f"fetch."
                                )
                            except Exception as exc:
                                content = (
                                    f"fetch_gene_set could not list collections for "
                                    f"{organism} in release {version!r}: {exc}"
                                )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": content,
                            })
                            analysis_log.append(AnalysisStep(
                                step_number=step_count, step_type="fetch_gene_set",
                                agent_reasoning=agent_reasoning,
                                code_description=f"Listed collections for {organism} {version}",
                            ))
                            continue

                        if args.get("list_sets"):
                            # Discovery path: never fabricate an identifier, look it up.
                            try:
                                available = await asyncio.to_thread(
                                    load_collection, version, collection, organism
                                )
                            except Exception as exc:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": (
                                        f"fetch_gene_set could not load {collection!r} "
                                        f"v{version!r} for {organism}: {exc}"
                                    ),
                                })
                                analysis_log.append(AnalysisStep(
                                    step_number=step_count, step_type="fetch_gene_set",
                                    agent_reasoning=agent_reasoning,
                                    code_description=(
                                        f"FAILED: browse {collection!r} v{version!r} "
                                        f"({organism})"
                                    ),
                                ))
                                continue
                            hits = (suggest_similar(set_id, available.keys(), limit=25)
                                    if set_id else sorted(available)[:25])
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": (
                                    f"{len(available)} sets in {collection} v{version} "
                                    f"({organism}). Matching: {', '.join(hits) or 'none'}"
                                ),
                            })
                            continue

                        if not set_id or not collection:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": (
                                    "fetch_gene_set needs both collection and set_id to "
                                    "fetch. Call with list_collections=true to see the "
                                    "identifiers this organism publishes, then "
                                    "list_sets=true to browse that collection."
                                ),
                            })
                            continue

                        if args.get("partition_by_direction"):
                            parts, error = await asyncio.to_thread(
                                resolve_partitioned_gene_set, set_id, organism,
                                version=version, collection=collection,
                            )
                            if error:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": f"fetch_gene_set could not partition: {error}",
                                })
                                analysis_log.append(AnalysisStep(
                                    step_number=step_count, step_type="fetch_gene_set",
                                    agent_reasoning=agent_reasoning,
                                    code_description=(
                                        f"FAILED: partition {set_id!r} -- {str(error)[:140]}"
                                    ),
                                ))
                                continue

                            lines = [
                                f"Split {set_id} into {len(parts)} direction-coherent "
                                f"panels. Score them SEPARATELY -- that is the point of "
                                f"the split. Register each as its own plan and score."
                            ]
                            for part in parts:
                                name = part.set_id.split("#")[-1]
                                var = (
                                    "gene_set_"
                                    + re.sub(r"[^0-9A-Za-z_]", "_", part.set_id)[:60]
                                )
                                try:
                                    await self.sandbox.execute(
                                        f"{var} = {part.genes!r}",
                                        save_to_history=False, skip_safety_check=True,
                                    )
                                    bound = f" bound as {var}"
                                except Exception as exc:
                                    logger.warning(
                                        f"[{self.session_id}] partition injection "
                                        f"failed: {exc}"
                                    )
                                    bound = f" genes: {', '.join(part.genes)}"
                                lines.append(
                                    f"[{name}] {part.n_genes} genes, "
                                    f"provenance {part.provenance}{bound}."
                                )
                                if name == "ambiguous":
                                    lines.append(
                                        "The ambiguous part is annotated to both "
                                        "directions; it cannot support a directional "
                                        "claim and is separated so it does not count "
                                        "toward either arm."
                                    )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": " ".join(lines),
                            })
                            analysis_log.append(AnalysisStep(
                                step_number=step_count, step_type="fetch_gene_set",
                                agent_reasoning=agent_reasoning,
                                code_description=(
                                    f"Partitioned {set_id} into "
                                    + ", ".join(
                                        f"{r.set_id.split('#')[-1]}({r.n_genes})"
                                        for r in parts)
                                ),
                            ))
                            continue

                        result, error = await asyncio.to_thread(
                            resolve_gene_set, set_id, organism,
                            version=version, collection=collection,
                        )
                        if error:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"fetch_gene_set failed: {error}",
                            })
                            # Record the attempt. Without this a failed fetch is
                            # invisible in session.json, and a later reader
                            # concludes the tool was never called -- which is the
                            # opposite of what happened in the first run.
                            analysis_log.append(AnalysisStep(
                                step_number=step_count, step_type="fetch_gene_set",
                                agent_reasoning=agent_reasoning,
                                code_description=f"FAILED: {set_id!r} -- {str(error)[:160]}",
                            ))
                            continue

                        # Hand the genes to the kernel as data rather than asking the
                        # model to retype them: retyping is where a fetched panel
                        # silently becomes a remembered one again.
                        var_name = f"gene_set_{re.sub(r'[^0-9A-Za-z_]', '_', set_id)[:60]}"
                        try:
                            await self.sandbox.execute(
                                f"{var_name} = {result.genes!r}",
                                save_to_history=False, skip_safety_check=True,
                            )
                            injected = True
                        except Exception as exc:
                            logger.warning(
                                f"[{self.session_id}] gene set injection failed: {exc}"
                            )
                            injected = False

                        lines = [
                            f"Fetched {result.set_id}: {result.n_genes} genes. "
                            f"Provenance: {result.provenance}. "
                            f"Pass this string as gene_list_source / panel_source."
                        ]
                        if injected:
                            lines.append(
                                f"The genes are bound in the sandbox as {var_name} -- use "
                                f"that variable directly rather than retyping them."
                            )
                        else:
                            lines.append(f"Genes: {', '.join(result.genes)}")
                        lines.extend(result.warnings())
                        lines.append(_method_rule_text("read-the-set-before-scoring-it"))
                        if direction_sibling_ids(result.set_id):
                            lines.append(
                                "This set may be direction-mixed. Call again with "
                                "partition_by_direction=true to split it into "
                                "direction-coherent panels using the ontology's own "
                                "regulation terms, rather than assembling one by hand."
                            )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": " ".join(lines),
                        })

                        analysis_log.append(AnalysisStep(
                            step_number=step_count, step_type="fetch_gene_set",
                            agent_reasoning=agent_reasoning,
                            code_description=(
                                f"Fetched {result.set_id} ({result.n_genes} genes) "
                                f"from {result.provenance}"
                            ),
                        ))

                    elif tool_name == "register_analysis_plan":
                        from ..models.schemas import ScoringPlan
                        from ..services.analysis_plan import (
                            canonical_panel, panel_hash, summarize_edits,
                            validate_edit_operations,
                        )
                        from ..services.gene_set_provenance import parse_provenance

                        panel = canonical_panel(args.get("panel_genes"))
                        if not panel:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": (
                                    "register_analysis_plan rejected: panel_genes is empty. "
                                    "Freeze the exact panel you intend to score."
                                ),
                            })
                            continue

                        _prov, prov_error = parse_provenance(args.get("panel_source"))
                        if prov_error:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"register_analysis_plan rejected: {prov_error}",
                            })
                            continue

                        edits, edit_error = validate_edit_operations(
                            args.get("edit_operations")
                        )
                        if edit_error:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"register_analysis_plan rejected: {edit_error}",
                            })
                            continue

                        plan = ScoringPlan(
                            score_name=args.get("score_name", ""),
                            method=args.get("method", ""),
                            hypothesis=args.get("hypothesis", ""),
                            falsifier=args.get("falsifier", ""),
                            panel_genes=panel,
                            panel_hash=panel_hash(panel),
                            panel_source=args.get("panel_source", ""),
                            baseline_genes=canonical_panel(args.get("baseline_genes")),
                            baseline_source=args.get("baseline_source", ""),
                            edit_operations=edits,
                            testing_family=[str(t) for t in (args.get("testing_family") or [])],
                            registered_at_step=step_count,
                            round_number=state.current_round,
                        )
                        state.scoring_plans.append(plan)

                        plan_lines = [
                            f"Plan frozen. plan_id={plan.id}. "
                            f"Panel: {len(panel)} gene(s), hash {plan.panel_hash}. "
                            f"Cite this plan_id when you call register_score."
                        ]
                        edit_summary = summarize_edits(edits)
                        if edit_summary:
                            plan_lines.append(edit_summary + ".")
                        if not plan.falsifier:
                            plan_lines.append(
                                "No falsifier was stated. A score with no result that "
                                "would count against it cannot support a conclusion "
                                "either way."
                            )
                        if not plan.testing_family:
                            plan_lines.append(
                                "No testing_family was declared. If this panel goes into a "
                                "shared enrichment call, its q-value depends on the other "
                                "sets in that call and cannot be interpreted alone."
                            )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": " ".join(plan_lines),
                        })

                        analysis_log.append(AnalysisStep(
                            step_number=step_count, step_type="register_analysis_plan",
                            agent_reasoning=agent_reasoning,
                            code_description=(
                                f"Plan {plan.id} for {plan.score_name}: "
                                f"{len(panel)} genes via {plan.method}"
                            ),
                        ))

                    elif tool_name == "register_score":
                        from ..models.schemas import ScoreDefinition
                        from ..services.analysis_plan import (
                            describe_drift, diff_panels, panel_hash,
                        )
                        from ..services.gene_set_provenance import (
                            assess_panel_power, parse_provenance, power_caveat,
                        )

                        gene_list = args.get("gene_list") or []
                        provenance, prov_error = parse_provenance(
                            args.get("gene_list_source")
                        )
                        gate_note = ""
                        if prov_error:
                            # Reject rather than record an unattributable panel: an
                            # unsourced gene list is indistinguishable downstream from
                            # a versioned one, which is how two runs of the same
                            # analysis ended up with different panels and no record.
                            if score_gate_reject_count < MAX_SCORE_GATE_REJECTIONS:
                                score_gate_reject_count += 1
                                logger.warning(
                                    f"[Bioinformatics] register_score rejected: provenance "
                                    f"({score_gate_reject_count}/{MAX_SCORE_GATE_REJECTIONS})"
                                )
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": (
                                        f"register_score rejected: {prov_error} "
                                        f"Re-call register_score with a valid gene_list_source. "
                                        + _method_rule_text("cite-the-resource-version")
                                    ),
                                })
                                continue
                            # Ceiling reached: record the score so the methodology
                            # section is not simply empty, but mark what is missing.
                            logger.warning(
                                "[Bioinformatics] register_score provenance gate: max "
                                "rejections reached, recording score as unattributed"
                            )
                            provenance, _ = parse_provenance(
                                "bespoke:provenance not supplied after repeated requests; "
                                "this panel is unattributed and not reproducible"
                            )
                            gate_note = (
                                "Recorded WITHOUT valid provenance after "
                                f"{MAX_SCORE_GATE_REJECTIONS} attempts; treat this panel "
                                "as unattributed."
                            )

                        # A panel score must cite the plan that froze the panel.
                        plan_id = str(args.get("plan_id", "") or "").strip()
                        plan = next(
                            (pl for pl in state.scoring_plans if pl.id == plan_id),
                            None,
                        )
                        drift = {}
                        if gene_list:
                            if not plan_id or plan is None:
                                if score_gate_reject_count < MAX_SCORE_GATE_REJECTIONS:
                                    score_gate_reject_count += 1
                                    known = ", ".join(pl.id for pl in state.scoring_plans[-5:])
                                    detail = (
                                        "this score uses a gene panel, so it must cite the "
                                        "plan_id returned by register_analysis_plan. Call "
                                        "register_analysis_plan FIRST with this panel, then "
                                        "re-call register_score with the plan_id it returns."
                                        if not plan_id else
                                        f"no frozen plan with plan_id={plan_id!r}. "
                                        f"Recent plan_ids: {known or 'none'}."
                                    )
                                    logger.warning(
                                        f"[Bioinformatics] register_score rejected: plan "
                                        f"({score_gate_reject_count}/{MAX_SCORE_GATE_REJECTIONS})"
                                    )
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": f"register_score rejected: {detail}",
                                    })
                                    continue
                                logger.warning(
                                    "[Bioinformatics] register_score plan gate: max rejections "
                                    "reached, recording score without a frozen plan"
                                )
                                gate_note = (gate_note + " " if gate_note else "") + (
                                    f"Recorded WITHOUT a frozen plan after "
                                    f"{MAX_SCORE_GATE_REJECTIONS} attempts; the panel was not "
                                    f"pre-specified, so it cannot be distinguished from one "
                                    f"chosen after seeing the result."
                                )
                                plan_id = ""
                            else:
                                drift = diff_panels(plan.panel_genes, gene_list)

                        n_matched = args.get("n_genes_matched")
                        try:
                            n_matched = int(n_matched) if n_matched is not None else None
                        except (TypeError, ValueError):
                            n_matched = None
                        power = assess_panel_power(len(gene_list), n_matched)

                        has_register_score = True
                        score_def = ScoreDefinition(
                            score_name=args.get("score_name", ""),
                            method=args.get("method", ""),
                            description=args.get("description", ""),
                            parameters=args.get("parameters", {}),
                            gene_list=gene_list,
                            gene_list_source=args.get("gene_list_source", ""),
                            provenance=provenance.to_dict(),
                            n_genes=len(gene_list),
                            n_genes_matched=n_matched,
                            power=power,
                            plan_id=plan_id,
                            panel_hash=panel_hash(gene_list),
                            plan_drift=drift,
                            round_number=state.current_round,
                        )
                        # Deduplicate: update existing score with same name, or append new
                        existing_idx = next(
                            (i for i, sd in enumerate(state.score_definitions) if sd.score_name == score_def.score_name),
                            None
                        )
                        if existing_idx is not None:
                            state.score_definitions[existing_idx] = score_def
                            await self._notify("status", f"Updated score: {score_def.score_name} ({score_def.method})")
                        else:
                            state.score_definitions.append(score_def)
                            await self._notify("status", f"Registered score: {score_def.score_name} ({score_def.method})")

                        action_word = "updated" if existing_idx is not None else "registered"
                        result_lines = [
                            f"Score '{score_def.score_name}' {action_word}. "
                            f"Method: {score_def.method}. "
                            f"Provenance: {provenance.citation()}. "
                            f"This will appear in the report methodology."
                        ]
                        if provenance.is_bespoke:
                            result_lines.append(
                                "This panel is recorded as hand-assembled. If a versioned "
                                "database set covers the same concept, prefer it: bespoke "
                                "panels are reported as such and carry less weight."
                            )
                        if n_matched is None and gene_list:
                            result_lines.append(
                                "n_genes_matched was not supplied, so power was assessed on "
                                "the nominal panel size. Report the number of panel genes "
                                "actually present in the matrix for an accurate assessment."
                            )
                        if gate_note:
                            # The escape must reach the report, not just the log.
                            score_def.plan_drift = dict(score_def.plan_drift or {})
                            score_def.plan_drift["gate_escape"] = gate_note.strip()
                        drift_text = describe_drift(drift) if drift else ""
                        if drift_text:
                            result_lines.append(
                                f"NOTE: the scored panel departs from plan {plan_id}: "
                                f"{drift_text}. This is recorded on the score and will "
                                f"appear in the report. If the change was deliberate, say "
                                f"why in the findings; if not, re-score the frozen panel. "
                                + _method_rule_text("prespecify-the-family-before-results")
                            )
                        if gate_note:
                            result_lines.append(gate_note.strip())
                        caveat = power_caveat(power)
                        if caveat:
                            result_lines.append(caveat)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": " ".join(result_lines),
                        })

                        analysis_log.append(AnalysisStep(
                            step_number=step_count, step_type="register_score",
                            agent_reasoning=agent_reasoning,
                            code_description=f"Score: {score_def.score_name} via {score_def.method}",
                        ))

                    elif tool_name == "register_mechanism_programs":
                        progs, err = validate_mechanism_program_payload(
                            args,
                            round_number=state.current_round,
                            phase=str(getattr(state.analysis_phase, "value", state.analysis_phase)),
                            stage=planning_stage,
                        )
                        if err:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"[REJECTED] {err} Please correct and call register_mechanism_programs again.",
                            })
                            analysis_log.append(AnalysisStep(
                                step_number=step_count, step_type="register_mechanism_programs",
                                agent_reasoning=agent_reasoning,
                                findings=[], findings_summary=f"REJECTED: {err}",
                            ))
                            continue
                        eligible_ids = {
                            execution.id
                            for execution in successful_evidence_executions(
                                self.code_history, round_number=state.current_round
                            )
                        }
                        invalid_program_ids = sorted({
                            evidence_id
                            for program in progs
                            for evidence_id in program.evidence_execution_ids
                            if evidence_id not in eligible_ids
                        })
                        if invalid_program_ids:
                            err = (
                                "Mechanism programs cite failed, contradictory, or unknown "
                                f"execution IDs: {invalid_program_ids}. Available IDs: "
                                f"{sorted(eligible_ids)[-20:]}"
                            )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"[REJECTED] {err}",
                            })
                            analysis_log.append(AnalysisStep(
                                step_number=step_count,
                                step_type="register_mechanism_programs",
                                agent_reasoning=agent_reasoning,
                                findings=[],
                                findings_summary=f"REJECTED: {err}",
                            ))
                            continue
                        mechanism_programs_registered = True
                        mechanism_programs.extend(progs)
                        # Handler is the single writer to AnalysisState (see plan decision 1).
                        state.data_driven_mechanism_programs.extend(progs)
                        if progs:
                            await self._notify("status", f"Registered {len(progs)} data-driven mechanism program(s)")
                            content = (
                                f"Registered {len(progs)} data-driven mechanism program(s). They will be "
                                "reviewed by the PI, mapped against literature, and may appear as CANDIDATE "
                                "mechanisms (not confirmed) in the report. You may now call report_findings."
                            )
                        else:
                            await self._notify("status", "No data-driven mechanism program registered (reason recorded)")
                            content = "Recorded that no data-driven mechanism program was registered. You may now call report_findings."
                        messages.append({
                            "role": "tool", "tool_call_id": tool_call.id, "content": content,
                        })
                        analysis_log.append(AnalysisStep(
                            step_number=step_count, step_type="register_mechanism_programs",
                            agent_reasoning=agent_reasoning,
                            code_description=f"{len(progs)} mechanism program(s)",
                        ))

                    elif tool_name == "validate_species_genes":
                        gene_names = args.get("gene_names", [])
                        state = self._current_analysis_state
                        if not state or not state.species:
                            tool_result = "No species set in session. Skipping validation."
                        else:
                            assessment = await self._assess_gene_species_match(gene_names, state.species)
                            if assessment.get("match", True):
                                tool_result = f"Gene names are consistent with {state.species}. Proceed with analysis."
                            else:
                                if self.species_validation_callback:
                                    action = await self.species_validation_callback(
                                        gene_names, state.species, assessment
                                    )
                                    if action.get("action") == "change_species":
                                        tool_result = f"Species updated to {action.get('new_species', '')}. Proceed."
                                    elif action.get("action") == "custom":
                                        self.analysis_context["species_user_directive"] = action.get("user_input", "")
                                        tool_result = f"User instruction received. Proceed with analysis."
                                    else:
                                        tool_result = f"Keeping species as {state.species}. Proceed with caution."
                                else:
                                    tool_result = (
                                        f"Warning: gene names may not match {state.species}. "
                                        f"{assessment.get('reasoning', '')}. Proceeding anyway."
                                    )
                            # Validation actually ran here (_assess_gene_species_match completed
                            # above); unlike the auto-validation hook, every branch in this tool
                            # call executes the assessment, so the flag can be set unconditionally.
                            if self._current_analysis_state:
                                self._current_analysis_state.species_genes_validated = True
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result,
                        })
                        continue

                    elif tool_name == "lookup_documentation":
                        function_path = args.get("function_path", "")
                        question = args.get("question", "")
                        method_question = args.get("method_question", "")

                        # Methodological lookups are answered from the rule corpus
                        # rather than from a docstring: the question is which test
                        # is defensible, not what the signature is.
                        method_answer = ""
                        if method_question:
                            try:
                                from ..services.method_rules import (
                                    format_rule, search_rules,
                                )

                                hits = search_rules(method_question, limit=3)
                                if hits:
                                    method_answer = "\n\n".join(
                                        format_rule(r) for r in hits
                                    )
                                else:
                                    method_answer = (
                                        "No rule in the corpus covers that question. "
                                        "Proceed with standard practice and state your "
                                        "reasoning in the findings."
                                    )
                            except Exception as exc:
                                logger.warning(
                                    f"[Bioinformatics] method rule lookup failed: {exc}"
                                )

                        if method_question and not function_path:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": method_answer or "Method guidance unavailable.",
                            })
                            analysis_log.append(AnalysisStep(
                                step_number=step_count,
                                step_type="lookup_documentation",
                                agent_reasoning=agent_reasoning,
                                code_description=f"Method lookup: {method_question[:80]}",
                            ))
                            continue

                        if self.technique_agent:
                            await self._notify("status", f"Looking up docs: {function_path}...")
                            try:
                                doc_result = await self.technique_agent.lookup_docs(
                                    self.sandbox, function_path, question=question
                                )
                            except Exception as e:
                                logger.warning(f"[Bioinformatics] lookup_documentation failed: {e}")
                                doc_result = f"Documentation lookup failed: {e}"
                        else:
                            doc_result = "Documentation lookup not available."

                        if method_answer:
                            doc_result = f"{doc_result}\n\n{method_answer}"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": doc_result
                        })

                        analysis_log.append(AnalysisStep(
                            step_number=step_count,
                            step_type="lookup_documentation",
                            agent_reasoning=agent_reasoning,
                            code_description=f"Docs lookup: {function_path}",
                        ))

                    elif tool_name == "generate_publication_figure":
                        has_figure_work = True
                        code = args.get("code", "")
                        figure_name = args.get("figure_name", f"figure_{len(state.publication_figures) + 1}")
                        description = args.get("description", "")
                        figure_type = args.get("figure_type", "other")

                        # Track figure generation attempts to prevent infinite retry loops
                        if "_figure_attempts" not in self.analysis_context:
                            self.analysis_context["_figure_attempts"] = {}
                        fig_attempts = self.analysis_context["_figure_attempts"]
                        fig_attempts[figure_name] = fig_attempts.get(figure_name, 0) + 1

                        if fig_attempts[figure_name] > 3:
                            logger.warning(f"[Bioinformatics] Skipping figure {figure_name}: exceeded 3 attempts")
                            tool_result = (
                                f"Figure {figure_name} has been skipped after {fig_attempts[figure_name]-1} failed attempts. "
                                "Please move on to the next analysis step or generate a different figure."
                            )
                        elif figure_iteration_count >= MAX_FIGURE_ITERATIONS:
                            # Budget exhausted: refuse figure generation, redirect to analysis
                            logger.warning(f"[Bioinformatics] Figure budget exhausted, rejecting {figure_name}")
                            tool_result = (
                                f"Figure generation budget exhausted ({MAX_FIGURE_ITERATIONS} figures already generated). "
                                "Do NOT generate any more figures. Focus on completing the remaining analysis steps "
                                "(differential expression, pathway enrichment, etc.) and then call report_findings."
                            )
                        else:
                            await self._notify("status", f"Generating publication figure: {figure_name} (attempt {fig_attempts[figure_name]})")

                            # Generate the figure with PI review
                            tool_result = await self._handle_generate_publication_figure(
                                code=code,
                                figure_name=figure_name,
                                description=description,
                                figure_type=figure_type,
                                state=state,
                                review_callback=review_callback
                            )

                        # Pass publication figure to supervisor for vision review
                        safe_fig_name = "".join(c if c.isalnum() or c == '_' else '_' for c in figure_name)
                        for pub_fig in state.publication_figures:
                            if pub_fig.name == safe_fig_name and pub_fig.png_base64:
                                _step_figures.append(pub_fig.png_base64)
                                break

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })

                        # Record generate_publication_figure step in analysis log
                        analysis_log.append(AnalysisStep(
                            step_number=step_count, step_type="generate_publication_figure",
                            agent_reasoning=agent_reasoning,
                            pub_figure_name=figure_name, pub_figure_description=description,
                            pub_figure_type=figure_type, pub_figure_code=code,
                            pub_figure_result=(tool_result or "")[:2000],
                        ))

                    else:
                        # Unknown tool name — most commonly an LLM hallucination
                        # or a recovered stringified tool_call with a bogus name.
                        # We MUST emit a role=tool response message so the
                        # assistant tool_call does not become orphan; otherwise
                        # the next request fails with a 400 from the upstream.
                        known = ", ".join(sorted(KNOWN_TOOL_NAMES))
                        err_msg = (
                            f"ERROR: Unknown tool '{tool_name}'. "
                            f"This tool does not exist. Available tools: {known}. "
                            f"Please retry using one of the available tools."
                        )
                        logger.warning(
                            f"[Bioinformatics] Unknown tool '{tool_name}' called "
                            f"(tool_call_id={tool_call.id}); responding with error to keep "
                            f"conversation well-formed"
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": err_msg,
                        })
                        analysis_log.append(AnalysisStep(
                            step_number=step_count, step_type="unknown_tool",
                            agent_reasoning=agent_reasoning,
                            text_content=err_msg[:2000],
                        ))

                except UpstreamExhaustedError:
                    raise
                except Exception as _exc_tool:
                    logger.exception(
                        "[Bioinformatics] Tool '%s' failed on args=%r; continuing",
                        tool_call.name, tool_call.arguments,
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error processing tool '{tool_call.name}': {_exc_tool}",
                    })
                    analysis_log.append(AnalysisStep(
                        step_number=step_count,
                        step_type="tool_error",
                        agent_reasoning=agent_reasoning,
                        text_content=f"Tool '{tool_call.name}' failed: {_exc_tool}",
                    ))

            # Clear reasoning after all tool calls in this iteration are processed
            agent_reasoning = None

            # Reset text-only counter when a substantive tool call was made
            if has_substantive_work and response.has_tool_calls:
                text_response_count = 0

            # === SUPERVISOR monitoring (every step) ===
            if _supervisor_callback and analysis_log:
                try:
                    sup_result = await _supervisor_callback(
                        analysis_log[-1],   # Latest step
                        _step_figures       # Figures from this iteration (may be empty)
                    )
                    if sup_result.feedback:
                        # De-duplicate: skip if the same feedback (first 120 chars)
                        # was already injected recently to avoid wasting tokens on
                        # repeated identical warnings.
                        sup_key = sup_result.feedback[:120]
                        if sup_key not in self._recent_sup_feedback:
                            messages.append({
                                "role": "user",
                                "content": f"[Supervisor] {sup_result.feedback}"
                            })
                            self._recent_sup_feedback.add(sup_key)
                        else:
                            logger.debug(f"[Bioinformatics] Skipping duplicate supervisor feedback: {sup_key[:60]}...")

                    if sup_result.force_skip:
                        current_err_type = extract_error_type(analysis_log[-1].code_error) if analysis_log[-1].code_error else None
                        if self._force_skip_error_type != current_err_type:
                            self._force_skip_counter = 0
                            self._force_skip_error_type = current_err_type

                        self._force_skip_counter += 1

                        if self._force_skip_counter <= 2:
                            messages.append({
                                "role": "user",
                                "content": (
                                    "[Supervisor] MANDATORY: You must skip the current step immediately. "
                                    "Do NOT retry this code or attempt to fix it. "
                                    "Move on to the next analysis step in your plan. "
                                    "This instruction is not optional — you will be forced to "
                                    "report findings if you continue on this path."
                                )
                            })
                        else:
                            # 3rd force-skip on same error type: the agent has ignored
                            # two mandatory-skip instructions. Force a hard exit from
                            # the tool-call loop by injecting report_findings and breaking.
                            logger.warning(
                                f"[Bioinformatics] Force-skip ignored after {self._force_skip_counter} "
                                f"attempts for {current_err_type}; forcing report_findings"
                            )
                            messages.append({
                                "role": "user",
                                "content": (
                                    "[System] You have repeatedly failed to skip the problematic step "
                                    "despite mandatory instructions. You MUST call report_findings now "
                                    "with whatever results you have so far. Do NOT attempt any more code execution."
                                )
                            })
                            self._force_skip_counter = 0
                            self._force_skip_error_type = None
                            force_break = True

                except Exception as e:
                    logger.warning(f"[Bioinformatics] Supervisor callback error: {e}")
            # === END SUPERVISOR ===

            # Force-break: supervisor force-skip was ignored 3 times; the loop
            # must exit now. Mark completed so the return path produces a valid
            # status rather than falling through to max_steps_reached with empty findings.
            if force_break:
                logger.info("[Bioinformatics] Force-break triggered by supervisor force-skip")
                if not all_findings:
                    all_findings.append(
                        "[Supervisor] Analysis was forcibly stopped: the agent was stuck "
                        "on a repeated error despite mandatory-skip instructions and "
                        "PI guidance. Results below are partial."
                    )
                completed = True
                break

            # Check if plan revision was triggered during this step
            if _plan_revision_needed:
                # The consult_pi plan-revision path breaks the per-tool_call loop
                # early, so sibling tool_calls in this assistant message may lack
                # results. Backfill placeholders so the saved checkpoint stays
                # resumable (orphaned tool_calls cause an upstream HTTP 400).
                append_missing_tool_results(
                    messages,
                    [tc.id for tc in response.tool_calls],
                    note="[System: plan revision was triggered; this tool call was not executed.]",
                )
                # Plan revision is a semantic execution boundary. Persist the
                # exact transcript and Jupyter state now instead of relying on
                # the five-step periodic checkpoint, which could replay work.
                try:
                    await _save_executor_checkpoint(
                        f'step_{step_count}_plan_revision',
                        save_sandbox=True,
                    )
                except Exception as e:
                    logger.warning(
                        "[Bioinformatics] Plan-revision checkpoint save failed at "
                        "step %s: %s",
                        step_count,
                        e,
                    )
                logger.info("[Bioinformatics] Plan revision needed, pausing execution")
                # Result dict will be populated after the loop exits
                break

            # Metadata-only iterations (no substantive work) don't count toward step budget.
            # When both figure and register_score are present, use the figure budget cap.
            if not has_substantive_work and response.has_tool_calls:
                if has_figure_work:
                    figure_iteration_count += 1
                    if figure_iteration_count <= MAX_FIGURE_ITERATIONS:
                        step_count -= 1
                        logger.debug(f"[Bioinformatics] Metadata-only step excluded ({figure_iteration_count}/{MAX_FIGURE_ITERATIONS}), step_count={step_count}")
                    else:
                        logger.warning(f"[Bioinformatics] Figure iteration budget ({MAX_FIGURE_ITERATIONS}) exhausted, accepting last version and moving on")
                elif has_register_score:
                    step_count -= 1
                    logger.debug(f"[Bioinformatics] register_score-only step excluded, step_count={step_count}")

        if llm_error:
            status = "error"
        elif _plan_revision_needed:
            status = "paused_for_revision"
        elif completed:
            status = "partial" if evidence_gate_degraded else "completed"
        else:
            status = "max_steps_reached"

        # Fallback: extract findings from last assistant text if report_findings
        # was never called AND we ran out of step budget. We deliberately do NOT
        # fall back for text-only exhaustion (that path now raises
        # UpstreamExhaustedError and never reaches here) or for `llm_error`
        # exits, where fabricated findings would mask real failures and push
        # the workflow into the next stage with empty Stage-1 results.
        if not all_findings and not completed and status == "max_steps_reached":
            logger.warning("[Bioinformatics] No findings reported, attempting fallback extraction...")
            last_assistant_content = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    last_assistant_content = msg["content"]
                    break
            if last_assistant_content:
                all_findings.append(f"[Fallback extraction] {last_assistant_content[:3000]}")
                status = "completed_with_fallback"
                logger.info("[Bioinformatics] Extracted findings from last text response (fallback)")

        _saved_fig_count = count_saved_figures(analysis_log)
        logger.info(f"[Bioinformatics] Analysis {status}: {step_count} steps executed, {len(all_findings)} findings, {len(all_figures)} inline figures, {_saved_fig_count} saved figures")

        # Clear checkpoint only on explicit completion via report_findings
        # Preserve checkpoint for max_steps_reached and completed_with_fallback to allow resumption
        if completed:
            checkpoint_mgr.clear()

        return {
            "status": status,
            "findings": all_findings,
            "figures": all_figures,
            "steps_executed": step_count,
            "code_history": self.code_history,
            "analysis_log": analysis_log,
            "success": status != "error",
            "mechanism_programs": mechanism_programs,
            "evidence_verified": bool(reported_evidence_ids) and not evidence_gate_degraded,
            "evidence_execution_ids": reported_evidence_ids,
            **({"error": llm_error} if llm_error else {}),
            "plan_revision_needed": _plan_revision_needed,
            "plan_revision_context": _plan_revision_context if _plan_revision_needed else {},
        }

    
    
    
    


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
        - Show at least 5 nodes and make the directional flow obvious with arrows.
        - Include cellular processes only when they are present in the supplied evidence.
        - Do NOT wrap the output in markdown code fences (no ``` lines).
        - Output ONLY the diagram — no prose explanation before or after.
    """).strip()



