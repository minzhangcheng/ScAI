"""
Bioinformatics Agent - Handles snRNA-seq analysis with iterative code execution
Supports autonomous technical steps and PI consultation for biological decisions
With step-level checkpointing for recovery from interruptions
"""
from typing import Dict, Any, List, Optional, Callable, Awaitable, Tuple
import json
import logging
import base64
import io
import uuid
import re
import textwrap
from pathlib import Path
from datetime import datetime

import aiofiles

from .base import BaseAgent, AgentConfig
from ..utils.sandbox import (
    validate_code_safety, DANGEROUS_PATTERNS, extract_code_from_markdown,
    LocalJupyterSandbox, CodeExecutionResult,
)
from ..utils.checkpoint import CheckpointManager

# Configure logger
logger = logging.getLogger(__name__)

from ..models.schemas import (
    AgentType, AnalysisState, MessageRole, CodeExecution,
    PublicationFigure, FigureReviewStatus, AnalysisStep
)
from ..utils.llm_client import OpenAICompatibleClient, ToolCall
from ..utils.error_parsing import strip_ansi

# Pre-compiled pattern for detecting manual cell_type assignment in code
_CELL_TYPE_ASSIGN_RE = re.compile(r"""\.obs\[['"]cell_type['"]\]\s*=""")

# Reusable sandbox template snippet: safely close all matplotlib figures
_TPL_SAFE_CLOSE_ALL = """\
try:
    plt.close('all')
except Exception:
    pass"""

# Define the tools for bioinformatics agent
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Generate and execute Python code for single-cell analysis. Use for technical steps like data loading, QC, normalization, clustering, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute"
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of what this code does"
                    },
                    "step_type": {
                        "type": "string",
                        "enum": ["technical", "biological"],
                        "description": "Whether this is a technical step (can proceed autonomously) or requires biological interpretation"
                    }
                },
                "required": ["code", "description", "step_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consult_pi",
            "description": "Request guidance from PI Agent on biological decisions like: analysis direction, which genes/pathways to analyze, whether to do trajectory analysis, result interpretation. Do NOT use for cell type annotation - use annotate_clusters instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The biological question or decision that needs PI guidance"
                    },
                    "context": {
                        "type": "string",
                        "description": "Relevant context from current analysis (e.g., cluster markers, current findings)"
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Possible options or choices if applicable"
                    }
                },
                "required": ["question", "context"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_literature",
            "description": "Search literature for biological information like marker genes, cell type signatures, pathway information",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to search for in literature"
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "report_findings",
            "description": "Report key findings and conclusions. Call this ONLY when you have completed ALL assigned tasks. In the exploration phase, you MUST complete the full pipeline including annotate_clusters BEFORE calling this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of key findings"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Summary of the analysis results"
                    },
                    "suggestions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Suggestions for next steps or further analysis"
                    }
                },
                "required": ["findings", "summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_publication_figure",
            "description": "Generate a publication-quality figure with PNG thumbnail and high-res PDF. The figure will be automatically reviewed by the PI Agent. Use this for important figures that should be included in the final report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python plotting code using matplotlib, seaborn, or scanpy. Should produce a single figure."
                    },
                    "figure_name": {
                        "type": "string",
                        "description": "Name for the figure file (English, no spaces, e.g., 'umap_cell_types')"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of what the figure shows (will be used for figure legend)"
                    },
                    "figure_type": {
                        "type": "string",
                        "enum": ["umap", "heatmap", "violin", "dotplot", "barplot", "trajectory", "volcano", "other"],
                        "description": "Type of figure being generated"
                    }
                },
                "required": ["code", "figure_name", "description", "figure_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "annotate_clusters",
            "description": "Annotate all cell clusters at once using literature-based evidence. Call this ONCE after clustering and rank_genes_groups. Returns cell type annotations with confidence levels and literature evidence. Much more efficient than per-cluster consult_pi calls. You do NOT need to pass cluster_markers — they will be auto-extracted from adata.uns['rank_genes_groups'] if omitted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_markers": {
                        "type": "object",
                        "description": "Optional. Dict mapping cluster ID (string) to list of top marker genes. Example: {\"0\": [\"Cd79a\", \"Ms4a1\", \"Cd79b\"], \"1\": [\"Cd3d\", \"Cd3e\", \"Cd8a\"]}. If omitted, markers will be auto-extracted from adata.uns['rank_genes_groups'].",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "tissue": {
                        "type": "string",
                        "description": "Tissue or organ type (e.g., 'kidney', 'liver', 'brain'). Extract from the research context or data metadata."
                    },
                    "organism": {
                        "type": "string",
                        "description": "Organism (e.g., 'mouse', 'human'). Extract from the research context or data metadata."
                    }
                },
                "required": ["tissue", "organism"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "register_score",
            "description": "Register how a score was computed for the final report methodology section. Call this IMMEDIATELY AFTER computing any score (gene signature scores, pathway enrichment, TF activity, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "score_name": {
                        "type": "string",
                        "description": "Name of the score as stored in adata (e.g., 'Fibrosis_Score', 'EMT_signature')"
                    },
                    "method": {
                        "type": "string",
                        "description": "Computation method (e.g., 'sc.tl.score_genes', 'gseapy.ssgsea', 'dc.run_ulm')"
                    },
                    "description": {
                        "type": "string",
                        "description": "What this score measures biologically (1-2 sentences)"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Key parameters used (e.g., {\"ctrl_size\": 50, \"n_bins\": 25})"
                    },
                    "gene_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Gene names used for the score (if applicable, top 50)"
                    },
                    "gene_list_source": {
                        "type": "string",
                        "description": "Source of the gene list (e.g., 'MSigDB Hallmark EMT', 'top 50 DEGs', 'PMID:12345678')"
                    }
                },
                "required": ["score_name", "method", "description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_documentation",
            "description": "Look up API documentation for an installed Python function/class. Returns a concise summary with signature, key parameters, and usage example. Use when unsure about correct API usage before writing code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "function_path": {
                        "type": "string",
                        "description": "Full dotted path to the function or module (e.g., 'scanpy.tl.umap', 'decoupler.op.collectri', 'gseapy.enrichr')"
                    },
                    "question": {
                        "type": "string",
                        "description": "Optional: specific question about the API (e.g., 'what parameters control batch correction?')"
                    }
                },
                "required": ["function_path"]
            }
        }
    },
]


class BioinformaticsAgent(BaseAgent):
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
        llm_client: OpenAICompatibleClient,
        model: str,
        system_prompt: str,
        sandbox: LocalJupyterSandbox,
        literature_callback: Optional[Callable[[str], Awaitable[str]]] = None,
        pi_callback: Optional[Callable[[str, str], Awaitable[str]]] = None,
        annotate_clusters_callback: Optional[Callable[[Dict, str, str], Awaitable[Dict[str, Any]]]] = None,
        message_callback: Optional[Callable[[str, str], Awaitable[None]]] = None,
        figure_review_callback: Optional[Callable[[PublicationFigure], Awaitable[Dict[str, Any]]]] = None,
        supervisor_callback: Optional[Callable[['AnalysisStep', List[str]], Awaitable[Optional[str]]]] = None,
        technique_agent=None,
        config: Optional[AgentConfig] = None
    ):
        super().__init__(
            agent_type=AgentType.BIOINFORMATICS,
            llm_client=llm_client,
            model=model,
            system_prompt=system_prompt,
            config=config
        )
        self.sandbox = sandbox
        self.literature_callback = literature_callback
        self.pi_callback = pi_callback
        self.annotate_clusters_callback = annotate_clusters_callback
        self.message_callback = message_callback
        self.figure_review_callback = figure_review_callback
        self.supervisor_callback = supervisor_callback
        self.technique_agent = technique_agent

        # Track execution history
        self.code_history: List[CodeExecution] = []
        self.last_result: Optional[CodeExecutionResult] = None
        self.analysis_context: Dict[str, Any] = {}
        self._recent_sup_feedback: set = set()

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
    
    def _build_notebook_context(self) -> str:
        """Build context string from notebook history"""
        if not self.code_history:
            return "Notebook is empty. This is the first code cell."

        # Prevent unbounded memory growth across rounds
        MAX_CODE_HISTORY = 100
        if len(self.code_history) > MAX_CODE_HISTORY:
            self.code_history = self.code_history[-MAX_CODE_HISTORY:]

        context_parts = ["=== Notebook History (last 10 cells) ===\n"]

        recent_history = self.code_history[-10:]
        start_idx = len(self.code_history) - len(recent_history)
        
        for i, execution in enumerate(recent_history):
            context_parts.append(f"--- Cell {start_idx + i + 1} ---")
            context_parts.append(f"```python\n{execution.code}\n```")
            
            if execution.output:
                if len(execution.output) > 2000:
                    output = f"[... first {len(execution.output) - 2000} chars truncated]\n{execution.output[-2000:]}"
                else:
                    output = execution.output
                context_parts.append(f"Output:\n{output}")
            
            if execution.error:
                context_parts.append(f"Error: {execution.error}")
            
            if execution.figures:
                context_parts.append(f"[Generated {len(execution.figures)} figure(s)]")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
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
        if seq_files:
            data_parts.append("Sequencing Data Files:")
            for f in seq_files:
                group = state.get_group_by_id(f.group_id)
                group_label = group.name if group else "Ungrouped"
                sample_label = f"Sample {f.sample_number}" if f.sample_number else ""
                label = f"{group_label} {sample_label}".strip()

                data_parts.append(f"  {label} ({f.data_type}): {f.file_path}")
                if f.data_type == "scRNA-seq":
                    # Generate a safe variable name for loading (deduplicate)
                    base_name = (group_label.lower().replace(" ", "_") +
                                (f"_s{f.sample_number}" if f.sample_number else ""))
                    var_name_counts[base_name] = var_name_counts.get(base_name, 0) + 1
                    var_name = f"{base_name}_{var_name_counts[base_name]}" if var_name_counts[base_name] > 1 else base_name
                    data_parts.append(f"    Load: adata_{var_name} = sc.read_h5ad('{f.file_path}')")
                    data_parts.append(f"    adata_{var_name}.obs_names_make_unique()")
                    data_parts.append(f"    adata_{var_name}.var_names_make_unique()")
                elif f.data_type == "loom":
                    data_parts.append(f"    Loom file for RNA velocity analysis")
        else:
            data_parts.append("⚠️ No sequencing data uploaded")

        # Add explicit data constraint to prevent hallucinating non-existent groups
        if state.treatment_groups:
            group_names = ', '.join(g.name for g in state.treatment_groups)
            data_parts.append(f"\n**CRITICAL DATA CONSTRAINT:** All analysis and figures MUST use only the actual experimental groups listed above. Do NOT fabricate data for groups that don't exist. If a figure requires group comparisons, use ONLY: {group_names}.")

        return "\n".join(data_parts)
    
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

    async def _discover_adata_in_sandbox(self) -> Tuple[Optional[str], Optional[str]]:
        """Discover the active AnnData variable name and clustering key in the sandbox.

        Iterates over sandbox globals to find an AnnData object with a known clustering
        column (leiden, louvain, cluster).

        Returns:
            Tuple of (variable_name, clustering_key), or (None, None) on failure.
        """
        if not self.sandbox:
            return (None, None)

        discover_code = (
            "import json as _json, anndata as _ad\n"
            "_result = {}\n"
            "for _name, _obj in list(globals().items()):\n"
            "    if isinstance(_obj, _ad.AnnData) and not _name.startswith('_'):\n"
            "        for _key in ['leiden', 'louvain', 'cluster']:\n"
            "            if _key in _obj.obs.columns:\n"
            "                _result = {'var': _name, 'cluster_key': _key}\n"
            "                break\n"
            "        if _result:\n"
            "            break\n"
            "print(_json.dumps(_result))\n"
        )
        try:
            result = await self.sandbox.execute(discover_code, save_to_history=False, skip_safety_check=True)
            if result.success and result.output and result.output.strip():
                parsed = json.loads(result.output.strip())
                var_name = parsed.get("var")
                cluster_key = parsed.get("cluster_key")
                if var_name and cluster_key:
                    logger.info(f"[Bioinformatics] Discovered AnnData var={var_name}, cluster_key={cluster_key}")
                    return (var_name, cluster_key)
        except Exception as e:
            logger.warning(f"[Bioinformatics] AnnData discovery failed: {e}")

        return (None, None)

    async def run_autonomous_analysis(
        self,
        state: AnalysisState,
        initial_instruction: str,
        max_steps: int = 200,
        figure_review_callback: Optional[Callable[[PublicationFigure], Awaitable[Dict[str, Any]]]] = None,
        supervisor_callback: Optional[Callable] = None,
        resume_from_checkpoint: bool = True
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
                        execution_time=0.0,
                    ))
                logger.info(f"[Bioinformatics] Restored {len(self.code_history)} cells into code_history from notebook state")

        # Build initial prompt
        system_prompt = f"""{self.system_prompt}

You are conducting single-cell RNA-seq analysis. You have access to tools:
1. execute_code - Run Python code (scanpy, anndata, numpy, pandas, matplotlib available)
   When writing plotting code, do NOT call plt.close() or plt.show() - the system handles figure saving and cleanup automatically.
2. consult_pi - Ask PI for biological decisions (analysis direction, pathway selection, interpretation)
3. search_literature - Search for marker genes, pathways, biological knowledge
4. annotate_clusters - Annotate ALL cell clusters at once with literature evidence (use INSTEAD of per-cluster consult_pi)
5. report_findings - Report conclusions ONLY after ALL assigned tasks are complete (including annotate_clusters in exploration)
6. generate_publication_figure - Generate publication-quality figures with PI review
7. register_score - Register a computed score's methodology for the final report
8. lookup_documentation - Look up API docs for an installed Python function.
   Returns concise summary (signature, key params, example).
   Use when unsure about API usage, especially for: decoupler, gseapy, scvelo, cellrank.
   Example: lookup_documentation(function_path="decoupler.op.collectri")
   IMPORTANT: Figure code MUST use actual analysis data from the sandbox (e.g., adata object).
   NEVER generate dummy/fabricated data. Only plot groups that exist in the experimental design.
   Do NOT include plt.close(), plt.show(), or plt.savefig() in the figure code - the system handles saving automatically.

   **PUBLICATION FIGURE MANDATORY CHECKLIST:**
   - When using sc.pl.umap/tsne, ALWAYS set show=False and capture the axes: `ax = sc.pl.umap(adata, ..., show=False, return_fig=True)`
   - ALWAYS manually add ax.set_xlabel('UMAP1') and ax.set_ylabel('UMAP2') to ALL embedding plots
   - If UMAP legends overlap data, use legend_loc='right margin' — NEVER manually set legend coordinates
   - All text elements must have fontsize >= 12pt (title >= 14pt)
   - Legend must NOT overlap data — place outside plot or use bbox_to_anchor
   - Use consistent naming across ALL panels (e.g., always 'Cholangiocyte', never abbreviate in some panels)
   - No grid lines unless explicitly needed for the figure type
   - Always call ax.set_xticks() BEFORE ax.set_xticklabels() to ensure label count matches tick count

   **COMPOSITION / PROPORTION BARPLOT RULES:**
   - When plotting cell type proportions across conditions (e.g., WT vs DKO):
     1. The cell type categories on the y-axis must be CONDITION-INDEPENDENT (e.g., 'Hepatocyte', not 'Hepatocyte_WT')
     2. If cell types in adata.obs contain condition suffixes (e.g., 'Hepatocyte_WT', 'Hepatocyte_DKO'), you MUST strip/merge them first
     3. Compute proportions as: (cells of type X in condition A) / (total cells in condition A)
     4. A correct composition plot should show MIXED colors in each bar — if any bar is 100% one color, check for tautological grouping
   - For pandas Categorical columns: call .cat.add_categories('NewCat') BEFORE assigning a new category value
   - Example pattern for merging:
     ct = adata.obs['cell_type'].astype(str)
     for cond in conditions:
         ct = ct.str.replace(f'_{{cond}}', '', regex=False)

SECURITY RESTRICTIONS:
- Do NOT use globals(), exec(), eval(), __import__(), or importlib — these are blocked by the sandbox. subprocess is only allowed for `subprocess.run(['pip', 'install', 'package_name'])`.
- ALWAYS use standard `import xxx` or `from xxx import yyy` syntax. NEVER use `__import__('xxx')` — it will be blocked.
- If a package import fails (ModuleNotFoundError), install it using `subprocess.run(['pip', 'install', 'package_name'])`, then import it normally. Do NOT use importlib.
- Do NOT write files to disk using open(..., 'w') or similar — direct file writes are blocked by the sandbox. Saving via library methods (e.g. pandas df.to_csv(), scanpy sc.pl.*, matplotlib savefig) to the workspace directory IS allowed. Store intermediate data in adata.uns, adata.obs, adata.var, or Python variables.
- Do NOT call annotate_clusters in the same response as execute_code that generates markers.
  You must FIRST execute code and see the printed output, THEN call annotate_clusters in a separate step.

WORKFLOW GUIDELINES:
- Technical steps can be done autonomously: data loading, QC, normalization, PCA, clustering, UMAP, basic DE
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
  * If batch effects are apparent (samples cluster by batch rather than cell type), consult PI about correction strategy (Harmony, BBKNN, scVI, Combat, etc.)
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
2. After PI confirms/modifies, execute the computation via execute_code
3. IMMEDIATELY after, call register_score to record the methodology
This applies to: sc.tl.score_genes, gseapy.ssgsea, gseapy.prerank, gseapy.enrichr, decoupleR, custom scoring functions.

GENE MODULE SCORING — USE FULL GENE SET (CRITICAL):
  sc.tl.score_genes() on HVG-filtered data will miss most genes. Always:
  1. Preserve raw before HVG filtering: adata.raw = adata.copy()  (do this ONCE, right after normalization)
  2. Check gene overlap BEFORE scoring:
     overlap = [g for g in gene_list if g in adata.raw.var_names]
     print(f"Gene overlap: {{len(overlap)}}/{{len(gene_list)}}")
  3. Score with use_raw=True: sc.tl.score_genes(adata, gene_list, score_name='...', use_raw=True)
  4. If overlap < 50% of gene_list, investigate: wrong organism? wrong gene naming convention?

SPECIES-AWARE GENE NAMES:
  Mouse genes: Title case (Cd79a, Ptprc, Gapdh)
  Human genes: UPPER CASE (CD79A, PTPRC, GAPDH)
  Check convention: print(adata.var_names[:10])
  Convert human→mouse: [g[0].upper() + g[1:].lower() for g in gene_list]

PACKAGE API LOOKUP:
  If unsure about the correct API for any package (decoupler, gseapy, scvelo, cellrank, etc.),
  call lookup_documentation BEFORE writing code.
  Example: lookup_documentation(function_path="decoupler.op.collectri")
  The system will also automatically look up documentation if you hit the same API error twice.

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
5. Call annotate_clusters(cluster_markers=markers, tissue="kidney", organism="mouse")
   where markers is the LITERAL dict content you saw in the print output.
6. Apply the returned mapping: adata.obs['cell_type'] = adata.obs['leiden'].astype(str).map(cluster_to_celltype).fillna('Unknown')
7. Generate UMAP colored by cell_type

{self._build_data_context(state)}

=== Working Directory ===
Your Jupyter kernel is running in: {self.sandbox.working_dir}
A variable `_WORKSPACE_DIR` (pathlib.Path) is pre-defined in the kernel and points to this directory.
All output files MUST be saved using `_WORKSPACE_DIR` as the root. NEVER use relative paths like './' or '../'.
- Figures: saved automatically by the system — do NOT call plt.savefig() yourself.
- Tables/CSV results: save to `_WORKSPACE_DIR / 'tables'`
- H5AD files: save to `_WORKSPACE_DIR`
When writing files, ALWAYS use the `_WORKSPACE_DIR` variable for the path:
  (_WORKSPACE_DIR / 'tables').mkdir(exist_ok=True)
  df.to_csv(_WORKSPACE_DIR / 'tables' / 'filename.csv')
  adata.write(_WORKSPACE_DIR / 'output.h5ad')
Do NOT use string literals for paths (e.g., './output.csv', '/workspace/results/').
Do NOT call open('filename', 'w') with bare filenames — use open(_WORKSPACE_DIR / 'tables' / 'filename', 'w').

{self._build_notebook_context()}

{self._build_analysis_context()}

**PACKAGE INSTALLATION**:
If you need a Python package that is not currently installed:
- Use `subprocess.run(['pip', 'install', 'package_name'])` to install it
- After installation, import the package normally in the same or next code cell
- Only install well-known scientific/bioinformatics packages
- Example: `subprocess.run(['pip', 'install', 'palettable'])` then `import palettable`

**CRITICAL: DATA PREPROCESSING (MUST DO BEFORE PCA)**
Before running sc.tl.pca() or any dimensionality reduction:
1. Check for NaN: print(f"NaN in X: {{np.isnan(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X).sum()}}")
2. If NaN exists, fix it:
   - sc.pp.filter_genes(adata, min_cells=3)
   - adata.X = np.nan_to_num(adata.X, nan=0.0) if dense
   - adata.X.data = np.nan_to_num(adata.X.data, nan=0.0) if sparse
3. NEVER skip this step - PCA will fail with NaN values!

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

        # Initialize or restore state from checkpoint
        if checkpoint and checkpoint_valid:
            messages = checkpoint.get('messages', [])
            # Validate message structure: each must have role and content
            valid_messages = all(
                isinstance(m, dict) and "role" in m and ("content" in m or "tool_calls" in m)
                for m in messages
            ) if messages else False
            if len(messages) < 2 or not valid_messages:
                logger.warning("[Bioinformatics] Checkpoint messages invalid (len=%d, valid=%s), using fresh messages",
                               len(messages), valid_messages)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analysis instruction: {initial_instruction}"}
                ]
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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analysis instruction: {initial_instruction}"}
            ]
            all_findings = []
            all_figures = []
            step_count = 0
            analysis_log = []
            # Reset per-run state to avoid pollution from previous rounds
            # (process() does this but workflow calls run_autonomous_analysis directly)
            self.code_history = []
            self.analysis_context = {}
            self.last_result = None

        completed = False
        annotation_completed = False  # Mandatory step tracker for exploration phase
        llm_error: Optional[str] = None
        text_response_count = 0
        MAX_TEXT_RESPONSES = 2
        figure_iteration_count = 0
        MAX_FIGURE_ITERATIONS = 5  # Max figure-only iterations excluded from step count
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 3  # Force report_findings after this many consecutive code errors

        while step_count < max_steps and not completed:
            step_count += 1

            # Save checkpoint periodically
            if checkpoint_mgr.should_checkpoint(step_count):
                try:
                    checkpoint_data = {
                        'step_name': f'step_{step_count}',
                        'session_id': state.id,
                        'round_number': state.current_round,
                        'step_number': step_count,
                        'total_steps': max_steps,
                        'messages': messages,
                        'findings': all_findings,
                        'figures': all_figures,
                        'code_history': [ce.model_dump() for ce in self.code_history],
                        'analysis_log': [step.model_dump() for step in analysis_log],
                        'analysis_context': self.analysis_context,
                        'completed': False
                    }
                    await checkpoint_mgr.save(checkpoint_data)
                    # Also save sandbox session for Python environment recovery
                    await self.sandbox.save_session()
                except Exception as e:
                    logger.warning(f"[Bioinformatics] Checkpoint save failed at step {step_count}: {e}")
            
            self._set_llm_context()

            # Trim message history to prevent unbounded growth
            MAX_MESSAGES = 80
            if len(messages) > MAX_MESSAGES:
                system_msg = messages[0]
                # Find the cut index: keep roughly MAX_MESSAGES-1 tail messages
                cut = len(messages) - (MAX_MESSAGES - 1)
                # Walk the cut point backward to avoid orphaning tool_result messages.
                # A "tool" role message requires its preceding assistant message
                # that contains the corresponding tool_use block.
                while cut < len(messages) and messages[cut].get("role") == "tool":
                    cut -= 1
                if cut < 1:
                    cut = 1
                messages = [system_msg] + messages[cut:]
                logger.info(f"Trimmed message history to {len(messages)} messages")

            try:
                response = await self.llm_client.chat_with_tools(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    temperature=self.config.temperature,  # Use configured temperature
                    max_tokens=32768
                )
            except Exception as e:
                llm_error = str(e)
                await self._notify("error", f"LLM call failed: {llm_error}")
                break

            # Capture agent reasoning from response content (before tool calls)
            agent_reasoning = response.content[:2000] if response.content else None

            if not response.has_tool_calls:
                if response.content:
                    await self._notify("response", response.content)
                    messages.append({"role": "assistant", "content": response.content})

                # Record text-only response in analysis log
                analysis_log.append(AnalysisStep(
                    step_number=step_count, step_type="text_response",
                    agent_reasoning=agent_reasoning,
                    text_content=(response.content or "")[:2000],
                ))
                agent_reasoning = None

                text_response_count += 1
                if text_response_count >= MAX_TEXT_RESPONSES:
                    logger.info(f"[Bioinformatics] {MAX_TEXT_RESPONSES} text-only responses, exiting loop")
                    break

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

                if tool_name == "execute_code":
                    has_substantive_work = True
                    code = args.get("code", "")
                    description = args.get("description", "")
                    step_type = args.get("step_type", "technical")

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

                    # Guard: block manual cell_type assignment when annotation
                    # has not been completed through the annotate_clusters workflow
                    if (
                        state.analysis_phase == "exploration"
                        and not annotation_completed
                        and _CELL_TYPE_ASSIGN_RE.search(code)
                    ):
                        guard_msg = (
                            "Manual assignment to adata.obs['cell_type'] is not allowed "
                            "before cell type annotations have been confirmed through "
                            "the annotate_clusters workflow. Please call annotate_clusters first."
                        )
                        logger.warning(f"[Bioinformatics] Blocked manual cell_type assignment: annotation not confirmed")
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
                    figures_dir_repr = repr(str(figures_dir))
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    indented_code = textwrap.indent(code, '    ')

                    wrapped_code = f'''
import matplotlib.pyplot as plt
import matplotlib.figure as _mpl_figure
from pathlib import Path
from IPython.display import display

# Close any stale figures from previous executions to prevent poisoning
{_TPL_SAFE_CLOSE_ALL}

# Set PDF vectorized fonts
plt.rcParams.update({{
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}})

# Create figures directory
_figures_dir = Path({figures_dir_repr})
_figures_dir.mkdir(exist_ok=True)

# Track figures explicitly saved by user code: map fig_number → list of saved paths
_explicitly_saved_figs = {{}}
_prev_fig_savefig = _mpl_figure.Figure.savefig
def _tracking_savefig(self, fname=None, *_a, **_kw):
    try:
        _explicitly_saved_figs.setdefault(self.number, [])
        if fname is not None:
            _explicitly_saved_figs[self.number].append(str(fname))
    except Exception as _track_err:
        print(f"[Auto-save] Debug: figure tracking failed: {{_track_err}}")
    return _prev_fig_savefig(self, fname, *_a, **_kw)
_mpl_figure.Figure.savefig = _tracking_savefig

# Temporarily disable plt.close/plt.show to keep figures alive for auto-save
_orig_close = plt.close
_orig_show = plt.show
plt.close = lambda *a, **kw: None
plt.show = lambda *a, **kw: None

# Execute user code (with try-finally to guarantee restore)
try:
{indented_code}
finally:
    plt.close = _orig_close
    plt.show = _orig_show
    _mpl_figure.Figure.savefig = _prev_fig_savefig

# Detect and save all active figures
try:
    _fig_nums = plt.get_fignums()
    for _fig_num in _fig_nums:
        try:
            _fig = plt.figure(_fig_num)

            # Display figure to trigger capture via display_data message (always)
            display(_fig)

            _fig_id = f"fig_{timestamp}_{{_fig_num}}"

            # For figures already saved by user code, skip JPG but supplement PDF if missing
            if _fig_num in _explicitly_saved_figs:
                _saved_paths = _explicitly_saved_figs[_fig_num]
                # Emit [Auto-saved] for explicitly saved images so they get tracked
                for _sp in _saved_paths:
                    _sp_lower = _sp.lower()
                    if _sp_lower.endswith(('.jpg', '.jpeg')):
                        print(f"[Auto-saved] JPG: {{_sp}}")
                    elif _sp_lower.endswith('.png'):
                        print(f"[Auto-saved] PNG: {{_sp}}")
                _has_pdf = any(_p.lower().endswith('.pdf') for _p in _saved_paths)
                if not _has_pdf and _saved_paths:
                    # Derive PDF path from the first saved image path
                    _src_path = Path(_saved_paths[0])
                    _pdf_supplement = _src_path.with_suffix('.pdf')
                    try:
                        _fig.savefig(str(_pdf_supplement), dpi=300, bbox_inches='tight', format='pdf')
                        print(f"[Auto-save] Supplemented PDF: {{_pdf_supplement}}")
                    except Exception as _pdf_err:
                        print(f"[Auto-save] Warning: could not supplement PDF: {{_pdf_err}}")
                else:
                    print(f"[Auto-save] Skipped figure {{_fig_num}} (already saved by code with PDF)")
                continue

            _jpg_path = _figures_dir / f"{{_fig_id}}.jpg"
            _pdf_path = _figures_dir / f"{{_fig_id}}.pdf"

            # Save JPG (for web display — smaller file size than PNG)
            _fig.savefig(_jpg_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none',
                         format='jpeg', pil_kwargs={{'quality': 85, 'optimize': True}})

            # Save PDF (vector format for publication quality download)
            _fig.savefig(_pdf_path, dpi=300, bbox_inches='tight', format='pdf')

            print(f"[Auto-saved] JPG: {{_jpg_path}}")
            print(f"[Auto-saved] PDF: {{_pdf_path}}")
        except Exception as _save_err:
            print(f"[Auto-save] Warning: could not save figure {{_fig_num}}: {{_save_err}}")
            try:
                plt.close(_fig_num)
            except Exception as _close_err:
                print(f"[Auto-save] Debug: plt.close({{_fig_num}}) failed: {{_close_err}}")
except Exception as _autosave_block_err:
    print(f"[Auto-save] Warning: figure auto-save failed: {{_autosave_block_err}}")

# Close all figures to free memory
{_TPL_SAFE_CLOSE_ALL}
'''

                    result = await self.sandbox.execute(wrapped_code)

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

                    code_exec = CodeExecution(
                        code=code,
                        output=result.output,
                        error=result.error,
                        figures=result.figures,
                        execution_time=0.0
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
                    if result.error:
                        await self._notify("error", result.error)
                        consecutive_errors += 1
                    else:
                        consecutive_errors = 0

                    tool_result = f"Execution {'successful' if not result.error else 'failed'}.\n"
                    if result.output:
                        tool_result += f"Output:\n{result.output}\n"
                    if result.error:
                        tool_result += f"Error: {result.error}\n"
                        # Hint: if a module is missing, remind the agent to pip install
                        if "ModuleNotFoundError" in result.error:
                            missing = ""
                            m = re.search(r"No module named ['\"]([^'\"]+)['\"]", result.error)
                            if m:
                                missing = m.group(1).split(".")[0]
                            tool_result += (
                                f"\nHint: The package '{missing}' is not installed. "
                                f"Install it by running: subprocess.run(['pip', 'install', '{missing}']), "
                                f"then retry the import.\n"
                            ) if missing else (
                                "\nHint: A required package is not installed. "
                                "Install it using: subprocess.run(['pip', 'install', '<package_name>']).\n"
                            )
                    if result.figures:
                        tool_result += f"Generated {len(result.figures)} figure(s).\n"
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                    # Circuit breaker: too many consecutive errors → force report or redirect
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        if state.analysis_phase == "exploration" and not annotation_completed:
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
                            logger.warning(f"[Bioinformatics] {consecutive_errors} consecutive code errors, forcing report")
                            messages.append({
                                "role": "user",
                                "content": (
                                    "[System] Too many consecutive code execution errors. "
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

                    n_clusters = len(cluster_markers)

                    # Discover the active AnnData variable and clustering key in sandbox
                    adata_var, cluster_key = await self._discover_adata_in_sandbox()

                    if adata_var is None:
                        tool_result = (
                            "Error: Could not find an AnnData object with a clustering column "
                            "(leiden/louvain/cluster) in the sandbox. Before calling annotate_clusters, "
                            "you must: (1) load data into an AnnData variable, (2) run clustering "
                            "(e.g., sc.tl.leiden(adata)), and (3) run differential expression "
                            "(sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon'))."
                        )
                        logger.warning("[Bioinformatics] annotate_clusters called but no AnnData with clustering found")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })
                        continue

                    # Auto-extract markers from kernel adata if LLM didn't provide them
                    if n_clusters == 0 and self.sandbox:
                        logger.info(f"[Bioinformatics] cluster_markers empty, auto-extracting from {adata_var}.uns['rank_genes_groups']")
                        extract_code = (
                            "import json as _json\n"
                            "_markers = {}\n"
                            "try:\n"
                            "    import scanpy as sc\n"
                            f"    for _cl in {adata_var}.obs['{cluster_key}'].cat.categories:\n"
                            f"        _df = sc.get.rank_genes_groups_df({adata_var}, group=str(_cl))\n"
                            "        _df = _df[_df['pvals_adj'] < 0.05]\n"
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
                            f"{adata_var}.uns['rank_genes_groups'] failed. You must first run "
                            f"sc.tl.rank_genes_groups({adata_var}, '{cluster_key}', method='wilcoxon') "
                            f"and ensure it completes successfully before calling "
                            f"annotate_clusters."
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })
                        continue

                    await self._notify("status", f"Annotating {n_clusters} clusters using literature evidence ({organism} {tissue})...")

                    annotation_result: Dict[str, Any] = {}
                    if self.annotate_clusters_callback:
                        annotation_result = await self.annotate_clusters_callback(
                            cluster_markers, tissue, organism,
                            analysis_log=analysis_log,
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
                        tool_result = (
                            f"[ANNOTATION FAILED] {error_detail}\n"
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
                        result_lines.append("[NOTE: These annotations were already confirmed by the user. No re-annotation needed. Proceed to downstream analysis.]\n")
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
                    ))
                    annotation_completed = True

                elif tool_name == "report_findings":
                    # Enforce mandatory pipeline in exploration phase
                    if state.analysis_phase == "exploration" and not annotation_completed:
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
                    has_substantive_work = True
                    findings = args.get("findings", [])
                    summary = args.get("summary", "")
                    suggestions = args.get("suggestions", [])

                    all_findings.extend(findings)

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
                    ))

                    completed = True
                    break

                elif tool_name == "register_score":
                    has_register_score = True
                    from ..models.schemas import ScoreDefinition
                    score_def = ScoreDefinition(
                        score_name=args.get("score_name", ""),
                        method=args.get("method", ""),
                        description=args.get("description", ""),
                        parameters=args.get("parameters", {}),
                        gene_list=args.get("gene_list", []),
                        gene_list_source=args.get("gene_list_source", ""),
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
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Score '{score_def.score_name}' {action_word}. Method: {score_def.method}. This will appear in the report methodology."
                    })

                    analysis_log.append(AnalysisStep(
                        step_number=step_count, step_type="register_score",
                        agent_reasoning=agent_reasoning,
                        code_description=f"Score: {score_def.score_name} via {score_def.method}",
                    ))

                elif tool_name == "lookup_documentation":
                    function_path = args.get("function_path", "")
                    question = args.get("question", "")

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

            # Clear reasoning after all tool calls in this iteration are processed
            agent_reasoning = None

            # Reset text-only counter when a substantive tool call was made
            if has_substantive_work and response.has_tool_calls:
                text_response_count = 0

            # === SUPERVISOR monitoring (every step) ===
            if _supervisor_callback and analysis_log:
                try:
                    supervisor_feedback = await _supervisor_callback(
                        analysis_log[-1],   # Latest step
                        _step_figures       # Figures from this iteration (may be empty)
                    )
                    if supervisor_feedback:
                        # De-duplicate: skip if the same feedback (first 120 chars)
                        # was already injected recently to avoid wasting tokens on
                        # repeated identical warnings.
                        sup_key = supervisor_feedback[:120]
                        if sup_key not in self._recent_sup_feedback:
                            messages.append({
                                "role": "user",
                                "content": f"[Supervisor] {supervisor_feedback}"
                            })
                            self._recent_sup_feedback.add(sup_key)
                        else:
                            logger.debug(f"[Bioinformatics] Skipping duplicate supervisor feedback: {sup_key[:60]}...")
                except Exception as e:
                    logger.warning(f"[Bioinformatics] Supervisor callback error: {e}")
            # === END SUPERVISOR ===

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
        else:
            status = "completed" if completed else "max_steps_reached"

        # Fallback: extract findings from last assistant text if report_findings was never called
        if not all_findings and not completed:
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

        logger.info(f"[Bioinformatics] Analysis {status}: {step_count} steps executed, {len(all_findings)} findings, {len(all_figures)} figures")

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
            **({"error": llm_error} if llm_error else {}),
        }

    async def execute_step(self, state: AnalysisState, instruction: str) -> Dict[str, Any]:
        """Execute a single analysis step (legacy interface)"""
        return await self.run_autonomous_analysis(state, instruction, max_steps=5)
    
    async def process(self, state: AnalysisState, instruction: str, **kwargs) -> Dict[str, Any]:
        """Main entry point for workflow"""
        # Reset per-run state to avoid pollution from previous rounds
        self.code_history = []
        self.analysis_context = {}
        self._recent_sup_feedback: set = set()  # Reset supervisor dedup set per round
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
            "steps_executed": result["steps_executed"]
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

    # ============================================
    # Publication Figure Generation
    # ============================================

    async def _handle_generate_publication_figure(
        self,
        code: str,
        figure_name: str,
        description: str,
        figure_type: str,
        state: AnalysisState,
        review_callback: Optional[Callable[[PublicationFigure], Awaitable[Dict[str, Any]]]] = None
    ) -> str:
        """
        Handle generate_publication_figure tool call.
        Generates the figure, saves in dual format, and optionally gets PI review.

        Args:
            code: Python plotting code
            figure_name: Name for the figure file
            description: Figure description
            figure_type: Type of figure
            state: Analysis state to store the figure
            review_callback: Optional callback for PI review

        Returns:
            Tool result message
        """
        logger.info(f"[Bioinformatics] Generating publication figure: {figure_name}")

        # Validate code before execution
        is_valid, error_msg = self._validate_code(code)
        if not is_valid:
            return f"Error: Code validation failed - {error_msg}"

        # Ensure figures directory exists
        figures_dir = Path(self.sandbox.working_dir) / "figures"

        # Clean up figure name
        safe_name = "".join(c if c.isalnum() or c == '_' else '_' for c in figure_name)
        jpg_path = figures_dir / f"{safe_name}.jpg"
        pdf_path = figures_dir / f"{safe_name}.pdf"

        # Wrap code to save both JPG and PDF (use repr() for paths to prevent injection)
        figures_dir_repr = repr(str(figures_dir))
        png_path_repr = repr(str(jpg_path))  # Variable name kept for template compatibility
        pdf_path_repr = repr(str(pdf_path))
        indented_code = textwrap.indent(code, '    ')
        wrapped_code = f'''
import matplotlib.pyplot as plt
from pathlib import Path

# Close any stale figures from previous executions to prevent poisoning
{_TPL_SAFE_CLOSE_ALL}

# Set publication-quality parameters
plt.rcParams.update({{
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'axes.grid': False,
    'axes.linewidth': 1.2,
    'pdf.fonttype': 42,  # Ensure fonts are vectorized in PDF
    'ps.fonttype': 42,
}})

# Create figures directory
_figures_dir = {figures_dir_repr}
Path(_figures_dir).mkdir(exist_ok=True)

# Temporarily disable plt.close/plt.show to keep figures alive
_orig_close = plt.close
_orig_show = plt.show
plt.close = lambda *a, **kw: None
plt.show = lambda *a, **kw: None

# Execute the plotting code (with try-finally to guarantee restore)
try:
{indented_code}
finally:
    plt.close = _orig_close
    plt.show = _orig_show

# Get current figure and save
try:
    fig = plt.gcf()
    _jpg_path = {png_path_repr}
    _pdf_path = {pdf_path_repr}

    fig.savefig(_jpg_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none',
                format='jpeg', pil_kwargs={{'quality': 85, 'optimize': True}})
    print(f"Saved JPG: {{_jpg_path}}")
    fig.savefig(_pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved PDF: {{_pdf_path}}")
except Exception as _save_err:
    print(f"[Auto-save] Warning: could not save figure: {{_save_err}}")

{_TPL_SAFE_CLOSE_ALL}
'''

        # Execute the code
        result = await self.sandbox.execute(wrapped_code)

        if result.error:
            result.error = strip_ansi(result.error)
            error_msg = f"Figure generation failed: {result.error}"
            logger.error(f"[Bioinformatics] {error_msg}")
            await self._notify("error", error_msg)
            return error_msg

        # Check if files were created
        if not jpg_path.exists():
            error_msg = f"JPG file was not created: {jpg_path}"
            logger.error(f"[Bioinformatics] {error_msg}")
            return error_msg

        # Detect blank figures (empty white JPG with bbox_inches='tight' is typically <2KB)
        if jpg_path.stat().st_size < 2000:
            error_msg = f"Figure appears to be blank (file size: {jpg_path.stat().st_size} bytes). The plotting code may not have generated visible content."
            logger.warning(f"[Bioinformatics] {error_msg}")
            return error_msg

        # Read JPG and compress to base64
        png_b64 = self._compress_image_to_base64(jpg_path, max_width=800)

        # Create PublicationFigure object (png_path field stores JPG path for web display)
        figure = PublicationFigure(
            id=str(uuid.uuid4()),
            name=safe_name,
            description=description,
            figure_type=figure_type,
            png_path=str(jpg_path),
            pdf_path=str(pdf_path),
            png_base64=png_b64,
            review_status=FigureReviewStatus.PENDING
        )

        # Send figure to frontend for display
        await self._notify("figure", png_b64)

        # If review callback provided, get PI review
        # Allow up to 3 revisions (4 reviews total: initial + 3 revisions)
        if review_callback:
            revision_count = 0
            current_code = code
            revision_history = []  # Accumulate all rounds of PI feedback

            while revision_count <= 3:
                logger.info(f"[Bioinformatics] Submitting figure for PI review (attempt {revision_count + 1})")
                await self._notify("status", f"PI reviewing figure: {figure_name}...")

                review = await review_callback(figure)

                if review.get('approved', False):
                    quality_score = review.get('quality_score', 5)
                    # Mark figures approved under leniency (after multiple revisions with non-perfect score)
                    if revision_count >= 2 and quality_score < 8:
                        figure.review_status = FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS
                        logger.info(f"[Bioinformatics] Figure {figure_name} accepted with reservations (score={quality_score}, revisions={revision_count})")
                    else:
                        figure.review_status = FigureReviewStatus.APPROVED
                        logger.info(f"[Bioinformatics] Figure {figure_name} approved by PI")
                    figure.revision_history = revision_history
                    self._upsert_publication_figure(state, figure)
                    await self._notify("status", f"Figure {figure_name} approved by PI")
                    return f"Figure {figure_name} has been approved by PI and saved.\nQuality score: {quality_score}/10\nFeedback: {review.get('feedback', 'Approved')}"

                # PI requested revision
                feedback = review.get('feedback', '')
                suggestions = review.get('suggestions', [])
                must_fix = review.get('must_fix', [])

                # If PI flagged a data/analysis issue that can't be fixed by revising plot code,
                # break out of the revision loop and defer to the main analysis flow
                if review.get('requires_reanalysis', False):
                    logger.warning(f"[Bioinformatics] Figure {figure_name} requires upstream reanalysis, skipping revision loop")
                    await self._notify("status", f"Figure {figure_name}: PI identified data-level issue, deferring to analysis")
                    figure.review_status = FigureReviewStatus.REVISION_REQUESTED
                    figure.pi_feedback = f"[REQUIRES REANALYSIS] {feedback}"
                    self._upsert_publication_figure(state, figure)
                    return (
                        f"Figure {figure_name} has a data-level issue that cannot be fixed by revising the plot code. "
                        f"PI feedback: {feedback}\n"
                        f"Please address this in the analysis pipeline and regenerate the figure later."
                    )

                # Record this round's feedback in cumulative history
                revision_history.append({
                    "round": revision_count + 1,
                    "feedback": feedback,
                    "suggestions": suggestions,
                    "must_fix": must_fix,
                    "quality_score": review.get('quality_score', None)
                })

                figure.pi_feedback = feedback
                figure.revision_history = revision_history

                if revision_count >= 3:
                    # Out of revision budget
                    break

                # Detect stalled revision loop: if the quality score hasn't
                # improved across the last 2 rounds, further revisions are
                # unlikely to help — accept the current version to avoid
                # burning time and tokens.
                if len(revision_history) >= 2:
                    prev_score = revision_history[-2].get("quality_score")
                    curr_score = revision_history[-1].get("quality_score")
                    if prev_score is not None and curr_score is not None and curr_score <= prev_score:
                        logger.warning(
                            f"[Bioinformatics] Figure {figure_name}: quality score stalled "
                            f"({prev_score} -> {curr_score}), accepting current version"
                        )
                        figure.review_status = FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS
                        self._upsert_publication_figure(state, figure)
                        await self._notify("status", f"Figure {figure_name} accepted (revision stalled)")
                        return (
                            f"Figure {figure_name} accepted with reservations (quality score did not improve "
                            f"after revision: {prev_score} -> {curr_score}).\n"
                            f"Last feedback: {feedback}\n"
                            f"Please move on to the next analysis step."
                        )

                # Try to revise the figure
                await self._notify("status", f"Revising figure based on PI feedback...")
                logger.info(f"[Bioinformatics] Revising figure {figure_name}: {feedback}")

                # Generate revised code with full revision history
                try:
                    revised_code = await self._revise_figure_code(
                        current_code, feedback, suggestions, must_fix, figure_type, state, figure.png_base64,
                        revision_history=revision_history
                    )
                except Exception as revise_err:
                    logger.error(f"[Bioinformatics] _revise_figure_code failed: {revise_err}, keeping current version")
                    await self._notify("status", f"Figure revision failed ({revise_err}), keeping current version")
                    break

                # Only increment revision count after successful code generation
                revision_count += 1
                figure.revision_count = revision_count

                # Validate revised code before execution
                is_valid, error_msg = self._validate_code(revised_code)
                if not is_valid:
                    logger.warning(f"[Bioinformatics] Revised code failed validation: {error_msg}")
                    break

                # Re-execute with revised code (rebuild wrapper instead of str.replace to avoid corruption)
                indented_revised_code = textwrap.indent(revised_code, '    ')
                revised_wrapped = f'''
import matplotlib.pyplot as plt
from pathlib import Path

# Close any stale figures from previous executions to prevent poisoning
{_TPL_SAFE_CLOSE_ALL}

plt.rcParams.update({{
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'axes.grid': False,
    'axes.linewidth': 1.2,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}})

_figures_dir = {figures_dir_repr}
Path(_figures_dir).mkdir(exist_ok=True)

# Temporarily disable plt.close/plt.show to keep figures alive
_orig_close = plt.close
_orig_show = plt.show
plt.close = lambda *a, **kw: None
plt.show = lambda *a, **kw: None

# Execute revised code (with try-finally to guarantee restore)
try:
{indented_revised_code}
finally:
    plt.close = _orig_close
    plt.show = _orig_show

try:
    fig = plt.gcf()
    _jpg_path = {png_path_repr}
    _pdf_path = {pdf_path_repr}

    fig.savefig(_jpg_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none',
                format='jpeg', pil_kwargs={{'quality': 85, 'optimize': True}})
    fig.savefig(_pdf_path, dpi=300, bbox_inches='tight', format='pdf')
except Exception as _save_err:
    print(f"[Auto-save] Warning: could not save figure: {{_save_err}}")

{_TPL_SAFE_CLOSE_ALL}
'''
                result = await self.sandbox.execute(revised_wrapped)

                if not result.error and jpg_path.exists():
                    # Detect blank revised figure
                    if jpg_path.stat().st_size < 2000:
                        logger.warning(f"[Bioinformatics] Revised figure appears blank (size: {jpg_path.stat().st_size} bytes)")
                        break
                    # Update figure with new image
                    figure.png_base64 = self._compress_image_to_base64(jpg_path, max_width=800)
                    await self._notify("figure", figure.png_base64)
                    current_code = revised_code
                else:
                    result.error = strip_ansi(result.error) if result.error else result.error
                    logger.warning(f"[Bioinformatics] Revision failed: {result.error}")
                    # Auto-retry: feed error back to LLM for a single fix attempt
                    error_msg = str(result.error)[:1000]
                    logger.info(f"[Bioinformatics] Auto-retrying revision with error context")
                    try:
                        retry_code = await self._revise_figure_code(
                            revised_code,
                            f"The previous code revision caused a runtime error:\n{error_msg}\nFix this error while keeping ALL the style improvements from the revision.",
                            [f"Fix the runtime error: {error_msg}"],
                            ["The code must execute without errors"],
                            figure_type, state, figure.png_base64,
                            revision_history=revision_history
                        )
                        is_valid_retry, err_retry = self._validate_code(retry_code)
                        if is_valid_retry:
                            indented_retry_code = textwrap.indent(retry_code, '    ')
                            retry_wrapped = f'''
import matplotlib.pyplot as plt
from pathlib import Path

# Close any stale figures from previous executions to prevent poisoning
{_TPL_SAFE_CLOSE_ALL}

plt.rcParams.update({{
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'axes.grid': False,
    'axes.linewidth': 1.2,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}})

_figures_dir = {figures_dir_repr}
Path(_figures_dir).mkdir(exist_ok=True)

_orig_close = plt.close
_orig_show = plt.show
plt.close = lambda *a, **kw: None
plt.show = lambda *a, **kw: None

try:
{indented_retry_code}
finally:
    plt.close = _orig_close
    plt.show = _orig_show

try:
    fig = plt.gcf()

    _jpg_path = {png_path_repr}
    _pdf_path = {pdf_path_repr}

    fig.savefig(_jpg_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none',
                format='jpeg', pil_kwargs={{'quality': 85, 'optimize': True}})
    fig.savefig(_pdf_path, dpi=300, bbox_inches='tight', format='pdf')
except Exception as _save_err:
    print(f"[Auto-save] Warning: could not save figure: {{_save_err}}")

{_TPL_SAFE_CLOSE_ALL}
'''
                            retry_result = await self.sandbox.execute(retry_wrapped)
                            if not retry_result.error and jpg_path.exists() and jpg_path.stat().st_size >= 2000:
                                logger.info(f"[Bioinformatics] Auto-retry succeeded for {figure_name}")
                                figure.png_base64 = self._compress_image_to_base64(jpg_path, max_width=800)
                                await self._notify("figure", figure.png_base64)
                                current_code = retry_code
                                continue  # Back to review loop
                            else:
                                logger.warning(f"[Bioinformatics] Auto-retry also failed: {retry_result.error}")
                    except Exception as retry_err:
                        logger.warning(f"[Bioinformatics] Auto-retry exception: {retry_err}")
                    break  # Both attempts failed

            # After 3 attempts, save with revision_requested status
            figure.review_status = FigureReviewStatus.REVISION_REQUESTED
            self._upsert_publication_figure(state, figure)
            logger.warning(f"[Bioinformatics] Figure {figure_name} not fully approved after {revision_count} revisions")
            return f"Figure {figure_name} saved after {revision_count} revision attempts. Status: needs further revision.\nLast feedback: {figure.pi_feedback}"

        # No review callback, just save the figure
        self._upsert_publication_figure(state, figure)
        return f"Figure {figure_name} has been generated and saved.\nJPG: {jpg_path}\nPDF: {pdf_path}"

    def _upsert_publication_figure(self, state, figure):
        """Replace existing figure with same name, or append if new."""
        existing_idx = next(
            (i for i, f in enumerate(state.publication_figures) if f.name == figure.name),
            None
        )
        if existing_idx is not None:
            state.publication_figures[existing_idx] = figure
        else:
            state.publication_figures.append(figure)

    def _compress_image_to_base64(self, image_path: Path, max_width: int = 800) -> str:
        """
        Compress an image and convert to base64 for efficient transmission.

        Args:
            image_path: Path to the image file
            max_width: Maximum width for the compressed image

        Returns:
            Base64 encoded JPEG string
        """
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                # Calculate resize ratio
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_size = (max_width, int(img.height * ratio))
                    # Handle PIL version compatibility
                    try:
                        resample_method = Image.Resampling.LANCZOS
                    except AttributeError:
                        resample_method = Image.LANCZOS  # For older PIL versions
                    img = img.resize(new_size, resample_method)

                # Convert to RGB if necessary (handle RGBA, P modes)
                if img.mode in ('RGBA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        # Safely get alpha channel - check split() returns enough channels
                        channels = img.split()
                        if len(channels) >= 4:
                            background.paste(img, mask=channels[3])
                        else:
                            background.paste(img)
                    else:
                        background.paste(img)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Save to buffer as JPEG
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85, optimize=True)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')

        except Exception as e:
            logger.exception("[Bioinformatics] Failed to compress image")
            # Fallback: read raw file and encode
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')

    async def _revise_figure_code(
        self,
        original_code: str,
        feedback: str,
        suggestions: List[str],
        must_fix: List[str],
        figure_type: str,
        state: AnalysisState = None,
        image_base64: str = None,
        revision_history: List[Dict[str, Any]] = None
    ) -> str:
        """
        Generate revised plotting code based on PI feedback.

        Args:
            original_code: The original plotting code
            feedback: Overall feedback from PI
            suggestions: List of improvement suggestions
            must_fix: List of critical issues that must be fixed
            figure_type: Type of figure
            state: Analysis state with treatment group info
            image_base64: Base64 encoded current figure image for visual review
            revision_history: Accumulated feedback from all revision rounds (text only)

        Returns:
            Revised Python plotting code
        """
        suggestions_text = "\n".join(f"- {s}" for s in suggestions) if suggestions else "None"
        must_fix_text = "\n".join(f"- {m}" for m in must_fix) if must_fix else "None"

        exp_design_constraint = ""
        if state and state.treatment_groups:
            group_names = ', '.join(g.name for g in state.treatment_groups)
            exp_design_constraint = f"""
**CRITICAL - Experimental Design Constraint:**
The actual experimental groups are: {group_names}
You must ONLY use data from these groups. Do NOT create or reference groups that don't exist.
If the PI feedback asks you to add groups not in the experimental design, IGNORE that part of the feedback and only plot data for the actual groups.
"""

        figure_context = "**Current Figure:** An image of the current figure is attached. Review it carefully to understand what needs to be changed.\n\n" if image_base64 else ""

        # Build revision history section (show previous rounds, exclude current which is shown below)
        history_section = ""
        if revision_history and len(revision_history) > 1:
            past_rounds = revision_history[:-1]
            history_parts = []
            for entry in past_rounds:
                round_num = entry.get('round', '?')
                hist_feedback = entry.get('feedback', '')
                hist_suggestions = entry.get('suggestions', [])
                hist_must_fix = entry.get('must_fix', [])
                round_text = f"### Round {round_num}\n"
                round_text += f"- Feedback: {hist_feedback}\n"
                if hist_suggestions:
                    round_text += f"- Suggestions: {'; '.join(str(s) for s in hist_suggestions)}\n"
                if hist_must_fix:
                    round_text += f"- Must Fix: {'; '.join(str(m) for m in hist_must_fix)}\n"
                history_parts.append(round_text)
            history_section = (
                "**Previous Revision History (CRITICAL - read carefully):**\n"
                "The following issues were identified and some were FIXED in prior rounds.\n"
                "You MUST:\n"
                "1. Keep ALL previously applied fixes (do NOT revert them)\n"
                "2. Address the NEW issues from the current round\n"
                "3. Before submitting, mentally verify each previous fix is still present in your code\n\n"
                + "\n".join(history_parts) + "\n"
            )

        prompt = f"""Please revise the following plotting code based on PI feedback.

{figure_context}**Original Code:**
```python
{original_code}
```

**Figure Type:** {figure_type}
{exp_design_constraint}
{history_section}**PI Feedback (Current Round):** {feedback}

**Improvement Suggestions (Current Round):**
{suggestions_text}

**Critical Issues (MUST FIX - Current Round):**
{must_fix_text}

**Requirements:**
1. Fix all critical issues listed above
2. Address as many suggestions as possible
3. CRITICAL: Do NOT revert previously applied fixes. If Round 1 changed grid lines to dashed, keep them dashed. If Round 2 added axis labels, keep them.
4. Maintain the same overall figure structure
5. Keep using the same data variables (assume they exist), but REMOVE any fabricated/dummy data for groups not in the experimental design
6. Before calling set_xticklabels()/set_yticklabels(), ALWAYS call set_xticks()/set_yticks() first with matching count
7. For pandas Categorical columns, use .cat.add_categories() before assigning new category values
8. Use legend_.get_legend_handles_labels() instead of legend_.legendHandles (deprecated in newer matplotlib)
9. Return ONLY the revised Python code, no explanations

**Common fixes for publication figures:**
- Add/improve axis labels and titles
- Adjust font sizes (plt.rcParams['font.size'] = 12)
- Add legends with proper placement
- Use colorblind-friendly palettes
- Add scale bars or annotations
- Improve color contrast
- Remove unnecessary elements

Return only the Python code:"""

        self._set_llm_context()
        # Use slightly lower temperature for code revision (needs precision)
        revision_temp = min(self.config.temperature, 0.2)
        if image_base64 and self.has_vision:
            # Use vision API so the agent can see the current figure
            try:
                response = await self.llm_client.chat_with_vision(
                    model=self.model,
                    text_prompt=prompt,
                    images=[image_base64],
                    temperature=revision_temp,
                    max_tokens=32768
                )
            except Exception as vision_err:
                _is_vision_unsupported = (
                    "NotFoundError" in type(vision_err).__name__
                    or "404" in str(vision_err)
                    or "1210" in str(vision_err)  # ZhipuAI: API params error (non-vision model)
                )
                if _is_vision_unsupported:
                    # Model does not support vision input — disable for this session and fall back
                    logger.warning(
                        f"[Bioinformatics] Model {self.model} does not support vision "
                        f"({vision_err}), disabling vision for this session"
                    )
                    self.has_vision = False
                    prompt = prompt.replace(
                        "**Current Figure:** An image of the current figure is attached. "
                        "Review it carefully to understand what needs to be changed.\n\n",
                        "**Current Figure:** (Image not available — model does not support vision. "
                        "Revise code based on PI feedback text only.)\n\n",
                    )
                    response = await self.llm_client.chat(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=revision_temp,
                        max_tokens=32768
                    )
                else:
                    raise
        else:
            response = await self.llm_client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=revision_temp,
                max_tokens=32768
            )

        response_text = response.content if response and response.content else None
        if not response_text:
            logger.warning("[Bioinformatics] Empty response from LLM for code revision")
            return original_code  # Return unchanged code

        # Extract code from markdown response
        code = extract_code_from_markdown(response_text)

        return code.strip()
