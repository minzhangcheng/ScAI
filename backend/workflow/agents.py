"""Agent node methods: pathology, bioinformatics, biochemistry, literature."""
import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..models.schemas import AgentType, AnalysisPhase, AnalysisStep, MessageRole, TaskStatus
from .helpers import _get_task_text, _normalize_tasks, cache_hit, cache_invalidate, cache_store, STAGE1_KEYWORDS
from .retry import with_retry

if TYPE_CHECKING:
    from .state import WorkflowState
else:
    WorkflowState = dict

logger = logging.getLogger(__name__)


def _reset_pi_instructions_for_stage2(pi_instructions: Dict[str, Any]) -> Dict[str, Any]:
    cleared = dict(pi_instructions)
    cleared["bioinformatics_tasks"] = []
    cleared["pathology_tasks"] = []
    cleared["biochemistry_tasks"] = []
    cleared["literature_topics"] = []
    cleared["overview"] = ""
    return cleared


def _stable_code_hash(code: str) -> str:
    return hashlib.sha1(code.encode("utf-8")).hexdigest()


class AgentNodesMixin:
    """Mixin providing agent-node execution methods. Composed into SingleCellAnalysisWorkflow."""

    async def _require_core_agent(
        self,
        state: 'WorkflowState',
        agent_name: str,
        node_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Block until the named core agent is configured.

        Returns None to indicate the caller may proceed. If the user chose to
        skip, returns a partial state dict the caller should `return` immediately
        (the corresponding result_key set to {'status': 'skipped'}). If the user
        chose to terminate, raises WorkflowTerminatedError.
        """
        from ..constants import CORE_AGENTS, LlmFailureAction, NODE_TO_RESULT_KEY
        from .core import WorkflowTerminatedError

        if agent_name not in CORE_AGENTS:
            return None

        provider = getattr(self, "_settings_provider", None)
        if provider is None:
            return None

        while True:
            model = provider.get_model(agent_name)
            if model:
                return None
            action = await self._wait_for_disabled_agent_intervention(
                agent_name, node_name, state["analysis_state"]
            )
            if action == LlmFailureAction.RETRY:
                continue
            if action == LlmFailureAction.TERMINATE:
                raise WorkflowTerminatedError(
                    f"User terminated workflow due to disabled {agent_name}"
                )
            # SKIP
            result_key = NODE_TO_RESULT_KEY.get(node_name, f"{node_name}_result")
            skip_nodes = list(state.get("skip_nodes", []))
            if node_name not in skip_nodes:
                skip_nodes.append(node_name)
            return {
                **state,
                "skip_nodes": skip_nodes,
                "failed_node": None,
                result_key: {"findings": [], "status": "skipped"},
            }

    @with_retry("pathology_analysis")
    async def _pathology_analysis(self, state: WorkflowState) -> WorkflowState:
        """Pathology agent analyzes images"""
        analysis_state = state["analysis_state"]

        cached = cache_hit(analysis_state, AgentType.PATHOLOGY)
        if cached is not None:
            logger.info(f"[{self.session_id}] pathology cache hit, skipping re-execution")
            await self._send_message("result", cached, AgentType.PATHOLOGY)
            return {
                **state,
                "analysis_state": analysis_state,
                "pathology_result": cached,
                "failed_node": None,
            }

        if not self.pathology_agent or not self._settings_provider.get_model("pathology"):
            logger.info(f"[{self.session_id}] Pathology agent disabled, skipping")
            skipped_result = {"findings": [], "status": "skipped"}
            cache_store(analysis_state, AgentType.PATHOLOGY, skipped_result)
            return {**state, "failed_node": None, "pathology_result": skipped_result}

        instructions = state["pi_instructions"]

        tasks = instructions.get("pathology_tasks", [])
        if not tasks:
            tasks = ["Perform standard pathology analysis"]

        pathology_files = analysis_state.get_files_by_section("pathology")
        logger.info(f"[{self.session_id}] Starting pathology analysis with {len(pathology_files)} images")
        await self._send_message("status", "Pathology analysis starting...", AgentType.PATHOLOGY)
        
        # Check if images are available
        if not pathology_files:
            no_data_result = {"findings": [], "status": "no_data"}
            await self._send_message(
                "result",
                {
                    "status": "no_data",
                    "message": "No pathology images uploaded. Please upload H&E or other stained sections for analysis.",
                    "findings": []
                },
                AgentType.PATHOLOGY
            )
            cache_store(analysis_state, AgentType.PATHOLOGY, no_data_result)
            return {
                **state,
                "analysis_state": analysis_state,
                "pathology_result": no_data_result
            }
        
        # Log image count
        await self._send_message(
            "status", 
            f"Analyzing {len(pathology_files)} pathology image(s)...", 
            AgentType.PATHOLOGY
        )
        
        # Dynamic PI planning: refine pathology tasks based on biochemistry results
        overview = instructions.get("overview", "")
        if overview:  # Guard: skip for adjusted_instructions (no overview field)
            completed = self._collect_current_round_results(state)
            if completed:
                await self._send_message("status", "PI is refining pathology tasks...", AgentType.PI)
                tasks = await self.pi_agent.plan_agent_tasks(
                    agent_name="pathology",
                    original_tasks=tasks,
                    completed_results=completed,
                    overall_strategy=overview,
                    research_objective=analysis_state.initial_instruction,
                    state=analysis_state,
                )

        # Enrich instruction with previous findings for round 2+
        task_instruction = "\n".join(_normalize_tasks(tasks))
        previous_context = self._build_previous_findings_context(analysis_state, "pathology")
        if previous_context:
            task_instruction += previous_context

        # Add biochemistry context for cross-reference
        biochemistry_result = state.get("biochemistry_result")
        if biochemistry_result and biochemistry_result.get("findings"):
            biochem_findings = biochemistry_result["findings"][:5]
            task_instruction += (
                "\n\nCurrent round biochemistry findings:\n"
                + "\n".join(f"- {f}" for f in biochem_findings)
            )

        # Let exceptions propagate to @with_retry decorator for proper retry handling
        result = await self.pathology_agent.process(
            analysis_state,
            task_instruction
        )

        await self._send_message(
            "result",
            result,
            AgentType.PATHOLOGY
        )

        # Add message to state
        analysis_state.add_message(
            AgentType.PATHOLOGY,
            MessageRole.ASSISTANT,
            json.dumps(result, ensure_ascii=False)
        )

        cache_store(analysis_state, AgentType.PATHOLOGY, result)
        return {
            **state,
            "analysis_state": analysis_state,
            "pathology_result": result
        }
    
    @with_retry("bioinformatics_analysis")
    async def _bioinformatics_analysis(self, state: WorkflowState) -> WorkflowState:
        """
        Bioinformatics analysis with PI-controlled iterative execution

        Flow:
        1. PI provides high-level analysis goals
        2. Bioinformatics agent runs autonomously for technical steps
        3. Agent consults PI only for biological decisions
        4. Agent reports findings when analysis phase is complete
        """
        self._current_graph_state = state
        analysis_state = state["analysis_state"]

        skip = await self._require_core_agent(state, "bioinformatics", "bioinformatics_analysis")
        if skip is not None:
            return skip

        cached = cache_hit(analysis_state, AgentType.BIOINFORMATICS)
        if cached is not None:
            logger.info(f"[{self.session_id}] bioinformatics cache hit, skipping re-execution")
            await self._send_message("result", cached, AgentType.BIOINFORMATICS)
            # If the cached result was from stage2, the normal path would have
            # transitioned planning_stage to "full". Carry that forward here so
            # downstream routing (_after_bioinformatics) sees the correct stage.
            result_state = {
                **state,
                "analysis_state": analysis_state,
                "bioinformatics_result": cached,
                "failed_node": None,
            }
            planning_stage = state.get("planning_stage", "full")
            if planning_stage == "stage2":
                result_state["planning_stage"] = "full"
                result_state["planning_context"] = None
            return result_state

        logger.info(f"[{self.session_id}] Starting bioinformatics analysis")
        seq_files = analysis_state.get_files_by_section("sequencing")
        logger.info(f"[{self.session_id}] Sequencing files: {len(seq_files)}")

        await self._send_message("status", "Starting bioinformatics analysis...", AgentType.BIOINFORMATICS)
        
        # Check if data is available
        if not analysis_state.has_sequencing_data():
            waiting_result = {"findings": [], "status": "waiting_data"}
            await self._send_message(
                "result",
                {"message": "Please upload sequencing data first", "findings": []},
                AgentType.BIOINFORMATICS
            )
            cache_store(analysis_state, AgentType.BIOINFORMATICS, waiting_result)
            return {
                **state,
                "analysis_state": analysis_state,
                "bioinformatics_result": waiting_result
            }
        
        # Incorporate human guidance from workflow state (set by _initialize_round)
        human_guidance = ""
        if state.get("human_guidance"):
            human_guidance = f"\n\nHuman researcher guidance: {state['human_guidance']}"
        
        # Planning stage support: filter tasks and adjust instructions based on stage
        planning_stage = state.get("planning_stage", "full")
        
        # Create snapshot before execution
        await self._create_auto_snapshot(analysis_state, f"pre_execution_{planning_stage}")
        
        # Step 1: PI provides analysis goals
        await self._send_message("status", "PI is defining analysis goals...", AgentType.PI)
        
        # Get PI instructions from workflow state
        pi_instructions = state.get("pi_instructions", {})
        bioinf_tasks = pi_instructions.get("bioinformatics_tasks", [])
        merge_strategy = pi_instructions.get("merge_strategy", "")
        cluster_comparison_strategy = pi_instructions.get("cluster_comparison_strategy", "")

        # Stage1 task filtering: only QC + clustering + annotation tasks
        if planning_stage == "stage1":
            stage1_tasks = [t for t in bioinf_tasks if any(
                kw in _get_task_text(t) for kw in STAGE1_KEYWORDS
            )]
            if not stage1_tasks:
                stage1_tasks = ["Execute scRNA-seq QC, clustering, and cell type annotation"]
            bioinf_tasks = stage1_tasks

        # Dynamic PI planning: refine bioinformatics tasks based on completed results
        overview = pi_instructions.get("overview", "")
        if overview:
            completed = self._collect_current_round_results(state)
            if completed:
                await self._send_message("status", "PI is refining bioinformatics tasks...", AgentType.PI)
                bioinf_tasks = await self.pi_agent.plan_agent_tasks(
                    agent_name="bioinformatics",
                    original_tasks=bioinf_tasks,
                    completed_results=completed,
                    overall_strategy=overview,
                    research_objective=analysis_state.initial_instruction,
                    state=analysis_state,
                )

        # Build previous findings and code history context for round 2+
        previous_bio_findings = self._build_previous_findings_context(analysis_state, "bioinformatics")
        code_history_context = ""
        if analysis_state.current_round > 1:
            code_history_context = self._build_code_history_summary(analysis_state)

        # Include current round pathology findings if available
        pathology_context = ""
        pathology_result = state.get("pathology_result")
        if pathology_result and pathology_result.get("findings"):
            path_findings = pathology_result["findings"][:5]
            pathology_context = (
                "\n\nCurrent round pathology findings:\n"
                + "\n".join(f"- {f}" for f in path_findings)
            )

        # Include current round biochemistry findings if available
        biochemistry_context = ""
        biochemistry_result = state.get("biochemistry_result")
        if biochemistry_result and biochemistry_result.get("findings"):
            biochem_findings = biochemistry_result["findings"][:5]
            biochemistry_context = (
                "\n\nCurrent round biochemistry findings:\n"
                + "\n".join(f"- {f}" for f in biochem_findings)
            )

        # Build comprehensive instruction for bioinformatics agent
        # Phase-aware instruction for bioinformatics agent
        # Detect data types for dynamic instruction (single pass)
        has_scrna = analysis_state.has_scrnaseq()
        has_bulk = analysis_state.has_bulk_rnaseq()

        phase_instruction = ""
        if planning_stage == "stage1":
            # Stage 1: QC + Clustering + Annotation ONLY
            phase_instruction = """
## STAGE 1 - EXPLORATION PHASE (QC + Clustering + Annotation ONLY)

You are in STAGE 1 of the exploration phase. Execute ONLY the following analyses:

MANDATORY STEPS:
1. Load the input data (h5ad file)
2. Basic QC: filter cells by min_genes, filter genes by min_cells, compute QC metrics
3. Doublet detection assessment (scrublet or equivalent)
4. Batch effect assessment: check if multiple samples/batches exist, evaluate need for correction
5. Normalization (scanpy.pp.normalize_total), log1p
6. Highly variable gene selection (scanpy.pp.highly_variable_genes)
7. PCA (scanpy.pp.pca)
8. UMAP (scanpy.pp.neighbors + scanpy.tl.umap)
9. Clustering (scanpy.tl.leiden with appropriate resolution)
10. Cell type annotation: find marker genes per cluster (scanpy.tl.rank_genes_groups), then call annotate_clusters tool

WHAT NOT TO DO (these will be done in Stage 2):
- Do NOT run condition-level differential expression (DEG)
- Do NOT run GO/KEGG pathway enrichment
- Do NOT run cell-cell communication analysis (liana)
- Do NOT generate publication figures (save that for Stage 2)
- Do NOT run trajectory/velocity analysis

IMPORTANT: After completing steps 1-10, call report_findings with a summary of:
- Number of cells/genes after QC
- Batch effects detected (if any) and correction applied
- Number of clusters found
- Key marker genes per cluster
- Annotation results (cell types identified)

These Stage 1 findings will be used by PI to plan Stage 2 downstream analysis.
"""
        elif analysis_state.analysis_phase == AnalysisPhase.EXPLORATION:
            phase_instruction = """
## EXPLORATION PHASE INSTRUCTIONS
This is Round 1 — comprehensive, unbiased exploration. You MUST:
1. Complete the FULL standard pipeline (QC → normalization → clustering → annotation)
2. After annotation: merge biologically similar clusters (e.g., stress hepatocytes + hepatocytes,
   macrophage subtypes) into unified cell type groups before DEG analysis
3. Perform condition-level DEG (treatment vs control) for ALL merged cell type groups
4. Run GO/KEGG pathway enrichment on ALL DEG results
5. Keep analyses simple and high-throughput (DEG + GO/KEGG)
6. Do NOT skip any cell type — but DO merge related subtypes/states of the same cell type
7. Complete cell communication analysis (liana)
8. For EACH cell type group's DEG results, generate a visualization figure (volcano plot or bar chart)
9. For EACH cell type group's KEGG/GO enrichment, generate a bar plot of top enriched pathways
"""
            if has_bulk:
                phase_instruction += """
10. **MANDATORY — Bulk RNA-seq DEG Analysis** (DO NOT SKIP):
    - Load the bulk RNA-seq count matrix and perform DEG analysis using pydeseq2
      (CRITICAL: If pydeseq2 raises ImportError, fallback to scipy.stats.ttest_ind with statsmodels multipletests for FDR correction)
    - Generate a volcano plot for bulk DEG results
    NOTE: report_findings will be REJECTED if bulk DEG analysis is not performed.

11. **MANDATORY — Bulk/scRNA-seq Cross-Validation** (DO NOT SKIP):
    - Aggregate scRNA-seq data to pseudobulk level OR use condition-level DEGs across all cells
    - Find overlapping DEGs between bulk and scRNA-seq results using set intersection
    - Visualize the overlap with a Venn diagram (from matplotlib_venn import venn2)
    - For overlapping DEGs, visualize their cell-type-specific expression in scRNA-seq (e.g., dotplot or heatmap)
    NOTE: This is a SEPARATE mandatory gate. report_findings will be REJECTED if cross-validation is not performed.

12. **MANDATORY — Cell-Type Deconvolution** (DO NOT SKIP):
    - Use the annotated scRNA-seq data as a reference to perform cell-type deconvolution on the bulk RNA-seq matrix.
    - Prioritize pure Python methods like scipy.optimize.nnls or sklearn.linear_model.LinearRegression(positive=True).
    - If needed and available, use advanced tools like scaden (prioritize scipy.optimize.nnls).
    - Generate a stacked bar plot or box plot showing the inferred cell-type proportions across bulk samples/conditions.
    NOTE: This is a SEPARATE mandatory gate. report_findings will be REJECTED if deconvolution is not performed.
"""
        else:
            direction_ctx = analysis_state.current_round_direction or "PI-directed"
            phase_instruction = f"""
## INNOVATION PHASE INSTRUCTIONS
This is a targeted, hypothesis-driven round.
Research direction: {direction_ctx}

ANALYSIS REQUIREMENTS:
1. Execute ALL tasks assigned by PI below — do not skip any
2. For each task, generate at least one publication-quality figure
3. Use advanced analysis methods as specified (GSEA, gene scoring, trajectory, TF activity, etc.)
4. After completing PI's tasks, if you identify additional relevant analyses, execute them
5. When performing DEG, always include volcano plots and heatmaps of top genes
6. When performing pathway analysis, include bar plots of top enriched terms
7. For any scoring analysis (gene signatures, TF activity), show UMAP feature plots AND violin/box plots by condition
8. Report detailed statistical results (p-values, effect sizes, gene counts) — not just summaries
9. Before computing any score, consult PI about scoring method and gene list source
10. After computing any score, call register_score to document methodology for the report
"""
            if has_bulk:
                phase_instruction += """
11. **MANDATORY — Bulk RNA-seq Direction Validation** (if bulk data exists; DO NOT SKIP):
    Bulk RNA-seq data remains available and loaded from Round 1. When this round
    produces new gene signatures, DEG sets, or pathway conclusions, use bulk data
    as an independent validation layer. Choose validation methods appropriate to
    your specific findings and the bulk sample design. Reuse bulk results from
    Round 1 (see the data context section for the exact variable names) rather than
    re-running the pipeline.
    NOTE: report_findings will be REJECTED if bulk direction validation is not performed.
"""

            phase_instruction += """
You may skip irrelevant cell populations and focus on those relevant to the hypothesis.
Do NOT re-run completed steps from Round 1 (QC, clustering, cell annotation, etc.).
"""
            if has_bulk:
                phase_instruction += """BUT: "completed steps" does NOT include bulk validation — each direction must independently
validate its findings on bulk data, even if Round 1 already ran a bulk DEG pipeline.
"""

        # Build merge strategy context from PI instructions
        merge_context = ""
        if merge_strategy:
            merge_context += f"\n\nPI's cluster merge strategy:\n{merge_strategy}"
        if cluster_comparison_strategy:
            merge_context += f"\n\nPI's cluster comparison strategy:\n{cluster_comparison_strategy}"

        if has_scrna and has_bulk:
            data_desc = "single-cell RNA-seq and bulk RNA-seq"
        elif has_bulk:
            data_desc = "bulk RNA-seq"
        else:
            data_desc = "single-cell RNA-seq"

        if bioinf_tasks:
            goals_text = "\n".join(f"- {task}" for task in _normalize_tasks(bioinf_tasks))
        elif has_bulk and not has_scrna:
            goals_text = "- Perform standard bulk RNA-seq analysis (pydeseq2, pathway enrichment)"
        elif has_bulk and has_scrna:
            goals_text = ("- Perform standard scRNA-seq analysis pipeline\n"
                          "- Perform bulk RNA-seq DEG analysis (pydeseq2) and cross-validate with scRNA-seq pseudobulk DEGs")
        else:
            goals_text = "- Perform standard scRNA-seq analysis pipeline"

        # Ensure bulk direction validation appears in goals even if PI didn't assign a bulk-specific task
        if analysis_state.analysis_phase == AnalysisPhase.INNOVATION and has_bulk and has_scrna:
            goals_lower = goals_text.lower()
            bulk_validation_mentioned = any(
                kw in goals_lower for kw in ["bulk", "ssgsea", "deconvolution", "cross-validat", "pseudobulk"]
            )
            if not bulk_validation_mentioned:
                goals_text += (
                    "\n- Validate direction-level findings on bulk RNA-seq data "
                    "(reuse the bulk variables already loaded in Round 1; see the data context section "
                    "for their exact names)"
                )

        instruction = f"""Analyze {data_desc} data for project: {analysis_state.name}

Research context: {analysis_state.initial_instruction}
{phase_instruction}

PI's analysis goals for this round:
{goals_text}
{merge_context}
{human_guidance}
{pathology_context}
{biochemistry_context}
{previous_bio_findings}
{code_history_context}

This is Round {analysis_state.current_round}. The Jupyter kernel retains all variables from previous rounds.
Focus on PI's specific goals above — do NOT re-run completed steps (QC, clustering, cell annotation, etc.).
Call report_findings ONLY after completing ALL assigned tasks above."""

        await self._send_message("prompt", instruction, AgentType.PI)
        
        # Step 2: Bioinformatics agent runs autonomously with figure review support
        await self._send_message("status", "Bioinformatics agent is running autonomous analysis...", AgentType.BIOINFORMATICS)

        # Reset supervisor state for this round
        if self.supervisor_agent:
            self.supervisor_agent.reset_for_round(planning_stage=planning_stage)

        # Set current state references for _pi_callback plan deviation detection
        self._current_analysis_state = analysis_state
        self._current_graph_state = state

        max_steps = self.config.bioinfo_max_steps
        result = await self.bioinformatics_agent.run_autonomous_analysis(
            analysis_state,
            instruction,
            max_steps=max_steps,
            figure_review_callback=self._figure_review_callback,
            supervisor_callback=self._supervisor_callback,
            planning_stage=planning_stage,  # Pass stage context to agent
        )

        # Collect supervisor stats
        if self.supervisor_agent:
            supervision_stats = self.supervisor_agent.get_summary_stats()
            await self._send_message("supervisor_summary", json.dumps(supervision_stats), AgentType.SUPERVISOR)
            logger.info(
                f"[{self.session_id}] Supervisor: {supervision_stats['total_steps_monitored']} steps, "
                f"{supervision_stats['steps_with_technical_issues']} issues, "
                f"checkpoints: {supervision_stats['checkpoints_reached']}"
            )

        # Check result status and log accordingly
        result_status = result.get("status", "unknown")
        if result_status == "error":
            logger.error(f"[{self.session_id}] Bioinformatics analysis failed: {result.get('error', 'Unknown error')}")
            await self._send_message("error", f"Bioinformatics analysis encountered an error: {result.get('error', 'Unknown error')}", AgentType.BIOINFORMATICS)
        elif result_status in ("timeout", "max_steps_reached"):
            logger.warning(f"[{self.session_id}] Bioinformatics analysis reached step limit after {result.get('steps_executed', 0)} steps")
            await self._send_message("status", "Bioinformatics analysis reached step limit, proceeding with partial results", AgentType.BIOINFORMATICS)

        # Collect results
        all_findings = result.get("findings", [])
        all_figures = result.get("figures", [])

        existing_code_ids = {c.id for c in analysis_state.code_history}
        existing_code_hashes = {_stable_code_hash(c.code) for c in analysis_state.code_history}
        for code_exec in result.get("code_history", []):
            code_hash = _stable_code_hash(code_exec.code)
            if code_exec.id not in existing_code_ids and code_hash not in existing_code_hashes:
                analysis_state.code_history.append(code_exec)
                existing_code_ids.add(code_exec.id)
                existing_code_hashes.add(code_hash)

        # Store analysis log (dedup by step id to handle node retries)
        existing_step_ids = {s.id for s in analysis_state.analysis_log}
        for step in result.get("analysis_log", []):
            if isinstance(step, dict):
                step = AnalysisStep(**step)
            if step.id not in existing_step_ids:
                step.round_number = max(analysis_state.current_round, 1)
                analysis_state.analysis_log.append(step)
                existing_step_ids.add(step.id)
        # Cap log size to prevent unbounded memory growth
        _MAX_ANALYSIS_LOG = 2000
        if len(analysis_state.analysis_log) > _MAX_ANALYSIS_LOG:
            analysis_state.analysis_log = analysis_state.analysis_log[-_MAX_ANALYSIS_LOG:]

        # Dedup publication_figures by name to handle node retries
        seen_fig_names = {}
        for i, fig in enumerate(analysis_state.publication_figures):
            seen_fig_names[fig.name] = i  # later index wins (keeps latest)
        if len(seen_fig_names) < len(analysis_state.publication_figures):
            analysis_state.publication_figures = [
                analysis_state.publication_figures[i]
                for i in sorted(seen_fig_names.values())
            ]

        # Step 3: PI summarizes results
        await self._send_message("status", "PI is reviewing analysis results...", AgentType.PI)

        summary = await self.pi_agent.summarize_bioinformatics_analysis(
            analysis_state,
            [{"findings": all_findings, "figures": all_figures}]
        )

        final_result = {
            "findings": all_findings,
            "figures": all_figures,
            "summary": summary,
            "steps_executed": result.get("steps_executed", 0),
            "status": result.get("status", "completed")
        }

        logger.info(f"[{self.session_id}] Bioinformatics analysis completed: {len(all_findings)} findings, {len(all_figures)} figures, {result.get('steps_executed', 0)} steps")

        await self._send_message(
            "result",
            {
                "findings": all_findings,
                "summary": summary,
                "steps_executed": result.get("steps_executed", 0)
            },
            AgentType.BIOINFORMATICS
        )

        analysis_state.add_message(
            AgentType.BIOINFORMATICS,
            MessageRole.ASSISTANT,
            json.dumps({"findings": all_findings, "summary": summary}, ensure_ascii=False)
        )

        # Stage transition: if stage1 just completed, set up for stage2
        result_state = {
            **state,
            "analysis_state": analysis_state,
            "bioinformatics_result": final_result
        }

        if planning_stage == "stage1":
            # Don't cache the stage1 result — stage2 re-enters this node and
            # must run fresh. Biochem/pathology/literature cache survives so
            # stage2 reuses them.
            result_state["planning_stage"] = "stage2"
            result_state["plan_approved"] = False
            result_state["pi_instructions"] = _reset_pi_instructions_for_stage2(
                state.get("pi_instructions", {})
            )
            result_state["planning_context"] = {
                "stage1_summary": final_result.get("summary", ""),
                "stage1_findings": final_result.get("findings", []),
                "stage1_figures": final_result.get("figures", []),
                "stage1_steps_executed": final_result.get("steps_executed", 0),
            }
            analysis_state.current_plan = None
            cache_invalidate(analysis_state, AgentType.BIOINFORMATICS)
        else:
            cache_store(analysis_state, AgentType.BIOINFORMATICS, final_result)
            if planning_stage == "stage2":
                result_state["planning_stage"] = "full"
                result_state["planning_context"] = None  # Clear stage1 context after stage2 completes

        # Check for mid-execution plan revision from agent result
        if isinstance(final_result, dict) and final_result.get("plan_revision_needed"):
            result_state["plan_revision_needed"] = True
            result_state["plan_revision_context"] = final_result.get("plan_revision_context", {})
            logger.info(f"[{self.session_id}] Mid-execution plan revision triggered")

        return result_state
    
    @with_retry("biochemistry_analysis")
    async def _biochemistry_analysis(self, state: WorkflowState) -> WorkflowState:
        """Biochemistry agent analyzes experimental results"""
        analysis_state = state["analysis_state"]

        cached = cache_hit(analysis_state, AgentType.BIOCHEMISTRY)
        if cached is not None:
            logger.info(f"[{self.session_id}] biochemistry cache hit, skipping re-execution")
            await self._send_message("result", cached, AgentType.BIOCHEMISTRY)
            return {
                **state,
                "analysis_state": analysis_state,
                "biochemistry_result": cached,
                "failed_node": None,
            }

        if not self.biochemistry_agent or not self._settings_provider.get_model("biochemistry"):
            logger.info(f"[{self.session_id}] Biochemistry agent disabled, skipping")
            skipped_result = {"findings": [], "status": "skipped"}
            cache_store(analysis_state, AgentType.BIOCHEMISTRY, skipped_result)
            return {**state, "failed_node": None, "biochemistry_result": skipped_result}

        instructions = state["pi_instructions"]

        tasks = instructions.get("biochemistry_tasks", [])

        bio_files = analysis_state.get_files_by_section("biochemistry")
        logger.info(f"[{self.session_id}] Starting biochemistry analysis with {len(analysis_state.biochemistry_indicators)} indicators and {len(bio_files)} files")
        logger.info(f"[{self.session_id}] Biochemistry tasks: {tasks}")

        await self._send_message("status", "Biochemistry analysis in progress...", AgentType.BIOCHEMISTRY)
        
        # Enrich instruction with previous findings for round 2+
        task_instruction = "\n".join(_normalize_tasks(tasks)) if tasks else "Analyze existing biochemistry indicators"
        previous_context = self._build_previous_findings_context(analysis_state, "biochemistry")
        if previous_context:
            task_instruction += previous_context

        # Biochemistry runs first — no current-round cross-agent results available.
        # The biochemistry agent's _get_other_agent_context() will fall back to
        # previous round results from state.round_results for round 2+.
        current_round_context = {}

        result = await self.biochemistry_agent.process(
            analysis_state,
            task_instruction,
            current_round_context=current_round_context
        )
        
        await self._send_message(
            "result",
            result,
            AgentType.BIOCHEMISTRY
        )

        analysis_state.add_message(
            AgentType.BIOCHEMISTRY,
            MessageRole.ASSISTANT,
            json.dumps(result, ensure_ascii=False)
        )

        cache_store(analysis_state, AgentType.BIOCHEMISTRY, result)
        return {
            **state,
            "analysis_state": analysis_state,
            "biochemistry_result": result
        }

    @with_retry("literature_research")
    async def _literature_research(self, state: WorkflowState) -> WorkflowState:
        """Literature agent researches relevant topics.

        Deep Research mode: merges all topics into one comprehensive query and
        makes a single DR call. If some topics are insufficiently covered,
        falls back to PubMed+LLM for the gaps.

        PubMed mode: searches each topic individually (up to 3).
        """
        analysis_state = state["analysis_state"]

        skip = await self._require_core_agent(state, "literature", "literature_research")
        if skip is not None:
            return skip

        cached = cache_hit(analysis_state, AgentType.LITERATURE)
        if cached is not None:
            logger.info(f"[{self.session_id}] literature cache hit, skipping re-execution")
            await self._send_message("result", cached, AgentType.LITERATURE)
            return {
                **state,
                "analysis_state": analysis_state,
                "literature_result": cached,
                "failed_node": None,
            }

        instructions = state["pi_instructions"]

        topics = _normalize_tasks(instructions.get("literature_topics", []))
        logger.info(f"[{self.session_id}] Starting literature research with topics: {topics}")

        # Dynamic PI planning: refine literature topics based on all completed results
        overview = instructions.get("overview", "")
        if overview and topics:
            completed = self._collect_current_round_results(state)
            if completed:
                await self._send_message("status", "PI is refining literature topics...", AgentType.PI)
                topics = await self.pi_agent.plan_agent_tasks(
                    agent_name="literature",
                    original_tasks=topics,
                    completed_results=completed,
                    overall_strategy=overview,
                    research_objective=analysis_state.initial_instruction,
                    state=analysis_state,
                )

        if not topics:
            # Default topics based on other findings
            path_findings = state.get("pathology_result", {}).get("findings", []) if state.get("pathology_result") else []
            bio_findings = state.get("bioinformatics_result", {}).get("findings", []) if state.get("bioinformatics_result") else []
            biochem_findings = state.get("biochemistry_result", {}).get("findings", []) if state.get("biochemistry_result") else []

            all_source_findings = path_findings + bio_findings + biochem_findings
            if all_source_findings:
                # Use truncated findings as search keywords (full findings are too long for search queries)
                short_findings = [
                    (f[:80] if isinstance(f, str) else str(f)[:80])
                    for f in all_source_findings[:3]
                ]
                topics = [f"single cell sequencing {' '.join(short_findings)}"]
            else:
                # No findings from other agents — build topic from research objective and treatment groups
                objective = analysis_state.initial_instruction or ""
                group_names = [g.name for g in (analysis_state.treatment_groups or []) if g.name]
                if objective:
                    topics = [f"single cell sequencing {objective[:120]}"]
                elif group_names:
                    topics = [f"single cell sequencing {' '.join(group_names[:3])}"]
                else:
                    logger.warning(f"[{self.session_id}] No findings or objective for literature search, skipping")
                    topics = []

        await self._send_message("status", "Literature research in progress...", AgentType.LITERATURE)

        # Collect current round findings from other agents for literature context
        current_round_findings: List[str] = []
        for result_key in ("biochemistry_result", "pathology_result", "bioinformatics_result"):
            result = state.get(result_key)
            if result and result.get("findings"):
                for f in result["findings"][:3]:
                    short = f[:100] if isinstance(f, str) else str(f)[:100]
                    current_round_findings.append(short)

        # Enrich topics with previous findings for round 2+
        previous_lit_context = self._build_previous_findings_context(analysis_state, "literature")

        # Check if Deep Research mode is available (merges topics into one call)
        is_deep_research = self.literature_agent.is_deep_research_available

        limited_topics = topics[:3]

        if is_deep_research and limited_topics:
            # === Deep Research: merge topics, one call ===
            merged_topic = "Please conduct a comprehensive literature review covering the following aspects:\n"
            for i, t in enumerate(limited_topics, 1):
                merged_topic += f"{i}. {t}\n"
            if previous_lit_context:
                merged_topic += f"\n{previous_lit_context}\nFocus on aspects NOT already covered above."

            logger.info(f"[{self.session_id}] Deep Research mode: merging {len(limited_topics)} topics into one query")
            dr_result = await self.literature_agent.process(
                analysis_state, merged_topic, use_deep_research=True,
                current_round_findings=current_round_findings
            )
            results = [dr_result]

            # Evaluate coverage — fallback to PubMed for gaps
            if len(limited_topics) > 1:
                gaps = await self._evaluate_dr_coverage(limited_topics, dr_result)
                if gaps:
                    logger.info(f"[{self.session_id}] Deep Research gaps detected, PubMed fallback for: {gaps}")
                    await self._send_message("status", f"Supplementing {len(gaps)} topic(s) via PubMed...", AgentType.LITERATURE)
                    # Temporarily switch to PubMed mode so _do_research() routes
                    # to _multi_round_pubmed() instead of retrying Deep Research
                    for gap_topic in gaps:
                        enriched = gap_topic
                        if previous_lit_context:
                            enriched = f"{gap_topic}\n{previous_lit_context}\nFocus on aspects NOT already covered."
                        gap_result = await self.literature_agent.process(
                            analysis_state, enriched, use_deep_research=True,
                            current_round_findings=current_round_findings,
                            mode_override="pubmed_llm",
                        )
                        results.append(gap_result)
        else:
            # === PubMed mode: per-topic search ===
            results = []
            for topic in limited_topics:
                enriched_topic = topic
                if previous_lit_context:
                    enriched_topic = f"{topic}\n{previous_lit_context}\nFocus on aspects NOT already covered above."
                result = await self.literature_agent.process(
                    analysis_state,
                    enriched_topic,
                    # True routes to _do_research() for full PubMed pipeline,
                    # not the external Deep Research API (guarded by is_deep_research_available).
                    use_deep_research=True,
                    current_round_findings=current_round_findings
                )
                results.append(result)

        # Combine results
        combined_result = {
            "topics": topics,
            "findings": [],
            "reports": []
        }

        for r in results:
            combined_result["findings"].extend(r.get("findings", []))
            if r.get("report"):
                combined_result["reports"].append(r["report"])

        logger.info(f"[{self.session_id}] Literature research completed: {len(combined_result['findings'])} findings from {len(topics)} topics")

        await self._send_message(
            "result",
            combined_result,
            AgentType.LITERATURE
        )

        analysis_state.add_message(
            AgentType.LITERATURE,
            MessageRole.ASSISTANT,
            json.dumps({"findings": combined_result["findings"]}, ensure_ascii=False)
        )

        cache_store(analysis_state, AgentType.LITERATURE, combined_result)
        return {
            **state,
            "analysis_state": analysis_state,
            "literature_result": combined_result
        }

    async def _evaluate_dr_coverage(self, topics: List[str], dr_result: Dict) -> List[str]:
        """Evaluate which topics were NOT sufficiently covered by Deep Research.

        Returns a list of gap topic strings for PubMed+LLM fallback.
        """
        report = dr_result.get("report", "")
        if not report:
            return topics  # No report at all — fall back on everything

        topics_list = "\n".join(f"{i+1}. {t}" for i, t in enumerate(topics))
        prompt = f"""The following Deep Research report was generated:

{report[:3000]}

The original research topics were:
{topics_list}

Which topics are NOT sufficiently covered in the report? Return a JSON array of topic strings that need additional research. If all topics are well covered, return an empty array [].

Return ONLY the JSON array, nothing else."""

        try:
            response = await self.literature_agent.chat_without_history(prompt, temperature=0.2)
            # Strip markdown code fences if present (e.g. ```json ... ```)
            cleaned = response.strip()
            if cleaned.startswith('```'):
                first_newline = cleaned.find('\n')
                if first_newline != -1:
                    cleaned = cleaned[first_newline + 1:]
                else:
                    cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            gaps = json.loads(cleaned)
            if isinstance(gaps, list):
                return [g for g in gaps if isinstance(g, str)]
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[{self.session_id}] DR coverage evaluation failed: {e}")
        return []
    
