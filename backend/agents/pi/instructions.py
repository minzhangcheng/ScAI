"""Round-opening instructions handed to the other agents.

Moved verbatim out of backend/agents/pi_agent.py; composed into PIAgent.
"""
from typing import Dict, Any
import json
import logging

from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY

# Configure logger
logger = logging.getLogger(__name__)
from ...models.schemas import AnalysisState

from .shared import (  # noqa: F401
    _CONTROL_SIGNAL_RESULTS,
    _BRIEF_PREFERRED_TYPES,
    _BRIEF_QC_MARKERS,
    _DIRECTION_CANDIDATE_LIST_FIELDS,
    _normalize_supporting_data,
    _normalize_direction_candidate,
    _normalize_integration_scored_candidates,
    _bounded_brief_fallback,
    _format_database_priors_block,
    _build_default_bio_tasks,
)


class InstructionsMixin:
    """Round-opening instructions handed to the other agents."""

    async def initial_instruction(
        self,
        state: AnalysisState,
        user_instruction: str
    ) -> Dict[str, Any]:
        """
        Generate initial instructions for all agents

        Args:
            state: Current analysis state
            user_instruction: User's initial instruction

        Returns:
            Dict with instructions for each agent
        """
        logger.info(f"[PI] Generating initial instructions for project: {state.name}")
        prompt = f"""
## Project Information
Project Name: {state.name}
Research Objective: {user_instruction}

## Available Data
{state.build_experimental_design_summary()}

You need to develop the first round analysis plan for the team.

IMPORTANT: You must assign tasks based ONLY on the data that is actually available (listed above).
- Do NOT instruct agents to perform experiments or staining that are not in the available data.
- Each agent should analyze the data types listed for their section.
- If a section has no data files, assign minimal or no tasks for that agent.
- You MAY include a brief "recommended_future_experiments" list suggesting additional experiments/staining that would strengthen the analysis, but the primary tasks must use only available data.

Please assign specific tasks for each direction (listed in execution order):

1. Biochemistry analysis: Analyze the available biochemistry data/indicators. What patterns and abnormalities should be assessed?
2. Pathology analysis: Analyze the available histology images. What morphological features and pathological changes should be evaluated?
3. Bioinformatics analysis: Analyze the available sequencing data. Standard workflow plus cell types and pathways to focus on.
4. Literature research: What specific topics should be searched in PubMed?
   - Each topic should be a focused research question (e.g., "Role of <pathway> signaling in <cell type> state transitions under <condition>")
   - Avoid overly broad or overly specific topics
   - Include 2-4 topics covering different aspects (mechanism, phenotype, therapeutic relevance)

Reply in JSON format as follows:
{{
    "overview": "Overview of overall research strategy",
    "biochemistry_tasks": ["task1", "task2", ...],
    "pathology_tasks": ["task1", "task2", ...],
    "bioinformatics_tasks": ["task1", "task2", ...],
    "literature_topics": ["topic1", "topic2", ...],
    "recommended_future_experiments": ["experiment1", "experiment2", ...]
}}
"""
        
        response = await self.chat(prompt, temperature=0.3)

        # Use base class safe JSON parser
        result = self._safe_parse_json(response, default={})

        # If we got valid JSON with expected fields, return it
        if result and "overview" in result:
            return result

        # Return default structure if parsing fails
        return {
            "overview": response or "Failed to generate initial instructions",
            "pathology_tasks": ["Analyze tissue sections, assess overall morphological changes"],
            "bioinformatics_tasks": _build_default_bio_tasks(state, detailed=False),
            "biochemistry_tasks": ["Summarize existing biochemical indicators"],
            "literature_topics": ["Literature related to the research topic and phenotype"]
        }

    async def initial_exploration_instruction(
        self,
        state: AnalysisState,
        user_instruction: str
    ) -> Dict[str, Any]:
        """
        Generate exploration-phase instructions (Round 1) — unbiased, comprehensive.

        The PI is constrained to plan without hypothesis-driven bias,
        focusing on broad data extraction rather than targeted investigation.
        """
        logger.info(f"[PI] Generating exploration-phase instructions for project: {state.name}")
        prompt = f"""
## Project Information
Project Name: {state.name}
Species: {state.species or "Not yet determined"}
Research Objective: {user_instruction}

## Available Data
{state.build_experimental_design_summary()}

You are generating the EXPLORATION PHASE plan (Round 1). This round performs comprehensive, unbiased data exploration.

CRITICAL CONSTRAINTS:
- Do NOT let the research objective replace, narrow, or bias the mandatory broad exploration tasks.
- The goal is to discover what the data shows, NOT to confirm pre-existing hypotheses by default.
- Treat mechanisms explicitly named in the research objective as prespecified hypotheses. Preserve
  them for an explicit evidence assessment after the unbiased baseline analyses, reporting support,
  contradiction, or insufficient evidence rather than silently omitting or assuming them.
- Prioritize broad, efficient, high-information analyses (e.g., DEG + GO/KEGG) over targeted deep analyses.
- Do NOT instruct agents to perform experiments or staining that are not in the available data.

## Fixed Mandatory Pipeline (MUST be included in bioinformatics_tasks)
1. Quality Control (QC) and filtering
2. Doublet removal assessment (if data quality suggests it is needed)
3. Batch effect assessment and correction (if multiple samples/batches exist)
4. Normalization, HVG selection, PCA, UMAP
5. Clustering (leiden/louvain)
6. Cell type annotation via annotate_clusters tool
7. Condition-level DEG (treatment vs control) + GO/KEGG enrichment for all cell types
   (merge closely related states into a shared parent population only when the observed markers support it)
8. Cell-cell communication analysis (liana)
9. If bulk RNA-seq data is available:
   a. Load bulk expression matrix and perform differential expression analysis (pydeseq2)
   b. Cross-validate bulk DEG with scRNA-seq pseudobulk DEG
   c. Generate comparison visualizations (Venn diagram, correlation plot)
   d. Perform cell-type deconvolution on the bulk RNA-seq data using the annotated scRNA-seq data as a reference to estimate cell-type proportions (prioritize pure Python NNLS).

Note: These tasks span both Stage 1 (steps 1–6: QC through cell type annotation) and Stage 2 (steps 7–9: DEG, enrichment, cell-cell communication, bulk integration). The bioinformatics agent will automatically split and schedule them across the appropriate stages — do NOT try to compress everything into a single stage.

## Your Additional Planning (UNBIASED exploration)
Beyond the fixed pipeline, plan these additional analyses:
1. Decide which cluster comparisons are biologically meaningful for cross-cluster DEG
   - Do NOT compare biologically unrelated populations merely because they are both present
   - Focus on: same cell type across conditions, related subtypes within a lineage
2. Decide which subtypes should be merged for additional condition-level comparisons
   (e.g., merge data-supported subtypes that share a coherent parent identity)
3. Keep all analyses simple and high-throughput

IMPORTANT: You must assign tasks based ONLY on the data that is actually available.
You MAY include a brief "recommended_future_experiments" list, but primary tasks must use only available data.

Please assign specific tasks for each direction (listed in execution order):

1. Biochemistry analysis: Analyze available biochemistry data/indicators
2. Pathology analysis: Analyze available histology images
3. Bioinformatics analysis: The fixed pipeline above PLUS your additional unbiased exploration tasks
4. Literature research: Broad topics covering expected cell types and pathways in this tissue/disease context

Reply in JSON format:
{{
    "overview": "Exploration strategy overview (unbiased, comprehensive)",
    "biochemistry_tasks": ["task1", "task2", ...],
    "pathology_tasks": ["task1", "task2", ...],
    "bioinformatics_tasks": ["task1", "task2", ...],
    "literature_topics": ["topic1", "topic2", ...],
    "cluster_comparison_strategy": "Which clusters to compare and why (brief)",
    "merge_strategy": "Which subtypes to merge for condition comparison (brief)",
    "recommended_future_experiments": ["experiment1", ...]
}}
"""

        response = await self.chat(prompt, temperature=0.3)
        result = self._safe_parse_json(response, default={})

        if result and "overview" in result:
            return result

        return {
            "overview": response or "Failed to generate exploration instructions",
            "pathology_tasks": ["Analyze tissue sections, assess overall morphological changes"],
            "bioinformatics_tasks": _build_default_bio_tasks(state, detailed=True),
            "biochemistry_tasks": ["Summarize existing biochemical indicators"],
            "literature_topics": ["Cell types and pathways relevant to the tissue and disease context"]
        }

    async def innovation_instruction(
        self,
        state: AnalysisState,
        user_direction: str,
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Generate innovation-phase instructions based on user's chosen direction.

        The PI generates targeted, hypothesis-driven tasks focused on the selected direction.
        """
        logger.info(f"[PI] Generating innovation-phase instructions for direction: {user_direction[:100]}")

        # Build exploration context
        exploration_context = ""
        if state.exploration_report:
            exploration_context = f"""
## Exploration Phase Summary
Cluster Overview: {state.exploration_report.cluster_overview[:500]}
Key Pathway Changes: {json.dumps(state.exploration_report.key_pathway_changes[:5], ensure_ascii=False)}
Data Quality Warnings: {json.dumps(state.exploration_report.data_quality_warnings[:3], ensure_ascii=False)}
PI Direction Assessment: {state.exploration_report.pi_direction_assessment[:500]}
"""

        # Build user notes section
        notes_section = ""
        if notes and notes.strip():
            notes_section = f"\n## User Notes for This Direction\n{notes.strip()}\n"

        # Detect data types for conditional toolkit inclusion
        has_bulk_data = state.has_bulk_rnaseq()
        has_scrna_data = state.has_scrnaseq()

        # Conditionally add bulk-scRNA integration as the FIRST and recommended item
        if has_bulk_data and has_scrna_data:
            toolkit_intro = """
- **Bulk-scRNA direction validation (RECOMMENDED — include at least ONE bulk validation task)**:
  Use bulk RNA-seq data as an independent validation layer for this direction's new findings.
  Reuse the bulk variables already loaded in Round 1 (see the data context section for their exact
  names); do NOT re-run the bulk DEG pipeline from scratch.
"""
        elif has_bulk_data:
            toolkit_intro = """
- **Bulk direction validation (RECOMMENDED when bulk data exists)**:
  Validate new findings on existing bulk data. Reuse the bulk variables already loaded in
  Round 1 (see the data context section for their exact names); do NOT re-run the bulk DEG
  pipeline from scratch.
"""
        else:
            toolkit_intro = ""

        # Append additional targeted-analysis methods (Round 2+ enrichments)
        toolkit_intro += """- **Gene Expression Programs (Lightweight)**: For continuous cell states, prefer `sklearn.decomposition.NMF` on highly variable genes; for large datasets consider subsampling (e.g., ~3000 cells) to keep runtime tractable. Avoid full cNMF bootstraps unless explicitly requested.
- **Biomarker Discovery (Conditional)**: When the target population has clear binary/multiclass metadata (e.g., `len(adata.obs['Condition'].unique()) >= 2`), use `sklearn.ensemble.RandomForestClassifier` to extract top feature genes (~20 by default).
"""

        prompt = f"""
## Project: {state.name}
## Research Objective: {state.initial_instruction}
## User's Chosen Research Direction: {user_direction}
{exploration_context}
{notes_section}
## Available Data
{state.build_experimental_design_summary()}

You are now in the INNOVATION PHASE. Generate TARGETED, HYPOTHESIS-DRIVEN analysis tasks.
Focus specifically on: {user_direction}

Design specific analyses to deeply investigate this direction.

{MECHANISM_EVIDENCE_POLICY}

For each bioinformatics task, specify the CONCRETE analysis method and target:

Available analysis toolkit (use as many as relevant):
{toolkit_intro}- **Targeted DEG**: Specify which cell populations to compare, which contrasts, and threshold strategy
- **Pathway enrichment**: GO/KEGG for broad screening; GSEA/ssGSEA for quantitative pathway scoring; MSigDB hallmark gene sets
- **Gene signature & Metabolic scoring**: Score cells for specific biological or metabolic signatures using sc.tl.score_genes or pyscenic.aucell; choose gene sets based on data-driven findings, not prior assumptions
- **Trajectory & RNA Velocity**: Diffusion pseudotime (DPT), PAGA for lineage inference; scvelo or cellrank for directed cell fate mapping and driver gene identification (if spliced/unspliced data available)
- **Transcription factor & Gene Regulatory Networks (GRN)**: decoupler / pyscenic to infer TF activity; PyWGCNA or gene co-expression analysis to identify co-expressed gene modules and hub genes
- **Cell-cell communication**: liana / cellphonedb for ligand-receptor interactions; ligand-target analysis (via decoupler) to predict downstream target gene activation in receiver cells
- **Cellular Stemness & Plasticity**: cytotrace2-py for CytoTRACE-like scoring to infer differentiation potential and developmental states
- **Tumor/Malignancy estimation**: infercnvpy to detect large-scale chromosomal copy number variations (use only when oncology data is present)
- **Subpopulation re-clustering**: Re-cluster specific cell types at higher resolution to discover subtypes or functional states
- **Correlation analysis**: Gene-gene correlation, module scoring, or regression analysis within cell populations
- **Compositional analysis**: Compare cell type proportions between conditions with statistical testing (pertpy or sccoda)
- **Immune repertoire analysis**: scirpy for TCR/BCR clonotype expansion, diversity, and sharing analysis (use only when paired VDJ data is available)
- **Marker gene visualization**: Dot plots, violin plots, feature plots for key genes across conditions

IMPORTANT:
- Generate AT LEAST 4-6 bioinformatics tasks that form a LOGICAL, CASCADING analytical storyline.
- Design tasks so Task N builds upon the findings of Task N-1 where possible, rather than as isolated probes.
- For each bioinformatics task, you MUST specify three fields:
  1. Task & Method: what to analyze, the concrete method/library to use, and the target cell types.
  2. Rationale: WHY this is biologically relevant to the chosen direction (state the underlying hypothesis).
  3. Expected Outcome: what specific patterns or values would confirm OR reject the hypothesis.
- Include both hypothesis-confirming AND exploratory analyses.
- Do NOT repeat analyses already completed in Round 1 (basic pipeline, annotation, etc.).
- The Jupyter kernel retains all Round 1 variables — build upon existing results.
- **Literature Verification Loop**: in `literature_topics`, explicitly ask the Literature Agent to search PubMed for recent evidence that SUPPORTS or REFUTES each bioinformatics task's hypothesis, and to flag any contradictory evidence prominently in its summary.
- {self._LITERATURE_TOPIC_FORMAT_RULE}
"""

        prompt += f"""Reply in JSON format:
{{
    "overview": "Innovation strategy for: {user_direction}",
    "biochemistry_tasks": [...],
    "pathology_tasks": [...],
    "bioinformatics_tasks": [...],
    "literature_topics": [...]
}}
"""

        response = await self.chat(prompt, temperature=0.4)
        result = self._safe_parse_json(response, default={})

        if result and "overview" in result:
            return result

        return {
            "overview": f"Innovation phase: {user_direction}",
            "bioinformatics_tasks": [f"Perform targeted analysis on: {user_direction}"],
            "pathology_tasks": [],
            "biochemistry_tasks": [],
            "literature_topics": [f"Literature review on: {user_direction}"]
        }

    async def innovation_instruction_pi_directed(
        self,
        state: AnalysisState
    ) -> Dict[str, Any]:
        """
        PI autonomously selects a research direction using 3-dimension evaluation,
        then generates innovation-phase instructions.

        Dimensions:
        1. Innovation — is this well-studied or emerging? (Literature search)
        2. Novelty — is this expected or surprising? (PI reasoning)
        3. Reliability — is the supporting data reliable? (Data quality)
        """
        logger.info(f"[PI] PI-directed innovation: evaluating research directions")

        # Get candidates from exploration report
        candidates = []
        if state.exploration_report and state.exploration_report.research_direction_candidates:
            candidates = state.exploration_report.research_direction_candidates

        if not candidates:
            # Fallback: generate candidates from round results
            candidates = [{"direction": "Continue analysis based on Round 1 findings", "rationale": "No candidates available"}]

        # Build exploration context
        exploration_context = ""
        if state.exploration_report:
            exploration_context = f"""
Cluster Overview: {state.exploration_report.cluster_overview[:500]}
Key Pathway Changes: {json.dumps(state.exploration_report.key_pathway_changes[:5], ensure_ascii=False)}
Strongest Heterogeneity: {json.dumps(state.exploration_report.strongest_heterogeneity[:5], ensure_ascii=False)}
Data Quality Warnings: {json.dumps(state.exploration_report.data_quality_warnings[:3], ensure_ascii=False)}
"""

        prompt = f"""Evaluate the following research direction candidates for the innovation phase.

## Exploration Report
{exploration_context}

## Candidates
{json.dumps(candidates, ensure_ascii=False, indent=2)}

For EACH candidate, evaluate on 3 dimensions:

1. **Innovation**: Based on your knowledge of the biomedical literature, is this a well-studied area (low innovation) or an emerging/under-explored area (high innovation)?
   - Score: high / medium / low
   - Provide brief evidence for your assessment

2. **Novelty**: Given the experimental design and tissue context, is this finding expected (low novelty) or surprising/counter-intuitive (high novelty)?
   - Score: high / medium / low
   - IMPORTANT: You MUST explain WHY this is novel or expected

3. **Reliability**: Based on the data quality (sample sizes, statistical significance, effect sizes from the exploration report), how reliable is the supporting evidence?
   - Score: high / medium / low
   - Provide evidence from the data

DECISION RULE:
- Reliability is the BASELINE: low-reliability directions are NOT recommended regardless of other scores
- Among reliable directions: Innovation + Novelty together determine priority
- If multiple directions are equally ranked, prefer the one with higher novelty

Return JSON:
{{
    "evaluations": [
        {{
            "direction": "...",
            "innovation_score": "high/medium/low",
            "innovation_evidence": "...",
            "novelty_score": "high/medium/low",
            "novelty_reasoning": "WHY this is novel or expected",
            "reliability_score": "high/medium/low",
            "reliability_evidence": "...",
            "overall_priority": "high/medium/low"
        }}
    ],
    "selected_direction": "The highest-priority direction to pursue",
    "rationale": "Why this direction was selected over others"
}}
"""

        response = await self.chat(prompt, temperature=0.4)
        data = self._safe_parse_json(response, default={})

        selected = data.get("selected_direction", "")

        if not selected and candidates:
            selected = candidates[0].get("direction", "Continue analysis based on Round 1 findings")

        logger.info(f"[PI] Selected direction: {selected}")

        # Generate tasks for the selected direction
        result = await self.innovation_instruction(state, selected)
        result["_chosen_direction"] = selected
        return result
