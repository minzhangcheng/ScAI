"""Round-opening instructions handed to the other agents.

Moved verbatim out of backend/agents/pi_agent.py; composed into PIAgent.
"""
from typing import Any, Dict, List, Mapping
import json
import logging

from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from ...utils.formatting import as_list, coerce_json_text
from ...services.sample_design import sequencing_design_instruction

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


_INSTRUCTION_LIST_FIELDS = (
    "biochemistry_tasks",
    "pathology_tasks",
    "bioinformatics_tasks",
    "literature_topics",
    "recommended_future_experiments",
)


def build_out_of_scope_context(state: Any) -> str:
    """Render the follow-ups previous rounds ruled out as unrunnable here.

    A round records in ``out_of_scope`` the work that needs data, metadata, or
    experiments this session does not hold. Nothing consumed that list, while
    the findings the next planner *did* receive reported the same work as
    unfinished -- so the same provenance, manifest and replication steps were
    planned again round after round, and the executor spent its budget
    re-deriving a limit that no analysis step can lift.

    Naming them here closes that loop without deciding anything: the PI is told
    what was already ruled out, and still chooses what to plan. Pure so the
    carryover is testable without running a round.
    """
    seen: List[str] = []
    for review in (getattr(state, "post_execution_reviews", None) or []):
        if not isinstance(review, Mapping):
            continue
        plan = review.get("review_plan")
        if not isinstance(plan, Mapping):
            continue
        for item in (plan.get("out_of_scope") or []):
            text = str(item).strip()
            if text and text not in seen:
                seen.append(text)
    if not seen:
        return ""
    # Most recent last, and the cap keeps that end: a long session accumulates
    # far more of these than fit (41 distinct entries in one measured 8-round
    # run), and the rulings that bear on the round about to be planned are the
    # ones the latest review just made.
    lines = "\n".join(f"- {text}" for text in seen[-20:])
    return f"""
## Already ruled out in previous rounds
Earlier rounds determined that each of the following needs data, metadata, or
experiments this session does not hold. They are not unfinished work, and no
analysis step can supply what they are missing. Do NOT plan a step whose
purpose is to establish one of them: a round that tries spends its budget
re-deriving the same limit. Where one of them would block a task you would
otherwise propose, state the limit in that task's expected outcome and plan
what the available data CAN answer instead.
{lines}
"""


def _normalize_instruction_payload(value: Any) -> Dict[str, Any] | None:
    """Normalize the container shape of one PI round-opening response."""
    if not isinstance(value, dict) or "overview" not in value:
        return None
    normalized = dict(value)
    normalized["overview"] = coerce_json_text(normalized.get("overview"))
    for field in _INSTRUCTION_LIST_FIELDS:
        normalized[field] = as_list(normalized.get(field))
    for field in ("cluster_comparison_strategy", "merge_strategy"):
        if field in normalized:
            normalized[field] = coerce_json_text(normalized.get(field))
    return normalized


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
        normalized = _normalize_instruction_payload(result)
        if normalized is not None:
            return normalized

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

        additional_planning_block = """Beyond the baseline pipeline, plan these additional analyses:
1. Decide which cluster comparisons are biologically meaningful for cross-cluster DEG
   - Do NOT compare biologically unrelated populations merely because they are both present
   - Focus on: same cell type across conditions, related subtypes within a lineage
2. Decide which subtypes should be merged for additional condition-level comparisons
   (e.g., merge data-supported subtypes that share a coherent parent identity)
3. Keep all analyses simple and high-throughput"""
        literature_scope = (
            "Broad topics covering the cell types and pathways the data actually shows, in "
            "the tissue and experimental context stated above. Do not assume a disease "
            "context the design does not describe."
        )
        cluster_field_hint = '"Which clusters to compare and why (brief)"'
        merge_field_hint = '"Which subtypes to merge for condition comparison (brief)"'
        baseline_pipeline_block = """## Baseline Pipeline (MUST be included in bioinformatics_tasks)
1. Quality Control (QC) and filtering
2. Doublet removal assessment (if data quality suggests it is needed)
3. Batch effect assessment on the uncorrected representation; decide whether a
   correction/integration step is scientifically appropriate for the measured design
4. Normalization, HVG selection, PCA, UMAP
5. Clustering (leiden/louvain)
6. Cell type annotation via annotate_clusters tool
7. Assess condition-associated signals using methods and inferential strength that
   you judge appropriate for the measured design; state material limitations
   (merge closely related states into a shared parent population only when the observed markers support it)
8. Cell-cell communication analysis (liana)"""
        sample_design_facts = sequencing_design_instruction(state)
        prompt = f"""
## Project Information
Project Name: {state.name}
Species: {state.species or "Not yet determined"}
Research Objective: {user_instruction}

## Available Data
{state.build_experimental_design_summary()}

## Measured Sequencing-Design Facts
{sample_design_facts or "No additional design facts."}

You are generating the EXPLORATION PHASE plan (Round 1). This round performs comprehensive, unbiased data exploration.

CRITICAL CONSTRAINTS:
- Do NOT let the research objective replace, narrow, or bias the mandatory broad exploration tasks.
- The goal is to discover what the data shows, NOT to confirm pre-existing hypotheses by default.
- Treat mechanisms explicitly named in the research objective as prespecified hypotheses. Preserve
  them for an explicit evidence assessment after the unbiased baseline analyses, reporting support,
  contradiction, or insufficient evidence rather than silently omitting or assuming them.
- Prioritize broad, efficient, high-information analyses (e.g., DEG + GO/KEGG) over targeted deep analyses.
- Do NOT instruct agents to perform experiments or staining that are not in the available data.

{baseline_pipeline_block}
9. If bulk RNA-seq data is available:
   a. Analyze bulk expression using methods you judge appropriate for its measured design.
   b. Assess cross-modal concordance ONLY if another modality exists to compare against, and
      decide what inferential weight it supports — noting whether the modalities come from the
      same biological samples (a consistency check) or different ones (replication).
   c. Select and justify any comparison visualizations needed to communicate the
      cross-modal evidence.
   d. Decide whether a cell-composition analysis is scientifically supported. Deconvolution
      needs a reference the available inputs can supply; without single-cell data from this
      study or a justified external reference, it is not supported — say so rather than
      planning it.

Note: These tasks span two stages — the baseline establishment steps above, then the condition-level, enrichment and integration steps. The bioinformatics agent will automatically split and schedule them across the appropriate stages — do NOT try to compress everything into a single stage.

## Your Additional Planning (UNBIASED exploration)
{additional_planning_block}

IMPORTANT: You must assign tasks based ONLY on the data that is actually available.
You MAY include a brief "recommended_future_experiments" list, but primary tasks must use only available data.

Please assign specific tasks for each direction (listed in execution order):

1. Biochemistry analysis: Analyze available biochemistry data/indicators
2. Pathology analysis: Analyze available histology images
3. Bioinformatics analysis: The fixed pipeline above PLUS your additional unbiased exploration tasks
4. Literature research: {literature_scope}

Reply in JSON format:
{{
    "overview": "Exploration strategy overview (unbiased, comprehensive)",
    "biochemistry_tasks": ["task1", "task2", ...],
    "pathology_tasks": ["task1", "task2", ...],
    "bioinformatics_tasks": ["task1", "task2", ...],
    "literature_topics": ["topic1", "topic2", ...],
    "cluster_comparison_strategy": {cluster_field_hint},
    "merge_strategy": {merge_field_hint},
    "recommended_future_experiments": ["experiment1", ...]
}}
"""

        response = await self.chat(prompt, temperature=0.3)
        result = self._safe_parse_json(response, default={})

        normalized = _normalize_instruction_payload(result)
        if normalized is not None:
            return normalized

        return {
            "overview": response or "Failed to generate exploration instructions",
            "pathology_tasks": ["Analyze tissue sections, assess overall morphological changes"],
            "bioinformatics_tasks": _build_default_bio_tasks(state, detailed=True),
            "biochemistry_tasks": ["Summarize existing biochemical indicators"],
            "literature_topics": [
                "Cell types and pathways relevant to the stated experimental context"
            ]
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

        # What earlier rounds already ruled out. Without this the planner sees
        # only the findings, which report the same work as unfinished.
        out_of_scope_section = build_out_of_scope_context(state)

        # Detect data types for conditional toolkit inclusion
        has_bulk_data = state.has_bulk_rnaseq()
        scoring_entry = (
            "score cells for biological or metabolic signatures using sc.tl.score_genes "
            "or pyscenic.aucell"
        )
        correlation_unit = "computed within a cell population"
        marker_viz = "dot plots, violin plots, feature plots for key genes across conditions"
        task_target = "the target cell types"
        modality_toolkit = """- **Trajectory & RNA Velocity**: Diffusion pseudotime (DPT), PAGA for lineage inference; scvelo or cellrank for directed cell fate mapping (only where spliced/unspliced layers exist; these answer different questions and are not interchangeable)
- **Cell-cell communication**: liana / cellphonedb for ligand-receptor interactions; ligand-target analysis (via decoupler) to predict downstream target activation in receiver cells
- **Cellular Stemness & Plasticity**: cytotrace2-py for CytoTRACE-like scoring to infer differentiation potential
- **Tumor/Malignancy estimation**: infercnvpy to detect large-scale copy number variation (use only when oncology data is present)
- **Subpopulation re-clustering**: re-cluster specific cell types at higher resolution to discover subtypes or functional states
- **Compositional analysis**: compare cell type proportions between conditions with statistical testing (pertpy or sccoda)
"""

        # Conditionally add bulk-scRNA integration as the FIRST and recommended item
        if has_bulk_data:
            toolkit_intro = """
- **Bulk-scRNA cross-assay check (include one when the direction makes a claim bulk can test)**:
  Check whether the bulk data is CONCORDANT with this direction's findings. Before framing the
  result, establish from the experimental design whether the bulk and single-cell libraries came
  from the SAME biological samples or from different ones, and state which:
    * different biological samples -> independent replication, and may be described as such
    * the same samples -> a cross-assay consistency check ONLY. It shares the biological
      replicates, the animals/donors, and usually the batch, so it cannot independently confirm
      anything; agreement mostly rules out a single-assay technical artifact.
  Never call the same-sample case "independent validation". If the provenance cannot be
  established from the design, say so and use the weaker framing.
  Reuse the bulk variables already loaded in Round 1 (see the data context section for their exact
  names); do NOT re-run the bulk DEG pipeline from scratch.
"""
        else:
            toolkit_intro = ""

        # Append additional targeted-analysis methods (Round 2+ enrichments)
        toolkit_intro += """- **Gene Expression Programs**: For continuous cell states, matrix factorization (NMF, cNMF) or
  another program-discovery method — choose it, its rank, and its input feature set from the data
  and justify them; a rank picked without inspecting the reconstruction or stability is a guess.
  Subsampling to keep runtime tractable is fine, but state the number used and check the programs
  are recovered on a second subsample before interpreting them.
- **Biomarker Discovery (Conditional)**: A supervised classifier can nominate candidate features,
  but only where the design can support it. Before proposing one, check what the label actually
  separates: with one biological replicate per condition, ANY classifier separates the conditions
  perfectly by learning donor/batch identity, and the "biomarkers" it returns are that identity.
  Require independent biological replicates per class, hold out data or cross-validate at the
  BIOLOGICAL-UNIT level (never at the cell level, where cells of one sample leak across the
  split), report the out-of-fold performance next to a label-permuted baseline, and treat
  correlated-feature importance rankings as unstable. Any feature count you cut at is a reporting
  choice, not a result — state it.
"""

        prompt = f"""
## Project: {state.name}
## Research Objective: {state.initial_instruction}
## User's Chosen Research Direction: {user_direction}
{exploration_context}
{notes_section}{out_of_scope_section}
## Available Data
{state.build_experimental_design_summary()}

You are now in the INNOVATION PHASE. Generate TARGETED, HYPOTHESIS-DRIVEN analysis tasks.
Focus specifically on: {user_direction}

Design specific analyses to deeply investigate this direction.

{MECHANISM_EVIDENCE_POLICY}

For each bioinformatics task, specify the CONCRETE analysis method and target:

Available analysis toolkit (use as many as relevant):
{toolkit_intro}{modality_toolkit}- **Targeted differential expression**: specify which populations or sample groups to
  compare, which contrasts, and the threshold strategy — fixed in advance, not after seeing
  the result
- **Pathway enrichment**: GO/KEGG for broad screening; GSEA/ssGSEA for quantitative pathway scoring; MSigDB hallmark gene sets
- **Gene signature & Metabolic scoring**: {scoring_entry}; choose gene sets based on data-driven findings, not prior assumptions
- **Transcription factor & Gene Regulatory Networks (GRN)**: decoupler / pyscenic to infer TF activity; PyWGCNA or gene co-expression analysis to identify co-expressed gene modules and hub genes
- **Correlation analysis**: gene-gene correlation, module scoring, or regression, {correlation_unit}
- **Immune repertoire analysis**: scirpy for TCR/BCR clonotype expansion, diversity, and sharing analysis (use only when paired VDJ data is available)
- **Key-gene visualization**: {marker_viz}

IMPORTANT:
- Generate as many bioinformatics tasks as this direction actually needs — typically a handful.
  Do not pad the list to reach a count; a direction that is answered by two well-chosen analyses
  is answered.
- Sequence tasks so a later one builds on an earlier one where that genuinely helps, but do NOT
  build a chain that only holds together if every earlier step comes out positive. For any task
  that depends on a prior result, state what should happen if that prior result is null or
  ambiguous: which downstream tasks stop, and what gets reported instead. A storyline that
  survives its own negative results is a plan; one that does not is an assumption.
- For each bioinformatics task, you MUST specify three fields:
  1. Task & Method: what to analyze, the concrete method/library to use, and {task_target}.
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
            "literature_topics": [f"Literature review on: {user_direction}"],
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
