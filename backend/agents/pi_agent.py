"""
PI Agent - Principal Investigator that coordinates all agents
"""
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import json
import re
import logging

from .base import BaseAgent
from ..utils.settings_provider import AgentSettingsProvider
from ..utils.direction_evidence import build_fallback_direction_evidence

# Configure logger
logger = logging.getLogger(__name__)
from ..models.schemas import (
    AgentType, AnalysisPhase, AnalysisState, MessageRole, RoundResult, ExplorationReport,
    PublicationFigure, FigureReviewStatus, GraphicAbstract,
    DATA_TYPE_SCRNASEQ,
    CrossModalSynthesis, CrossModalMechanismCandidate, CrossModalEvidenceItem,
)
from ..utils.llm_client import OpenAICompatibleClient, UpstreamExhaustedError
from ..constants import (
    CONTINUE_SIGNAL,
    PI_DECIDE_SIGNAL,
    PI_REVIEW_FALLBACK_DECISION,
    REASONING_MODEL_MIN_TOKENS,
    STOP_KEYWORDS,
)

# Synthetic control sentinels that bypass the stop/continue LLM classifier and
# deterministically continue the workflow → (response text, reason) per signal.
_CONTROL_SIGNAL_RESULTS = {
    CONTINUE_SIGNAL: ("Continuing to next round of analysis.", "User requested to continue"),
    PI_DECIDE_SIGNAL: ("Proceeding with PI-selected research directions.", "PI to autonomously select directions"),
}


def _format_database_priors_block(database_priors) -> str:
    if not database_priors:
        return ""
    lines = []
    for cid, hits in database_priors.items():
        parts = []
        for h in hits:
            ov = ", ".join(h.get("overlap", []))
            parts.append(f"{h.get('cell_type','?')}: padj={h.get('p_adj', 1.0):.2g} (overlap: {ov})")
        if parts:
            lines.append(f"Cluster {cid} \u2192 " + "; ".join(parts))
    if not lines:
        return ""
    return ("\n\n=== DATABASE PRIORS (offline ORA, supportive evidence) ===\n"
            "These are quantitative priors from offline marker databases; literature remains primary. "
            "If a prior contradicts the literature, favor biological plausibility and the literature.\n"
            + "\n".join(lines) + "\n")


def _build_default_bio_tasks(state: AnalysisState, detailed: bool = False) -> list:
    """Build default bioinformatics task list based on available sequencing data types."""
    has_bulk = state.has_bulk_rnaseq()
    has_scrna = state.has_scrnaseq()
    tasks = []
    if has_scrna:
        if detailed:
            tasks.append(
                "Execute standard scRNA-seq analysis: QC, normalization, clustering, annotation, "
                "then condition-level DEG + GO/KEGG for ALL clusters, cell communication analysis"
            )
        else:
            tasks.append("Execute standard scRNA-seq/snRNA-seq analysis workflow")
    if has_bulk:
        tasks.append("Perform bulk RNA-seq DEG analysis using pydeseq2 (fallback: scipy.stats.ttest_ind + FDR correction if pydeseq2 unavailable)")
        if has_scrna:
            tasks.append(
                "Cross-validate bulk DEGs with scRNA-seq pseudobulk DEGs: "
                "compute set intersection, generate Venn diagram (from matplotlib_venn import venn2), "
                "and visualize overlapping DEGs' cell-type-specific expression. "
                "CRITICAL: Perform cell-type deconvolution of bulk RNA-seq data using scRNA-seq as a reference "
                "(prioritize scipy.optimize.nnls or sklearn Non-negative Least Squares; if unavailable, try scaden) to estimate cell-type proportion changes."
            )
    if not tasks:
        tasks.append("Execute standard sequencing analysis workflow")
    return tasks


class PIAgent(BaseAgent):
    """
    Principal Investigator Agent

    Responsibilities:
    - Coordinate overall research strategy
    - Analyze and synthesize results from all agents
    - Resolve conflicts between agents
    - Make decisions on next steps
    - Determine when to stop analysis
    """

    _LITERATURE_TOPIC_FORMAT_RULE = (
        "Format rule for `literature_topics`: output a JSON array of PLAIN "
        "natural-language English strings — NOT Python dicts and NOT dict-like "
        "strings. Each topic is one self-contained descriptive sentence, e.g. "
        "\"Mitochondrial copper deficiency and cytochrome c oxidase impairment in "
        "hepatocytes\". Do NOT wrap topics as {'Topic': ..., 'Rationale': ...}; "
        "if you want to convey rationale, fold it into the sentence itself."
    )

    def __init__(
        self,
        settings_provider: 'AgentSettingsProvider',
    ):
        super().__init__(
            agent_type=AgentType.PI,
            settings_provider=settings_provider,
        )
        
    async def process(
        self,
        state: AnalysisState,
        instruction: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Process instruction and coordinate analysis"""
        
        self.refresh_runtime_config_if_needed()
        # Get context from state
        context = self._build_context(state)
        
        # Build prompt
        prompt = f"""
{context}

Current instruction: {instruction}

Based on the above information:
1. Analyze results from all aspects
2. Identify key findings and potential issues
3. Arbitrate if there are conflicts
4. Plan next steps for analysis

Reply in JSON format with the following fields:
- summary: Summary of the current round
- findings: List of key findings
- conflicts: Conflicts found and solutions
- next_steps: Tasks for each Agent to execute next
- should_continue: Whether to continue analysis (true/false)
- reason: Reason for continuing/stopping
"""
        
        response = await self.chat(prompt, temperature=0.5)
        
        # Parse response
        result = self._parse_response(response)
        
        return result
    
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
- Do NOT let the research objective bias your exploration tasks.
- The goal is to discover what the data shows, NOT to confirm pre-existing hypotheses.
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
   (merge similar subtypes/states into unified groups first, e.g., merge activated/exhausted states of the same lineage back into the parent cell type)
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
   - Do NOT compare unrelated cell types (e.g., T cells vs B cells)
   - Focus on: same cell type across conditions, related subtypes within a lineage
2. Decide which subtypes should be merged for additional condition-level comparisons
   (e.g., merge all macrophage subtypes into one group for condition comparison)
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

    async def _infer_species(self, prompt: str, log_context: str) -> Dict[str, Any]:
        """Shared helper for species inference calls.

        Args:
            prompt: The LLM prompt to send
            log_context: Context string for logging (e.g., "project: X" or "user hint: Y")

        Returns:
            Dict with species, confidence, suggestions, reasoning
        """
        logger.info(f"[PI] Inferring species: {log_context}")
        try:
            response = await self.chat(prompt, temperature=0.1)
            result = self._safe_parse_json(response, default={})
            if not result.get("species"):
                return {"species": "", "confidence": "low",
                        "suggestions": result.get("suggestions", []),
                        "reasoning": result.get("reasoning", "Could not determine species from project info")}
            return {
                "species": result.get("species", ""),
                "confidence": result.get("confidence", "low"),
                "suggestions": result.get("suggestions", [result["species"]]),
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            logger.warning(f"[PI] Species inference failed: {e}")
            return {"species": "", "confidence": "low", "suggestions": [],
                    "reasoning": f"Auto-inference failed: {e}"}

    async def infer_species(self, state: AnalysisState) -> Dict[str, Any]:
        """Infer species from project info.

        Returns {"species": str, "confidence": "high"|"low",
                 "suggestions": [...], "reasoning": str}
        """
        prompt = f"""Determine the species/organism for this research project.

Project Name: {state.name}
Research Objective: {state.initial_instruction}
Available Data:
{state.build_experimental_design_summary()}

Based on ALL available information (project name, research description, sample/group names, data file names), determine the species.

Common species in single-cell research:
- Homo sapiens (human) — gene names are ALL UPPERCASE (TP53, BRCA1, CD4)
- Mus musculus (mouse) — gene names Capitalize first letter (Tp53, Brca1, Cd4)
- Rattus norvegicus (rat) — similar to mouse naming

Return JSON:
{{
    "species": "Scientific name (e.g., Homo sapiens, Mus musculus)",
    "confidence": "high" or "low",
    "suggestions": ["Most likely species", "Second option if uncertain"],
    "reasoning": "Brief explanation of how you determined the species"
}}

Set confidence to "high" ONLY if you are very certain from the project information.
If there is any ambiguity, set confidence to "low".
"""
        return await self._infer_species(prompt, f"project: {state.name}")

    async def infer_species_from_hint(self, state: AnalysisState, user_hint: str) -> Dict[str, Any]:
        """Re-infer species incorporating user hint. Same return format as infer_species."""
        prompt = f"""Determine the species/organism for this research project based on the user's input.

Project Name: {state.name}
Research Objective: {state.initial_instruction}
User's input about the species: {user_hint}

Based on the user's input and project information, determine the species.
Use the scientific name (e.g., "Homo sapiens", "Mus musculus", "Rattus norvegicus").

Return JSON:
{{
    "species": "Scientific name",
    "confidence": "high" or "low",
    "suggestions": ["Most likely species"],
    "reasoning": "Brief explanation"
}}
"""
        return await self._infer_species(prompt, f"user hint: {user_hint[:100]}")

    async def decide_species_action(
        self, user_input: str, current_species: str, assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Decide what action to take based on user's free-text input about gene-species mismatch.

        Returns {"action": "change_species"|"keep"|"custom", "new_species": str}
        """
        prompt = f"""A gene-species mismatch was detected. The current species is "{current_species}".
Assessment: {assessment.get('reasoning', '')}
Likely actual species: {assessment.get('likely_species', '')}

The user responded: "{user_input}"

Classify the user's intent into one of these actions:
1. "change_species" — user wants to change the species. Extract the new species name.
2. "keep" — user wants to keep the current species and proceed.
3. "custom" — user gave a specific instruction that doesn't fit the above.

If the user's input looks like a species name (e.g., "human", "mouse", "Homo sapiens"), 
treat it as "change_species" and convert to scientific name.

Return JSON: {{"action": "change_species"|"keep"|"custom", "new_species": "scientific name if changing"}}
"""
        try:
            response = await self.chat(prompt, temperature=0.1)
            result = self._safe_parse_json(response, default={"action": "keep"})
            return {
                "action": result.get("action", "keep"),
                "new_species": result.get("new_species", ""),
            }
        except UpstreamExhaustedError:
            raise
        except Exception:
            return {"action": "keep", "new_species": ""}

    async def review_cluster_annotations(
        self,
        state: AnalysisState,
        annotations: Dict[str, Any],
        tissue: str,
        organism: str,
        triage: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        PI reviews Literature Agent's cluster annotations and assigns confidence scores.
        """
        self.refresh_runtime_config_if_needed()
        logger.info(f"[PI] Reviewing cluster annotations for {organism} {tissue}")

        ann_lines = []
        for cid, info in annotations.items():
            if isinstance(info, dict):
                ct = info.get("cell_type", "Unknown")
                markers = ", ".join(info.get("key_markers", [])[:10])
                evidence = (info.get("evidence", ""))[:200]
                ann_lines.append(
                    f"- Cluster {cid}: {ct} | Markers: {markers} | Evidence: {evidence}"
                )
        ann_text = "\n".join(ann_lines) if ann_lines else "No annotations available"

        triage = triage or {}
        triage_lines = []
        for it in triage.get("suspected_contamination", []):
            triage_lines.append(f"- Cluster {it.get('cluster_id')}: suspected technical/contamination — "
                                 f"{it.get('reason','')} (mark low confidence; pass this reason through verbatim)")
        for it in triage.get("context_dependent_state", []):
            triage_lines.append(f"- Cluster {it.get('cluster_id')}: context-dependent state variant of "
                                 f"{it.get('parent_celltype','?')} (confidence low/medium)")
        triage_block = ("\n## Triage notes from search phase\n" + "\n".join(triage_lines)) if triage_lines else ""

        prompt = f"""You are reviewing cell type annotations generated by a literature search for {organism} {tissue}.

## Cluster Annotations from Literature Search
{ann_text}
{triage_block}

Biological judgement principles:
- This dataset may be single-nucleus RNA-seq (snRNA-seq): long nuclear genes (Nrxn1/Nrxn3/Reln)
  appear easily; treat neuronal calls in non-neural tissue with caution.
- marker set coherence: require positive + negative markers + tissue plausibility + intra-cluster
  consistency; do not call a type on 1-2 markers alone.
- condition-skewed clusters are likely state variants, not new cell types.
- suspected technical/contamination clusters: keep low confidence and pass the provided reason through.
- rare immune populations (e.g. pDC: Irf8+Siglech+Tcf4+Bcl11a) must NOT be down-graded merely for being rare.
- distinguish cholangiocyte vs ductular reaction / hepatobiliary progenitor when hepatocyte/progenitor
  markers co-occur with Pkhd1/Glis3/Spp1.

For EACH cluster, evaluate:
1. Is the cell type annotation correct given the marker genes and tissue context?
2. Assign a confidence level:
   - "high": Markers strongly and specifically match the annotated cell type in this tissue
   - "medium": Markers partially match or could belong to a related cell type
   - "low": Markers are ambiguous, contradictory, or insufficient for confident annotation
3. For clusters with "medium" or "low" confidence, provide:
   - alternative_cell_types: other possible cell types these markers could indicate
   - low_confidence_reason: explanation of why confidence is not high

Reply in JSON format (include ALL clusters):
{{
    "<cluster_id>": {{
        "cell_type": "confirmed or corrected cell type",
        "confidence": "high/medium/low",
        "alternative_cell_types": ["Alternative Type 1", "Alternative Type 2"],
        "low_confidence_reason": "Reason for low/medium confidence (empty string if high)"
    }},
    ...
}}
"""

        response = await self.chat(prompt, temperature=0.2)
        result = self._safe_parse_json(response, default={})

        if not result or not isinstance(result, dict):
            return {}

        # 确定性后处理: 对 suspected_contamination 簇强制低置信 + reason 透传 + 候选入 alt
        for it in triage.get("suspected_contamination", []):
            cid = str(it.get("cluster_id"))
            if cid in result and isinstance(result[cid], dict):
                result[cid]["confidence"] = "low"
                reason = it.get("reason", "")
                if reason:
                    result[cid]["low_confidence_reason"] = reason
                cand = it.get("candidate_celltype")
                if cand:
                    alts = result[cid].setdefault("alternative_cell_types", [])
                    if cand not in alts:
                        alts.append(cand)
        return result

    async def generate_annotation_validation_questions(
        self,
        cluster_markers: Dict[str, Any],
        annotations: Dict[str, Any],
        tissue: str,
        organism: str,
        triage: Optional[Dict[str, Any]] = None,
        focused_evidence: Optional[Dict[str, Any]] = None,
        database_priors: Optional[Dict[str, Any]] = None,
        metadata_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """PI produces biological validation QUESTIONS (not code). Returns {} on
        failure (caller treats as skip)."""
        self.refresh_runtime_config_if_needed()
        import json as _j
        triage = triage or {}; focused_evidence = focused_evidence or {}
        database_priors = database_priors or {}; metadata_context = metadata_context or {}

        ann_lines = []
        for cid, info in annotations.items():
            if isinstance(info, dict):
                ann_lines.append(
                    f"- Cluster {cid}: {info.get('cell_type','Unknown')} [conf={info.get('confidence','?')}] "
                    f"markers={', '.join(info.get('key_markers', [])[:8])} "
                    f"alt={info.get('alternative_cell_types', [])} reason={info.get('low_confidence_reason','')}")
        ann_text = "\n".join(ann_lines) or "No annotations."

        prompt = f"""You are the PI planning a focused annotation-VALIDATION round for {organism} {tissue}.
Raise biological validation QUESTIONS for clusters whose labels are uncertain, triage-flagged,
technical/QC-suspect, or possibly a state rather than a lineage. Do NOT write code; a bioinformatics
agent will translate your questions into concrete checks against real columns.

## Preliminary annotations
{ann_text}

## Triage from search phase
{_j.dumps(triage, ensure_ascii=False)[:1500]}

## Focused literature evidence
{_j.dumps(focused_evidence, ensure_ascii=False)[:1500]}

## Database priors (marker-based cell-type priors)
{_j.dumps(database_priors, ensure_ascii=False)[:1000]}

## Available analysis metadata (real columns you may reference)
{_j.dumps(metadata_context, ensure_ascii=False)[:1500]}

For each uncertain cluster, ask whether the candidate label is lineally supported, whether it might be
technical/mixed/doublet, whether a state label fits better, or whether exclusion-marker evidence is needed.
You MAY suggest tissue markers from your reasoning, but do not assume columns that aren't listed.

Reply with STRICT JSON only:
{{
  "overall_goal": "Validate preliminary cell annotations before user confirmation.",
  "scope": "annotation_validation_only",
  "tasks": [{{"task_id": "<id>", "cluster_ids": ["<id>"], "question": "<biological question>",
      "biological_concern": "<why>", "suggested_checks": ["<high-level check>"],
      "decision_rule": "<evidence -> label/confidence>"}}],
  "required_outputs": ["per_cluster_validation_summary", "suggested_annotation_changes", "generated_tables_or_figures"],
  "constraints": ["No DEG/pathway/communication/trajectory/velocity.",
                  "No clustering/filtering/normalization changes to the main object."]
}}
If every annotation is high-confidence and unflagged, return {{"scope": "annotation_validation_only", "tasks": []}}."""

        result = self._safe_parse_json(await self.chat(prompt, temperature=0.2), default={})
        if not result or "parse_error" in result:
            return {}
        return result if isinstance(result, dict) else {}

    async def revise_annotations_after_validation(
        self,
        annotations: Dict[str, Any],
        validation_plan: Dict[str, Any],
        validation_result: Dict[str, Any],
        literature_context: str,
        tissue: str,
        organism: str,
    ) -> Dict[str, Any]:
        """PI revises annotations using validation evidence. Returns
        {"annotations": <revised>, "validation_summary_for_user": <str>}. On parse
        failure returns originals unchanged with a generic summary."""
        self.refresh_runtime_config_if_needed()
        import json as _j, copy as _copy
        revised = _copy.deepcopy(annotations)

        ann_lines = [f"- Cluster {cid}: {info.get('cell_type','Unknown')} [conf={info.get('confidence','?')}]"
                     for cid, info in annotations.items() if isinstance(info, dict)]
        ann_text = "\n".join(ann_lines) or "No annotations."

        prompt = f"""You are the PI revising preliminary cell annotations for {organism} {tissue}
using data-grounded validation evidence. Update labels/confidence ONLY where evidence warrants.
Never remove a cluster: if evidence is poor, use a broad or technical/QC label with low confidence.

## Preliminary annotations
{ann_text}

## Validation plan (questions asked)
{_j.dumps(validation_plan, ensure_ascii=False)[:2000]}

## Validation evidence (restricted bioinformatics checks)
{_j.dumps(validation_result, ensure_ascii=False)[:5000]}

## Literature context
{(literature_context or '')[:1500]}

Per cluster you may update: cell_type, subtypes, confidence, evidence, alternative_cell_types, low_confidence_reason.
Reply with STRICT JSON only:
{{
  "revisions": {{"<id>": {{"cell_type": "...", "confidence": "high|medium|low", "evidence": "...",
      "subtypes": [], "alternative_cell_types": [], "low_confidence_reason": "..."}}}},
  "validation_summary_for_user": "<2-4 sentences: what changed and what remains uncertain>"
}}
Only include clusters you are actually changing in "revisions"."""

        parsed = self._safe_parse_json(await self.chat(prompt, temperature=0.2), default={})
        if not isinstance(parsed, dict):
            return {"annotations": revised,
                    "validation_summary_for_user": "Validation completed; no annotation changes were applied."}
        for cid, change in (parsed.get("revisions") or {}).items():
            if cid in revised and isinstance(revised[cid], dict) and isinstance(change, dict):
                for k, v in change.items():
                    if v not in (None, ""):
                        revised[cid][k] = v
        summary = parsed.get("validation_summary_for_user") or \
            "Validation completed; annotations reviewed against data evidence."
        return {"annotations": revised, "validation_summary_for_user": str(summary)}

    async def review_bioinfo_plan(
        self,
        state: AnalysisState,
        plan: 'AnalysisPlan',
        dialogue_round: int = 1,
    ) -> Dict[str, Any]:
        """Review bioinformatics agent's analysis plan.

        Returns:
            Dict with keys: decision ("approved"/"revise"/"reject"), feedback (str),
            modifications (List[Dict] with step_id + suggestion)
        """
        steps_text = "\n".join(
            f"  Step {s.id}: {s.title}\n"
            f"    Method: {s.method}\n"
            f"    Input: {s.input}\n"
            f"    Output: {s.expected_output}\n"
            f"    Rationale: {s.rationale}\n"
            f"    Depends on: {s.depends_on}"
            for s in plan.steps
        )

        decision_points = "\n".join(f"  - {d}" for d in plan.key_decision_points)

        stage_label = (plan.stage or "full")
        stage_scope_section = ""
        if stage_label == "stage1":
            stage_scope_section = """
## Stage Scope (Stage 1)
This plan is for **Stage 1** only. Stage 1 covers: data loading of count matrices, QC, doublet handling, normalization / HVG, batch correction, dimensionality reduction, clustering, and cell type annotation.

Stage 1 must NOT contain steps whose only purpose is to feed Stage 2 analyses, including:
- Loading Loom files or assigning spliced / unspliced layers (RNA velocity belongs to Stage 2)
- Trajectory / pseudotime inputs (e.g. diffusion maps staged for Slingshot / Monocle)
- Cell-cell communication preprocessing (CellChat / liana / NicheNet ligand-receptor tables)
- Pre-staging DEG contrasts, gene signatures, or pathway databases for Stage 2 enrichment

If any step's method, input, or rationale references "for downstream", "to support Stage 2", "for velocity", "for trajectory", "for communication", or any Stage 2 analysis by name, you MUST mark the plan as **revise** and list that step under modifications with the suggestion to defer it to Stage 2."""
        elif stage_label == "stage2":
            stage_scope_section = """
## Stage Scope (Stage 2)
This plan is for **Stage 2**. Stage 2 covers downstream analyses (DEG, GO/KEGG, cell communication, RNA velocity, trajectory, etc.) and may load any data those analyses need. Do NOT flag Stage 2-specific steps as out of scope here."""

        prompt = f"""You are reviewing a bioinformatics analysis plan for a single-cell RNA-seq study.

## Plan Overview
{plan.overview}

## Plan Stage
{stage_label}

## Planned Steps
{steps_text}

## Key Decision Points
{decision_points}

## Context
- Phase: {plan.phase}
- Round: {state.current_round}
- Data: {state.build_experimental_design_summary()}
{stage_scope_section}

## Review Criteria
1. **Stage scope**: Does every step belong in the declared stage? See "Stage Scope" above. Steps that prep data for a later stage are out of scope and require **revise**.
2. **Scientific validity**: Are the planned methods appropriate for the data and biological questions?
3. **Completeness**: Are any critical steps missing for THIS stage?
4. **Logical order**: Do dependencies make sense? Are steps correctly ordered?
5. **Redundancy**: Are there unnecessary or duplicate steps?
6. **Risk**: Are there steps likely to fail or produce unreliable results?

## Decision Options
- **approved**: Plan is in-scope and scientifically sound, no changes needed
- **revise**: Plan needs specific modifications (list them; scope-leak steps must be called out by step_id)
- **reject**: Plan is fundamentally flawed and needs complete redesign (explain why)

Respond in JSON:
{{
    "decision": "approved|revise|reject",
    "feedback": "Overall assessment and reasoning",
    "modifications": [
        {{"step_id": 3, "suggestion": "Use harmony for batch correction instead of BBKNN because..."}},
        ...
    ]
}}

IMPORTANT: Be specific in your feedback. Reference exact step IDs. If you approve, modifications must be empty. If you reject, explain what the bioinformatics agent should rethink."""

        response = await self.chat(prompt, temperature=0.2)
        result = self._safe_parse_json(response, default={})
        
        if not result:
            result = {
                "decision": PI_REVIEW_FALLBACK_DECISION,
                "feedback": "Plan needs review (fallback — PI response could not be parsed)",
                "modifications": [],
            }

        result.setdefault("decision", PI_REVIEW_FALLBACK_DECISION)
        result.setdefault("feedback", "")
        result.setdefault("modifications", [])
        
        logger.info(f"[PI] Plan review decision: {result['decision']} (round {dialogue_round})")
        return result

    async def generate_final_plan_from_dialogue(
        self,
        analysis_state: AnalysisState,
        dialogue_history: List[Dict[str, Any]],
    ) -> 'AnalysisPlan':
        """P1-10: Unilaterally generate a final analysis plan based on the
        planning dialogue so far. Used as a fallback when MAX_PLANNING_ROUNDS
        is exhausted without bioinfo-PI agreement — PI decides alone instead
        of looping further.

        Returns an AnalysisPlan whose status is "pi_reviewing"; the workflow
        will then mark plan_approved=True so the user-review gate sees a
        concrete plan.
        """
        from ..models.schemas import AnalysisPlan, PlanStep

        phase = analysis_state.analysis_phase.value
        stage = "full"
        if analysis_state.current_plan is not None and analysis_state.current_plan.stage:
            stage = analysis_state.current_plan.stage

        recent = dialogue_history[-5:] if dialogue_history else []
        history_summary = "\n\n".join(
            f"Round {entry.get('round', '?')}: PI decision={entry.get('pi_decision', '?')}\n"
            f"  PI feedback: {str(entry.get('pi_feedback', ''))[:400]}\n"
            f"  Plan attempted: {str(entry.get('plan'))[:600]}"
            for entry in recent
        ) or "(no prior dialogue)"

        prompt = f"""You are the PI deciding on a FINAL analysis plan unilaterally because
the bioinformatics agent and PI could not reach agreement within the maximum
allowed planning rounds. You must now commit to a plan based on the dialogue
history below — no more discussion is permitted.

## Phase
{phase}

## Stage
{stage}

## Experimental Design
{analysis_state.build_experimental_design_summary()}

## Dialogue History (most recent rounds)
{history_summary}

## Your Task
Produce a final AnalysisPlan in JSON form. Synthesize the best ideas from the
dialogue, resolve any disagreements in favor of scientific soundness, and keep
the plan executable within the current stage's scope.

Respond ONLY with JSON in this schema:
{{
    "overview": "Short summary of the plan",
    "steps": [
        {{"id": 1, "title": "...", "method": "...", "input": "...",
          "expected_output": "...", "rationale": "...", "depends_on": []}},
        ...
    ],
    "key_decision_points": ["...", "..."],
    "expected_runtime": "~N steps"
}}"""

        response = await self.chat(prompt, temperature=0.2)
        parsed = self._safe_parse_json(response, default={})

        steps: List[PlanStep] = []
        if isinstance(parsed, dict):
            for s in parsed.get("steps", []) or []:
                if not isinstance(s, dict):
                    continue
                steps.append(PlanStep(
                    id=s.get("id", len(steps) + 1),
                    title=s.get("title", ""),
                    method=s.get("method", ""),
                    input=s.get("input", ""),
                    expected_output=s.get("expected_output", ""),
                    rationale=s.get("rationale", ""),
                    depends_on=s.get("depends_on", []) or [],
                ))

        if not steps:
            raise RuntimeError(
                "PI failed to generate a final plan from dialogue (no parseable steps)"
            )

        overview = ""
        decision_points: List[str] = []
        expected_runtime = f"~{len(steps)} steps"
        if isinstance(parsed, dict):
            overview = parsed.get("overview", "") or ""
            decision_points = parsed.get("key_decision_points", []) or []
            expected_runtime = parsed.get("expected_runtime", expected_runtime) or expected_runtime

        plan = AnalysisPlan(
            phase=phase,
            stage=stage,
            overview=overview,
            steps=steps,
            key_decision_points=decision_points,
            expected_runtime=expected_runtime,
            status="pi_reviewing",
            revision_of=(analysis_state.current_plan.id
                         if analysis_state.current_plan else None),
            revision_reason="MAX_PLANNING_ROUNDS exhausted — PI finalized unilaterally",
        )
        logger.info(
            f"[PI] Generated final plan unilaterally from dialogue: "
            f"{len(steps)} steps, phase={phase}, stage={stage}"
        )
        return plan

    async def process_user_plan_feedback(
        self,
        user_input: str,
        current_plan: 'AnalysisPlan',
    ) -> Dict[str, Any]:
        """Process user feedback on an analysis plan.

        PI first judges the user's intent, then decides next action.

        Returns:
            Dict with keys: 
            - intent: "confirm" | "modify" | "unclear"
            - action: "proceed_to_execution" | "revise_with_bioinfo" | "clarify_with_user"
            - guidance_for_bioinfo: str (only if action=revise_with_bioinfo)
            - clarification_question: str (only if action=clarify_with_user)
        """
        steps_summary = "\n".join(
            f"  Step {s.id}: {s.title} — {s.method}"
            for s in current_plan.steps
        )

        prompt = f"""A user has reviewed a bioinformatics analysis plan and provided feedback.

## Current Plan Summary
{steps_summary}

## User's Feedback
{user_input}

## Your Task
First, determine the user's INTENT:
- **confirm**: User explicitly approves the plan (keywords: "确认", "同意", "没问题", "可以", "approve", "confirm", "ok", "good", "looks good") or says to proceed/start
- **modify**: User wants specific changes to the plan (adding/removing/changing steps, different methods, etc.)
- **unclear**: You cannot confidently determine if the user is confirming or requesting changes

Then, based on intent, decide your ACTION:
- If **confirm** → action="proceed_to_execution", no further dialogue with bioinformatics needed
- If **modify** → action="revise_with_bioinfo", provide guidance_for_bioinfo summarizing what to change
- If **unclear** → action="clarify_with_user", provide a clarification_question to ask

## CRITICAL RULES
- If the user says ANYTHING that means "yes/go ahead/proceed/start/execute", treat it as **confirm**
- Do NOT overthink — if user says "好的" or "可以" or "go", just confirm
- Only classify as **modify** if the user gives SPECIFIC change requests
- Only classify as **unclear** if genuinely ambiguous

Respond in JSON:
{{
    "intent": "confirm|modify|unclear",
    "action": "proceed_to_execution|revise_with_bioinfo|clarify_with_user",
    "guidance_for_bioinfo": "Specific changes to make (only for revise_with_bioinfo, empty otherwise)",
    "clarification_question": "Question to ask user (only for clarify_with_user, empty otherwise)"
}}"""

        response = await self.chat(prompt, temperature=0.1)
        result = self._safe_parse_json(response, default={})
        
        if not result:
            result = {
                "intent": "confirm",
                "action": "proceed_to_execution",
                "guidance_for_bioinfo": "",
                "clarification_question": "",
            }
        
        result.setdefault("intent", "confirm")
        result.setdefault("action", "proceed_to_execution")
        result.setdefault("guidance_for_bioinfo", "")
        result.setdefault("clarification_question", "")
        
        return result

    async def judge_plan_deviation(
        self,
        deviation_description: str,
        current_plan: 'AnalysisPlan',
        affected_step_ids: List[int],
    ) -> Dict[str, Any]:
        """Judge whether a mid-execution deviation is minor or major.

        Returns:
            Dict with keys:
            - severity: "minor" | "major"
            - reasoning: str
            - suggested_action: str
            - needs_user_popup: bool
        """
        affected_steps = [s for s in current_plan.steps if s.id in affected_step_ids]
        affected_text = "\n".join(
            f"  Step {s.id}: {s.title} ({s.method})"
            for s in affected_steps
        )

        prompt = f"""During bioinformatics analysis execution, a deviation from the plan was discovered.

## Deviation Description
{deviation_description}

## Affected Plan Steps
{affected_text}

## Remaining Plan Overview
{current_plan.overview}

## Your Task
Classify this deviation as **minor** or **major**:

**Minor** (no user popup needed):
- Switching between equivalent analysis methods (e.g., gseapy → clusterProfiler)
- Adjusting parameters based on data characteristics
- Adding/removing a visualization step
- Skipping a step that data genuinely doesn't support
- Adding a small complementary analysis

**Major** (requires user popup):
- The core hypothesis/direction needs to change (e.g., "thought it was T cell driven, actually myeloid")
- A completely new analysis branch is needed (e.g., "need to add trajectory analysis")
- A core objective of the plan must be removed
- The fundamental experimental design interpretation has changed

Respond in JSON:
{{
    "severity": "minor|major",
    "reasoning": "Brief explanation of your classification",
    "suggested_action": "What the bioinformatics agent should do next",
    "needs_user_popup": true|false
}}"""

        response = await self.chat(prompt, temperature=0.1)
        result = self._safe_parse_json(response, default={})
        
        if not result:
            result = {}
        
        result.setdefault("severity", "minor")
        result.setdefault("reasoning", "Unable to classify (fallback)")
        result.setdefault("suggested_action", "Proceed with best judgment")
        result.setdefault("needs_user_popup", False)
        
        return result

    # Shared "Rules:" footer for PI-authored Deep Research briefs. The
    # citation-format directive is intentionally excluded — the Literature Agent
    # appends its own (parser-coupled) footer.
    _BRIEF_RULES = (
        "Rules:\n"
        "- Write the brief as direct instructions to a deep-research agent (what to search and synthesize).\n"
        "- Do NOT include any citation or reference-format instructions.\n"
        "- Do NOT quote or restate the full experimental background."
    )

    @staticmethod
    def _summarize_cluster_markers(
        cluster_markers: Dict[str, List[str]]
    ) -> Tuple[int, str]:
        """Summarize markers across ALL clusters → (n_clusters, markers_summary)."""
        n_clusters = len(cluster_markers)
        markers_summary = "\n".join(
            f"  Cluster {cid}: {', '.join(genes[:8])}"
            for cid, genes in cluster_markers.items()
        )
        return n_clusters, markers_summary

    @staticmethod
    def _brief_background_block(state: Optional[AnalysisState]) -> str:
        """Background section for a PI brief: experimental goal as reference only ('' if none)."""
        if state and getattr(state, "initial_instruction", ""):
            return (
                "\n\nBackground (reference only — decide what is relevant, do NOT "
                f"copy verbatim):\n{state.initial_instruction[:500]}"
            )
        return ""

    async def generate_annotation_search_query(
        self,
        cluster_markers: Dict[str, List[str]],
        organism: str,
        tissue: str,
        state: Optional[AnalysisState] = None,
    ) -> str:
        """Generate an optimized PubMed search query for cell type annotation.

        Uses PI's strong model to create high-quality queries with proper MeSH terms
        and Boolean operators, instead of delegating to Literature Agent's weak model.
        """
        logger.info(f"[PI] Generating annotation search query for {organism} {tissue}")

        n_clusters, markers_summary = self._summarize_cluster_markers(cluster_markers)

        # Experimental background from state
        background = ""
        if state and state.initial_instruction:
            background = f"\nExperimental background:\n{state.initial_instruction[:500]}"

        prompt = f"""You are a principal investigator planning a PubMed literature search to identify cell types
in a single-cell RNA sequencing dataset.

Organism: {organism}
Tissue: {tissue}
Number of clusters: {n_clusters}
{background}

Cluster marker genes (top genes per cluster):
{markers_summary}

Generate an optimized PubMed search query to find literature about cell types in {organism} {tissue}
that will help annotate these clusters. The query should retrieve articles about:
- Known cell type markers and signatures in this tissue
- Single-cell RNA sequencing studies of this tissue
- Cell type classification and identification

Rules for PubMed query:
1. Use MeSH terms where appropriate (e.g., "Single-Cell Analysis"[MeSH], plus a tissue MeSH term matching the dataset's organ such as "Liver"[MeSH], "Kidney"[MeSH], "Lung"[MeSH], "Brain"[MeSH])
2. Use Boolean operators (AND, OR, NOT)
3. Keep the query focused but not overly restrictive — aim for 20-200 results
4. Include species and tissue/organ as filters
5. Pick 2-3 of the most informative marker genes (genes that are highly specific to known cell types)
6. Use field tags like [Title/Abstract], [MeSH] when helpful
7. Prefer broader terms over highly specific ones to avoid zero-result queries

Return ONLY the PubMed query string, nothing else."""

        response = await self.chat_without_history(prompt, temperature=0.2)
        # Clean up response
        query = response.strip().strip('"').strip("'").strip("`")
        query = re.sub(r'^```\w*\n?', '', query)
        query = re.sub(r'\n?```$', '', query)
        return query.strip()

    async def generate_annotation_research_brief(
        self,
        cluster_markers: Dict[str, List[str]],
        organism: str,
        tissue: str,
        state: Optional[AnalysisState] = None,
    ) -> str:
        """Write a FOCUSED Deep Research brief for cell-type annotation.

        Produces prose instructions for the deep-research model, using the
        experimental background as reference only (no injection). Returns the
        stripped brief; an empty string signals the caller to fall back.
        """
        n_clusters, markers_summary = self._summarize_cluster_markers(cluster_markers)
        background = self._brief_background_block(state)

        prompt = f"""You are a principal investigator writing a Deep Research brief to identify cell types
in a single-cell RNA sequencing dataset.

Organism: {organism}
Tissue: {tissue}
Number of clusters: {n_clusters}

Cluster marker genes (top genes per cluster):
{markers_summary}{background}

This is a cell-type annotation task. Decide the search scope, then write a FOCUSED Deep Research
brief about marker genes and cell-type signatures for {organism} {tissue} that will help map these
clusters to known cell types. Do NOT expand into disease mechanisms, treatments, or
experimental-group comparisons unless a specific marker directly requires it.

{self._BRIEF_RULES}

Return ONLY the research brief text."""

        response = await self.chat_without_history(prompt, temperature=0.3)
        return (response or "").strip()

    async def generate_round_research_brief(
        self,
        topic: str,
        current_round_findings: Optional[List[str]] = None,
        state: Optional[AnalysisState] = None,
    ) -> str:
        """Write a Deep Research brief for a regular research round.

        PI judges breadth (broad survey vs focused deep-dive) from the topic and
        findings, drawing on experimental background only where relevant. Returns
        the stripped brief; an empty string signals the caller to fall back.
        """
        findings_block = ""
        if current_round_findings:
            joined = "; ".join(current_round_findings[:5])
            findings_block = f"\n\nFindings so far this round:\n{joined}"

        background = self._brief_background_block(state)

        prompt = f"""You are a principal investigator writing a Deep Research brief for a biomedical
literature review.

Research direction (topic):
{topic}{findings_block}{background}

Given the research direction and the findings so far, judge whether this needs a broad survey or a
focused deep-dive, then write the Deep Research brief accordingly. Draw on the experimental
background only where it is directly relevant to the topic.

{self._BRIEF_RULES}

Return ONLY the research brief text."""

        response = await self.chat_without_history(prompt, temperature=0.3)
        return (response or "").strip()

    async def evaluate_annotation_search_results(
        self,
        cluster_markers: Dict[str, List[str]],
        organism: str,
        tissue: str,
        articles: List[Dict[str, str]],
        search_history: List[Dict[str, Any]],
        state: Optional[AnalysisState] = None,
    ) -> Dict[str, Any]:
        """Evaluate PubMed search results with four-bucket triage.

        Uses PI's strong model to classify every cluster into:
        covered / rare_unresolved / suspected_contamination / context_dependent_state.
        """
        markers_summary = "\n".join(
            f"  Cluster {cid}: {', '.join(genes[:8])}"
            for cid, genes in cluster_markers.items()
        )

        articles_summary = "\n".join(
            f"- [{a.get('year', 'N/A')}] {a['title']}\n  Abstract: {a.get('abstract', 'N/A')[:200]}"
            for a in articles[:15]
        )

        previous_queries = "\n".join(
            f"  Round {h['round']}: {h['query']}"
            for h in search_history
        )

        background = ""
        if state is not None and getattr(state, "initial_instruction", ""):
            background = f"\n## Experimental Context (use to judge condition-driven states)\n{state.initial_instruction[:500]}\n"
            try:
                background += state.build_experimental_design_summary()
            except Exception:
                pass

        prompt = f"""You are a principal investigator evaluating whether PubMed search results are sufficient
to annotate cell clusters in a {organism} {tissue} single-cell/single-nucleus RNA sequencing dataset.

## Cluster Marker Genes (need to identify cell types for ALL clusters)
{markers_summary}
{background}
## Previous Queries (DO NOT repeat these)
{previous_queries}

## Articles Found ({len(articles)} total, showing first 15 with abstracts)
{articles_summary}

Classify EVERY cluster into exactly one of four buckets, judging against the experimental
context (treatment groups / conditions) and tissue biology:
- covered: marker matches a known cell type already well supported by retrieved literature.
- rare_unresolved: a real, tissue-plausible but rare cell type with insufficient literature.
- suspected_contamination: markers biologically implausible for this tissue, OR technical-noise
  signature (batch effect, low-quality cluster, high mitochondrial/ribosomal fraction, doublet, ambient RNA).
- context_dependent_state: a KNOWN cell type's state variant driven by experimental context
  (e.g. condition-skewed cluster), NOT an independent new type — physiological reasoning suffices,
  no literature deep-dive. Do NOT use this bucket for tumor cells under study, developmental
  time-series target stages, or models where the baseline is equally treated.

Respond in JSON format:
{{
    "sufficient": true or false,
    "reasoning": "which clusters lack literature support",
    "refined_query": "a NEW PubMed query (MUST differ from all previous queries)",
    "missing_aspects": ["cell types / markers not yet covered"],
    "covered": ["<cluster_id>", ...],
    "rare_unresolved": [{{"cluster_id": "..", "candidate_celltype": "Cell Ontology term or null",
        "candidate_celltype_confidence": "high|medium|low", "synonyms": ["CL official aliases, may be empty"],
        "top_markers": ["2-3 markers"], "triage_confidence": "high|medium|low",
        "evidence_need": "why the main loop cannot cover this cluster + what evidence is needed"}}],
    "suspected_contamination": [{{"cluster_id": "..", "candidate_celltype": "term or null",
        "top_markers": ["markers"], "triage_confidence": "high|medium|low",
        "reason": "why implausible / which technical-noise signature",
        "evidence_need": "what to verify"}}],
    "context_dependent_state": [{{"cluster_id": "..", "parent_celltype": "known type",
        "context": "the experimental condition driving the state", "top_markers": ["markers"],
        "triage_confidence": "high|medium|low", "state_evidence_note": "optional light background, NOT a deep-dive"}}]
}}

IMPORTANT:
- Classify ALL clusters; do not omit any cluster from the four buckets.
- refined_query MUST be valid PubMed (AND/OR), and MUST NOT duplicate any previous query.
- candidate_celltype MUST be a normalized Cell Ontology term; if you cannot, set it to null — do NOT invent.
- synonyms may be empty; do NOT fabricate synonyms when unsure.
- Every non-covered entry MUST include evidence_need."""

        response = await self.chat_without_history(prompt, temperature=0.3)
        result = self._safe_parse_json(response, default={})
        if not result or "sufficient" not in result:
            result = {"sufficient": True}
        for key in ("covered", "rare_unresolved", "suspected_contamination", "context_dependent_state"):
            result.setdefault(key, [])

        # --- Deterministic post-processing: normalize cluster IDs and enforce one bucket per cluster ---
        all_ids = {str(c) for c in cluster_markers.keys()}
        buckets = ("rare_unresolved", "suspected_contamination", "context_dependent_state")

        result["covered"] = [str(c) for c in result.get("covered", []) if str(c) in all_ids]
        covered_ids = set(result["covered"])

        cleaned_by_bucket = {bucket: [] for bucket in buckets}
        assigned_noncovered = set()

        # Higher priority buckets go first. Context states should not also be rare/contamination;
        # suspected contamination should not also be rare unresolved.
        priority_order = ("context_dependent_state", "suspected_contamination", "rare_unresolved")
        for bucket in priority_order:
            for item in result.get(bucket, []):
                if not isinstance(item, dict):
                    continue
                cid = str(item.get("cluster_id"))
                if cid not in all_ids:
                    continue
                if cid in covered_ids:
                    continue
                if cid in assigned_noncovered:
                    continue
                cleaned_by_bucket[bucket].append(item)
                assigned_noncovered.add(cid)

        for bucket in buckets:
            result[bucket] = cleaned_by_bucket[bucket]

        triaged = covered_ids | assigned_noncovered
        result["untriaged_cluster_ids"] = sorted(all_ids - triaged)
        result["triage_complete"] = bool(all_ids) and not (all_ids - triaged)
        result["triage_low_confidence"] = bool(all_ids) and (len(assigned_noncovered) / len(all_ids) > 0.40)
        return result

    async def annotate_clusters_from_literature(
        self,
        cluster_markers: Dict[str, List[str]],
        organism: str,
        tissue: str,
        literature_report: str,
        triage: Optional[Dict[str, Any]] = None,
        focused_evidence: Optional[Dict[str, Any]] = None,
        database_priors: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Annotate cell clusters by matching marker genes to literature-derived signatures.
        """
        self.refresh_runtime_config_if_needed()
        logger.info(f"[PI] Annotating {len(cluster_markers)} clusters from literature for {organism} {tissue}")

        tissue_part = f" {tissue}" if tissue else ""
        markers_text = "\n".join(
            f"Cluster {cid}: {', '.join(genes[:15])}"
            for cid, genes in cluster_markers.items()
        )

        triage = triage or {}
        focused_evidence = focused_evidence or {}
        hint_lines: List[str] = []
        for it in triage.get("context_dependent_state", []):
            cid = it.get("cluster_id")
            hint_lines.append(
                f"[Note: Cluster {cid} classified as context-dependent state — parent: "
                f"{it.get('parent_celltype','?')}, driven by {it.get('context','experimental context')}. "
                f"Annotate as a state variant of the parent, no literature deep-dive expected.]")
        for bucket in ("rare_unresolved", "suspected_contamination"):
            for it in triage.get(bucket, []):
                cid = it.get("cluster_id")
                fe = focused_evidence.get(str(cid)) or focused_evidence.get(cid) or {}
                status = fe.get("status")
                cand = it.get("candidate_celltype") or "marker-defined population"
                if status in ("strong", "limited"):
                    arts = "\n".join(f"- {a.get('title','')}: {a.get('abstract','')[:300]}"
                                     for a in fe.get("articles", [])[:5])
                    tag = "" if status == "strong" else " (limited evidence)"
                    hint_lines.append(f"[Focused evidence for Cluster {cid} (candidate: {cand}){tag}]\n{arts}")
                elif status == "contradictory":
                    if bucket == "rare_unresolved":
                        hint_lines.append(f"[Cluster {cid}: focused evidence contradicts candidate {cand}; "
                                          f"consider alternative cell identities]")
                    else:
                        hint_lines.append(f"[Cluster {cid}: focused evidence supports suspected contamination "
                                          f"or non-target origin]")
                else:
                    reason = fe.get("exhaustion_reason")
                    error = fe.get("error")
                    if reason in ("search_failed", "judge_failed"):
                        detail = f" Error: {error}" if error else ""
                        hint_lines.append(
                            f"[Focused evidence unavailable for Cluster {cid} due to runtime failure "
                            f"({reason}).{detail} Do not interpret as absence of literature.]"
                        )
                    elif reason == "no_pubmed_results":
                        hint_lines.append(
                            f"[Focused search found no PubMed results for Cluster {cid} — this is evidence scarcity, "
                            f"not a runtime failure.]"
                        )
                    elif reason == "empty_query":
                        hint_lines.append(
                            f"[Focused search skipped for Cluster {cid} — no candidate term or marker query terms were available.]"
                        )
                    elif reason == "no_relevant_evidence":
                        arts = "; ".join(a.get("title", "") for a in fe.get("articles", [])[:3])
                        suffix = f" Retrieved articles were not relevant: {arts}" if arts else " Retrieved articles were not relevant."
                        hint_lines.append(f"[Focused evidence exhausted for Cluster {cid}.{suffix}]")
                    else:
                        hint_lines.append(f"[Focused evidence exhausted for Cluster {cid} — no relevant literature found]")
        hints_block = ("\n\n=== PER-CLUSTER TRIAGE & FOCUSED EVIDENCE ===\n" + "\n".join(hint_lines) + "\n") if hint_lines else ""
        priors_block = _format_database_priors_block(database_priors)

        prompt = f"""Based on the following literature review about cell types in {organism}{tissue_part}, \
annotate each cell cluster by matching its marker genes to known cell type signatures.

=== LITERATURE REVIEW ===
{literature_report[:12000]}
{hints_block}{priors_block}
=== CLUSTER MARKER GENES (from sc.tl.rank_genes_groups) ===
{markers_text}

For each cluster, provide:
1. The most likely cell type identity
2. Possible subtypes if applicable
3. Key marker genes that support this annotation (from the cluster's markers)
4. Confidence level (high/medium/low) based on how well markers match literature
5. Brief evidence/reasoning citing the literature

IMPORTANT:
- Be specific about cell types (e.g., "CD8+ cytotoxic T cells" not just "T cells")
- If a cluster has ambiguous markers, list the top 2 candidates
- Consider the tissue context ({tissue}) when making assignments
- Use "Unknown" only if markers truly don't match any known cell type

Return a JSON object with this EXACT structure:
{{
    "annotations": {{
        "<cluster_id>": {{
            "cell_type": "Primary cell type name",
            "subtypes": ["possible subtype 1", "possible subtype 2"],
            "key_markers": ["gene1", "gene2", "gene3"],
            "confidence": "high|medium|low",
            "evidence": "Brief reasoning with literature support"
        }}
    }},
    "tissue_summary": "Brief summary of expected cell types in this tissue"
}}
"""

        response = await self.chat_without_history(prompt, temperature=0.2)
        parsed = self._safe_parse_json(response, default={})

        if parsed and "annotations" in parsed:
            return parsed

        # Retry once with a more explicit prompt if first attempt fails
        retry_prompt = prompt + "\n\nIMPORTANT: Respond with ONLY the JSON object, no additional text or markdown formatting."
        response_retry = await self.chat_without_history(retry_prompt, temperature=0.2)
        parsed_retry = self._safe_parse_json(response_retry, default={})

        if parsed_retry and "annotations" in parsed_retry:
            return parsed_retry

        # Both attempts failed
        logger.warning(f"[PI] Failed to parse annotation matching JSON after 2 attempts. Response preview: {response[:500]}")
        return {
            "status": "error",
            "error": "Failed to parse structured annotations after 2 attempts",
            "raw_response": response[:2000],
            "annotations": {},
            "tissue_summary": ""
        }

    async def generate_exploration_report(
        self,
        state: AnalysisState,
        round_result: RoundResult
    ) -> ExplorationReport:
        """
        Generate structured exploration report (Report A) after Phase 1.
        Delegates to generate_raw_direction_candidates and score_directions_with_evidence.
        """
        logger.info(f"[PI] Generating exploration report")
        raw_candidates = await self.generate_raw_direction_candidates(state, round_result)
        fallback_evidence = build_fallback_direction_evidence(
            raw_candidates,
            status="skipped",
            reason="Literature evidence was not provided to direct generate_exploration_report call.",
        )
        return await self.score_directions_with_evidence(state, round_result, raw_candidates, fallback_evidence)

    def _format_cluster_annotation_summary(self, state) -> str:
        if not (state.annotation_state and state.annotation_state.cluster_annotations):
            return "No annotation data"
        lines = [
            f"- Cluster {ca.cluster_id}: {ca.cell_type} "
            f"[confidence: {ca.confidence}] markers: {', '.join(ca.key_markers[:10])}"
            for ca in state.annotation_state.cluster_annotations
        ]
        return "\n".join(lines)

    async def generate_raw_direction_candidates(
        self,
        state: AnalysisState,
        round_result: RoundResult,
    ) -> List[Dict[str, Any]]:
        """
        Generate raw research direction candidates without final scoring.
        Called by generate_exploration_report as a separate substep.
        """
        logger.info(f"[PI] Generating raw direction candidates")
        self.reset_history()

        ann_summary = self._format_cluster_annotation_summary(state)

        mech_programs = getattr(state, "data_driven_mechanism_programs", []) or []
        mech_block = ""
        if mech_programs:
            lines = ["## Data-Driven Mechanism Programs",
                     "Use these programs as candidate evidence, not as conclusions.",
                     "Do not overclaim named mechanisms unless supported by data and literature.",
                     "Prefer directions grounded in concrete observations and cross-modal support.",
                     "For every mechanism-derived direction, include alternative hypotheses and missing evidence.",
                     ""]
            for p in mech_programs[-8:]:
                lines.append(f"- {p.to_prompt_text(style='direction', max_observations=2, max_genes=12, max_missing=2)}")
            mech_block = "\n".join(lines)

        prompt = f"""Based on the Round 1 exploration results, propose 3-5 raw research direction candidates.

## Project Context
- Name: {state.name}
- Research Objective: {state.initial_instruction}
- Species: {state.species or "Not specified"}
{state.build_experimental_design_summary()}

## Round 1 Results
**Bioinformatics findings**: {json.dumps(round_result.bioinformatics_findings[:15], ensure_ascii=False)}
**Pathology findings**: {json.dumps(round_result.pathology_findings[:5], ensure_ascii=False)}
**Biochemistry findings**: {json.dumps(round_result.biochemistry_findings[:5], ensure_ascii=False)}
**Literature findings**: {json.dumps(round_result.literature_findings[:5], ensure_ascii=False)}
**PI Summary**: {round_result.pi_summary[:1000]}

## Cell Annotations
{ann_summary}

{mech_block}

For each candidate, provide ALL of the following fields:
- direction_id: Unique identifier like "dir_001"
- direction: A specific, actionable research question
- core_observation: Concrete Round 1 data observation motivating the candidate. Must cite specific data elements such as cluster ID, cell type, gene, pathway, fold-change, p-value, correlation, module score, or sample-level observation.
- main_hypothesis: Primary testable hypothesis.
- alternative_hypotheses: At least two plausible alternatives unless the data clearly excludes alternatives. Consider different transcriptional regulators, different ions/metals, different biological process classes, different causal direction, and technical artifacts.
- artifact_risks: Specific technical or interpretive risks such as batch effect, doublets, low-quality cluster, annotation uncertainty, ambient RNA, dropout, or confounded design.
- rationale: Why this direction is interesting based on the data
- supporting_data: What specific data supports this direction
- key_genes: List of relevant genes
- key_cell_types: List of relevant cell types
- key_pathways: List of relevant pathways
- key_uncertainties: What remains uncertain or needs validation
- suggested_literature_topics: Natural-language topics for literature search
- suggested_validation_analysis: Computational follow-up analyses.
- suggested_wet_lab_validation: Wet-lab validation experiments.
- decision_criteria: Results that would support or weaken the hypothesis.
- biology_expectedness: one of "expected", "partially_expected", "unexpected", "unknown"
- functional_impact_hypothesis: Brief hypothesis about functional/clinical impact
- prerequisite_dependencies: List of direction_ids that should be investigated first (empty list if none)
- organism: {state.species or "Not specified"}
- tissue: Tissue context from the experiment
- experimental_context: The experimental design context

Do not produce generic core_observation text like "the data suggest interesting patterns". The core_observation must name the specific data element that triggered the direction.

IMPORTANT: Do NOT assign final innovation, literature novelty, data reliability, or priority scores. Those will be evaluated in a separate step with literature evidence.

Return a JSON object with a single key "candidates" containing the list of candidate objects:
{{
    "candidates": [
        {{
            "direction_id": "dir_001",
            "direction": "...",
            "hypothesis": "...",
            ...
        }}
    ]
}}
"""
        response = await self.chat(prompt, temperature=0.3)
        data = self._safe_parse_json(response, default={})
        candidates = data.get("candidates", [])
        if not candidates:
            logger.warning("[PI] Raw direction candidates empty, using fallback")
            candidates = [{
                "direction_id": "dir_001",
                "direction": "Continue analysis based on Round 1 findings",
                "hypothesis": "Further exploration needed",
                "rationale": "No specific candidates generated",
                "supporting_data": "",
                "key_genes": [],
                "key_cell_types": [],
                "key_pathways": [],
                "key_uncertainties": [],
                "suggested_literature_topics": [],
                "suggested_methods": [],
                "biology_expectedness": "unknown",
                "functional_impact_hypothesis": "",
                "prerequisite_dependencies": [],
                "organism": state.species or "Not specified",
                "tissue": "",
                "experimental_context": "",
            }]
        return candidates

    async def score_directions_with_evidence(
        self,
        state: AnalysisState,
        round_result: RoundResult,
        raw_candidates: List[Dict[str, Any]],
        literature_evidence: List[Dict[str, Any]],
    ) -> ExplorationReport:
        """
        Score raw direction candidates with literature evidence to produce a complete ExplorationReport.
        """
        logger.info(f"[PI] Scoring {len(raw_candidates)} direction candidates with evidence")
        self.reset_history()

        ann_summary = self._format_cluster_annotation_summary(state)

        prompt = f"""You are the PI scoring research direction candidates with literature evidence.

## Project Context
- Name: {state.name}
- Research Objective: {state.initial_instruction}

## Round 1 Results
**Bioinformatics findings**: {json.dumps(round_result.bioinformatics_findings[:15], ensure_ascii=False)}
**Pathology findings**: {json.dumps(round_result.pathology_findings[:5], ensure_ascii=False)}
**Biochemistry findings**: {json.dumps(round_result.biochemistry_findings[:5], ensure_ascii=False)}
**Literature findings**: {json.dumps(round_result.literature_findings[:5], ensure_ascii=False)}
**PI Summary**: {round_result.pi_summary[:1000]}

## Cell Annotations
{ann_summary}

## Raw Direction Candidates
{json.dumps(raw_candidates, ensure_ascii=False, indent=2)}

## Literature Evidence
{json.dumps(literature_evidence, ensure_ascii=False, indent=2) if literature_evidence else "No literature evidence provided."}

### PI SCORING RULES

Apply ALL of the following scoring rules:

1. **Data reliability is the baseline**: If data_reliability=low, this direction cannot be a top recommendation unless the user explicitly chooses it.
2. **Failed literature confidence forces unknown novelty**: If literature_confidence=failed, literature_novelty must be "unknown".
3. **Speculative combinations**: High literature_novelty with low data_reliability is speculative and must not be the default recommendation; add a risk flag.
4. **Strong combination dominates**: high data reliability + high data novelty + high mechanistic importance should dominate overall_priority.
5. **Complementarity**: when recommending multiple directions, they must be complementary (e.g., one describes network architecture, one tests causal origin), not redundant.
6. **Independence**: functional_impact must be assessed separately from mechanistic_importance.
7. **Independence**: biology_expectedness must be assessed separately from literature_novelty and data_novelty.
8. **Prerequisites**: if a direction has prerequisites, its prerequisite direction must also be selected or explicitly noted as already satisfied by prior results.
9. **Composite priority**: overall_priority derives from the interaction of data_reliability, data_novelty, and functional_impact; when dimensions conflict, give explicit reasoning.
10. **Conflict reconciliation**: when literature_novelty and biology_expectedness conflict, reconcile the discrepancy explicitly in literature_support_summary, data_support_summary, or risk_flags.
11. **Negative-validation caveat**: when an evidence record has negative_validation_confidence=medium, any literature_novelty=high must be explicitly justified in literature_support_summary because at least one evidence axis is incomplete.

### SCORING DIMENSIONS

For each candidate, score ALL of these dimensions:

1. **data_reliability**: "high" / "medium" / "low" — Based on sample sizes, effect sizes, statistical significance, and data quality.
2. **literature_novelty**: "high" / "medium" / "low" / "unknown" — How well-studied is this area? "unknown" only when literature evidence failed.
3. **biology_expectedness**: "expected" / "partially_expected" / "unexpected" / "unknown" — Given known pathways/mechanisms, is the finding expected, partly expected, or unexpected? This is PI reasoning, not PubMed coverage.
4. **data_novelty**: "high" / "medium" / "low" — How unexpected or surprising is this finding in the data?
5. **mechanistic_importance**: "high" / "medium" / "low" — How central is the proposed mechanism to the phenotype/disease?
6. **functional_impact**: "high" / "medium" / "low" — What is the likely functional or clinical impact?
7. **data_sufficiency**: "sufficient" / "partial" / "insufficient" — Is there enough data to support this direction?
8. **methodological_gap**: "none" / "minor" / "major" — What methodological gaps exist?

Also derive:
- **overall_priority**: "high" / "medium" / "low" — Composite priority based on ALL dimensions.
- **risk_flags**: List of specific risk flags (e.g., "cross_species_extrapolation", "low_statistical_power").
- **prerequisite_dependencies**: Which other direction_ids must be completed first.
- **evidence_ref**: Must match the direction_id from literature evidence.

Scoring rubric:

biological_plausibility:
- high: mechanism is consistent with established biology and direct pathway/literature evidence.
- medium: biologically possible but depends on indirect or analogous evidence.
- low: contradicts known biology, lacks meaningful support, or requires a novel pathway with no precedent.

validation_feasibility:
- high: routine assays such as qPCR, Western, IF/IHC, ELISA, standard flow cytometry, or standard perturbation.
- medium: specialized equipment or expertise such as ICP-MS, Seahorse, FACS sorting, spatial transcriptomics, advanced imaging, or animal generation.
- low: technically prohibitive, unavailable reagents/models, or cannot distinguish competing hypotheses.

If direction evidence contains novelty_status=known_area, literature_novelty should usually be low or medium unless novelty comes from a new model, species, tissue, cell type, perturbation, or validation strategy.
If direction evidence contains novelty_status=unknown_due_to_timeout or unknown_due_to_failure, literature_novelty must be unknown and recommendation must be conservative.
Support evidence and contradictory evidence must have equal weight in the scoring decision.

Finally, produce the **direction_dependency_graph** — a mapping from each direction_id to a list of direction_ids it depends on (from prerequisite_dependencies).

Return a complete structured report JSON:
{{
    "cluster_overview": "...",
    "data_quality_warnings": [...],
    "strongest_heterogeneity": [...],
    "key_pathway_changes": [...],
    "abnormal_markers": [...],
    "low_confidence_annotations": [...],
    "pi_direction_assessment": "...",
    "research_direction_candidates": [
        {{
            "direction_id": "dir_001",
            "direction": "...",
            "core_observation": "...",
            "main_hypothesis": "...",
            "alternative_hypotheses": ["..."],
            "artifact_risks": ["..."],
            "supporting_data": [{{"source": "bioinformatics", "evidence": "...", "strength": "high|medium|low"}}],
            "contradictory_or_missing_evidence": ["..."],
            "contradicts_stated_premise": false,
            "validation_plan": {{"next_analysis": ["..."], "wet_lab_validation": ["..."], "decision_criteria": ["..."]}},
            "biological_plausibility": "high|medium|low",
            "validation_feasibility": "high|medium|low",
            "artifact_risk": "high|medium|low",
            "recommendation": "prioritize|secondary|needs_more_data|avoid",
            "pi_review_summary": "...",
            "rationale": "...",
            "data_support_summary": "...",
            "literature_support_summary": "...",
            "data_reliability": "high|medium|low",
            "literature_novelty": "high|medium|low|unknown",
            "biology_expectedness": "expected|partially_expected|unexpected|unknown",
            "data_novelty": "high|medium|low",
            "mechanistic_importance": "high|medium|low",
            "functional_impact": "high|medium|low",
            "data_sufficiency": "sufficient|partial|insufficient",
            "methodological_gap": "none|minor|major",
            "overall_priority": "high|medium|low",
            "risk_flags": ["..."],
            "prerequisite_dependencies": ["dir_id", "..."],
            "evidence_ref": "dir_001"
        }}
    ],
    "raw_direction_candidates": <echo the raw candidates exactly as provided above>,
    "direction_literature_evidence": <echo the evidence matrix exactly as provided above>,
    "direction_dependency_graph": {{"dir_001": ["dir_002"], ...}},
    "direction_selection_summary": {{}}
}}
"""
        response = await self.chat(prompt, temperature=0.3)
        data = self._safe_parse_json(response, default={})

        # Build ExplorationReport from parsed data. Use from_llm_dict so a single
        # type-mismatched field (e.g. the LLM emitting a string for the dict-typed
        # direction_selection_summary) is dropped rather than raising a
        # ValidationError that would abort Report-A generation and silently skip
        # the exploration->innovation transition + direction-selection gate.
        report = ExplorationReport.from_llm_dict(data)

        # Normalize: research_direction_candidates must hold the FINAL scored candidates.
        scored = data.get("research_direction_candidates") or []
        # Recover if the LLM placed scored candidates under raw_direction_candidates by mistake.
        if not any(isinstance(c, dict) and c.get("overall_priority") for c in scored):
            recovered = [c for c in (data.get("raw_direction_candidates") or [])
                         if isinstance(c, dict) and c.get("overall_priority")]
            if recovered:
                scored = recovered
        # Guarantee direction_id + evidence_ref on every scored candidate.
        raw_by_index = {i: rc for i, rc in enumerate(raw_candidates)}
        for i, c in enumerate(scored):
            if not isinstance(c, dict):
                continue
            if not c.get("direction_id"):
                c["direction_id"] = (raw_by_index.get(i, {}) or {}).get("direction_id", f"dir_{i+1:03d}")
            if not c.get("evidence_ref"):
                c["evidence_ref"] = c["direction_id"]
        # Build round context for soft retry of missing critical fields.
        round_context = "\n".join([
            "Bioinformatics findings: " + json.dumps(round_result.bioinformatics_findings[:15], ensure_ascii=False),
            "Pathology findings: " + json.dumps(round_result.pathology_findings[:5], ensure_ascii=False),
            "Biochemistry findings: " + json.dumps(round_result.biochemistry_findings[:5], ensure_ascii=False),
            "Literature findings: " + json.dumps(round_result.literature_findings[:5], ensure_ascii=False),
            "PI summary: " + (round_result.pi_summary or "")[:1000],
        ])
        for i, candidate in enumerate(scored):
            if not isinstance(candidate, dict):
                continue
            if candidate.get("overall_priority") != "high" and candidate.get("recommendation") != "prioritize":
                continue
            missing = self._critical_direction_fields_missing(candidate)
            if not missing:
                continue
            try:
                scored[i] = await self._soft_retry_direction_candidate(candidate, round_context, missing)
            except Exception as retry_err:
                self._append_unique_flag(candidate, "critical_field_retry_failed")
                candidate.setdefault("limitations", []).append(f"Critical field retry failed: {str(retry_err)[:120]}")
        scored = self._apply_direction_guardrails(scored, raw_candidates, literature_evidence)
        report.research_direction_candidates = scored

        # raw_direction_candidates always holds the ORIGINAL raw input.
        report.raw_direction_candidates = raw_candidates
        # Evidence matrix always preserved.
        if not report.direction_literature_evidence:
            report.direction_literature_evidence = literature_evidence

        # Dependency-cycle validation MUST propagate — a cyclic graph must not be exposed.
        if report.direction_dependency_graph:
            self._validate_direction_dependency_graph(report.direction_dependency_graph)

        return report

    def _validate_direction_dependency_graph(self, graph: Dict[str, List[str]]) -> None:
        """Validate direction dependency graph for cycles using DFS."""
        visiting = set()
        visited = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError(f"Cycle detected in direction dependency graph at {node}")
            if node in visited:
                return
            visiting.add(node)
            for dep in graph.get(node, []) or []:
                visit(dep)
            visiting.remove(node)
            visited.add(node)

        for node in graph:
            visit(node)

    @staticmethod
    def _candidate_has_specific_core_observation(candidate: Dict[str, Any]) -> bool:
        text = str(candidate.get("core_observation") or "").strip()
        if len(text) < 25:
            return False
        generic = (
            "interesting pattern", "interesting patterns", "data suggests",
            "data suggest", "further analysis", "needs investigation"
        )
        lowered = text.lower()
        if any(g in lowered for g in generic) and not any(ch.isdigit() for ch in text):
            return False
        markers = (
            "cluster", "log2", "fold", "p=", "padj", "rho", "correlation",
            "score", "cell", "gene", "pathway", "module", "pmid"
        )
        return any(m in lowered for m in markers) or any(ch.isdigit() for ch in text)

    @staticmethod
    def _append_unique_flag(candidate: Dict[str, Any], flag: str) -> None:
        flags = candidate.get("risk_flags")
        if not isinstance(flags, list):
            flags = []
        if flag not in flags:
            flags.append(flag)
        candidate["risk_flags"] = flags

    @staticmethod
    def _select_directions(candidates: list) -> list:
        return [d.get("direction", "") for d in candidates if isinstance(d, dict) and d.get("direction")]

    @classmethod
    def _apply_direction_guardrails(
        cls,
        scored: List[Dict[str, Any]],
        raw_candidates: List[Dict[str, Any]],
        literature_evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        evidence_by_id = {
            str(e.get("direction_id", "")): e
            for e in literature_evidence or []
            if isinstance(e, dict)
        }
        for candidate in scored:
            if not isinstance(candidate, dict):
                continue
            if not cls._candidate_has_specific_core_observation(candidate):
                candidate["recommendation"] = "needs_more_data"
                cls._append_unique_flag(candidate, "generic_core_observation")
            # A "prioritize" recommendation is downgraded if any quality
            # dimension is disqualifying. Add a (field, bad_value) tuple to extend.
            downgrade_conditions = (
                ("data_reliability", "low"),
                ("artifact_risk", "high"),
                ("data_sufficiency", "insufficient"),
            )
            if candidate.get("recommendation") == "prioritize" and any(
                str(candidate.get(field, "")).lower() == bad_value
                for field, bad_value in downgrade_conditions
            ):
                candidate["recommendation"] = "needs_more_data"
            if candidate.get("contradicts_stated_premise") is True:
                cls._append_unique_flag(candidate, "contradicts_premise")
            ev = evidence_by_id.get(str(candidate.get("evidence_ref") or candidate.get("direction_id") or ""), {})
            novelty_status = str(ev.get("novelty_status", "")).lower()
            if novelty_status in {"unknown_due_to_timeout", "unknown_due_to_failure"}:
                candidate["literature_novelty"] = "unknown"
                if candidate.get("recommendation") == "prioritize":
                    candidate["recommendation"] = "needs_more_data"
                cls._append_unique_flag(
                    candidate,
                    "literature_evidence_timeout" if novelty_status == "unknown_due_to_timeout" else "literature_evidence_failed",
                )
        return scored

    @staticmethod
    def _critical_direction_fields_missing(candidate: Dict[str, Any]) -> List[str]:
        missing = []
        for key in (
            "core_observation", "alternative_hypotheses", "artifact_risks",
            "contradictory_or_missing_evidence", "validation_plan", "recommendation",
        ):
            value = candidate.get(key)
            if value is None or value == "" or value == [] or value == {}:
                missing.append(key)
        return missing

    async def _soft_retry_direction_candidate(
        self,
        candidate: Dict[str, Any],
        round_context: str,
        missing_fields: List[str],
    ) -> Dict[str, Any]:
        prompt = f"""Repair one research direction candidate by filling only missing critical fields.

Round context:
{round_context[:3000]}

Candidate JSON:
{json.dumps(candidate, ensure_ascii=False, indent=2)}

Missing fields: {', '.join(missing_fields)}

Return ONLY the complete candidate JSON object. Preserve existing fields unless they directly conflict with the repaired fields. Required repaired fields:
- core_observation: specific Round 1 data element, not generic prose
- alternative_hypotheses: list of plausible alternatives
- artifact_risks: list of technical or interpretive risks
- contradictory_or_missing_evidence: list of missing or conflicting evidence
- validation_plan: object with next_analysis, wet_lab_validation, decision_criteria lists
- recommendation: prioritize|secondary|needs_more_data|avoid
"""
        response = await self.chat_without_history(prompt, temperature=0.2)
        repaired = self._safe_parse_json(response, default={})
        if isinstance(repaired, dict) and repaired.get("direction"):
            merged = dict(candidate)
            merged.update(repaired)
            return merged
        return candidate

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

    async def select_best_directions(
        self,
        state: AnalysisState
    ) -> Dict[str, Any]:
        """
        PI autonomously selects the best research directions from exploration report candidates.
        Used when user clicks 'Let PI Decide' or countdown expires.

        Returns:
            DirectionSelectionDecision-shaped dict with selected_directions, rejected_directions, selection_rationale.
        """
        logger.info(f"[PI] Selecting best research directions")
        self.reset_history()

        candidates = []
        if state.exploration_report and state.exploration_report.research_direction_candidates:
            candidates = state.exploration_report.research_direction_candidates

        if not candidates:
            return {
                "selected_directions": [{"direction": "Continue analysis based on Round 1 findings", "notes": ""}],
                "rejected_directions": [],
                "selection_rationale": "No candidates available; using default direction.",
            }

        exploration_context = ""
        dependency_graph_section = ""
        if state.exploration_report:
            exploration_context = f"""
Cluster Overview: {state.exploration_report.cluster_overview[:500]}
Key Pathway Changes: {json.dumps(state.exploration_report.key_pathway_changes[:5], ensure_ascii=False)}
Strongest Heterogeneity: {json.dumps(state.exploration_report.strongest_heterogeneity[:5], ensure_ascii=False)}
Data Quality Warnings: {json.dumps(state.exploration_report.data_quality_warnings[:3], ensure_ascii=False)}
PI Direction Assessment: {state.exploration_report.pi_direction_assessment[:500]}
"""
            if state.exploration_report.direction_dependency_graph:
                dependency_graph_section = f"""
## Direction Dependency Graph
{json.dumps(state.exploration_report.direction_dependency_graph, ensure_ascii=False, indent=2)}

Select directions that respect dependency ordering: dependent directions should only be selected if their prerequisites are also selected.
"""

        prompt = f"""You are the PI selecting the best research direction(s) for the innovation phase.

## Exploration Report
{exploration_context}
{dependency_graph_section}
## Candidates
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Select the best direction(s) — usually 1 to 2 — that together provide the most comprehensive and impactful outcomes while respecting prerequisite dependencies.

SELECTION CRITERIA:
1. **Reliability** is the baseline: low-reliability directions should be avoided
2. **Innovation + Novelty** together determine priority
3. Selected directions should ideally be complementary (not redundant)
4. Consider the dependency graph: if a direction has prerequisites, ensure its prerequisites are also selected or the direction is independently viable
5. For each rejected direction, provide an explanation

Return JSON:
{{
    "selected_directions": [
        {{
            "direction": "The exact direction text from candidates",
            "notes": "Brief PI notes on why this was selected and what to focus on"
        }},
        {{
            "direction": "The exact direction text from candidates",
            "notes": "Brief PI notes on why this was selected and what to focus on"
        }}
    ],
    "rejected_directions": [
        {{
            "direction": "The exact direction text from rejected candidates",
            "reason_not_selected": "Why this direction was not selected",
            "follow_up_value": "high|medium|low",
            "dependency_note": "Whether it has unmet prerequisites or is suitable as follow-up"
        }}
    ],
    "selection_rationale": "Why these directions were chosen over others"
}}
"""

        response = await self.chat(prompt, temperature=0.3)
        data = self._safe_parse_json(response, default={})

        selected = data.get("selected_directions", [])
        rejected = data.get("rejected_directions", [])
        rationale = data.get("selection_rationale", "")

        if not selected or len(selected) < 1:
            # Fallback: pick first 2 candidates
            selected = [
                {"direction": c.get("direction", ""), "notes": ""}
                for c in candidates[:2]
            ]
            # Build rejected list from remaining candidates
            selected_dirs_set = {d.get("direction", "") for d in selected}
            rejected = [
                {"direction": c.get("direction", ""), "reason_not_selected": "Not selected in fallback",
                 "follow_up_value": "low", "dependency_note": ""}
                for c in candidates
                if c.get("direction", "") not in selected_dirs_set
            ]
            if not rationale:
                rationale = "Fallback: selected first available candidates."

        # Ensure at least 1 direction
        if not selected:
            selected = [{"direction": "Continue analysis based on Round 1 findings", "notes": ""}]

        # Validate dependency graph if present
        dep_graph = state.exploration_report.direction_dependency_graph if state.exploration_report else {}
        if dep_graph:
            # A cyclic graph must not be used to present direction options (spec §377).
            self._validate_direction_dependency_graph(dep_graph)

        # Prerequisite closure (multi-hop): keep adding prerequisites until none are missing.
        text_by_id = {c.get("direction_id"): c.get("direction", "") for c in candidates if isinstance(c, dict)}
        id_by_text = {c.get("direction", ""): c.get("direction_id") for c in candidates if isinstance(c, dict)}
        cand_by_id = {c.get("direction_id"): c for c in candidates if isinstance(c, dict)}
        selected_texts = {d.get("direction", "") for d in selected}
        changed = True
        while changed:
            changed = False
            for d in list(selected):
                dir_id = id_by_text.get(d.get("direction", ""))
                cand = cand_by_id.get(dir_id)
                if not cand:
                    continue
                for prereq_id in cand.get("prerequisite_dependencies", []) or []:
                    prereq_text = text_by_id.get(prereq_id)
                    if prereq_text and prereq_text not in selected_texts:
                        selected.append({"direction": prereq_text,
                                         "notes": f"Added to satisfy prerequisite of '{d.get('direction', '')}'."})
                        selected_texts.add(prereq_text)
                        changed = True

        # A direction promoted into selected (e.g. as a prerequisite) must not also appear as rejected.
        rejected = [r for r in rejected
                    if isinstance(r, dict) and r.get("direction", "") not in selected_texts]

        logger.info(f"[PI] Selected {len(selected)} directions: {[d.get('direction', '')[:50] for d in selected]}")
        return {
            "selected_directions": selected,
            "rejected_directions": rejected,
            "selection_rationale": rationale,
        }

    async def plan_supplementary_analysis(
        self, state: AnalysisState, user_request: str, context: str
    ) -> Dict[str, Any]:
        """Plan supplementary analysis tasks based on user's request at a HITL point.

        Args:
            state: Current analysis state
            user_request: User's free-text description of what additional analysis they want
            context: "annotation" or "direction" - which HITL checkpoint

        Returns:
            Dict with keys: bioinformatics_tasks, literature_topics, summary
        """
        context_desc = (
            "cell type annotation confirmation (the user wants more information before confirming cluster annotations)"
            if context == "annotation"
            else "research direction selection (the user wants more information before choosing innovation directions)"
        )

        prompt = f"""You are the PI Agent coordinating a supplementary analysis requested by the user.

Current analysis checkpoint: {context_desc}

User's request for additional analysis:
"{user_request}"

Current analysis state:
- Round: {state.current_round}
- Phase: {state.analysis_phase}
- Clusters annotated: {len(state.annotation_state.cluster_annotations) if state.annotation_state else 0}

Based on the user's request, plan specific supplementary analysis tasks.
Return a JSON object with these keys (empty lists for agents that don't need to run):
{{
    "bioinformatics_tasks": ["task1", "task2"],
    "literature_topics": ["topic1"],
    "summary": "Brief explanation of what will be analyzed"
}}

Keep tasks focused and actionable. The user is waiting for results to make a decision,
so prioritize speed - plan only the minimum necessary analysis."""

        response = await self.chat(prompt)
        plan = self._safe_parse_json(response, default={})
        if not plan or not isinstance(plan, dict):
            plan = {
                "bioinformatics_tasks": [user_request],
                "literature_topics": [],
                "summary": user_request,
            }

        logger.info(f"[PI] Supplementary analysis plan: {plan.get('summary', '')[:100]}")
        return plan

    async def analyze_direction_correlations(
        self,
        state: AnalysisState,
        directions_with_results: List[Dict]
    ) -> str:
        """
        Analyze correlations and complementarity between multiple research directions.
        Called after all selected directions have been analyzed.

        Args:
            state: Current analysis state
            directions_with_results: List of dicts with keys:
                - direction: str
                - notes: str
                - round_summary: str (PI summary from that direction's round)

        Returns:
            Correlation analysis text (markdown formatted)
        """
        logger.info(f"[PI] Analyzing correlations between {len(directions_with_results)} directions")
        self.reset_history()

        directions_text = ""
        for i, d in enumerate(directions_with_results, 1):
            directions_text += f"""
### Direction {i}: {d.get('direction', 'Unknown')}
**User Notes**: {d.get('notes', 'None')}
**Analysis Summary**: {d.get('round_summary', 'No summary available')[:800]}
"""

        prompt = f"""You are the PI analyzing the relationships between multiple research directions that were independently investigated.

## Project: {state.name}
## Research Objective: {state.initial_instruction}

## Directions Analyzed
{directions_text}

Analyze the correlations, complementarity, and potential synergies between these research directions:

1. **Shared Mechanisms**: Do any directions share underlying molecular mechanisms or pathways?
2. **Complementary Evidence**: Do findings from one direction support or strengthen conclusions from another?
3. **Contradictions**: Are there any contradictory findings between directions that need resolution?
4. **Integrated Hypothesis**: Can you formulate an integrated hypothesis that connects the key findings from all directions?
5. **Recommended Next Steps**: Based on the combined findings, what would you recommend as the most impactful follow-up studies?

Provide your analysis in a clear, structured markdown format.
"""

        response = await self.chat(prompt, temperature=0.3)
        return response

    @staticmethod
    def _format_completed_results(completed_results: Dict[str, List[str]]) -> str:
        """Render completed-agent findings into a markdown summary for planning
        prompts. Each agent contributes up to 5 findings, truncated to 200 chars."""
        parts = []
        for key, findings in (completed_results or {}).items():
            label = key.replace("_findings", "").capitalize()
            parts.append(
                f"### {label} Findings\n"
                + "\n".join(f"- {f[:200]}" for f in findings[:5])
            )
        return "\n\n".join(parts)

    async def plan_agent_tasks(
        self,
        agent_name: str,
        original_tasks: List[str],
        completed_results: Dict[str, List[str]],
        overall_strategy: str = "",
        research_objective: str = "",
        state: Optional[AnalysisState] = None,
    ) -> List[str]:
        """Dynamically refine tasks for an agent based on already-completed results.

        Uses self.chat() to accumulate context within the round, building a
        coherent chain of thought across successive agent planning calls.

        Args:
            agent_name: Name of the target agent (e.g., "pathology", "bioinformatics")
            original_tasks: Tasks originally assigned by PI planning
            completed_results: Dict of findings from already-completed agents
            overall_strategy: Overview/strategy string for context
            research_objective: The project's research objective
            state: Optional AnalysisState for data availability context

        Returns:
            Refined list of task strings. Falls back to original_tasks on error.
        """
        if not completed_results:
            return original_tasks

        results_summary = self._format_completed_results(completed_results)

        data_summary = state.build_experimental_design_summary() if state else "Not available"

        prompt = f"""You are the PI coordinating a multi-agent analysis.

## Research Objective
{research_objective}

## Species
{state.species if state else "Not specified"}

## Available Data
{data_summary}

## Overall Strategy
{overall_strategy or "Not available"}

## Completed Agent Results (this round)
{results_summary}

## Original Tasks for {agent_name.capitalize()} Agent
{chr(10).join(f"- {t}" for t in original_tasks)}

Based on the completed results, refine the tasks for the {agent_name} agent.
You may: keep tasks unchanged, add specific focus areas, modify tasks to leverage
insights, or add cross-referencing instructions.
Ensure refined tasks are grounded in the available data. Do NOT add tasks requiring data that is not available.

Return a JSON object:
{{
    "refined_tasks": ["task1", "task2", ...]
}}

Keep tasks actionable and specific. Do NOT remove core tasks unless truly unnecessary."""

        try:
            response = await self.chat(prompt, temperature=0.3)
            result = self._safe_parse_json(response, default={})
            refined = result.get("refined_tasks", [])
            if refined and isinstance(refined, list) and all(isinstance(t, str) for t in refined):
                logger.info(
                    f"[PI] Dynamic planning refined {len(original_tasks)} -> "
                    f"{len(refined)} tasks for {agent_name}"
                )
                return refined
        except Exception as e:
            logger.warning(f"[PI] Dynamic planning failed for {agent_name}: {e}")

        return original_tasks

    async def plan_deep_research_tasks(
        self,
        topics: List[str],
        completed_results: Dict[str, List[str]],
        overall_strategy: str = "",
        research_objective: str = "",
        state: Optional[AnalysisState] = None,
    ) -> Dict[str, Any]:
        """Decide whether literature research is complex enough to split into
        multiple independent Deep Research (DR) sub-tasks.

        Returns ``{"decompose": bool, "tasks": [str, ...]}``. When ``decompose``
        is True the caller issues one DR call per task (serial, capped at 2).
        When False the caller merges all topics into a single DR call (legacy
        behavior). Falls back to ``{"decompose": False, "tasks": topics}`` on any
        parse/LLM error.
        """
        self.refresh_runtime_config_if_needed()
        MAX_DR_TASKS = 2

        results_summary = self._format_completed_results(completed_results) or "None yet."

        prompt = f"""You are the PI coordinating a literature review that will be \
executed with a Deep Research engine (each call autonomously runs multi-round \
web/literature search and synthesis).

## Research Objective
{research_objective}

## Overall Strategy
{overall_strategy or "Not available"}

## Completed Agent Results (this round)
{results_summary}

## Candidate Literature Topics
{chr(10).join(f"- {t}" for t in topics)}

        Decide whether this literature work truly REQUIRES splitting into TWO independent \
Deep Research tasks, or whether ONE comprehensive Deep Research query is sufficient.

DEFAULT to a single merged query (decompose=false). A single Deep Research call \
already runs multi-round web/literature search and synthesis internally, so it \
covers the vast majority of cases.

Set decompose=true ONLY when it is GENUINELY NECESSARY — i.e. the topics span TWO \
clearly distinct questions that each demand their own separate, deep investigation \
and cannot be adequately covered together. The bar for splitting is HIGH; when in \
doubt, prefer a single merged query.

Return a JSON object:
{{
    "decompose": true/false,
    "tasks": ["focused task 1", "focused task 2", ...]
}}

If decompose is true, provide EXACTLY 2 self-contained tasks (each a complete \
standalone research brief). If decompose is false, "tasks" may echo the candidate \
topics."""

        try:
            response = await self.chat(prompt, temperature=0.3)
            result = self._safe_parse_json(response, default={})
            decompose = bool(result.get("decompose", False))
            tasks = result.get("tasks", [])
            if not (isinstance(tasks, list) and all(isinstance(t, str) for t in tasks) and tasks):
                tasks = list(topics)
                decompose = False
            tasks = tasks[:MAX_DR_TASKS]
            # Decomposition only makes sense with 2+ tasks.
            if decompose and len(tasks) < 2:
                decompose = False
            logger.info(
                f"[PI] Deep Research planning: decompose={decompose}, "
                f"{len(tasks)} task(s)"
            )
            return {"decompose": decompose, "tasks": tasks}
        except Exception as e:
            logger.warning(f"[PI] plan_deep_research_tasks failed: {e}")
            return {"decompose": False, "tasks": list(topics)}

    async def process_human_guidance(
        self,
        state: AnalysisState,
        human_input: str
    ) -> Dict[str, Any]:
        """
        Process human researcher's guidance and adjust analysis direction

        Args:
            state: Current analysis state
            human_input: Human researcher's guidance/instruction
            
        Returns:
            Dict with:
            - response: PI's response to human
            - should_stop: Whether to stop analysis
            - adjusted_instructions: New instructions for agents (if applicable)
        """
        # Build context from current state
        context = self._build_context(state)
        
        prompt = f"""
## Current Analysis Status
Project: {state.name}
Current Round: {state.current_round}
Research Objective: {state.initial_instruction}

{context}

## Human Researcher Guidance
The human researcher has provided the following guidance:
"{human_input}"

As the PI Agent, please:
1. Carefully interpret what the human researcher wants
2. Decide if they want to stop/conclude the analysis, or adjust the direction
3. If adjusting direction, provide new specific instructions for each agent

Common stop/conclude keywords: stop, terminate, end, finish, conclude, summarize

Respond in JSON format:
{{
    "interpretation": "Your interpretation of what the human wants",
    "response": "Your response to the human researcher (be helpful and professional)",
    "should_stop": true/false,
    "reason": "Why you decided to stop or continue",
    "adjusted_instructions": {{
        "pathology_tasks": ["task1", "task2"],
        "bioinformatics_tasks": ["task1", "task2"],
        "biochemistry_tasks": ["task1", "task2"],
        "literature_topics": ["topic1", "topic2"]
    }},
    "focus_areas": ["key area 1", "key area 2"],
    "deprioritize": ["area to skip or reduce focus"]
}}

If should_stop is true, adjusted_instructions can be empty.
If adjusting direction, provide specific and actionable tasks.
"""
        
        response = await self.chat(prompt, temperature=0.3)

        # Use base class safe JSON parser
        data = self._safe_parse_json(response, default={})

        if data and "interpretation" in data:
            return {
                "interpretation": data.get("interpretation", ""),
                "response": data.get("response", "Acknowledged. Adjusting analysis direction."),
                "should_stop": data.get("should_stop", False),
                "reason": data.get("reason", ""),
                "adjusted_instructions": data.get("adjusted_instructions"),
                "focus_areas": data.get("focus_areas", []),
                "deprioritize": data.get("deprioritize", [])
            }
        else:
            logger.warning(f"[PI] No valid JSON found in response, using fallback parsing. Response: {response[:200] if response else 'None'}...")
        
        # Fallback: check for stop keywords manually (English + Chinese)
        human_lower = human_input.lower()

        if any(keyword in human_lower for keyword in STOP_KEYWORDS):
            return {
                "interpretation": "Human requested to stop the analysis",
                "response": "Understood. I will conclude the current analysis and generate a final report.",
                "should_stop": True,
                "reason": "Human explicitly requested to stop/conclude",
                "adjusted_instructions": None
            }
        
        # Default: treat as guidance to continue with adjustments
        return {
            "interpretation": human_input,
            "response": f"Acknowledged. I will incorporate your guidance: {human_input}",
            "should_stop": False,
            "reason": "Continuing with adjusted focus based on human guidance",
            "adjusted_instructions": {
                "pathology_tasks": [f"Focus on: {human_input}"],
                "bioinformatics_tasks": [f"Prioritize analysis related to: {human_input}"],
                "biochemistry_tasks": [f"Review biochemistry aspects of: {human_input}"],
                "literature_topics": [f"Search literature on: {human_input}"]
            }
        }
    
    async def identify_data_needs(
        self,
        state: AnalysisState,
        round_result: RoundResult
    ) -> Dict[str, List[str]]:
        """
        Identify what additional data would help the analysis
        
        Args:
            state: Current analysis state
            round_result: Results from the current round
            
        Returns:
            Dict with lists of needed data by category
        """
        prompt = f"""
## Current Analysis State
Project: {state.name}
Round: {round_result.round_number}
Research Objective: {state.initial_instruction}

## Round Summary
{round_result.pi_summary}

## Key Findings
Pathology: {round_result.pathology_findings}
Bioinformatics: {round_result.bioinformatics_findings}
Biochemistry: {round_result.biochemistry_findings}

## Next Steps Planned
{round_result.next_steps}

Based on the current findings and planned next steps, what additional data would help deepen the analysis?

Please identify specific data needs in JSON format:
{{
    "pathology": [
        "Special staining to assess specific features (e.g., fibrosis, lipid, immune cells)",
        "Additional tissue sections from different regions",
        ...
    ],
    "biochemistry": [
        "Tissue/organ function markers",
        "Metabolic indicators relevant to the phenotype",
        "Oxidative stress or inflammatory markers",
        ...
    ],
    "other": [
        "Western blot for specific proteins",
        "qPCR validation of key genes",
        ...
    ]
}}

Only include data that would significantly help answer the research question.
If no additional data is needed, return empty lists.
"""
        
        response = await self.chat(prompt, temperature=0.3)

        # Use base class safe JSON parser
        data = self._safe_parse_json(response, default={})

        return {
            "pathology": data.get("pathology", []),
            "biochemistry": data.get("biochemistry", []),
            "other": data.get("other", [])
        }
    
    async def process_user_input(
        self,
        state: AnalysisState,
        user_message: str,
        uploaded_files: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user input after a round completes.
        Classify uploaded files and determine next actions.
        
        Args:
            state: Current analysis state
            user_message: User's text message
            uploaded_files: List of uploaded file info (type, filename, id)
            
        Returns:
            Dict with:
            - action: 'continue', 'stop', 'analyze_new_data'
            - file_assignments: Which files go to which agents
            - instructions: Updated instructions for agents
            - response: Response to user
        """
        uploaded_files = uploaded_files or []
        logger.info(f"[PI] Processing user input: {user_message[:100] if user_message else 'no message'}... ({len(uploaded_files)} files)")

        # Quick path: synthetic control sentinels must deterministically continue
        # and NOT be sent to the stop/continue LLM classifier — that path can
        # non-deterministically return action="stop" and end the analysis after
        # Round 1. [PI_DECIDE] (set by no-wait mode in synthesis.py) means "let
        # the PI autonomously pick research directions"; _pi_planning's
        # select_best_directions() later consumes it via user_chosen_direction.
        if user_message and user_message.strip() in _CONTROL_SIGNAL_RESULTS:
            response, reason = _CONTROL_SIGNAL_RESULTS[user_message.strip()]
            logger.info(f"[PI] Control sentinel {user_message.strip()} detected, proceeding to next round")
            return {
                "action": "continue",
                "response": response,
                "file_assignments": {},
                "biochemistry_values": {},
                "adjusted_instructions": None,
                "focus_areas": [],
                "reason": reason,
            }

        # Build file description
        file_descriptions = []
        for f in uploaded_files:
            file_descriptions.append(f"{f.get('type', 'unknown')}: {f.get('filename', 'unknown')}")
        
        prompt = f"""
## Project Context
Project Name: {state.name}
Research Objective: {state.initial_instruction}

## User Input Processing

The user has provided input after Round {state.current_round} analysis.

User message: "{user_message}"

Uploaded files:
{chr(10).join(file_descriptions) if file_descriptions else "None"}

## Task
1. Determine what the user wants:
   - Continue to next round?
   - Stop and generate final report?
   - Analyze new data first?
   - Adjust analysis direction?

2. If files were uploaded, classify each file:
   - Images of H&E, Sirius Red, Oil Red O, etc. → pathology
   - Images of Western blots, gels → biochemistry
   - Text describing biochemistry results → biochemistry
   - Other data files → determine appropriate agent

3. Extract any specific instructions from user message

Respond in JSON format:
{{
    "action": "continue" | "stop" | "analyze_new_data",
    "response": "Your response to the user",
    "file_assignments": {{
        "pathology": [
            {{"filename": "xxx.png", "image_type": "Sirius Red", "reason": "Fibrosis assessment"}}
        ],
        "biochemistry": [
            {{"filename": "xxx.txt", "data_type": "metabolic markers", "reason": "Enzyme activity values"}}
        ]
    }},
    "biochemistry_values": {{
        "Marker1": {{"value": 120, "unit": "U/L"}},
        "Marker2": {{"value": 95, "unit": "U/L"}}
    }},
    "adjusted_instructions": {{
        "pathology_tasks": ["specific task"],
        "bioinformatics_tasks": ["specific task"],
        "biochemistry_tasks": ["specific task"],
        "literature_topics": ["specific topic"]
    }},
    "focus_areas": ["area to emphasize"],
    "reason": "Why this action was chosen"
}}

Note: Extract biochemistry values from user message if they mention specific numbers (e.g., "ALT is 120 U/L").
"""
        
        response = await self.chat(prompt, temperature=0.3)

        # Use base class safe JSON parser
        data = self._safe_parse_json(response, default={})

        if data and "action" in data:
            return {
                "action": data.get("action", "continue"),
                "response": data.get("response", "Acknowledged"),
                "file_assignments": data.get("file_assignments", {}),
                "biochemistry_values": data.get("biochemistry_values", {}),
                "adjusted_instructions": data.get("adjusted_instructions"),
                "focus_areas": data.get("focus_areas", []),
                "reason": data.get("reason", "")
            }
        
        # Fallback: check for stop keywords (English + Chinese)
        user_lower = user_message.lower()
        if any(keyword in user_lower for keyword in STOP_KEYWORDS):
            return {
                "action": "stop",
                "response": "Understood. Generating final report.",
                "file_assignments": {},
                "biochemistry_values": {},
                "adjusted_instructions": None,
                "focus_areas": [],
                "reason": "User requested to stop"
            }
        
        # Default: continue
        return {
            "action": "continue" if not uploaded_files else "analyze_new_data",
            "response": "Acknowledged. Proceeding with analysis.",
            "file_assignments": {
                "pathology": [{"filename": f["filename"], "image_type": "user_upload"} 
                             for f in uploaded_files if f.get("type") == "image"]
            },
            "biochemistry_values": {},
            "adjusted_instructions": None,
            "focus_areas": [],
            "reason": "Continuing analysis"
        }
    
    async def synthesize_round(
        self,
        state: AnalysisState,
        pathology_result: Optional[str] = None,
        bioinformatics_result: Optional[str] = None,
        biochemistry_result: Optional[str] = None,
        literature_result: Optional[str] = None,
        research_direction: str = "",
        cross_modal_context: str = "",
    ) -> RoundResult:
        """
        Synthesize results from all agents for a round

        Args:
            state: Current analysis state
            *_result: Results from each agent

        Returns:
            RoundResult with synthesis
        """
        self.refresh_runtime_config_if_needed()
        logger.info(f"[PI] Synthesizing round {state.current_round} results")

        # Reset message history — the prompt below embeds full state context,
        # so prior chat history is unnecessary and wastes tokens
        self.reset_history()

        # Build previous rounds context for redundancy avoidance
        previous_context = ""
        if state.round_results:
            prev_parts = []
            for prev_round in state.round_results:
                all_findings = (
                    prev_round.pathology_findings[:3]
                    + prev_round.bioinformatics_findings[:3]
                    + prev_round.biochemistry_findings[:3]
                    + prev_round.literature_findings[:3]
                )
                prev_parts.append(
                    f"### Round {prev_round.round_number}\n"
                    f"**Summary**: {prev_round.pi_summary[:500] if prev_round.pi_summary else 'N/A'}\n"
                    f"**Key findings**: {'; '.join(str(f) if not isinstance(f, str) else f for f in all_findings) if all_findings else 'None'}\n"
                    f"**Tasks assigned**: {'; '.join(prev_round.next_steps[:8]) if prev_round.next_steps else 'None'}"
                )
            previous_context = "\n\n## Previous Rounds Summary\n" + "\n\n".join(prev_parts)

        cross_modal_section = ""
        if cross_modal_context and cross_modal_context.strip():
            cross_modal_section = (
                "\n## Cross-Modal Evidence Audit\n"
                "The following cross-modal evidence matrix was computed from this round's "
                "multi-modal results. Use it to surface mechanisms supported across modalities:\n"
                f"{cross_modal_context}\n"
            )

        prompt = f"""
## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}
{previous_context}
{cross_modal_section}
This is the results summary for Round {state.current_round}. Please synthesize:

## Biochemistry Analysis Results
{biochemistry_result or "No new results"}

## Pathology Analysis Results
{pathology_result or "No new results"}

## Bioinformatics Analysis Results
{bioinformatics_result or "No new results"}

## Literature Research Results
{literature_result or "No new results"}

Please provide comprehensive analysis in JSON format:
{{
    "summary": "Summary of this round",
    "biochemistry_findings": ["biochemistry finding 1", ...],
    "pathology_findings": ["pathology finding 1", ...],
    "bioinformatics_findings": ["bioinformatics finding 1", ...],
    "literature_findings": ["literature finding 1", ...],
    "conflicts": [
        {{"description": "conflict description", "resolution": "solution", "agents": ["agent1", "agent2"]}}
    ],
    "next_biochemistry": ["next biochemistry task"],
    "next_pathology": ["next pathology task"],
    "next_bioinformatics": ["next bioinformatics task"],
    "next_literature": ["next literature topic"],
    "new_findings": true/false,
    "should_stop": true/false,
    "stop_reason": "reason for stopping (if applicable)",
    "agents_for_next_round": ["bioinformatics", ...],
    "reanalysis_agents": ["pathology", ...]  // Only if re-analysis of existing data is needed
}}

IMPORTANT: When generating next_* tasks:
- Do NOT repeat tasks already completed in previous rounds
- Only propose NEW directions, deeper investigations, or follow-up analyses
- If a previous round already completed standard processing (QC, clustering, cell annotation), do NOT re-assign these
- Focus next_* on specific hypotheses, pathway analyses, or validation tasks based on current findings

IMPORTANT for next_literature topics:
- Each topic should be a specific, searchable research question (e.g., "Role of <cell type> state transition in <tissue> under <condition>")
- Include biological context (disease name, tissue, species) in each topic
- Focus on filling knowledge GAPS identified in this round's literature findings
- Avoid overly broad topics or raw gene lists

IMPORTANT: agents_for_next_round vs reanalysis_agents:
- agents_for_next_round: which agents should RUN in the next round (general control)
  - "bioinformatics" MUST ALWAYS be included (it is the core analysis agent)
  - Only include "pathology"/"biochemistry" if NEW data needs processing
- reanalysis_agents: agents needing re-analysis of EXISTING data with different parameters/perspectives
  - Only use when NO new data but re-analysis is needed (e.g., "pathology" to re-examine existing images)
  - Example: "pathology" to re-examine existing images with different staining analysis
- EXCLUDE agents with no meaningful new work to save time and resources
"""

        # Add phase-specific synthesis instructions
        if state.analysis_phase == AnalysisPhase.EXPLORATION:
            prompt += """
EXPLORATION PHASE — additional synthesis requirements:
- Summarize ALL findings comprehensively, without hypothesis bias
- Highlight unexpected or surprising findings specifically
- Flag data quality concerns (clusters with very few cells, poor QC metrics)
- Note any low-confidence cell annotations
- In next_* tasks: suggest potential research directions for the Innovation phase
"""
        else:
            direction_ctx = research_direction or state.current_round_direction or "PI-selected direction"
            prompt += f"""
INNOVATION PHASE — focused synthesis:
- Current research direction: {direction_ctx}
- Prioritize findings relevant to the current direction for hypothesis evaluation,
  but additionally surface any cross-modal mechanisms that are supported by evidence
  from two or more independent sources, even if those mechanisms are outside the
  current direction. Flag such mechanisms with their evidence sources.
- Evaluate whether the hypothesis is supported or contradicted by the evidence
- Suggest deeper follow-up analyses within this specific direction
"""

        # Try up to 2 times if response is empty
        max_retries = 2
        response = ""
        for attempt in range(max_retries):
            response = await self.chat(prompt, temperature=0.3)
            if response and response.strip():
                break
            logger.warning(f"[PI] Empty response on synthesis attempt {attempt + 1}, retrying...")
            # Remove failed attempt from history to avoid pollution
            self.rollback_last_turn()
            await asyncio.sleep(1)  # Brief delay before retry

        # Use base class safe JSON parser
        data = self._safe_parse_json(response, default={})

        if data and "summary" in data:
            # Ensure bioinformatics is always included in agents_for_next_round
            agents_next = data.get("agents_for_next_round",
                ["bioinformatics", "pathology", "biochemistry", "literature"])
            if "bioinformatics" not in agents_next:
                agents_next.insert(0, "bioinformatics")

            reanalysis_next = data.get("reanalysis_agents", [])

            return RoundResult(
                round_number=state.current_round,
                pathology_findings=data.get("pathology_findings", []),
                bioinformatics_findings=data.get("bioinformatics_findings", []),
                biochemistry_findings=data.get("biochemistry_findings", []),
                literature_findings=data.get("literature_findings", []),
                pi_summary=data.get("summary", ""),
                conflicts=data.get("conflicts", []),
                next_steps=self._extract_next_steps(data),
                new_findings=data.get("new_findings", True),
                agents_for_next_round=agents_next,
                reanalysis_agents=reanalysis_next,
            )

        # If we have a response but couldn't parse JSON, use it as summary
        if response and response.strip():
            return RoundResult(
                round_number=state.current_round,
                pi_summary=response,
                new_findings=True,
                reanalysis_agents=[]
            )

        # If response is completely empty, create a fallback summary
        logger.error(f"[PI] Failed to get synthesis response after {max_retries} attempts")
        return RoundResult(
            round_number=state.current_round,
            pi_summary=f"Round {state.current_round} analysis completed. Results collected from agents pending synthesis.",
            new_findings=True,
            reanalysis_agents=[]
        )

    async def build_cross_modal_synthesis(
        self,
        state: AnalysisState,
        current_direction: str,
        current_agent_raw: Dict[str, Any],
        literature_context: str,
        biochemistry_raw: str,
    ) -> "CrossModalSynthesis":
        """Run a single LLM call over a wide input surface (raw agent result
        dicts + bounded literature context + raw biochemistry measurements) to
        produce a cross-modal evidence matrix and honestly-graded mechanism
        candidates. Mechanism-agnostic: no preset pathway/gene/mechanism names.
        """
        self.refresh_runtime_config_if_needed()
        self.reset_history()

        # Accumulated round results (findings + pi_summary) for cross-round context
        rounds_text = []
        for r in state.round_results:
            all_findings = (
                (r.pathology_findings or [])[:5]
                + (r.bioinformatics_findings or [])[:5]
                + (r.biochemistry_findings or [])[:5]
                + (r.literature_findings or [])[:5]
            )
            rounds_text.append(
                f"### Round {r.round_number} (direction: {r.direction or 'N/A'})\n"
                f"PI summary: {(r.pi_summary or 'N/A')[:600]}\n"
                f"Findings: {'; '.join(str(f) for f in all_findings) if all_findings else 'None'}\n"
                f"Suggested future work (next_steps): {'; '.join((r.next_steps or [])[:10]) if r.next_steps else 'None'}"
            )
        accumulated = "\n\n".join(rounds_text) if rounds_text else "None"

        try:
            current_raw_text = json.dumps(current_agent_raw, ensure_ascii=False, default=str)
        except Exception:
            current_raw_text = str(current_agent_raw)
        # Safety cap on raw dump (~24K chars) to avoid runaway prompts
        current_raw_text = current_raw_text[:24000]

        prompt = f"""You are the PI performing a cross-modal mechanism audit. Correlate evidence ACROSS the four data modalities (biochemistry, pathology, bioinformatics, literature) and propose candidate biological mechanisms, each with honestly-graded evidence.

## Project Context
Project: {state.name}
Research objective: {state.initial_instruction}
Current research direction: {current_direction or "N/A"}

## Accumulated Round Results
{accumulated}

## Current Round Agent Results (raw)
{current_raw_text or "None"}

## Literature Context (current round, bounded)
{literature_context or "None"}

## Biochemistry Raw Measurements (current round)
{biochemistry_raw or "None"}

## Output JSON Schema
{{
    "matrix_markdown": "Markdown table string",
    "candidates": [{{
        "theme": "...",
        "mechanism_statement": "...",
        "supporting_evidence": [{{"source": "biochemistry|pathology|bioinformatics|literature", "evidence": "concrete observation", "strength": "strong|moderate|weak", "limitation": "..."}}],
        "evidence_consistency": "high|medium|low|conflicting",
        "confidence": "high|medium|low",
        "novelty_relative_to_current_direction": "high|medium|low",
        "relation_to_current_direction": "central|complementary|downstream|alternative|uncovered",
        "missing_evidence": ["..."],
        "report_recommendation": "headline|secondary|future_direction|do_not_emphasize"
    }}],
    "excluded_or_low_support_themes": ["..."]
}}

## Constraints
- Do NOT use any preset mechanism names, pathway names, gene sets, or example mechanism categories. Derive ALL row labels and mechanism statements from the evidence only. You MAY cite specific genes, pathways, PMIDs, or values that appear in the input evidence.
- The matrix_markdown column headers must reflect the four agent sources plus an evidence-consistency column, in the same language as this prompt. Do not hardcode labels.
- Each candidate must draw evidence from at least TWO different sources, and at least one must be an experimental-data agent (biochemistry, pathology, or bioinformatics) — not purely literature.
- Each supporting_evidence item must cite a concrete observation (value, gene, pathway, morphological feature, PMID), not a vague summary.
- STRICT evidentiary separation: items that appear only as suggested future experiments (in next_steps or data_requests) can ONLY populate missing_evidence, never supporting_evidence. A candidate built mainly from suggested future experiments must have confidence="low" and report_recommendation="future_direction" unless independent observed data also supports it.
- Populate missing_evidence for every candidate so the report can present mechanisms as "supported but incomplete" rather than conclusive.
- Return a SINGLE JSON object. Do NOT include example candidates or invented mechanisms.
"""

        response = await self.chat(prompt, temperature=0.3)
        return self._parse_cross_modal_response(response)

    def _parse_cross_modal_response(self, response: str) -> "CrossModalSynthesis":
        data = self._safe_parse_json(response, default=None)
        if not isinstance(data, dict) or "parse_error" in data:
            # Entire response unparseable: keep raw text capped at 20K chars
            raw = data.get("raw_response", response) if isinstance(data, dict) else response
            return CrossModalSynthesis(
                matrix_markdown=(raw or "")[:20000],
                candidates=[],
                excluded_or_low_support_themes=[],
            )
        matrix = data.get("matrix_markdown")
        if not isinstance(matrix, str):
            matrix = ""
        candidates = []
        for entry in (data.get("candidates") or []):
            try:
                candidates.append(CrossModalMechanismCandidate(**entry))
            except Exception:
                logger.warning("[PI] Skipping unparseable cross-modal candidate entry")
                continue
        excluded = data.get("excluded_or_low_support_themes") or []
        if not isinstance(excluded, list):
            excluded = []
        return CrossModalSynthesis(
            matrix_markdown=matrix,
            candidates=candidates,
            excluded_or_low_support_themes=excluded,
        )

    async def generate_final_summary(self, state: AnalysisState) -> str:
        """
        Generate final summary of the entire analysis

        Args:
            state: Final analysis state

        Returns:
            Final summary text
        """
        logger.info(f"[PI] Generating final summary after {len(state.round_results)} rounds")
        # Reset message history — the prompt below embeds full state context
        self.reset_history()
        # Build summary of all rounds
        rounds_summary = []
        for result in state.round_results:
            rounds_summary.append(f"""
### Round {result.round_number}
**Pathology findings**: {', '.join(str(f) for f in result.pathology_findings) if result.pathology_findings else 'None'}
**Bioinformatics findings**: {', '.join(str(f) for f in result.bioinformatics_findings) if result.bioinformatics_findings else 'None'}
**Biochemistry findings**: {', '.join(str(f) for f in result.biochemistry_findings) if result.biochemistry_findings else 'None'}
**Literature findings**: {', '.join(str(f) for f in result.literature_findings) if result.literature_findings else 'None'}
**PI Summary**: {result.pi_summary}
""")

        # Cross-modal mechanism candidate context (primary; lightweight literature fallback).
        # O5: gate on at least one *candidate*, not merely a CrossModalSynthesis object.
        # A synthesis with an empty candidates list carries no narrative value, so fall
        # back to the lightweight literature summary instead of emitting an empty header.
        cross_modal_block = ""
        # O5 + consistency with planning.py: filter out do_not_emphasize candidates
        # from the final summary narrative (they remain in the appendix for traceability).
        cm_parts = []
        for r in state.round_results:
            cms = r.cross_modal_synthesis
            if not cms or not cms.candidates:
                continue
            displayable = [c for c in cms.candidates if c.report_recommendation != "do_not_emphasize"]
            if not displayable:
                continue
            if not cm_parts:
                cm_parts.append("\n## Cross-Modal Mechanism Candidates (persisted audit)")
            cm_parts.append(f"\n### From Round {r.round_number}")
            for c in displayable:
                ev = "; ".join(
                    f"{e.source}: {e.evidence} ({e.strength})" for e in c.supporting_evidence
                ) or "N/A"
                cm_parts.append(
                    f"- **{c.theme}** — {c.mechanism_statement}\n"
                    f"  Evidence: {ev}\n"
                    f"  Confidence: {c.confidence}; consistency: {c.evidence_consistency}; "
                    f"relation: {c.relation_to_current_direction}; recommendation: {c.report_recommendation}\n"
                    f"  Missing evidence: {'; '.join(c.missing_evidence) or 'None'}"
                )
        if cm_parts:
            cross_modal_block = "\n".join(cm_parts)
        else:
            # Fallback: lightweight literature context summary
            lit_parts = []
            for rep in state.literature_reports:
                excerpt = (rep.report or "")[:500]
                sigs = ""
                if rep.mechanism_signals:
                    sigs = " | signals: " + "; ".join(s.mechanism for s in rep.mechanism_signals)
                lit_parts.append(f"- {rep.topic}: {excerpt}{sigs}")
            if lit_parts:
                cross_modal_block = "\n## Literature Context Summary\n" + "\n".join(lit_parts)

        mech_lines = []
        for r in state.round_results[-3:]:
            for p in (getattr(r, "data_driven_mechanism_programs", []) or [])[:3]:
                mech_lines.append(f"- {p.to_prompt_text(style='compact', max_observations=2, max_genes=8, max_alternatives=2, max_missing=2)}")
        mechanism_block = ""
        if mech_lines:
            mechanism_block = (
                "\n## Data-Driven Mechanism Programs (candidates, NOT confirmed)\n"
                "Classify each program as one of: covered (already a current direction's "
                "mechanism), integrated (cross-modally supported), uncovered candidate "
                "(distinct evidence not yet pursued), or future/new direction. Use hypothesis "
                "wording; never present a program as a confirmed mechanism.\n"
                + "\n".join(mech_lines)
            )

        prompt = f"""
Based on all the analysis results below, write a complete analysis report:

## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}

## Results from All Rounds
{''.join(rounds_summary)}
{cross_modal_block}{mechanism_block}

Please write a well-structured analysis report including:
1. Research Overview
2. Key Findings
   - Pathological phenotype
   - Single-cell transcriptomic features
   - Biochemical indicator changes
3. Molecular Mechanism Analysis
4. Conclusions
5. Limitations and Future Directions
6. Cross-Modal Mechanism Candidates: for any candidate above, present it with its
   evidence sources, confidence, and explicitly labeled evidence gaps. Present
   low-confidence candidates as "hypotheses with evidence gaps", NOT confirmed conclusions.

Please write in academic paper style.
"""

        # Try up to 2 times if response is empty
        max_retries = 2
        response = ""
        for attempt in range(max_retries):
            response = await self.chat(prompt, temperature=0.3)
            if response and response.strip():
                return response
            logger.warning(f"[PI] Empty response on final summary attempt {attempt + 1}, retrying...")
            self.rollback_last_turn()
            await asyncio.sleep(1)

        # If all retries failed, generate a basic summary from round results
        logger.error(f"[PI] Failed to generate final summary after {max_retries} attempts, using fallback")
        fallback_parts = [
            f"# Analysis Summary for {state.name}\n",
            f"## Research Objective\n{state.initial_instruction}\n",
            f"## Analysis Rounds: {len(state.round_results)}\n"
        ]
        for r in state.round_results:
            fallback_parts.append(f"\n### Round {r.round_number}\n{r.pi_summary or 'No summary available'}\n")

        return "\n".join(fallback_parts)

    def _build_context(self, state: AnalysisState) -> str:
        """Build context string from state"""
        context_parts = []
        
        # Add project info
        context_parts.append(f"## Project Information")
        context_parts.append(f"Project Name: {state.name}")
        context_parts.append(f"Research Objective: {state.initial_instruction}")
        
        # Add current round info
        context_parts.append(f"\nCurrent Round: {state.current_round}")
        
        # Add previous round summaries
        if state.round_results:
            context_parts.append("\n## Historical Analysis Results")
            for result in state.round_results[-3:]:  # Last 3 rounds
                context_parts.append(f"\n### Round {result.round_number}\n{result.pi_summary}")
        
        return "\n".join(context_parts)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse PI response to extract structured data"""
        # Use base class safe JSON parser
        result = self._safe_parse_json(response, default={})

        if result and "summary" in result:
            return result

        return {
            "summary": response if response else "",
            "findings": [],
            "conflicts": [],
            "next_steps": [],
            "should_continue": True,
            "reason": ""
        }
    
    def _extract_next_steps(self, data: Dict[str, Any]) -> List[str]:
        """Extract next steps from parsed data"""
        steps = []
        
        if "next_biochemistry" in data:
            for task in data["next_biochemistry"]:
                steps.append(f"[Biochemistry] {task}")

        if "next_pathology" in data:
            for task in data["next_pathology"]:
                steps.append(f"[Pathology] {task}")

        if "next_bioinformatics" in data:
            for task in data["next_bioinformatics"]:
                steps.append(f"[Bioinformatics] {task}")

        if "next_literature" in data:
            for topic in data["next_literature"]:
                steps.append(f"[Literature] {topic}")
        
        return steps
    
    # ============================================
    # Iterative Bioinformatics Analysis Control
    # ============================================
    
    async def plan_bioinformatics_workflow(
        self,
        state: AnalysisState
    ) -> List[str]:
        """
        Plan the bioinformatics analysis workflow based on research goals
        
        Returns:
            List of high-level analysis steps to perform
        """
        # Build data availability info from unified data_files
        seq_files = [f for f in state.data_files if f.section == "sequencing"]
        data_info = []
        for f in seq_files:
            group = state.get_group_by_id(f.group_id)
            group_label = group.name if group else "Ungrouped"
            sample_label = f"Sample {f.sample_number}" if f.sample_number else ""
            label = f"{group_label} {sample_label}".strip()
            data_info.append(f"- {label} ({f.data_type}): {f.file_path}")

        # Determine analysis characteristics
        seq_group_ids = set(f.group_id for f in seq_files if f.data_type == DATA_TYPE_SCRNASEQ and f.group_id)
        has_multiple_groups = len(seq_group_ids) > 1
        has_velocity = any(f.data_type == "loom" for f in seq_files)

        # Experimental design summary
        design_summary = state.build_experimental_design_summary()

        prompt = f"""
## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}

## Experimental Design
{design_summary}

## Uploaded Sequencing Data
{chr(10).join(data_info) if data_info else "No sequencing data uploaded"}

## Data Characteristics
- Multiple groups for comparison: {"Yes (" + str(len(seq_group_ids)) + " groups), differential/comparative analysis required" if has_multiple_groups else "Single group only"}
- RNA velocity data: {"Yes, cell trajectory analysis possible" if has_velocity else "No"}

Please plan the single-cell sequencing analysis workflow for this project. Return an ordered list of analysis steps.
Note: The first step must be loading data.
{"If multiple groups exist, include steps for merging/integrating data, batch correction, and multi-group comparison analysis." if has_multiple_groups else ""}

Format requirement: Return a JSON array, each element is a brief description of an analysis step.
Example: ["Load data", "Quality control", "Normalization", ...]

Return only the JSON array, no other content.
"""

        response = await self.chat(prompt, temperature=0.3)

        try:
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                steps = json.loads(json_match.group())
                if isinstance(steps, list):
                    return steps
        except json.JSONDecodeError:
            pass

        # Default workflow if parsing fails
        default_steps = [
            "Load data and view basic information",
            "Quality control: filter low-quality cells and genes",
            "Data normalization and log transformation",
            "Identify highly variable genes",
            "PCA dimensionality reduction",
            "Build neighbor graph and UMAP visualization",
            "Leiden clustering",
            "Identify marker genes for each cluster",
            "Cell type annotation",
        ]

        if has_multiple_groups:
            default_steps.extend([
                "Multi-group differential expression analysis",
                "Pathway enrichment analysis",
                "Cell proportion comparison across groups"
            ])

        if has_velocity:
            default_steps.append("RNA velocity analysis")

        return default_steps
    
    async def generate_bioinformatics_instruction(
        self,
        state: AnalysisState,
        current_step: str,
        previous_results: List[Dict[str, Any]],
        step_index: int,
        total_steps: int
    ) -> str:
        """
        Generate specific instruction for bioinformatics agent
        
        Args:
            state: Current analysis state
            current_step: The current step description
            previous_results: Results from previous steps
            step_index: Current step number (0-indexed)
            total_steps: Total number of planned steps
            
        Returns:
            Specific instruction string for the bioinformatics agent
        """
        # Build context from previous results
        prev_context = ""
        if previous_results:
            prev_context = "\n## Previous Analysis Results\n"
            for i, result in enumerate(previous_results[-5:]):  # Last 5 results
                findings = result.get("findings", [])
                desc = result.get("description", "")
                if findings or desc:
                    prev_context += f"\nStep {i+1} ({desc}):\n"
                    for finding in findings:
                        prev_context += f"- {finding}\n"
        
        # Build data path info for first step
        data_paths_info = ""
        if step_index == 0 or "load" in current_step.lower() or "read" in current_step.lower():
            seq_files = [f for f in state.data_files if f.section == "sequencing"]
            data_paths = []
            for f in seq_files:
                group = state.get_group_by_id(f.group_id)
                group_label = group.name if group else "Ungrouped"
                sample_label = f"Sample {f.sample_number}" if f.sample_number else ""
                label = f"{group_label} {sample_label}".strip()
                data_paths.append(f"{label} ({f.data_type}): {f.file_path}")
            if data_paths:
                design = state.build_experimental_design_summary()
                data_paths_info = f"\n## Experimental Design\n{design}\n\n## Data File Paths\n" + "\n".join(data_paths) + "\n"
        
        prompt = f"""
## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}

## Current Progress
Step {step_index + 1}/{total_steps}: {current_step}
{data_paths_info}
{prev_context}

Please generate specific analysis instructions for the Bioinformatics Agent to execute "{current_step}".

Instruction requirements:
1. Be specific and clear about what the Agent should do
2. If loading data, must use the exact file paths provided above
3. Include necessary parameter suggestions (thresholds, method choices, etc.)
4. Specify expected outputs (charts, numbers, etc.)
5. If performing differential analysis, specify comparison groups (treatment vs control)

Return the instruction content directly, no additional formatting needed.
"""
        
        instruction = await self.chat(prompt, temperature=0.3)
        return instruction.strip()
    
    async def evaluate_bioinformatics_result(
        self,
        state: AnalysisState,
        step_description: str,
        result: Dict[str, Any],
        remaining_steps: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate bioinformatics result and decide next action
        
        Args:
            state: Current analysis state
            step_description: What step was just completed
            result: Result from the bioinformatics agent
            remaining_steps: List of remaining planned steps
            
        Returns:
            Dict with:
            - continue_analysis: bool - whether to continue
            - next_step: str - next step to execute (if continue)
            - modify_plan: bool - whether to modify the remaining steps
            - new_steps: List[str] - new steps to insert (if modifying)
            - findings: str - PI's interpretation of the result
            - issues: List[str] - any issues or concerns
        """
        findings = result.get("findings", [])
        error = result.get("error", "")
        success = result.get("success", False)
        
        prompt = f"""
## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}

## Completed Analysis Step
{step_description}

## Execution Result
Success: {"Yes" if success else "No"}
Error: {error or "None"}
Findings: {json.dumps(findings, ensure_ascii=False) if findings else "No specific findings"}

## Remaining Planned Steps
{json.dumps(remaining_steps, ensure_ascii=False) if remaining_steps else "No more planned steps"}

Please evaluate this step's result and decide the next action.

Return in JSON format:
{{
    "continue_analysis": true/false,
    "findings": "PI's interpretation of this step's results",
    "issues": ["issue 1", "issue 2"],
    "need_retry": true/false,
    "retry_instruction": "If retry needed, provide modified instruction",
    "insert_steps": ["If additional analysis steps needed, list here"],
    "skip_steps": ["If some planned steps are no longer needed, list here"]
}}
"""
        
        response = await self.chat(prompt, temperature=0.3)

        # Use base class safe JSON parser
        data = self._safe_parse_json(response, default={})

        if data and "continue_analysis" in data:
            return {
                "continue_analysis": data.get("continue_analysis", True),
                "findings": data.get("findings", ""),
                "issues": data.get("issues", []),
                "need_retry": data.get("need_retry", False),
                "retry_instruction": data.get("retry_instruction", ""),
                "insert_steps": data.get("insert_steps", []),
                "skip_steps": data.get("skip_steps", [])
            }

        # Default: continue if successful
        return {
            "continue_analysis": success and len(remaining_steps) > 0,
            "findings": response[:500] if response else "",
            "issues": [] if success else [error],
            "need_retry": not success,
            "retry_instruction": "",
            "insert_steps": [],
            "skip_steps": []
        }
    
    async def summarize_bioinformatics_analysis(
        self,
        state: AnalysisState,
        all_results: List[Dict[str, Any]]
    ) -> str:
        """
        Summarize the complete bioinformatics analysis

        Args:
            state: Current analysis state
            all_results: All results from the bioinformatics analysis

        Returns:
            Summary string
        """
        # Collect all findings
        all_findings = []
        all_figures = 0

        for result in all_results:
            findings = result.get("findings", [])
            figures = result.get("figures", [])
            desc = result.get("description", "")

            if findings:
                all_findings.append(f"**{desc}**: " + "; ".join(findings))
            all_figures += len(figures)

        prompt = f"""
## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}

## Bioinformatics Analysis Completed
Total {len(all_results)} analysis steps executed
Generated {all_figures} figures

## Findings from Each Step
{chr(10).join(all_findings) if all_findings else "No specific findings recorded"}

Please synthesize the above results and write a concise bioinformatics analysis summary (200-300 words).
Focus on:
1. Cell type composition and changes
2. Major differential expression patterns
3. Key findings related to research objectives
4. Questions requiring further verification or analysis

Respond in English.
"""

        summary = await self.chat(prompt, temperature=0.3)
        return summary.strip()

    # ============================================
    # Publication Figure Review
    # ============================================

    def _build_revision_history_text(self, figure: PublicationFigure) -> str:
        """Build a text summary of all revision history for the PI review prompt."""
        if figure.revision_history:
            lines = []
            for entry in figure.revision_history:
                round_num = entry.get('round', '?')
                hist_feedback = entry.get('feedback', '')
                lines.append(f"  Round {round_num}: {hist_feedback}")
            return "- Previous revision feedback:\n" + "\n".join(lines)
        elif figure.pi_feedback:
            return f"- Previous feedback: {figure.pi_feedback}"
        return ""

    async def review_figure(
        self,
        state: AnalysisState,
        figure: PublicationFigure,
        image_base64: str
    ) -> Dict[str, Any]:
        """
        Review a figure generated by Bioinformatics Agent for publication quality.

        Args:
            state: Current analysis state
            figure: PublicationFigure object with metadata
            image_base64: Base64 encoded PNG image for review

        Returns:
            Dict with:
            - approved: bool - whether the figure passes review
            - feedback: str - overall feedback
            - suggestions: List[str] - specific improvement suggestions
            - quality_score: int - score from 1-10
        """
        logger.info(f"[{self.session_id}][PI] Reviewing figure: {figure.name}")

        prompt = f"""## Figure Review for Publication Quality

You are reviewing a figure for publication in a high-impact scientific journal.

**Research Context:**
Project: {state.name}
Objective: {state.initial_instruction[:500] if state.initial_instruction else 'Biomedical research'}

**Experimental Design (ACTUAL uploaded groups):**
{state.build_experimental_design_summary()}
Group names: {', '.join(g.name for g in state.treatment_groups) if state.treatment_groups else 'No groups defined'}

**CRITICAL CONSTRAINT:** The experiment contains ONLY the groups listed above. Do NOT suggest adding, referencing, or comparing groups that are not in the experimental design. If the figure shows groups not listed above, flag this as a critical data accuracy issue.

**Figure Information:**
- Name: {figure.name}
- Type: {figure.figure_type}
- Description: {figure.description}
- Revision count: {figure.revision_count}
{self._build_revision_history_text(figure)}

**Review Criteria:**
1. **Scientific Accuracy**: Does the figure accurately represent the data?
2. **Clarity**: Is the figure easy to understand? Are labels clear and readable?
3. **Aesthetics**: Is the color scheme appropriate? Is the layout clean?
4. **Completeness**: Are all necessary elements present (legend, axis labels, scale bars, titles)?
5. **Publication Quality**: Would this be acceptable in a high-impact journal (Nature, Cell, etc.)?
6. **Data Fidelity**: Does the figure show ONLY the actual experimental groups listed above? Are there any fabricated or non-existent groups/conditions?

**Respond in JSON format:**
```json
{{
    "approved": true/false,
    "quality_score": 1-10,
    "feedback": "Detailed assessment: describe EXACTLY what is wrong and what the expected result should look like. Reference specific visual elements (axis labels, color scheme, legend, font size, layout, etc.)",
    "strengths": ["strength 1", "strength 2"],
    "issues": ["Specific issue with exact description, e.g., 'X-axis label font size is too small (appears ~6pt, should be >=10pt)'"],
    "suggestions": ["Actionable code-level fix, e.g., 'Add plt.xlabel(..., fontsize=12)' or 'Change colormap from viridis to tab20 for discrete clusters'"],
    "must_fix": ["Critical issue that MUST be fixed. Be specific: what is wrong AND what it should look like instead"],
    "requires_reanalysis": false
}}
```

**requires_reanalysis**: Set to true ONLY if the issue cannot be fixed by modifying the plotting code alone — e.g., wrong clustering results, incorrect differential expression, data needs re-processing upstream. When true, the figure revision loop will be skipped and the issue will be deferred to the next analysis step.

**Critical review instructions:**
- Your feedback will be used to automatically revise the plotting code. Every issue MUST be described precisely enough for a programmer to fix it.
- BAD feedback: "The figure is not clear enough"
- GOOD feedback: "The cluster labels overlap in the upper-right quadrant. Use adjustText or reduce font size from 12 to 8."
- BAD feedback: "Colors are not good"
- GOOD feedback: "Clusters 3 and 7 use nearly identical shades of blue, making them indistinguishable. Use a qualitative palette like tab20."
- Be strict but fair. Only approve figures that meet publication standards.
- If this is a revision (revision_count > 0), focus on whether previous issues were actually addressed. Be more lenient on minor aesthetic issues after 2+ revisions.
- For UMAP/clustering plots: check color distinctness, label clarity, legend placement.
- For heatmaps: check color scale, row/column labels, clustering dendrograms if applicable.
- For violin/dotplots: check group comparisons, statistical annotations if needed.
"""

        # Apply progressive leniency for multiple revisions
        if figure.revision_count >= 3:
            prompt += """
**FINAL REVISION - MUST APPROVE unless critical issues exist:**
- This is the final revision attempt. APPROVE if quality_score >= 5.
- Only reject for: fabricated data, completely missing labels, or scientific errors.
- Aesthetic issues at this stage should be noted in feedback but should NOT prevent approval.
"""
        elif figure.revision_count >= 2:
            prompt += """
**IMPORTANT - REVISION LENIENCY POLICY (revision count >= 2):**
- This figure has been revised multiple times. Focus ONLY on critical issues.
- Approve if quality_score >= 6 AND there are no data accuracy/fabrication issues.
- Minor aesthetic preferences (color choices, exact font sizes, spacing) should NOT block approval.
- Only reject if there are scientific accuracy issues, missing essential labels, or data fidelity problems.
"""

        # Use vision capability to analyze the figure
        if self.has_vision:
            try:
                self._set_llm_context()
                response = await self.llm_client.chat_with_vision(
                    model=self.model,
                    text_prompt=prompt,
                    images=[image_base64],
                    temperature=0.3
                )
                response_text = response.content if response and response.content else ""
            except Exception as e:
                # Vision call failed — disable vision for the rest of this session to avoid
                # repeated failures (e.g. model does not support multimodal input)
                logger.warning(f"[PI] Vision API failed, disabling vision for this session: {e}")
                self.has_vision = False
                response_text = await self.chat(
                    f"{prompt}\n\n[Note: Unable to view image directly. Please provide general guidance based on figure type and description.]",
                    temperature=0.3
                )
        else:
            # User explicitly marked model as no-vision — skip vision call
            self._set_llm_context()
            response_text = await self.chat(
                f"{prompt}\n\n[Note: Vision not available for this model. Please provide general guidance based on figure type and description.]",
                temperature=0.3
            )

        # Use base class safe JSON parser
        data = self._safe_parse_json(response_text, default={})

        if data and "approved" in data:
            result = {
                "approved": bool(data.get("approved", False)),
                "quality_score": int(data.get("quality_score", 5)),
                "feedback": data.get("feedback", ""),
                "strengths": data.get("strengths", []),
                "issues": data.get("issues", []),
                "suggestions": data.get("suggestions", []),
                "must_fix": data.get("must_fix", []),
                "requires_reanalysis": bool(data.get("requires_reanalysis", False))
            }
            logger.info(f"[PI] Figure review result: approved={result['approved']}, score={result['quality_score']}, requires_reanalysis={result['requires_reanalysis']}")
            return result

        # Fallback
        return {
            "approved": False,
            "quality_score": 5,
            "feedback": "Unable to parse review response. Please regenerate the figure.",
            "strengths": [],
            "issues": [],
            "suggestions": ["Ensure figure meets publication standards"],
            "must_fix": [],
            "requires_reanalysis": False
        }

    # ============================================
    # Graphic Abstract Review & Selection
    # ============================================

    def _build_ga_revision_history_text(self, revision_history: List[Dict[str, Any]]) -> str:
        """Build a text summary of graphic abstract revision history for prompts."""
        if not revision_history:
            return ""
        lines = []
        for entry in revision_history:
            round_num = entry.get('round', '?')
            hist_feedback = entry.get('feedback', '')
            lines.append(f"  Round {round_num}: {hist_feedback}")
        return "- Previous revision feedback:\n" + "\n".join(lines)

    async def review_graphic_abstract(
        self,
        state: AnalysisState,
        image_base64: str,
        revision_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Review a graphic abstract for publication quality using vision API.

        Args:
            state: Current analysis state
            image_base64: Base64 encoded graphic abstract image
            revision_history: All previous rounds' feedback

        Returns:
            Dict with feedback, suggestions, must_fix, quality_score
        """
        logger.info(f"[{self.session_id}][PI] Reviewing graphic abstract (round {len(revision_history) + 1})")

        revision_text = self._build_ga_revision_history_text(revision_history)

        prompt = f"""## Graphic Abstract Review

You are reviewing a graphic abstract for a biomedical research publication.

**Research Context:**
Project: {state.name}
Objective: {state.initial_instruction[:500] if state.initial_instruction else 'Biomedical research'}

**Final Summary:**
{state.final_summary[:1000] if state.final_summary else 'Not available'}

**Review Round:** {len(revision_history) + 1}
{revision_text}

**Review Criteria:**
1. **Scientific Accuracy**: Does the graphic abstract correctly represent the key findings?
2. **Visual Communication**: Does it tell the research story at a glance?
3. **Layout & Flow**: Is there a clear visual flow (left-to-right or top-to-bottom)?
4. **Labeling**: Are key elements labeled clearly and legibly?
5. **Color Scheme**: Are colors professional and publication-appropriate?
6. **Completeness**: Are the main findings, cell types, pathways, and conclusions represented?
7. **Publication Quality**: Would this be acceptable for a high-impact journal?

**Respond in JSON format:**
```json
{{
    "quality_score": 1-10,
    "feedback": "Detailed assessment of the graphic abstract. Describe what works well and what needs improvement.",
    "strengths": ["strength 1", "strength 2"],
    "issues": ["specific issue 1", "specific issue 2"],
    "suggestions": ["Describe the visual change needed, e.g., 'Add arrows showing the pathway from treatment to outcome' or 'Use distinct colors for different cell types'"],
    "must_fix": ["Critical visual issue that MUST be addressed, e.g., 'The key finding about X pathway is not represented at all'"]
}}
```

**Review instructions:**
- Your feedback will be used by an image generation model to revise the graphic abstract. Describe visual changes clearly.
- BAD: "The figure is not informative enough"
- GOOD: "The graphic abstract should include a visual representation of the <key signaling pathway> connecting <source cell type> to <target cell type> as identified in the analysis"
- BAD: "Colors need improvement"
- GOOD: "Use distinct colors for tumor cells (red/pink) vs immune cells (blue/green) to improve visual differentiation"
{"- Since this is round " + str(len(revision_history) + 1) + ", focus on whether previous issues were addressed. Be more lenient on minor aesthetic issues." if revision_history else ""}
"""

        if not self.has_vision:
            # No vision capability — cannot meaningfully review graphic abstract
            logger.info(f"[PI] Skipping graphic abstract vision review (has_vision=False)")
            return {
                "quality_score": 6,
                "feedback": "Vision not available for this model. General improvement: ensure clear visual hierarchy and complete representation of key findings.",
                "strengths": [],
                "issues": [],
                "suggestions": ["Ensure all key findings are visually represented", "Use clear labels and professional colors"],
                "must_fix": []
            }

        try:
            self._set_llm_context()
            response = await self.llm_client.chat_with_vision(
                model=self.model,
                text_prompt=prompt,
                images=[image_base64],
                temperature=0.3
            )
            response_text = response.content if response and response.content else ""
        except Exception as e:
            logger.warning(f"[PI] Vision API failed for graphic abstract review: {e}")
            # Cannot meaningfully review without seeing the image
            return {
                "quality_score": 6,
                "feedback": "Unable to review image visually. General improvement: ensure clear visual hierarchy and complete representation of key findings.",
                "strengths": [],
                "issues": [],
                "suggestions": ["Ensure all key findings are visually represented", "Use clear labels and professional colors"],
                "must_fix": []
            }

        data = self._safe_parse_json(response_text, default={})

        if data and "feedback" in data:
            result = {
                "quality_score": int(data.get("quality_score", 5)),
                "feedback": data.get("feedback", ""),
                "strengths": data.get("strengths", []),
                "issues": data.get("issues", []),
                "suggestions": data.get("suggestions", []),
                "must_fix": data.get("must_fix", [])
            }
            logger.info(f"[PI] Graphic abstract review: score={result['quality_score']}")
            return result

        return {
            "quality_score": 5,
            "feedback": "Unable to parse review. Consider improving visual clarity and scientific completeness.",
            "strengths": [],
            "issues": [],
            "suggestions": ["Improve visual hierarchy and labeling"],
            "must_fix": []
        }

    async def select_best_graphic_abstract(
        self,
        state: AnalysisState,
        candidates: List[str],
        revision_history: List[Dict[str, Any]]
    ) -> int:
        """
        Select the best graphic abstract from all candidates using vision API.

        Args:
            state: Current analysis state
            candidates: List of base64 encoded candidate images
            revision_history: All rounds' feedback history

        Returns:
            Index of the best candidate (0-based)
        """
        logger.info(f"[{self.session_id}][PI] Selecting best graphic abstract from {len(candidates)} candidates")

        revision_text = self._build_ga_revision_history_text(revision_history)

        prompt = f"""## Select Best Graphic Abstract

You are reviewing {len(candidates)} versions of a graphic abstract for publication.

**Research Context:**
Project: {state.name}
Objective: {state.initial_instruction[:500] if state.initial_instruction else 'Biomedical research'}

**Final Summary:**
{state.final_summary[:800] if state.final_summary else 'Not available'}

**Revision History:**
{revision_text if revision_text else 'No previous feedback'}

**Images provided (in order):**
{chr(10).join(f'- Image {i+1}: {"Original" if i == 0 else f"Revision {i}"}' for i in range(len(candidates)))}

**Task:**
Compare ALL {len(candidates)} images and select the ONE that is best suited for journal publication.

Consider:
1. Scientific accuracy and completeness of key findings
2. Visual communication clarity
3. Layout, flow, and visual hierarchy
4. Labeling and readability
5. Professional appearance and color scheme
6. Overall impact and storytelling

**Respond in JSON format:**
```json
{{
    "selected_index": <0-based index of the best image>,
    "reason": "Brief explanation of why this version is the best"
}}
```
"""

        if not self.has_vision:
            # No vision — default to last candidate (most revised)
            logger.info(f"[PI] Skipping graphic abstract selection (has_vision=False), using last candidate")
            return len(candidates) - 1

        try:
            self._set_llm_context()
            response = await self.llm_client.chat_with_vision(
                model=self.model,
                text_prompt=prompt,
                images=candidates,
                temperature=0.2
            )
            response_text = response.content if response and response.content else ""
        except Exception as e:
            logger.warning(f"[PI] Vision API failed for graphic abstract selection: {e}")
            # Default to last candidate (most revised)
            return len(candidates) - 1

        data = self._safe_parse_json(response_text, default={})

        if data and "selected_index" in data:
            idx = int(data["selected_index"])
            if 0 <= idx < len(candidates):
                reason = data.get("reason", "No reason provided")
                logger.info(f"[PI] Selected graphic abstract {idx + 1}/{len(candidates)}: {reason}")
                return idx
            else:
                logger.warning(f"[PI] Invalid selected_index {idx}, defaulting to last")

        # Default to last candidate
        return len(candidates) - 1

    # ============================================
    # Text-Mode Graphic Abstract (ASCII diagram)
    # ============================================

    async def generate_text_diagram_prompt(
        self,
        state: AnalysisState
    ) -> str:
        """
        Generate an instruction for the Bioinformatics Agent to draw an ASCII/Unicode
        text-mode pathway diagram. Used when Painter Agent is disabled or fails.

        Returns empty string on failure so the caller can fall back to a hardcoded template.
        """
        logger.info(f"[{self.session_id}][PI] Generating text diagram prompt")

        all_findings = []
        for result in state.round_results:
            all_findings.extend(result.bioinformatics_findings[:3])
            all_findings.extend(result.pathology_findings[:2])
            all_findings.extend(result.biochemistry_findings[:2])

        prompt = f"""## Generate Text-Mode Diagram Instruction

You are directing a bioinformatics collaborator to draw an ASCII/Unicode flow diagram
(not a real image) that summarizes the key signaling pathways or cellular processes
discovered in this study.

**Research Project:**
- Name: {state.name}
- Objective: {state.initial_instruction[:400] if state.initial_instruction else 'Biomedical research'}

**Key Findings:**
{chr(10).join(f'- {f}' for f in all_findings[:10]) if all_findings else 'No specific findings'}

**Final Summary:**
{state.final_summary[:1200] if state.final_summary else 'Summary not yet generated'}

**Task:**
Write a 300-500 word instruction that tells the bioinformatics agent EXACTLY what to
draw. The output will be rendered inside a <pre> block on a web page, so it must be
pure ASCII/Unicode text.

Your instruction MUST:
1. List 5-10 key nodes (ligands, receptors, transcription factors, cell states, phenotypes)
2. State the directional relationships between them (A -> B, B phosphorylates C, etc.)
3. Mention cellular processes to include (proliferation, apoptosis, migration, ...)
4. Use English labels only (non-ASCII characters will break alignment)
5. NOT include drawing symbols — the diagram author will choose them

Return ONLY the instruction text, no markdown headers, no code fences.
"""

        try:
            text = await self.chat_without_history(prompt, temperature=0.4)
        except Exception as e:
            logger.warning(f"[PI] generate_text_diagram_prompt failed: {e}")
            return ""

        return (text or "").strip()

    async def review_text_diagram(
        self,
        state: AnalysisState,
        text_diagram: str,
        revision_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Review an ASCII/Unicode text-mode graphic abstract.

        Pure text, so vision capability is NOT required. Raises on any failure so the
        workflow can fall back to a lower-priority reviewer.
        """
        logger.info(f"[{self.session_id}][PI] Reviewing text diagram (round {len(revision_history) + 1})")

        revision_text = self._build_ga_revision_history_text(revision_history)

        system_prompt = (
            "You are a principal investigator reviewing a text-mode (ASCII/Unicode) "
            "pathway diagram prepared for a publication's graphic abstract. "
            "You read the diagram character by character; it is not an image."
        )

        prompt = f"""## Text Diagram Review

**Research Context:**
Project: {state.name}
Objective: {state.initial_instruction[:400] if state.initial_instruction else 'Biomedical research'}

**Final Summary:**
{state.final_summary[:1000] if state.final_summary else 'Not available'}

**Review Round:** {len(revision_history) + 1}
{revision_text}

**Diagram to review:**
```
{text_diagram}
```

**Review Criteria:**
1. Scientific accuracy — do the nodes and arrows match the key findings?
2. Completeness — are critical pathways, cell states, or processes missing?
3. Clarity — is the directional flow (arrows) unambiguous?
4. Layout — do lines align cleanly (watch for mis-aligned arrows, stray characters)?
5. Labels — are node labels in English, short, scientifically precise?

**Respond in JSON:**
```json
{{
    "quality_score": 1-10,
    "feedback": "Plain-English assessment of the diagram",
    "strengths": ["..."],
    "issues": ["..."],
    "suggestions": ["Describe text edits, e.g., 'Add a node for SMAD4 between SMAD2/3 and the nucleus'"],
    "must_fix": ["Critical omissions that block publication"]
}}
```

Describe edits in terms of nodes and arrows, NOT visual styling (color, size, etc.).
"""

        response_text = await self.chat_without_history(
            prompt, temperature=0.3, system_prompt=system_prompt
        )
        if not response_text or not response_text.strip():
            raise ValueError("PI text diagram review returned empty response")

        data = self._safe_parse_json(response_text, default={})
        if not data or "feedback" not in data:
            raise ValueError("PI text diagram review did not return parseable JSON with 'feedback'")

        result = {
            "quality_score": int(data.get("quality_score", 5)),
            "feedback": data.get("feedback", ""),
            "strengths": data.get("strengths", []),
            "issues": data.get("issues", []),
            "suggestions": data.get("suggestions", []),
            "must_fix": data.get("must_fix", [])
        }
        logger.info(f"[PI] Text diagram review: score={result['quality_score']}")
        return result

    async def select_best_text_diagram(
        self,
        state: AnalysisState,
        candidates: List[str],
        revision_history: List[Dict[str, Any]]
    ) -> int:
        """
        Select the best text-mode diagram from multiple ASCII candidates.

        Pure text, no vision needed. On failure returns the last (most revised) candidate.
        """
        logger.info(f"[{self.session_id}][PI] Selecting best text diagram from {len(candidates)} candidates")

        revision_text = self._build_ga_revision_history_text(revision_history)

        candidates_block = "\n\n".join(
            f"--- Candidate {i+1} ---\n{c}" for i, c in enumerate(candidates)
        )

        system_prompt = (
            "You are a principal investigator comparing multiple text-mode pathway "
            "diagrams and selecting the best one for a publication graphic abstract."
        )

        prompt = f"""## Select Best Text Diagram

**Research Context:**
Project: {state.name}
Objective: {state.initial_instruction[:400] if state.initial_instruction else 'Biomedical research'}

**Final Summary:**
{state.final_summary[:800] if state.final_summary else 'Not available'}

**Revision History:**
{revision_text if revision_text else 'No previous feedback'}

**Candidates ({len(candidates)} versions):**
{candidates_block}

**Task:**
Select the ONE candidate that is most scientifically accurate, complete, and cleanly
laid out. Prefer later revisions only if they genuinely addressed feedback; otherwise
an earlier version may win.

**Respond in JSON:**
```json
{{
    "selected_index": <0-based index>,
    "reason": "Brief justification"
}}
```
"""

        try:
            response_text = await self.chat_without_history(
                prompt, temperature=0.2, system_prompt=system_prompt
            )
        except Exception as e:
            logger.warning(f"[PI] select_best_text_diagram failed: {e}")
            return len(candidates) - 1

        data = self._safe_parse_json(response_text or "", default={})
        if data and "selected_index" in data:
            idx = int(data["selected_index"])
            if 0 <= idx < len(candidates):
                logger.info(f"[PI] Selected text diagram {idx + 1}/{len(candidates)}")
                return idx

        return len(candidates) - 1

    # ============================================
    # Integrated Report Generation
    # ============================================

    async def generate_integrated_report(
        self,
        state: AnalysisState
    ) -> str:
        """
        Generate an integrated report with embedded figure and table references.
        This replaces the separate Report generation and integrates it into the analysis flow.

        Args:
            state: Analysis state with all findings and figures

        Returns:
            Markdown formatted report with figure/table references
        """
        logger.info(f"[{self.session_id}][PI] Generating integrated report")

        # Collect approved figures
        figures_info = []
        for i, fig in enumerate(state.publication_figures):
            status = "Approved" if fig.review_status == FigureReviewStatus.APPROVED else "Pending"
            figures_info.append(
                f"Figure {i+1}: {fig.name} ({fig.figure_type}) - {fig.description} [{status}]"
            )

        # Collect tables
        tables_info = []
        for i, table in enumerate(state.publication_tables):
            tables_info.append(f"Table {i+1}: {table.name} - {table.description}")

        # Collect round results
        rounds_summary = []
        for result in state.round_results:
            rounds_summary.append(f"""
### Round {result.round_number}
**Key Findings:**
- Pathology: {', '.join(result.pathology_findings[:3]) if result.pathology_findings else 'N/A'}
- Bioinformatics: {', '.join(result.bioinformatics_findings[:5]) if result.bioinformatics_findings else 'N/A'}
- Biochemistry: {', '.join(result.biochemistry_findings[:3]) if result.biochemistry_findings else 'N/A'}

**Summary:** {result.pi_summary[:500] if result.pi_summary else 'N/A'}
""")

        mech_lines = []
        rounds = state.round_results[-3:]
        for rr in rounds:
            for p in (getattr(rr, "data_driven_mechanism_programs", []) or [])[:3]:
                mech_lines.append(f"- {p.to_prompt_text(style='direction', max_observations=2, max_genes=8, max_missing=2)}")
        mech_block = ""
        if mech_lines:
            mech_block = (
                "## Data-Driven Mechanism Programs\n"
                "Moderate/strong programs with at least two triggering observations MUST appear "
                "as candidate mechanisms (with missing evidence and hypothesis wording). Do NOT "
                "demote them to pure future work unless contradictory evidence is explicitly stated.\n"
                + "\n".join(mech_lines)
            )

        prompt = f"""## Generate Integrated Research Report

**Project Information:**
- Name: {state.name}
- Objective: {state.initial_instruction}
- Total Rounds: {len(state.round_results)}

**Available Figures:**
{chr(10).join(figures_info) if figures_info else "No figures generated"}

**Available Tables:**
{chr(10).join(tables_info) if tables_info else "No tables generated"}

**Analysis Rounds Summary:**
{''.join(rounds_summary)}

{mech_block}

**Task:**
Generate a comprehensive research report in Markdown format suitable for publication. The report should:

1. **Title**: Create an appropriate title for the research
2. **Abstract**: 150-250 words summarizing the research
3. **Introduction**: Background and research objectives (2-3 paragraphs)
4. **Results**:
   - Organize by major findings
   - Reference figures using format: (Figure X) or "As shown in Figure X, ..."
   - Reference tables using format: (Table X)
   - Include key statistics and observations
5. **Discussion**:
   - Interpret findings in context of research objectives
   - Compare with literature
   - Discuss implications
6. **Conclusions**: Key takeaways (1 paragraph)
7. **Methods Summary**: Brief overview of analysis methods

**Important:**
- Use academic writing style
- Include figure/table references naturally in the text
- Write in English
- Focus on biological/clinical significance, not just technical details
"""

        # Try up to 2 times if response is empty
        max_retries = 2
        for attempt in range(max_retries):
            report = await self.chat(prompt, temperature=0.3)
            if report and report.strip():
                return report.strip()
            logger.warning(f"[PI] Empty response on integrated report attempt {attempt + 1}, retrying...")
            self.rollback_last_turn()
            await asyncio.sleep(1)

        # Fallback if all retries failed
        logger.error(f"[PI] Failed to generate integrated report after {max_retries} attempts")
        return f"# {state.name}\n\n## Summary\n\n{state.final_summary or 'Analysis completed. Please see round results for details.'}"

    async def select_brief_figures(
        self,
        state: AnalysisState,
        all_figures: List[Dict[str, str]]
    ) -> List[str]:
        """
        Select figure names to display in the Brief Report (Result page).
        Uses AI judgment based on figure names/types and analysis context.

        Args:
            state: Analysis state (for research context)
            all_figures: All available figures from workspace, each with "name" and "type"

        Returns:
            List of figure names that should appear in the brief report
        """
        if not all_figures:
            return []

        figures_list = "\n".join(
            f"- {f['name']} (type: {f['type']})"
            for f in all_figures
        )

        prompt = f"""You are reviewing figures generated during a bioinformatics analysis.

Research objective: {state.initial_instruction}

All available figures:
{figures_list}

Select which figures should appear in the Brief Results display (shown to the researcher immediately after analysis).
Include figures that are:
- Key results: cell type UMAP with annotations, differential expression volcano plots, cell composition comparisons, key marker dotplots/violins
- Essential visualizations that directly answer the research question

Exclude figures that are:
- QC / technical diagnostics (qc_violin, pca_variance, hvg_plot, and similar)
- Intermediate/exploratory plots superseded by a final version (e.g. exclude umap_initial if umap_cell_types exists)
- Redundant older versions (prefer _v2 or _final over earlier versions of the same figure)

Return ONLY a JSON array of the selected figure names, with no other text.
Example: ["umap_cell_types", "cell_type_proportions", "volcano_<celltype>_<condition>_vs_<control>"]
"""

        try:
            self._set_llm_context()
            response = await self.llm_client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=REASONING_MODEL_MIN_TOKENS  # reasoning models need headroom or content comes back empty
            )
            text = (response.content or "").strip()
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.warning(f"[PI] select_brief_figures failed: {e}")

        # Fallback: include all figures
        return [f["name"] for f in all_figures]

    # ============================================
    # Graphic Abstract Prompt Generation
    # ============================================

    async def generate_graphic_abstract_prompt(
        self,
        state: AnalysisState
    ) -> str:
        """
        Generate a summary prompt for the Painter Agent to create a graphic abstract.

        Args:
            state: Analysis state with all findings

        Returns:
            Summary text for graphic abstract generation
        """
        logger.info(f"[{self.session_id}][PI] Generating graphic abstract prompt")

        # Extract key findings
        all_findings = []
        for result in state.round_results:
            all_findings.extend(result.bioinformatics_findings[:3])
            all_findings.extend(result.pathology_findings[:2])
            all_findings.extend(result.biochemistry_findings[:2])

        prompt = f"""## Generate Graphic Abstract Summary

**Research Project:**
- Name: {state.name}
- Objective: {state.initial_instruction}

**Key Findings:**
{chr(10).join(f'- {f}' for f in all_findings[:10]) if all_findings else "No specific findings"}

**Final Summary:**
{state.final_summary[:1500] if state.final_summary else "Summary not yet generated"}

**Task:**
Write a concise summary (100-150 words, STRICTLY under 200 words) that captures the essence of this research for a graphic abstract.

IMPORTANT: The downstream image generation model has a 30K token input limit.
Keep your summary brief and focused — long summaries will be truncated and lose important context.

The summary should:
1. State the main research question or hypothesis
2. Highlight 2-3 key discoveries (cell types, pathways, mechanisms)
3. Describe the main conclusion or clinical/biological significance
4. Be written in a way that can guide visual illustration

Focus on:
- Key cell types involved (e.g., epithelial cells, immune cells, stromal cells)
- Important molecular pathways or genes
- Treatment effects or disease mechanisms
- Comparisons between conditions (control vs treatment)

Return ONLY the summary text, no additional formatting.
"""

        # Try up to 2 times if response is empty
        max_retries = 2
        for attempt in range(max_retries):
            summary = await self.chat(prompt, temperature=0.4)
            if summary and summary.strip():
                return summary.strip()
            logger.warning(f"[PI] Empty response on graphic abstract prompt attempt {attempt + 1}, retrying...")
            self.rollback_last_turn()
            await asyncio.sleep(1)

        # Fallback if all retries failed
        logger.error(f"[PI] Failed to generate graphic abstract prompt after {max_retries} attempts")
        # Generate a basic summary from available data
        findings = []
        for r in state.round_results:
            findings.extend(r.bioinformatics_findings[:2])
            findings.extend(r.pathology_findings[:1])
        findings_text = "; ".join(findings[:5]) if findings else "Single-cell analysis completed"
        return f"Study: {state.name}. Objective: {state.initial_instruction[:200]}. Key findings: {findings_text}."
