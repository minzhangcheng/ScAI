"""
PI Agent - Principal Investigator that coordinates all agents
"""
from typing import Dict, Any, List, Optional
import asyncio
import json
import re
import logging

from .base import BaseAgent

# Configure logger
logger = logging.getLogger(__name__)
from ..models.schemas import (
    AgentType, AnalysisState, MessageRole, RoundResult, ExplorationReport,
    PublicationFigure, FigureReviewStatus, GraphicAbstract
)
from ..utils.llm_client import OpenAICompatibleClient


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

    def __init__(
        self,
        llm_client: OpenAICompatibleClient,
        model: str,
        system_prompt: str,
        config: Optional['AgentConfig'] = None
    ):
        from .base import AgentConfig
        super().__init__(
            agent_type=AgentType.PI,
            llm_client=llm_client,
            model=model,
            system_prompt=system_prompt,
            config=config or AgentConfig()
        )
        
    async def process(
        self,
        state: AnalysisState,
        instruction: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Process instruction and coordinate analysis"""
        
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
   - Each topic should be a focused research question (e.g., "Role of TGF-beta signaling in liver fibrosis progression")
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
            "bioinformatics_tasks": ["Execute standard scRNA-seq/snRNA-seq analysis workflow"],
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
7. ALL clusters: condition-level DEG (treatment vs control per cluster) + GO/KEGG enrichment
8. Cell-cell communication analysis (CellChat or NicheNet)

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
            "bioinformatics_tasks": [
                "Execute standard scRNA-seq analysis: QC, normalization, clustering, annotation, "
                "then condition-level DEG + GO/KEGG for ALL clusters, cell communication analysis"
            ],
            "biochemistry_tasks": ["Summarize existing biochemical indicators"],
            "literature_topics": ["Cell types and pathways relevant to the tissue and disease context"]
        }

    async def review_cluster_annotations(
        self,
        state: AnalysisState,
        annotations: Dict[str, Any],
        tissue: str,
        organism: str
    ) -> Dict[str, Dict[str, str]]:
        """
        PI reviews Literature Agent's cluster annotations and assigns confidence scores.

        Returns dict mapping cluster_id to adjusted annotation fields.
        """
        logger.info(f"[PI] Reviewing cluster annotations for {organism} {tissue}")

        # Build compact annotation summary for PI review
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

        prompt = f"""You are reviewing cell type annotations generated by a literature search for {organism} {tissue}.

## Cluster Annotations from Literature Search
{ann_text}

For EACH cluster, evaluate:
1. Is the cell type annotation correct given the marker genes and tissue context?
2. Assign a confidence level:
   - "high": Markers strongly and specifically match the annotated cell type in this tissue
   - "medium": Markers partially match or could belong to a related cell type
   - "low": Markers are ambiguous, contradictory, or insufficient for confident annotation
3. For clusters with "medium" or "low" confidence, provide:
   - alternative_cell_types: other possible cell types these markers could indicate
   - low_confidence_reason: explanation of why confidence is not high (e.g. shared markers with other types, insufficient marker specificity, tissue-atypical expression)

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

        return result

    async def generate_exploration_report(
        self,
        state: AnalysisState,
        round_result: RoundResult
    ) -> ExplorationReport:
        """
        Generate structured exploration report (Report A) after Phase 1.

        Summarizes all findings into a structured format for PI decision-making
        in the Innovation phase and for user review.
        """
        logger.info(f"[PI] Generating exploration report")
        self.reset_history()

        # Build annotation context
        ann_summary = "No annotation data"
        if state.annotation_state and state.annotation_state.cluster_annotations:
            ann_lines = []
            for ca in state.annotation_state.cluster_annotations:
                override_note = f" (user override: {ca.user_override})" if ca.user_override else ""
                ann_lines.append(
                    f"- Cluster {ca.cluster_id}: {ca.cell_type}{override_note} "
                    f"[confidence: {ca.confidence}] markers: {', '.join(ca.key_markers[:10])}"
                )
            ann_summary = "\n".join(ann_lines)

        prompt = f"""Based on the Round 1 exploration results, generate a structured exploration report.

## Round 1 Results
**Bioinformatics findings**: {json.dumps(round_result.bioinformatics_findings[:15], ensure_ascii=False)}
**Pathology findings**: {json.dumps(round_result.pathology_findings[:5], ensure_ascii=False)}
**Biochemistry findings**: {json.dumps(round_result.biochemistry_findings[:5], ensure_ascii=False)}
**Literature findings**: {json.dumps(round_result.literature_findings[:5], ensure_ascii=False)}
**PI Summary**: {round_result.pi_summary[:1000]}

## Cell Annotations
{ann_summary}

Generate a comprehensive structured report in JSON:
{{
    "cluster_overview": "Summary of all clusters: count, major cell type categories, proportions, condition distribution",
    "data_quality_warnings": ["Warning about clusters with very few cells (<50)", "Warning about poor QC metrics", ...],
    "strongest_heterogeneity": ["Which clusters/cell types show the strongest changes between conditions", ...],
    "key_pathway_changes": ["Significant pathway enrichment results across clusters", ...],
    "abnormal_markers": ["Clusters with unexpected marker expression patterns", ...],
    "low_confidence_annotations": ["Clusters with uncertain cell type identity", ...],
    "pi_direction_assessment": "Your overall assessment of the most promising research directions based on ALL the data",
    "research_direction_candidates": [
        {{
            "direction": "A specific research question, e.g., 'Macrophage polarization shift in disease group'",
            "rationale": "Why this is interesting based on the data (cite specific findings)",
            "expected_innovation": "High/Medium/Low — Is this a well-studied area (Low) or an emerging/under-explored area (High)? Provide brief evidence.",
            "expected_novelty": "High/Medium/Low — Is this finding expected (Low) or surprising/counter-intuitive (High)? Explain WHY.",
            "data_reliability": "High/Medium/Low — Based on sample sizes, statistical significance, and effect sizes, how reliable is the supporting evidence? Cite specific data."
        }},
        ...
    ]
}}

IMPORTANT:
- Include 3-5 research direction candidates
- Each candidate should be a specific, actionable research question (not vague)
- Base all assessments on actual data findings, not assumptions
- For EACH candidate, you MUST provide all 5 fields: direction, rationale, expected_innovation, expected_novelty, data_reliability
- Each evaluation field should start with the score (High/Medium/Low) followed by a brief explanation
"""

        response = await self.chat(prompt, temperature=0.3)
        data = self._safe_parse_json(response, default={})

        # Build ExplorationReport from parsed data
        valid_fields = {k: v for k, v in data.items() if k in ExplorationReport.model_fields}
        report = ExplorationReport(**valid_fields)

        return report

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
- **Targeted DEG**: Specify which cell populations to compare, which contrasts, and threshold strategy
- **Pathway enrichment**: GO/KEGG for broad screening; GSEA/ssGSEA for quantitative pathway scoring; MSigDB hallmark gene sets for cancer/metabolism
- **Gene signature scoring**: Score cells for specific gene signatures (e.g., ferroptosis, autophagy, EMT) using sc.tl.score_genes
- **Trajectory/pseudotime**: Diffusion pseudotime (DPT), PAGA for lineage inference — specify root cell type
- **Transcription factor activity**: decoupleR / pySCENIC to infer TF activity from expression
- **Cell-cell communication**: CellChat or CellPhoneDB focused on relevant cell types and signaling pathways
- **Subpopulation re-clustering**: Re-cluster specific cell types at higher resolution to find subtypes
- **Correlation analysis**: Gene-gene correlation, module scoring, or regression analysis within cell populations
- **Compositional analysis**: Compare cell type proportions between conditions with statistical testing
- **Marker gene visualization**: Dot plots, violin plots, feature plots for key genes across conditions

IMPORTANT:
- Generate AT LEAST 4-6 bioinformatics tasks to ensure comprehensive analysis
- Each task should specify: what to analyze, which method to use, and which cell types to focus on
- Include both hypothesis-confirming AND exploratory analyses
- Do NOT repeat analyses already completed in Round 1 (basic pipeline, annotation, etc.)
- The Jupyter kernel retains all Round 1 variables — build upon existing results

Reply in JSON format:
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
    ) -> List[Dict[str, str]]:
        """
        PI autonomously selects 2 best research directions from exploration report candidates.
        Used when user clicks 'Let PI Decide' or countdown expires.

        Returns:
            List of dicts, each with 'direction' and 'notes' keys.
        """
        logger.info(f"[PI] Selecting 2 best research directions")

        candidates = []
        if state.exploration_report and state.exploration_report.research_direction_candidates:
            candidates = state.exploration_report.research_direction_candidates

        if not candidates:
            return [{"direction": "Continue analysis based on Round 1 findings", "notes": ""}]

        exploration_context = ""
        if state.exploration_report:
            exploration_context = f"""
Cluster Overview: {state.exploration_report.cluster_overview[:500]}
Key Pathway Changes: {json.dumps(state.exploration_report.key_pathway_changes[:5], ensure_ascii=False)}
Strongest Heterogeneity: {json.dumps(state.exploration_report.strongest_heterogeneity[:5], ensure_ascii=False)}
Data Quality Warnings: {json.dumps(state.exploration_report.data_quality_warnings[:3], ensure_ascii=False)}
PI Direction Assessment: {state.exploration_report.pi_direction_assessment[:500]}
"""

        prompt = f"""You are the PI selecting the 2 BEST research directions for the innovation phase.

## Exploration Report
{exploration_context}

## Candidates
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Select exactly 2 directions that together will provide the most comprehensive and impactful research outcomes.

SELECTION CRITERIA:
1. **Reliability** is the baseline: low-reliability directions should be avoided
2. **Innovation + Novelty** together determine priority
3. The 2 selected directions should ideally be complementary (not redundant)

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
    "rationale": "Why these 2 directions were chosen over others"
}}
"""

        response = await self.chat(prompt, temperature=0.3)
        data = self._safe_parse_json(response, default={})

        selected = data.get("selected_directions", [])
        if not selected or len(selected) < 1:
            # Fallback: pick first 2 candidates
            selected = [
                {"direction": c.get("direction", ""), "notes": ""}
                for c in candidates[:2]
            ]

        # Ensure at least 1 direction
        if not selected:
            selected = [{"direction": "Continue analysis based on Round 1 findings", "notes": ""}]

        logger.info(f"[PI] Selected {len(selected)} directions: {[d['direction'][:50] for d in selected]}")
        return selected

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

        # Build completed results summary
        results_parts = []
        for key, findings in completed_results.items():
            label = key.replace("_findings", "").capitalize()
            truncated = findings[:5]
            results_parts.append(
                f"### {label} Findings\n"
                + "\n".join(f"- {f[:200]}" for f in truncated)
            )
        results_summary = "\n\n".join(results_parts)

        data_summary = state.build_experimental_design_summary() if state else "Not available"

        prompt = f"""You are the PI coordinating a multi-agent analysis.

## Research Objective
{research_objective}

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
        stop_keywords = ['stop', 'terminate', 'end', 'finish', 'conclude', 'summarize',
                         '停止', '终止', '结束', '总结']
        
        if any(keyword in human_lower for keyword in stop_keywords):
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

        # Quick path: [CONTINUE] marker means just continue without additional processing
        if user_message and user_message.strip() == '[CONTINUE]':
            logger.info("[PI] Continue marker detected, proceeding to next round")
            return {
                "action": "continue",
                "response": "Continuing to next round of analysis.",
                "file_assignments": {},
                "biochemistry_values": {},
                "adjusted_instructions": None,
                "focus_areas": [],
                "reason": "User requested to continue"
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
        stop_keywords = ['stop', 'terminate', 'end', 'finish', 'conclude', 'report',
                         '停止', '终止', '结束', '总结', '报告']
        
        if any(keyword in user_lower for keyword in stop_keywords):
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
    ) -> RoundResult:
        """
        Synthesize results from all agents for a round

        Args:
            state: Current analysis state
            *_result: Results from each agent

        Returns:
            RoundResult with synthesis
        """
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

        prompt = f"""
## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}
{previous_context}

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
    "agents_for_next_round": ["bioinformatics", ...]
}}

IMPORTANT: When generating next_* tasks:
- Do NOT repeat tasks already completed in previous rounds
- Only propose NEW directions, deeper investigations, or follow-up analyses
- If a previous round already completed standard processing (QC, clustering, cell annotation), do NOT re-assign these
- Focus next_* on specific hypotheses, pathway analyses, or validation tasks based on current findings

IMPORTANT for next_literature topics:
- Each topic should be a specific, searchable research question (e.g., "Role of macrophage polarization in NASH liver fibrosis")
- Include biological context (disease name, tissue, species) in each topic
- Focus on filling knowledge GAPS identified in this round's literature findings
- Avoid overly broad topics or raw gene lists

IMPORTANT: agents_for_next_round controls which agents are ACTIVATED in the next round:
- "bioinformatics" MUST ALWAYS be included (it is the core analysis agent)
- Only include "pathology" if new image analysis or re-examination is needed
- Only include "biochemistry" if biochemistry data needs further analysis or reinterpretation
- Only include "literature" if new literature search is needed for newly discovered topics
- EXCLUDE agents that have no meaningful new work to save time and resources
"""

        # Add phase-specific synthesis instructions
        if state.analysis_phase == "exploration":
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
- Focus your synthesis on findings relevant to this direction
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
            )

        # If we have a response but couldn't parse JSON, use it as summary
        if response and response.strip():
            return RoundResult(
                round_number=state.current_round,
                pi_summary=response,
                new_findings=True
            )

        # If response is completely empty, create a fallback summary
        logger.error(f"[PI] Failed to get synthesis response after {max_retries} attempts")
        return RoundResult(
            round_number=state.current_round,
            pi_summary=f"Round {state.current_round} analysis completed. Results collected from agents pending synthesis.",
            new_findings=True
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
        
        prompt = f"""
Based on all the analysis results below, write a complete analysis report:

## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}

## Results from All Rounds
{''.join(rounds_summary)}

Please write a well-structured analysis report including:
1. Research Overview
2. Key Findings
   - Pathological phenotype
   - Single-cell transcriptomic features
   - Biochemical indicator changes
3. Molecular Mechanism Analysis
4. Conclusions
5. Limitations and Future Directions

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
        seq_group_ids = set(f.group_id for f in seq_files if f.data_type == "scRNA-seq" and f.group_id)
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
            # Fallback to text-only if vision not available
            logger.warning(f"[PI] Vision API failed, using text-only review: {e}")
            response_text = await self.chat(
                f"{prompt}\n\n[Note: Unable to view image directly. Please provide general guidance based on figure type and description.]",
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
- GOOD: "The graphic abstract should include a visual representation of the TGF-beta signaling pathway connecting fibroblasts to immune cells"
- BAD: "Colors need improvement"
- GOOD: "Use distinct colors for tumor cells (red/pink) vs immune cells (blue/green) to improve visual differentiation"
{"- Since this is round " + str(len(revision_history) + 1) + ", focus on whether previous issues were addressed. Be more lenient on minor aesthetic issues." if revision_history else ""}
"""

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
Example: ["umap_cell_types", "cell_type_proportions", "volcano_hepatocytes_dko_vs_wt"]
"""

        try:
            self._set_llm_context()
            response = await self.llm_client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512
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
