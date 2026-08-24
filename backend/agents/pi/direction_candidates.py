"""Generation and evidence-scoring of research-direction candidates.

Moved verbatim out of backend/agents/pi_agent.py; composed into PIAgent.
"""
from typing import Dict, Any, List
import json
import logging

from ...utils.formatting import safe_parse_json_array
from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY

# Configure logger
logger = logging.getLogger(__name__)
from ...models.schemas import AnalysisState, RoundResult, ExplorationReport

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


class DirectionCandidatesMixin:
    """Generation and evidence-scoring of research-direction candidates."""

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

## Mechanism Naming and Hypothesis Policy
{MECHANISM_EVIDENCE_POLICY}

For each candidate, provide ALL of the following fields:
- direction_id: Unique identifier like "dir_001"
- direction: A specific, actionable research question
- core_observation: Concrete Round 1 data observation motivating the candidate. Must cite specific data elements such as cluster ID, cell type, gene, pathway, fold-change, p-value, correlation, module score, or sample-level observation.
- main_hypothesis: Primary testable hypothesis.
- alternative_hypotheses: Array of at least two strings unless the data clearly excludes alternatives. Consider different transcriptional regulators, different ions/metals, different biological process classes, different causal direction, and technical artifacts. Never return a single string.
- artifact_risks: Array of strings describing technical or interpretive risks such as batch effect, doublets, low-quality cluster, annotation uncertainty, ambient RNA, dropout, or confounded design. Never return a single string.
- rationale: Why this direction is interesting based on the data
- supporting_data: Array of objects with string fields source, evidence, and strength (high|medium|low). Never return a scalar.
- key_genes: Array of relevant gene strings. Never return a single string.
- key_cell_types: Array of relevant cell-type strings. Never return a single string.
- key_pathways: Array of relevant pathway strings. Never return a single string.
- key_uncertainties: Array of strings describing what remains uncertain or needs validation.
- suggested_literature_topics: Array of natural-language topics for literature search.
- suggested_validation_analysis: Array of computational follow-up analysis strings.
- suggested_wet_lab_validation: Array of wet-lab validation experiment strings.
- decision_criteria: Array of strings describing results that would support or weaken the hypothesis.
- biology_expectedness: one of "expected", "partially_expected", "unexpected", "unknown"
- functional_impact_hypothesis: Brief hypothesis about functional/clinical impact
- prerequisite_dependencies: Array of direction_id strings that should be investigated first (empty array if none)
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
        candidates = data.get("candidates", []) if isinstance(data, dict) else []
        if not isinstance(candidates, list):
            candidates = []
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
        return [
            _normalize_direction_candidate(candidate)
            for candidate in candidates
            if isinstance(candidate, dict)
        ]

    async def generate_integration_direction_candidates(
        self,
        state: "AnalysisState",
        cumulative_rounds: List["RoundResult"],
        integration_report: str,
        new_literature: str,
    ) -> List[Dict[str, Any]]:
        """NEW direction candidates from the CUMULATIVE integration of all
        innovation rounds (round 2..current). Excludes already-chosen directions
        (dedup). Output field schema matches generate_raw_direction_candidates."""
        self.reset_history()
        already_chosen = [d.get("direction", "") for d in (state.user_chosen_directions or [])]
        rounds_block = []
        for rr in cumulative_rounds:
            rounds_block.append(
                f"### Round {rr.round_number} — direction: {rr.direction or 'N/A'}\n"
                f"- Bioinformatics: {json.dumps(rr.bioinformatics_findings[:8], ensure_ascii=False)}\n"
                f"- Pathology: {json.dumps(rr.pathology_findings[:3], ensure_ascii=False)}\n"
                f"- Biochemistry: {json.dumps(rr.biochemistry_findings[:3], ensure_ascii=False)}\n"
                f"- Literature: {json.dumps(rr.literature_findings[:3], ensure_ascii=False)}\n"
                f"- Summary: {(rr.pi_summary or '')[:600]}"
            )
        prompt = f"""Based on the CUMULATIVE integration of ALL innovation rounds so far, propose 3-5 NEW research direction candidates that go BEYOND what has already been investigated.

## Project Context
- Name: {state.name}
- Research Objective: {state.initial_instruction}
- Species: {state.species or "Not specified"}
{state.build_experimental_design_summary()}

## Cross-Direction Integration Analysis
{integration_report[:4000]}

## All Innovation Rounds (cumulative, round 2..current)
{chr(10).join(rounds_block)}

## Newly Retrieved Literature (integration-facing search)
{new_literature[:3000]}

## Already-Investigated Directions (DO NOT repeat or trivially rephrase)
{json.dumps(already_chosen, ensure_ascii=False)}

## Mechanism Naming and Hypothesis Policy
{MECHANISM_EVIDENCE_POLICY}

Only propose directions genuinely NEW relative to the already-investigated list and grounded in concrete cumulative observations. If nothing sufficiently novel and evidence-supported exists, return an empty list [].

For each candidate, provide ALL fields with these exact JSON types:
- direction_id, direction, core_observation, main_hypothesis, rationale,
  functional_impact_hypothesis, organism, tissue, experimental_context: strings
- alternative_hypotheses, artifact_risks, key_genes, key_cell_types,
  key_pathways, key_uncertainties, suggested_literature_topics,
  suggested_validation_analysis, suggested_wet_lab_validation,
  decision_criteria: arrays of strings (never a single string)
- supporting_data: an array of objects, each with string fields
  source, evidence, and strength (high|medium|low); never a single string
- biology_expectedness: expected|partially_expected|unexpected|unknown
- prerequisite_dependencies: array of direction_id strings

Example field shapes:
{{"alternative_hypotheses": ["..."], "artifact_risks": ["..."],
  "supporting_data": [{{"source": "round 2 bioinformatics",
  "evidence": "...", "strength": "high"}}],
  "suggested_validation_analysis": ["..."],
  "suggested_wet_lab_validation": ["..."], "decision_criteria": ["..."]}}

Return a JSON array. Return [] if no sufficiently novel direction exists."""
        response = await self.chat(prompt, temperature=0.3)
        # This prompt contracts for a top-level JSON array; the dict parser drops valid arrays.
        data = safe_parse_json_array(response, default=[])
        if not isinstance(data, list):
            return []
        return [
            _normalize_direction_candidate(candidate)
            for candidate in data
            if isinstance(candidate, dict)
        ]

    async def score_integration_directions_with_evidence(
        self,
        state: "AnalysisState",
        raw_candidates: List[Dict[str, Any]],
        literature_evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Score integration-round candidates with literature novelty evidence.
        Returns a scored candidate LIST (not an ExplorationReport). literature_evidence
        is the list returned by LiteratureAgent.evaluate_direction_novelty_candidates."""
        if not raw_candidates:
            return []
        self.reset_history()
        prompt = f"""You are the PI scoring NEW research direction candidates that emerged from cross-direction integration.

## Project Context
- Name: {state.name}
- Research Objective: {state.initial_instruction}

## Raw Direction Candidates
{json.dumps(raw_candidates, ensure_ascii=False)[:8000]}

## Literature Novelty Evidence (per candidate)
{json.dumps(literature_evidence, ensure_ascii=False)[:6000]}

For each candidate, add:
- novelty_score: 0-10 (higher = more novel vs existing literature)
- evidence_strength: "strong" | "moderate" | "weak"
- confidence: "high" | "medium" | "low"
- report_recommendation: "headline" | "secondary" | "future_direction" | "do_not_emphasize"
- priority_rank: integer (1 = highest)
Keep every original field. Drop candidates that are NOT novel (already established in literature) or lack observed supporting data.

Return the surviving candidates as a JSON array, sorted by priority_rank ascending."""
        response = await self.chat(prompt, temperature=0.3)
        # This prompt contracts for a top-level JSON array; the dict parser drops valid arrays.
        data = safe_parse_json_array(response, default=[])
        if not isinstance(data, list):
            return []
        return _normalize_integration_scored_candidates(data, raw_candidates)

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
        raw_candidates = [
            _normalize_direction_candidate(candidate)
            for candidate in raw_candidates
            if isinstance(candidate, dict)
        ]
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
        # Merge score-only or shape-unstable responses onto canonical raw
        # candidates before any downstream validation or rendering.
        scored = _normalize_integration_scored_candidates(scored, raw_candidates)

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
        scored = _normalize_integration_scored_candidates(scored, raw_candidates)
        scored = self._apply_direction_guardrails(scored, raw_candidates, literature_evidence)
        report.research_direction_candidates = scored

        # Preserve the raw candidate content in its canonical normalized shape.
        report.raw_direction_candidates = raw_candidates
        # Evidence matrix always preserved.
        if not report.direction_literature_evidence:
            report.direction_literature_evidence = literature_evidence

        # Keep direction candidates available while removing only cycle-closing edges.
        if report.direction_dependency_graph:
            self._sanitize_direction_dependency_graph(report)

        return report

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
