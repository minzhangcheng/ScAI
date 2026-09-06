"""Round synthesis, cross-modal synthesis, and the final summary.

Moved verbatim out of backend/agents/pi_agent.py; composed into PIAgent.
"""
from typing import Dict, Any, List, Optional
import asyncio
import json
import logging

from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from ..prompt_blocks import build_integration_block
from ...utils.direction_payload import (
    build_direction_projection_block,
    collect_direction_projections,
)

# Configure logger
logger = logging.getLogger(__name__)
from ...models.schemas import (
    AnalysisPhase,
    AnalysisState,
    RoundResult,
    CrossModalSynthesis,
    CrossModalMechanismCandidate,
)

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


class SynthesisMixin:
    """Round synthesis, cross-modal synthesis, and the final summary."""

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
                should_stop=bool(data.get("should_stop", False)),
                stop_reason=str(data.get("stop_reason", "") or ""),
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
- {MECHANISM_EVIDENCE_POLICY}
- Do not invent mechanism names, pathways, gene sets, genes, PMIDs, or values that are absent from the supplied evidence.
- The matrix_markdown column headers must reflect the four agent sources plus an evidence-consistency column, in the same language as this prompt. Do not hardcode labels.
- Each candidate must draw evidence from at least TWO different sources, and at least one must be an experimental-data agent (biochemistry, pathology, or bioinformatics) — not purely literature.
- Each supporting_evidence item must cite a concrete observation (value, gene, pathway, morphological feature, PMID), not a vague summary.
- STRICT evidentiary separation: items that appear only as suggested future experiments (in next_steps or data_requests) can ONLY populate missing_evidence, never supporting_evidence. A candidate built mainly from suggested future experiments must have confidence="low" and report_recommendation="future_direction" unless independent observed data also supports it.
- Populate missing_evidence for every candidate, including the ones you grade high: an evidence gap is reported ALONGSIDE the verdict, not instead of it. A gap lowers confidence and bounds the claim; it does not turn a well-supported candidate back into an open question.
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
            execution_linkage = {
                True: "all structured findings linked to current successful receipts",
                False: "one or more findings lack a current successful receipt",
                None: "not recorded",
            }[result.bioinformatics_evidence_verified]
            rounds_summary.append(f"""
### Round {result.round_number}
**Round status**: {result.round_status}; evidence status: {result.evidence_status}
**Bioinformatics execution linkage**: {execution_linkage}
**Evidence notes**: {'; '.join(result.evidence_notes) if result.evidence_notes else 'None'}
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
        # Use the selected-direction projection rather than a recency slice. An
        # early direction remains part of the final PI context even after later
        # rounds, while the projection itself applies lineage and per-direction
        # bounds and retains source execution references and limiting evidence.
        for projection in collect_direction_projections(state):
            block = build_direction_projection_block(projection, max_chars=5000)
            if block:
                mech_lines.append(block)
        if not mech_lines:
            # Legacy records may have mechanism programs but no direction key.
            # Preserve that evidence without silently discarding older rounds.
            for result in state.round_results:
                for program in (getattr(result, "data_driven_mechanism_programs", []) or [])[:3]:
                    program_text = program.to_prompt_text(
                        style="compact",
                        max_observations=2,
                        max_genes=8,
                        max_alternatives=2,
                        max_missing=2,
                    )
                    mech_lines.append(f"- {program_text}")
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

        # integration insights into the final report so they are not lost.
        integration_block = build_integration_block(state, style="final")

        prompt = f"""
Based on all the analysis results below, write a complete analysis report:

## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}

## Results from All Rounds
{''.join(rounds_summary)}
{cross_modal_block}{mechanism_block}{integration_block}

## Mechanism Naming and Hypothesis Policy
{MECHANISM_EVIDENCE_POLICY}

Please write a well-structured analysis report including:
1. Research Overview
2. Key Findings
   - Pathological phenotype
   - Single-cell transcriptomic features
   - Biochemical indicator changes
3. Molecular Mechanism Analysis
4. Conclusions: where the evidence ranks the candidate mechanisms, name the
   most probable one and label it as an inference with its residual
   uncertainty. Do not restate as an open question something the analysis has
   already ranked.
5. Limitations and Future Directions
6. Cross-Modal Mechanism Candidates: for any candidate above, present it with its
   evidence sources, confidence, and explicitly labeled evidence gaps. Present
   low-confidence candidates as "hypotheses with evidence gaps", NOT confirmed conclusions.

Any round marked partial must remain explicitly partial. When bioinformatics execution
linkage is false or absent, state the concrete missing linkage and do not describe the
affected claims as validated, confirmed, causal, dominant, or independently replicated.

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
        partial_inputs = False
        for i, d in enumerate(directions_with_results, 1):
            findings = "\n".join(
                f"- {item}" for item in (d.get("available_findings") or [])
            ) or "- No retained findings"
            gaps = "\n".join(
                f"- {item}" for item in (d.get("evidence_gaps") or [])
            ) or "- No recorded gap"
            partial_inputs = partial_inputs or (
                d.get("round_status") != "completed"
                or d.get("evidence_status") != "substantive"
                or d.get("direction_coverage_eligible") is not True
            )
            directions_text += f"""
### Direction {i}: {d.get('direction', 'Unknown')}
**User Notes**: {d.get('notes', 'None')}
**Round / status**: {d.get('round_number', 'N/A')} / {d.get('round_status', 'unknown')}
**Evidence status**: {d.get('evidence_status', 'unknown')}
**Coverage eligible**: {d.get('direction_coverage_eligible', False)}
**Analysis Summary**: {d.get('round_summary', 'No summary available')[:2000]}
**Available Findings**:
{findings}
**Evidence Gaps**:
{gaps}
"""

        evidence_instruction = (
            "One or more directions are partial. Integrate only the available "
            "findings, do not infer missing results, and include one final "
            "'Evidence Coverage and Limitations' section that names the gaps."
            if partial_inputs else
            "All directions have substantive eligible evidence; still distinguish "
            "observations from hypotheses."
        )

        prompt = f"""You are the PI analyzing the relationships between multiple research directions that were independently investigated.

## Project: {state.name}
## Research Objective: {state.initial_instruction}

## Directions Analyzed
{directions_text}

## Evidence Handling
{evidence_instruction}

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
