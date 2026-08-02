"""Exploration report, integrated report, and research briefs.

Moved verbatim out of backend/agents/pi_agent.py; composed into PIAgent.
"""
from typing import List, Optional
import asyncio
import logging

from ...utils.direction_evidence import build_fallback_direction_evidence
from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from ...services.report.artifacts import archival_render_available, eligible_artifacts
from ..prompt_blocks import build_integration_block

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


class ReportsMixin:
    """Exploration report, integrated report, and research briefs."""

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

        # The integrated narrative is an archival report. Do not offer the model
        # stale, rejected, pending, or layout-blocked figures that downstream
        # rendering would withhold; otherwise it can cite a panel readers cannot
        # actually see.
        figures_info = []
        renderable_figures = [
            fig for fig in eligible_artifacts(state.publication_figures)
            if archival_render_available(fig)
        ]
        for i, fig in enumerate(renderable_figures):
            raw_status = getattr(fig.review_status, "value", fig.review_status)
            status = str(raw_status or "reportable").replace("_", " ").title()
            figures_info.append(
                f"Figure {i+1}: {fig.name} (declared type: {fig.figure_type}) - "
                f"{fig.description} [{status}]"
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

        # Cross-direction integration analyses (spec §13)
        integration_block = build_integration_block(state, style="integrated")

        prompt = f"""## Generate Integrated Research Report

**Project Information:**
- Name: {state.name}
- Objective: {state.initial_instruction}
- Total Rounds: {len(state.round_results)}

**Available Figures:**
{chr(10).join(figures_info) if figures_info else "No figures generated"}

The declared figure type and description above are authoritative. Do not infer
a different plot type from a filename, and keep every caption consistent with
the supplied metadata (for example, never describe a declared barplot as a UMAP).

**Available Tables:**
{chr(10).join(tables_info) if tables_info else "No tables generated"}

**Analysis Rounds Summary:**
{''.join(rounds_summary)}

{mech_block}

{integration_block}

**Mechanism Naming and Hypothesis Policy:**
{MECHANISM_EVIDENCE_POLICY}

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

    @staticmethod
    def _brief_background_block(state: Optional[AnalysisState]) -> str:
        """Background section for a PI brief: experimental goal as reference only ('' if none)."""
        if state and getattr(state, "initial_instruction", ""):
            return (
                "\n\nBackground (reference only — decide what is relevant, do NOT "
                f"copy verbatim):\n{state.initial_instruction[:500]}"
            )
        return ""
