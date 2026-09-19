"""Exploration report, integrated report, and research briefs.

Moved verbatim out of backend/agents/pi_agent.py; composed into PIAgent.
"""
from typing import List, Optional, Sequence
import asyncio
import logging

from ...utils.direction_evidence import build_fallback_direction_evidence
from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from ...services.report.artifacts import archival_render_available, eligible_artifacts
from ..prompt_blocks import build_integration_block

# Configure logger
logger = logging.getLogger(__name__)
from ...models.schemas import (
    AnalysisState,
    ExplorationReport,
    IntegratedReportFigureRef,
    RoundResult,
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


class ReportsMixin:
    """Exploration report, integrated report, and research briefs."""

    async def generate_stage_review_chapter(self, state, stage, catalog, context):
        """Use round-report structure while letting the PI select decision-time scope."""
        import json
        from ...services.report.round_chapters import chapter_prompt

        self.refresh_runtime_config_if_needed()
        scope = (
            "Stage 1: describe performed QC, preprocessing, clustering and candidate "
            "annotation evidence. Labels are pending user confirmation. Include existing "
            "candidate-label UMAPs and marker evidence; disclose any missing visualization. "
            "Do not describe candidate labels as accepted or invent an annotation plot."
            if stage == "stage1" else
            "Stage 2: describe the performed unbiased exploration and evidence for research "
            "directions, after a brief statement of the confirmed annotation basis. Do not "
            "repeat the Stage 1 QC/annotation gallery. Full-dataset confirmed annotation UMAP "
            "and foundation_annotation artifacts belong to the foundation section of the final "
            "report even if created late. Include newly available confirmed-label views here "
            "when they are needed for direction review and were absent from the Stage 1 "
            "review; that does not change their foundation placement in the final report. New downstream "
            "subset UMAPs belong with their downstream analysis. Explain uncertainties."
        )
        prompt = chapter_prompt(catalog, state=state) + """

## Decision-time review override
Generate an independent stage review, not the complete round chapter described above.
Return one JSON object with `stage_artifact_ids` (exact IDs selected for this stage)
and `report_chapter` using the block structure above. Use every SELECTED artifact once;
the full catalog is available evidence, not a requirement to include all artifacts.
Choose stage scope by scientific role and recorded analysis, not creation time or filename.
Keep methods, parameters, findings, adjacent figures, uncertainty, and conclusions in
actual analysis order, just as in a per-round report. Integrate supplementary evidence
beside its owning result. This review does not replace the full final-report chapter.
""" + scope + "\nRecorded review context:\n" + json.dumps(context, ensure_ascii=False, default=str)
        response = await self.chat_without_history(prompt)
        return self._safe_parse_json(response)

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
        state: AnalysisState,
        figure_refs: Optional[Sequence[IntegratedReportFigureRef]] = None,
        reporting_limitations: Optional[Sequence[str]] = None,
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
        if figure_refs is None:
            # Compatibility for direct callers. The workflow freezes and passes
            # refs before invoking this method; this branch must not invent a
            # separate report catalogue.
            renderable_figures = [
                fig for fig in eligible_artifacts(
                    state.publication_figures, state=state
                )
                if archival_render_available(fig)
            ]
            figure_refs = [
                IntegratedReportFigureRef(
                    number=index,
                    figure_id=fig.id,
                    name=fig.name,
                    description=fig.description,
                    figure_type=fig.figure_type,
                    render_sha256=fig.render_sha256,
                    pdf_sha256=fig.pdf_sha256,
                    logical_figure_id=getattr(fig, "logical_figure_id", None),
                    review_status=str(
                        getattr(fig.review_status, "value", fig.review_status)
                    ),
                    review_unavailable=bool(
                        getattr(fig, "review_unavailable", False)
                    ),
                    revision_count=int(getattr(fig, "revision_count", 0) or 0),
                    round_number=getattr(fig, "round_number", None),
                    diagnostic=bool(getattr(fig, "diagnostic", False)),
                    pi_report_note=(
                        getattr(fig, "pi_degraded_acceptance_reason", None)
                        or None
                    ),
                )
                for index, fig in enumerate(renderable_figures, start=1)
            ]

        for ref in figure_refs:
            status = (
                "Review Unavailable"
                if ref.review_unavailable
                else str(ref.review_status or "reportable").replace("_", " ").title()
            )
            figures_info.append(
                f"Figure {ref.number}: {ref.name} "
                f"(declared type: {ref.figure_type}) - {ref.description} [{status}]"
                + (f" PI reservation: {ref.pi_report_note}" if ref.pi_report_note else "")
            )

        if reporting_limitations is None:
            # Compatibility for direct callers. The workflow passes limitations
            # frozen with the pending catalogue instead of rereading live tasks.
            target_lineage = str(getattr(state, "restore_branch_id", None) or "main")
            figure_disposition_notes = [
                note
                for task in (getattr(state, "figure_revision_tasks", None) or [])
                if (
                    getattr(task, "lineage_id", None) is None
                    or str(getattr(task, "lineage_id", None) or "main")
                    == target_lineage
                )
                if (note := str(
                    getattr(task, "pi_report_note", None) or ""
                ).strip())
            ]
        else:
            figure_disposition_notes = [
                str(note).strip() for note in reporting_limitations
                if str(note).strip()
            ]

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
Disclose each supplied PI reservation where it affects interpretation.
Do not imply that an omitted or reserved figure passed ordinary review.

**Reader-facing PI limitations (preserve their meaning; do not create a metadata section):**
{chr(10).join(f'- {note}' for note in figure_disposition_notes) if figure_disposition_notes else "None"}

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
   - Reference only figures in the supplied catalogue
   - Use exact Markdown links: [Figure X](#figure-X)
   - Reference tables using format: (Table X)
   - Include key statistics and observations
5. **Discussion**:
   - Interpret findings in context of research objectives
   - Compare with literature
   - Discuss implications
6. **Conclusions**: Key takeaways (1 paragraph)

**Important:**
- Use academic writing style
- Include figure/table references naturally in the text
- Write in English
- Focus on biological/clinical significance, not just technical details
- Do not create a Methods, Methods Summary, Materials and Methods, Experimental
  Procedures, Declarations, Metadata, figure inventory, or figure-type list section
- Do not invent sample preparation, laboratory protocols, sequencing procedures,
  assays, instruments, reagents, or other experimental methods
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
