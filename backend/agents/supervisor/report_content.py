"""LLM-generated content for the 9-chapter detailed report.

Moved verbatim out of backend/agents/supervisor_agent.py; composed into SupervisorAgent.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Sequence
import asyncio
import json
import logging
import re
import uuid
from pathlib import Path

from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY

logger = logging.getLogger(__name__)

from openai import APIConnectionError

from ...models.schemas import AnalysisState, AnalysisStep
from ...utils.llm_client import OpenAICompatibleClient, UpstreamExhaustedError
from ...services.report.interaction_records import write_interaction_record_files
from ...services.report.artifacts import eligible_artifacts
from ...services.report.figure_attribution import attribute_figures_to_rounds
from ...services.report.figure_placement import enforce_figure_placement
from ...services.report.report_outline import (
    GATE_SECTION_WORD_LIMIT,
    OutlineSection,
    align_sections_to_outline,
    build_report_outline,
)
from ..prompt_blocks import build_integration_block


# ---------------------------------------------------------------------------
# Rich HTML report CSS (extracted to utils/report_css.py for separation of concerns)
# ---------------------------------------------------------------------------
from ...utils.report_css import REPORT_CSS as _REPORT_CSS  # noqa: E402

from .shared import (  # noqa: F401
    _SPECIFIC_SECTION_IDS,
    _SECTION_ID_TO_DEFAULT_TITLE,
    _PLACEHOLDER_SECTION_TITLES,
    SupervisorRecord,
    SupervisorEvalResult,
    _CRITICAL_OUTPUT_PATTERNS,
    _PERCENTAGE_PATTERN,
    _INF_NAN_PATTERN,
    _SUPERVISOR_SYSTEM_PROMPT,
    _serialize_record,
    persist_graphic_abstract,
)


class ReportContentMixin:
    """LLM-generated content for the 9-chapter detailed report."""

    async def generate_detailed_report(
        self,
        analysis_state: AnalysisState,
        report_model: str,
        output_dir: str,
        workspace_dir: str = None,
        progress_callback: Optional[Callable] = None,
        report_llm_client: Optional["OpenAICompatibleClient"] = None,
        chat_messages: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a detailed report from the supervisor's accumulated analysis records.

        Uses the supervision log (every step's code, output, figures, PI guidance,
        technical issues) as the primary source. Does NOT re-scan figures or re-run
        PI review — just summarizes what was recorded.

        Args:
            analysis_state: Full analysis state
            report_model: Strong model for report writing (e.g. PI model)
            output_dir: Directory to save report files
            workspace_dir: Workspace directory containing generated figures
            progress_callback: Optional progress reporting callback
            report_llm_client: LLM client matching report_model's endpoint.
                If None, falls back to supervisor's own llm_client.
            chat_messages: Optional chat messages to build chat sections from.

        Returns:
            Report dict compatible with existing frontend expectations.
        """
        self.refresh_runtime_config_if_needed()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        workspace_path = Path(workspace_dir) if workspace_dir else None

        language = "en"

        async def _progress(pct: float, step: str):
            if progress_callback:
                await progress_callback({"progress": pct, "step": step, "status": "generating"})

        await _progress(5, "Preparing analysis records...")

        # Collect all records
        all_records = self.get_supervision_log()
        stats = self.get_summary_stats()

        # Build structured analysis narrative for the LLM
        narrative = self._build_analysis_narrative(all_records, analysis_state, workspace_path)

        # Collect figure paths from workspace and publication figures
        figure_map = self._collect_figure_paths(all_records, workspace_path, state=analysis_state)

        # Collect table paths from workspace
        table_list = self._collect_table_paths(
            workspace_path,
            invalidated_stems=set(analysis_state.invalidated_artifact_stems or []),
        )

        # Plan the report from the rounds that ran, and bind every figure to the
        # round that produced it. Both are needed before the prompt is built and
        # again after the model answers, to check its placement against them.
        outline = build_report_outline(analysis_state)
        attribution = attribute_figures_to_rounds(
            figure_map.keys(),
            state=analysis_state,
            records=all_records,
            figure_paths=figure_map,
        )

        await _progress(15, "Generating report content...")

        # Generate report sections using strong model
        report_prompt = self._build_report_prompt(
            narrative, stats, analysis_state, figure_map, language,
            table_list=table_list, outline=outline, attribution=attribution,
        )

        used_fallback = False
        try:
            # Use the provided LLM client (matching report_model's endpoint),
            # falling back to supervisor's own client if not provided.
            client = report_llm_client or self.llm_client
            client.set_context(self.session_id, "report_generator")
            for attempt in range(2):  # Single retry on connection error
                try:
                    response = await client.chat(
                        model=report_model,
                        messages=[
                            {"role": "system", "content": self._get_report_system_prompt(language)},
                            {"role": "user", "content": report_prompt},
                        ],
                        temperature=0.3,
                        max_tokens=32768,
                        timeout=900.0,  # 15 min — large narratives need long generation time
                    )
                    report_content = response.content or ""
                    break
                except UpstreamExhaustedError:
                    raise  # propagate to caller for retry/user-intervention
                except APIConnectionError:
                    if attempt == 0:
                        logger.warning(
                            f"[{self.session_id}] Report generation connection error, "
                            f"retrying in 5s"
                        )
                        await asyncio.sleep(5)
                        continue
                    logger.exception(
                        f"[{self.session_id}] Report generation failed after retry"
                    )
                    report_content = self._fallback_report(narrative, stats, language)
                    used_fallback = True
                except Exception:
                    logger.exception(
                        f"[{self.session_id}] Report generation LLM call failed"
                    )
                    report_content = self._fallback_report(narrative, stats, language)
                    used_fallback = True
                    break
        finally:
            # Restore supervisor context in case of subsequent LLM calls
            self._set_llm_context()

        await _progress(60, "Parsing report structure...")

        # Do not persist an empty/placeholder body as a completed report. The
        # client layer already retries transient zero-token empties and raises
        # UpstreamExhaustedError on exhaustion; this validator is the remaining
        # net for the cases that do NOT raise there — a token-consuming empty
        # (returned as EMPTY_CONTENT_PLACEHOLDER) or content that parses to
        # nothing. Raise so the workflow HITL loop can Retry (optionally after
        # the user changes the report model) or Skip.
        if not self._validate_report_content(report_content):
            logger.error(
                f"[{self.session_id}] Report model returned empty/placeholder "
                f"content; refusing to save an empty completed report"
            )
            raise UpstreamExhaustedError(
                "Report generation returned empty content after retries",
                agent_name="report_generator",
                model=report_model,
                llm_server=getattr(client, "base_url", ""),
            )

        # Parse the LLM output into sections
        sections = self._parse_report_sections(report_content, language, outline=outline)

        # The fallback body is a degraded step summary with its own fixed
        # sections; forcing it through an outline it was never written against
        # would only scramble it. Its figures fall through to the supplementary
        # section, which is the honest presentation for a failed report.
        if not used_fallback:
            # Reading order follows the outline, not the order the model emitted.
            # Sections owning figures are materialised even when skipped, so their
            # figures have a home in the right place.
            sections_with_figures = {
                sid for sid in (
                    next((s.id for s in outline if rn in s.rounds), None)
                    for rn in set(attribution.values())
                ) if sid
            }
            sections = align_sections_to_outline(
                sections, outline, require_ids=sections_with_figures
            )

            # Placement is enforced in code, not left to the model's compliance:
            # it relocates cross-round references, drops repeats, and appends
            # figures the model never cited to the section owning their round.
            figure_order = {}
            for name, path in figure_map.items():
                try:
                    figure_order[name] = path.stat().st_mtime
                except OSError:
                    pass
            placement = enforce_figure_placement(
                sections, outline, attribution, figure_map.keys(),
                figure_order=figure_order,
            )
            if placement.unplaced:
                logger.info(
                    f"[{self.session_id}] {len(placement.unplaced)} figures had no "
                    f"owning section and fall through to Supplementary Figures"
                )

        # Build chat sections if chat_messages provided
        if chat_messages:
            try:
                from backend.services.report.chat_sections import build_chat_sections_async
                chat_sections = await build_chat_sections_async(
                    chat_messages,
                    figures_dir=str(Path(workspace_dir) / "figures") if workspace_dir else "",
                    tables_dir=str(Path(workspace_dir) / "tables") if workspace_dir else "",
                    llm_client=report_llm_client or self._provider.get_llm_client("pi"),
                    progress_callback=progress_callback,
                    model=self._provider.get_model("pi"),
                )
                sections.extend(chat_sections)
            except Exception as e:
                logger.warning(f"Failed to build chat sections: {e}", exc_info=True)

        # Extract abstract and conclusions
        abstract = ""
        conclusions = ""
        filtered_sections = []
        for sec in sections:
            sid = sec.get("id", "")
            if sid == "abstract":
                abstract = sec.get("content", "")
            elif sid == "conclusions":
                conclusions = sec.get("content", "")
            elif sid in ("supervisor_log", "figure_appendix"):
                continue  # Skip monitoring and figure appendix sections
            else:
                filtered_sections.append(sec)

        await _progress(70, "Embedding figures...")

        # Build figures list for report
        report_figures = []
        for fig_name, fig_path in figure_map.items():
            fig_entry = {
                "filename": fig_name,
                "path": str(fig_path),
                "category": self._categorize_figure(fig_name),
            }
            # Add PDF path if exists
            pdf_path = fig_path.with_suffix(".pdf")
            if pdf_path.exists():
                fig_entry["pdf_path"] = str(pdf_path)
            report_figures.append(fig_entry)

        await _progress(80, "Building report...")

        report_id = str(uuid.uuid4())
        record_files = write_interaction_record_files(analysis_state, reports_dir=output_path)
        appendix_data = self._build_appendix(analysis_state, stats, record_files, reports_dir=output_path)

        # Persist graphic abstract for downloadable zip. Wrapped so a
        # malformed/unopenable image degrades gracefully (ga_path stays None;
        # the HTML embed still uses the base64 from analysis_state) instead of
        # crashing report generation — matches services/report/generator.py.
        ga_obj = getattr(analysis_state, "graphic_abstract", None)
        ga_path = None
        try:
            ga_path = persist_graphic_abstract(ga_obj, output_path / "figures")
        except Exception:
            logger.exception("Failed to save graphic abstract")

        report = {
            "id": report_id,
            "session_id": analysis_state.id,
            "title": analysis_state.name,
            "abstract": abstract,
            "sections": filtered_sections,
            "conclusions": conclusions,
            "figures": report_figures,
            "tables": table_list,
            "graphic_abstract_path": ga_path,  # Path for zip download
            "generated_at": datetime.now().isoformat(),
            "status": "completed",
            "progress": 100,
            "current_step": "Complete",
            "appendix": appendix_data,
            "analysis_log": all_records,
            "score_definitions": [sd.model_dump(mode="json") for sd in analysis_state.score_definitions] if analysis_state.score_definitions else [],
        }

        # Sanitize NaN/Infinity
        report = self._sanitize(report)

        # Save JSON
        report_path = output_path / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        await _progress(90, "Generating HTML report...")

        # Temporarily attach analysis_state for HTML generation. It is NOT
        # serialized into report.json; it is popped back off after HTML rendering
        # so it never leaks into the on-disk report or downstream consumers.
        report["_analysis_state"] = analysis_state

        # Generate HTML
        html_path = self._generate_html_report(
            report, filtered_sections, abstract, conclusions,
            report_figures, figure_map, stats, language, output_path, workspace_path,
        )
        report.pop("_analysis_state", None)
        report["html_path"] = str(html_path)

        await _progress(100, "Complete")
        return report

    @staticmethod
    def _build_figure_allocation_text(
        outline: Sequence["OutlineSection"],
        figure_map: Dict[str, Path],
        attribution: Dict[str, int],
    ) -> str:
        """List, per section, the figures that section owns.

        A figure belongs to the round that produced it. Handing the model a flat
        list and asking it to pick "the most relevant section" made placement a
        semantic guess over filenames — which is how a round-3 panel named
        ``..._qc_overview`` came to open a report. The allocation below removes
        the guess; code re-checks it afterwards regardless.
        """
        owner_of_round: Dict[int, str] = {}
        for sec in outline:
            for rn in sec.rounds:
                owner_of_round.setdefault(int(rn), sec.id)

        per_section: Dict[str, List[str]] = {sec.id: [] for sec in outline}
        unattributed: List[str] = []
        for name in figure_map:
            rn = attribution.get(name)
            sid = owner_of_round.get(rn) if rn is not None else None
            if sid is None:
                unattributed.append(name)
            else:
                per_section[sid].append(name)

        lines: List[str] = []
        for sec in outline:
            names = per_section.get(sec.id) or []
            if not sec.rounds:
                lines.append(f"[{sec.id}] — no figures (this section carries no result figures)")
            elif names:
                lines.append(f"[{sec.id}] — round(s) {', '.join(str(r) for r in sec.rounds)}:")
                lines.extend(f"    - {n}" for n in names)
            else:
                lines.append(
                    f"[{sec.id}] — round(s) {', '.join(str(r) for r in sec.rounds)}: "
                    f"no figures were produced"
                )
        if unattributed:
            lines.append(
                "Figures whose round could not be determined (place by content, "
                "or omit):"
            )
            lines.extend(f"    - {n}" for n in unattributed)
        return "\n".join(lines) or "None"

    def _build_report_prompt(
        self,
        narrative: str,
        stats: Dict,
        state: AnalysisState,
        figure_map: Dict[str, Path],
        language: str,
        table_list: Optional[List[Dict]] = None,
        outline: Optional[Sequence["OutlineSection"]] = None,
        attribution: Optional[Dict[str, int]] = None,
    ) -> str:
        """Build the prompt for the report-writing LLM."""
        attribution = attribution or {}
        figure_list = "\n".join(
            f"- {name} ({self._categorize_figure(name)})"
            + (f" [round {attribution[name]}]" if name in attribution else "")
            for name in figure_map.keys()
        ) or ("None")

        # Collect per-round findings with phase labels. The section list no
        # longer derives from this loop — build_report_outline() reads the rounds
        # directly — so it only builds narrative text now.
        findings_parts = []
        for rr in state.round_results:
            phase_label = (rr.phase or "exploration").upper()
            header = (
                f"### Round {rr.round_number} ({phase_label}; "
                f"status={rr.round_status}; evidence={rr.evidence_status})"
            )

            # Get direction for innovation rounds (stored at synthesis time)
            direction = ""
            if rr.phase == "innovation":
                direction = getattr(rr, "direction", "") or ""
                if not direction and getattr(state, "user_chosen_direction", None):
                    direction = state.user_chosen_direction
                if direction:
                    header += f" — Direction: {direction}"

            findings_parts.append(header)
            if getattr(rr, "bioinformatics_evidence_verified", None) is False:
                findings_parts.append(
                    "**Evidence warning**: Bioinformatics claims are partial/unverified. "
                    "Do not describe them as validated, confirmed, causal, or dominant."
                )
            for note in (getattr(rr, "evidence_notes", None) or []):
                findings_parts.append(f"**Evidence note**: {note}")
            if rr.pi_summary:
                findings_parts.append(f"**PI Summary**: {rr.pi_summary[:400]}")
            for agent_name, findings_list in [
                ("Bioinformatics", rr.bioinformatics_findings),
                ("Pathology", rr.pathology_findings),
                ("Biochemistry", rr.biochemistry_findings),
                ("Literature", rr.literature_findings),
            ]:
                if findings_list:
                    findings_parts.append(f"**{agent_name}**:")
                    for f in findings_list[:8]:
                        findings_parts.append(f"- {f[:300]}")
            findings_parts.append("")

        findings_text = "\n".join(findings_parts) or ("None")

        # Collect data-driven mechanism programs from last 3 rounds
        mech_parts = []
        for rr in state.round_results[-3:]:
            progs = (getattr(rr, "data_driven_mechanism_programs", []) or [])[:3]
            for p in progs:
                prefix = (
                    "[UNVERIFIED ROUND EVIDENCE] "
                    if getattr(rr, "bioinformatics_evidence_verified", None) is False
                    else ""
                )
                mech_parts.append(f"- {prefix}{p.to_prompt_text(style='compact', max_observations=2, max_genes=8, max_alternatives=2, max_missing=2)}")
            if getattr(rr, "cross_modal_synthesis", None):
                for c in rr.cross_modal_synthesis.candidates[:3]:
                    mech_parts.append(
                        f"- (cross-modal) {c.theme}: {c.mechanism_statement} "
                        f"[confidence {c.confidence}; recommendation {c.report_recommendation}]"
                    )
        mechanism_text = "\n".join(mech_parts) or "None"

        # Cross-direction integration analyses (spec §13)
        integration_text = build_integration_block(state, style="detailed") or "None"

        # Section template, derived from the rounds that actually ran. The
        # outline orders the report by round, which is at once the chronological
        # and the argumentative unit; see services/report/report_outline.py.
        outline = outline or build_report_outline(state)
        base_sections = [f"[{sec.id}] - {sec.guidance}" for sec in outline]

        sections_text = "\n".join(base_sections)

        # Per-section figure allocation. Each section is told exactly which
        # figures are its own, so the model never has to infer round provenance
        # from a filename — the failure mode this replaces.
        figure_allocation = self._build_figure_allocation_text(
            outline, figure_map, attribution
        )

        # Integration gates are hinges, not chapters — the 300-500 word floor
        # that makes findings sections substantial would pad them into rivals of
        # the directions they merely connect, and their conclusions are the ones
        # most likely to be superseded.
        gate_ids = [sec.id for sec in outline if sec.is_brief]
        if gate_ids:
            gate_note = (
                f"- EXCEPTION to the length floor: {', '.join(gate_ids)} are transition "
                f"sections. Keep each under {GATE_SECTION_WORD_LIMIT} words, covering only "
                f"what the completed directions jointly showed and which new direction that "
                f"motivated. Label their conclusions provisional.\n"
            )
            brevity_note = " (except the transition sections noted below)"
        else:
            gate_note = ""
            brevity_note = ""

        # Build per-section title guidance from the outline
        section_title_lines = []
        for sec, sec_line in zip(outline, base_sections):
            default_title = sec.default_title
            section_title_lines.append(
                f"  {sec_line}\n"
                f"    → Use a descriptive scientific title (e.g., \"{default_title}\") "
                f"based on your analysis"
            )
        section_title_guide = "\n".join(section_title_lines)

        # Build table list text
        table_list = table_list or []
        table_list_text = "\n".join(
            f"- {tbl['filename']}: {tbl.get('name', '')} ({tbl.get('total_rows', '?')} rows, columns: {', '.join(tbl.get('columns', [])[:8])})"
            for tbl in table_list
        ) if table_list else "None"

        prompt = f"""Based on the following complete analysis records, generate a detailed scientific research report.

## Analysis Narrative
{narrative}

## Key Findings (Per Round)
{findings_text}

## Data-Driven Candidate Mechanisms (candidates, NOT confirmed)
{mechanism_text}

## Cross-Direction Integration Analyses
{integration_text}

## Mechanism Naming and Hypothesis Policy
{MECHANISM_EVIDENCE_POLICY}

## Available Figures
{figure_list}

## Figure Allocation (BINDING)
Every figure belongs to the round that produced it, and each section below owns
the figures of its round. Place each figure ONLY in the section listed here.
Do not move a figure to a section that "reads" like a better fit — a filename
describes the kind of plot, not the stage of the analysis it came from.

{figure_allocation}

## Available Tables (CSV)
{table_list_text}

## Instructions
Write the report in English.
Rounds marked partial/unverified must remain explicitly labeled as such. Never
upgrade them to validated, confirmed, causal, dominant, or independently
replicated conclusions, even if a PI summary or literature passage uses stronger wording.
Generate the report with the following sections. Use the EXACT section IDs in brackets.
Each section MUST have a descriptive, scientific title — do NOT use placeholder text like "Section Title".

{sections_text}

## Section Title Guidance
For each section, write a concrete descriptive title (like the examples below) instead of generic placeholder text:

{section_title_guide}

Format each section as:
## [section_id] Your Descriptive Scientific Title Here
Content...

CRITICAL: Replace "Your Descriptive Scientific Title Here" with an actual title that describes what this specific analysis found. NEVER write "Section Title" as a title.

When referencing a figure, use this exact format (each marker on its own line).
IMPORTANT: Use the EXACT filename from the Available Figures list above (including the correct extension — most are .jpg, NOT .png):
[FIGURE: exact_filename_from_list_above.jpg]
[FIGURE_CAPTION: A concise scientific caption (1-2 sentences) describing what this figure shows and key observations]

When referencing a data table, use this exact format (each marker on its own line):
[TABLE: exact_filename.csv]
[TABLE_CAPTION: A concise scientific caption (1-2 sentences) describing key data in this table and its significance]

Each figure and table appears EXACTLY ONCE, in the section the Figure Allocation
assigns it to. Do NOT create separate figure/table appendix or listing sections.

IMPORTANT:
- Write as a formal scientific research report, NOT a monitoring or system log.
- Make the report thorough and comprehensive with detailed scientific content (300-500 words minimum per section){brevity_note}.
- Innovation phase sections are CRITICAL — they contain the novel research findings and MUST be written in detail.
- Reference specific figures, genes, pathways, cell types, statistical results, and quantitative findings.
- Include ALL available figures, each in the section assigned to it by the Figure Allocation above, using [FIGURE: filename] markers.
- Include ALL available tables — assign each table to the most relevant section using [TABLE: filename.csv] markers.
- Every [FIGURE:] and [TABLE:] marker MUST be followed by its corresponding [FIGURE_CAPTION:] or [TABLE_CAPTION:] marker on the next line.
{gate_note}"""
        return prompt

    def _get_report_system_prompt(self, language: str) -> str:
        return (
            "You are a senior bioinformatics researcher writing a detailed scRNA-seq analysis report. "
            "The report should be professional, rigorous, and comprehensive, including key analysis "
            "decisions and PI guidance. Use Markdown format, referencing specific figure filenames "
            "and analysis step numbers."
        )

    def _build_analysis_narrative(
        self,
        records: List[Dict],
        state: AnalysisState,
        workspace_path: Optional[Path],
    ) -> str:
        """Build a structured text narrative from all supervision records for the report LLM."""
        lines = []
        lines.append(f"Project: {state.name}")
        lines.append(f"Instruction: {state.initial_instruction[:500]}")
        lines.append(f"Total rounds: {len(state.round_results)}")
        lines.append(f"Total steps monitored: {len(records)}")
        lines.append("")

        # Split records into per-round blocks using step number resets
        round_blocks: List[List[Dict]] = []
        current_block: List[Dict] = []
        prev_step = -1
        for rec in records:
            step_num = rec.get("step_number", 0)
            if step_num <= prev_step and current_block:
                # Step number reset → new round
                round_blocks.append(current_block)
                current_block = []
            current_block.append(rec)
            prev_step = step_num
        if current_block:
            round_blocks.append(current_block)

        # Build narrative with round/phase labels
        for block_idx, block in enumerate(round_blocks):
            round_num = block_idx + 1
            # Determine phase from round_results
            phase = "exploration"
            direction = ""
            if block_idx < len(state.round_results):
                rr = state.round_results[block_idx]
                phase = rr.phase or "exploration"
            if phase == "innovation":
                # Get direction from round result (stored at synthesis time)
                if block_idx < len(state.round_results):
                    direction = state.round_results[block_idx].direction or ""
                if not direction and getattr(state, "user_chosen_direction", None):
                    direction = state.user_chosen_direction

            header = f"========== ROUND {round_num} ({phase.upper()}) =========="
            if direction:
                header += f"\nResearch Direction: {direction}"
            lines.append(header)
            lines.append("")

            for rec in block:
                step_line = f"--- Step {rec['step_number']} [{rec['step_type']}] ---"
                lines.append(step_line)
                if rec.get("agent_reasoning"):
                    lines.append(f"Reasoning: {rec['agent_reasoning'][:300]}")
                if rec.get("code"):
                    lines.append(f"Code:\n```python\n{rec['code'][-600:]}\n```")
                if rec.get("code_output"):
                    lines.append(f"Output: {rec['code_output'][-400:]}")
                if rec.get("code_error"):
                    lines.append(f"Error: {rec['code_error'][:200]}")
                if rec.get("saved_figure_filenames"):
                    lines.append(f"Figures generated: {', '.join(rec['saved_figure_filenames'])}")
                if rec.get("literature_topic"):
                    lines.append(f"Literature search: {rec['literature_topic']}")
                    if rec.get("literature_result"):
                        lines.append(f"Literature findings: {rec['literature_result'][:200]}")
                if rec.get("technical_issues"):
                    lines.append(f"Technical issues: {'; '.join(rec['technical_issues'][:3])}")
                if rec.get("pi_escalated") and rec.get("pi_response"):
                    lines.append(f"PI guidance: {rec['pi_response'][:300]}")
                lines.append("")

        # Exploration report and user-chosen directions
        if state.exploration_report and hasattr(state.exploration_report, "research_direction_candidates"):
            candidates = state.exploration_report.research_direction_candidates or []
            if candidates:
                lines.append("=== Exploration Report: Proposed Research Directions ===")
                for idx, c in enumerate(candidates):
                    lines.append(f"{idx+1}. {c.get('direction', '')}")
                    lines.append(f"   Rationale: {c.get('rationale', '')[:200]}")
                lines.append("")

        if state.user_chosen_directions:
            lines.append("=== User-Selected Research Directions ===")
            for d in state.user_chosen_directions:
                lines.append(f"- {d.get('direction', '')} (Notes: {d.get('notes', '')})")
            lines.append("")

        # Publication figures
        eligible_figures = eligible_artifacts(state.publication_figures)
        if eligible_figures:
            lines.append("=== Publication Figures ===")
            for pf in eligible_figures:
                lines.append(f"- {pf.name}: {pf.description or ''}")
                if pf.png_path:
                    lines.append(f"  Path: {pf.png_path}")

        # Truncate to fit context
        full_text = "\n".join(lines)
        if len(full_text) > 80000:
            full_text = full_text[:80000] + "\n\n[... truncated ...]"
        return full_text

    def _build_step_context(self, step: AnalysisStep) -> str:
        """Build context string for a step (used in PI escalation for critical issues)."""
        parts = [f"Step {step.step_number} ({step.step_type})"]
        if step.code_output:
            parts.append(f"Output: {step.code_output}")
        if step.code_error:
            parts.append(f"Error: {step.code_error[:300]}")
        return "\n".join(parts)

    def _build_checkpoint_context(self, step: AnalysisStep) -> str:
        """Build rich context for biological checkpoint PI escalation."""
        parts = [f"Checkpoint at step {step.step_number} ({step.step_type})"]

        # Include recent literature searches
        recent_lit = [
            r for r in self.supervision_log
            if r.step_type == "search_literature" and r.literature_topic
        ]
        if recent_lit:
            parts.append("\nLiterature searched before this checkpoint:")
            for lit in recent_lit[-3:]:
                parts.append(f"  Topic: {lit.literature_topic}")
                if lit.literature_result:
                    parts.append(f"  Key findings: {lit.literature_result[:300]}")

        if step.code_output:
            parts.append(f"\nMost recent output:\n{step.code_output[-1000:]}")
        if step.annotation_result_summary:
            parts.append(f"\nAnnotation result: {step.annotation_result_summary}")
        if step.findings:
            parts.append(f"\nFindings: {'; '.join(step.findings[:5])}")

        return "\n".join(parts)

    def _parse_report_sections(
        self,
        content: str,
        language: str,
        outline: Optional[Sequence["OutlineSection"]] = None,
    ) -> List[Dict]:
        """Parse LLM-generated report content into structured sections."""
        sections = []
        # Titles and analysis types come from the outline the report was planned
        # from; the static tables remain the fallback for sections the model
        # invented and for callers that pass no outline.
        outline_titles = {sec.id: sec.default_title for sec in (outline or ())}
        outline_types = {sec.id: sec.analysis_type for sec in (outline or ())}
        # Split by ## [section_id] pattern
        pattern = re.compile(r'^##\s*\[(\w+)\]\s*(.+?)$', re.MULTILINE)
        matches = list(pattern.finditer(content))

        for i, match in enumerate(matches):
            section_id = match.group(1)
            title = match.group(2).strip()
            # Safety net: replace placeholder text with section-ID-derived fallback
            if not title or title.lower() in _PLACEHOLDER_SECTION_TITLES:
                title = outline_titles.get(section_id) or _SECTION_ID_TO_DEFAULT_TITLE.get(
                    section_id,
                    section_id.replace("_", " ").title(),
                )
                logger.warning(
                    f"[{self.session_id}] Section '{section_id}' had placeholder title "
                    f"'{match.group(2).strip()}', replaced with '{title}'"
                )
            # Content is between this match and the next
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start:end].strip()

            analysis_type = outline_types.get(section_id) or (
                "specific"
                if section_id in _SPECIFIC_SECTION_IDS
                or section_id.startswith("innovation_")
                or section_id.startswith("direction_")
                else "general"
            )
            sections.append({
                "id": section_id,
                "title": title,
                "content": section_content,
                "analysis_type": analysis_type,
                "description": "",
                "figures": [],
                "tables": [],
            })

        if not sections:
            # Fallback: treat entire content as one section
            sections.append({
                "id": "full_report",
                "title": "Analysis Report",
                "content": content,
                "analysis_type": "general",
                "description": "",
                "figures": [],
                "tables": [],
            })

        return sections

    @staticmethod
    def _validate_report_content(report_content: str) -> bool:
        """True when the model returned a usable report body. Rejects empty /
        whitespace-only content and the client's EMPTY_CONTENT_PLACEHOLDER. A
        substantive body needs a meaningful amount of non-whitespace text."""
        from backend.utils.llm_client import EMPTY_CONTENT_PLACEHOLDER
        if not report_content:
            return False
        stripped = report_content.strip()
        if not stripped or stripped == EMPTY_CONTENT_PLACEHOLDER:
            return False
        prose = stripped.lstrip("#-*> \n\t")
        return len(prose) >= 20

    def _fallback_report(self, narrative: str, stats: Dict, language: str) -> str:
        """Generate a minimal fallback report when LLM call fails.

        Extracts only step headers and reasoning from the raw narrative,
        stripping code blocks and outputs to keep the fallback readable.

        Note: Currently generates English text only. The language parameter is
        accepted for interface consistency but not used in this fallback path.
        """
        # Extract step summaries (reasoning only, no code/output)
        summary_lines = []
        for line in narrative.split('\n'):
            stripped = line.strip()
            if stripped.startswith('--- Step') or stripped.startswith('Reasoning:'):
                summary_lines.append(stripped)
            elif stripped.startswith('========== ROUND'):
                summary_lines.append('')
                summary_lines.append(stripped)
            if len(summary_lines) >= 60:
                break

        if summary_lines:
            overview = '\n'.join(summary_lines)
        else:
            overview = (
                "Report content generation failed due to API connection error. "
                "Please refer to the analysis log in the appendix for detailed records."
            )

        return f"""## [abstract] Abstract
This report was generated by the automated monitoring system. Analysis monitored {stats['total_steps_monitored']} steps, found {stats['steps_with_technical_issues']} technical issues, with {stats['pi_escalations']} PI escalations.

Note: The detailed narrative report could not be generated due to an API connection error. Below is a summary of the analysis steps performed. For full details including code and outputs, please refer to the appendix.

## [analysis_overview] Analysis Overview
{overview}

## [conclusions] Conclusions
Report generation model call failed due to API connection error. Please refer to the analysis log in the appendix for complete information, including all figures and tables generated during the analysis.
"""
