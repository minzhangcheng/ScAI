"""Detailed-report prompt, narrative, parsing, and fallback helpers."""
from __future__ import annotations

from ...constants import NO_OUTPUT_TOKEN_CAP
from ...services.report.quality import build_quality_disclosures
from ...services.report.round_chapters import (
    merge_reusable_sections, reusable_sections, reuse_context, saved_figure_captions,
    reserved_artifact_names, report_artifact_names,
)


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )

def _build_figure_allocation_text(
    outline: Sequence["OutlineSection"],
    figure_map: Dict[str, Path],
    attribution: Dict[str, int],
) -> str:
    """List, per section, the figures that section owns, in analysis order.

    A figure belongs to the round that produced it. Handing the model a flat
    list and asking it to pick "the most relevant section" made placement a
    semantic guess over filenames — which is how a round-3 panel named
    ``..._qc_overview`` came to open a report. The allocation below removes
    the guess; code re-checks it afterwards regardless.
    """
    # Only a round claimed by exactly one section allocates a figure. The
    # legacy (single-pass) outline organises by analysis type instead of by
    # round, so its four findings sections all declare every round; taking
    # the first claimant would allocate the entire report to quality control
    # and then tell the model the other three chapters produced nothing.
    claimants: Dict[int, List[str]] = {}
    for sec in outline:
        for rn in sec.rounds:
            claimants.setdefault(int(rn), []).append(sec.id)
    owner_of_round = {
        rn: ids[0] for rn, ids in claimants.items() if len(ids) == 1
    }

    per_section: Dict[str, List[str]] = {sec.id: [] for sec in outline}
    unattributed: List[str] = []
    for name in figure_map:
        rn = attribution.get(name)
        sid = owner_of_round.get(rn) if rn is not None else None
        if sid is None:
            unattributed.append(name)
        else:
            per_section[sid].append(name)

    shared = {rn for rn, ids in claimants.items() if len(ids) > 1}
    lines: List[str] = []
    for sec in outline:
        names = per_section.get(sec.id) or []
        rounds_text = ", ".join(str(r) for r in sec.rounds)
        if getattr(sec, "is_text_only", False) and not names:
            lines.append(
                f"[{sec.id}] — text-only synthesis/transition; no figure required"
            )
        elif not sec.rounds:
            lines.append(f"[{sec.id}] — no figures (this section carries no result figures)")
        elif names:
            lines.append(
                f"[{sec.id}] — round(s) {rounds_text}, analysis order "
                "(preserve exactly):"
            )
            lines.extend(f"    {index}. {name}" for index, name in enumerate(names, 1))
        elif shared.issuperset(int(r) for r in sec.rounds):
            lines.append(
                f"[{sec.id}] — round(s) {rounds_text}, shared with other "
                f"sections: choose by content from the list below"
            )
        else:
            lines.append(
                f"[{sec.id}] — round(s) {rounds_text}: "
                f"no figures were produced"
            )
    if unattributed:
        lines.append(
            "Figures the round alone does not place (choose by content, "
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
    figure_metadata_by_stem: Optional[Dict[str, Dict[str, Any]]] = None,
    figure_outline: Optional[Sequence["OutlineSection"]] = None,
) -> str:
    """Build the prompt for the report-writing LLM.

    ``outline`` is what the model is asked to WRITE. ``figure_outline``, when
    given, is the outline the figure allocation is computed from, and it must
    stay the whole report's outline even when only part of it is being written:
    allocation decides ownership by round, so a partial outline leaves every
    figure of an unwritten-here chapter with no claimant and offers it to the
    model as a free candidate inside a block labelled BINDING.
    """
    attribution = attribution or {}
    figure_metadata_by_stem = figure_metadata_by_stem or publication_figure_metadata(
        state.publication_figures, report_only=True, state=state,
    )
    figure_lines = []
    for name in figure_map:
        metadata = figure_metadata_by_stem.get(Path(name).stem, {})
        figure_type = metadata.get("type") or self._categorize_figure(name)
        line = f"- {name} (type: {figure_type})"
        if name in attribution:
            line += f" [round {attribution[name]}]"
        description = str(metadata.get("description") or "").strip()
        if description:
            line += f" — authoritative description: {description[:600]}"
        degraded_reason = str(
            metadata.get("pi_degraded_acceptance_reason") or ""
        ).strip()
        if degraded_reason:
            line += f" — PI accepted with reservations: {degraded_reason[:600]}"
        figure_lines.append(line)
    figure_list = "\n".join(figure_lines) or "None"

    # Collect per-round findings with phase labels. The section list no
    # longer derives from this loop — build_report_outline() reads the rounds
    # directly — so it only builds narrative text now.
    findings_parts = []
    for rr in state.round_results:
        phase_label = (rr.phase or "exploration").upper()
        header = f"### Round {rr.round_number} ({phase_label})"

        # Get direction for innovation rounds (stored at synthesis time)
        direction = ""
        if rr.phase == "innovation":
            direction = getattr(rr, "direction", "") or ""
            if not direction and getattr(state, "user_chosen_direction", None):
                direction = state.user_chosen_direction
            if direction:
                header += f" — Direction: {direction}"

        findings_parts.append(header)
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

    quality = build_quality_disclosures(state)
    limitation_items = []
    for item in [
        *(quality.get("blocking_issues") or []),
        *(quality.get("cautions") or []),
    ]:
        text = str(item or "").strip()
        if text and text not in limitation_items:
            limitation_items.append(text[:800])
    if getattr(state, "allow_partial_report", False):
        limitation_items.append(
            "The report was explicitly approved with unresolved evidence qualifications."
        )
    limitations_text = (
        "\n".join(f"- {item}" for item in limitation_items[:40])
        or "- No unresolved machine-recorded evidence qualification was retained."
    )

    # Collect data-driven mechanism programs from last 3 rounds
    mech_parts = []
    for rr in state.round_results[-3:]:
        progs = (getattr(rr, "data_driven_mechanism_programs", []) or [])[:3]
        for p in progs:
            prefix = (
                "[EXECUTION LINKAGE NOT RECORDED] "
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
        figure_outline or outline, figure_map, attribution
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

## Recorded Technical Limitations for the Existing Limitations Subsection
{limitations_text}

## Mechanism Naming and Hypothesis Policy
{MECHANISM_EVIDENCE_POLICY}

## Available Figures
{figure_list}

For registered publication figures, the declared type and authoritative
description above outrank any inference from the filename. Captions MUST stay
consistent with them; for example, never call a declared barplot a UMAP.

## Figure Allocation (BINDING)
Every figure belongs to the round that produced it, and each section below owns
the figures of its round. Place each figure ONLY in the section listed here.
Do not move a figure to a section that "reads" like a better fit — a filename
describes the kind of plot, not the stage of the analysis it came from.
Within each section, write the scientific narrative and place the figures in the
EXACT numbered order shown. That sequence follows the persisted analysis output
order; do not regroup figures by plot type or reorder them for rhetorical effect.

{figure_allocation}

## Available Tables (CSV)
{table_list_text}

## Instructions
Write the report in English.
Treat unresolved evidence conservatively throughout: never upgrade it to validated,
confirmed, causal, dominant, or independently replicated conclusions, even if a PI
summary or literature passage uses stronger wording.
In the Limitations subsection, state the expression representation each quantitative
result rests on (raw counts, library-size normalized + log, scaled, or pseudobulk
aggregated), and flag any result produced by a method run on a representation it does
not strictly assume — count-based models given normalized input, or continuous methods
given counts. Both directions change the result and neither raises an error, so it has
to be recorded rather than inferred.
Where two modalities agree, state whether they came from the same biological samples
(a cross-assay consistency check) or from different ones (independent replication).
Never describe the same-sample case as independent validation. Do not repeat workflow-status
labels in every findings section. Consolidate the supplied qualifications in exactly
one **Limitations** subsection near the end of [conclusions].
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
- Make the report thorough and comprehensive with detailed scientific content, aiming for roughly 300-500 words per section{brevity_note}. This is a target for substance, not a quota: a section whose analysis produced little should say so concisely rather than being padded to length.
- Innovation phase sections are CRITICAL and must be written in detail — detail meaning what those rounds actually established, at the claim level the evidence supports. Do NOT assume they produced novel findings: report a null, inconclusive, or merely confirmatory innovation round as exactly that. A round that ruled something out is a result worth stating plainly, not a section to pad.
- Reference specific figures, genes, pathways, cell types, statistical results, and quantitative findings.
- Include ALL available figures, each in the section assigned to it by the Figure Allocation above, using [FIGURE: filename] markers.
- Explicit text-only synthesis/transition sections require no figure when none was produced.
- Include ALL available tables — assign each table to the most relevant section using [TABLE: filename.csv] markers.
- Every [FIGURE:] and [TABLE:] marker MUST be followed by its corresponding [FIGURE_CAPTION:] or [TABLE_CAPTION:] marker on the next line.
{gate_note}"""
    return prompt

def _get_report_system_prompt(self, language: str) -> str:
    return (
        "You are a senior bioinformatics researcher writing a detailed analysis report. "
        "Describe the study in terms of the modalities it actually used — single-cell, "
        "bulk, imaging, biochemical, or a combination — and never frame it as a "
        "single-cell study unless single-cell data was analysed. "
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
                lines.append(
                    f"Error: {compact_traceback(rec['code_error'], max_chars=300)}"
                )
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
    eligible_figures = eligible_artifacts(state.publication_figures, state=state)
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

**Limitations** The report-writing model was unavailable, so no new PI interpretation could be generated.
"""

async def prepare_report_generation(
    self, analysis_state, report_model, output_dir, workspace_dir,
    progress_callback, report_llm_client, force_fallback,
):
    self.refresh_runtime_config_if_needed()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    reset_report_asset_directories(output_path)
    workspace_path = Path(workspace_dir) if workspace_dir else None

    language = "en"

    async def _progress(
        pct: float,
        step: str,
        status: str = "generating",
    ):
        if progress_callback:
            await progress_callback({
                "progress": pct,
                "step": step,
                "status": status,
            })

    await _progress(5, "Preparing analysis records...")

    # Collect all records
    all_records = self.get_supervision_log()
    stats = self.get_summary_stats()

    # Build structured analysis narrative for the LLM
    narrative = self._build_analysis_narrative(all_records, analysis_state, workspace_path)

    # Collect figure paths from workspace and publication figures
    figure_map = self._collect_figure_paths(all_records, workspace_path, state=analysis_state)
    figure_metadata_by_stem = publication_figure_metadata(
        analysis_state.publication_figures,
        report_only=True,
        state=analysis_state,
    )

    # Collect table paths from workspace
    table_list = self._collect_table_paths(
        workspace_path,
        publication_tables=analysis_state.publication_tables,
        invalidated_stems=excluded_artifact_stems(
            analysis_state.publication_tables,
            eligible_artifacts(analysis_state.publication_tables),
            invalidated_stems=analysis_state.invalidated_artifact_stems,
        ),
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

    chapter_artifact_ids = {
        a.artifact_id for r in analysis_state.round_results if r.report_chapter
        for a in r.report_chapter.artifacts
    }
    artifact_names = report_artifact_names(
        analysis_state, figure_map, table_list, artifact_ids=chapter_artifact_ids,
    )
    saved_sections = reusable_sections(analysis_state, outline, figure_map, table_list, names=artifact_names)
    saved_ids = {section["id"] for section in saved_sections}
    writing_outline = [section for section in outline if section.id not in saved_ids]
    saved_context = reuse_context(saved_sections)
    reserved = reserved_artifact_names(
        saved_sections, figure_map, [table["filename"] for table in table_list],
    )
    reserved_names = reserved["FIGURE"] | reserved["TABLE"]
    writing_figures = {name: path for name, path in figure_map.items() if name not in reserved["FIGURE"]}
    writing_tables = [table for table in table_list if table["filename"] not in reserved["TABLE"]]

    await _progress(15, "Generating report content...")

    # Generate report sections using strong model
    report_prompt = self._build_report_prompt(
        narrative, stats, analysis_state, writing_figures, language,
        table_list=writing_tables, outline=writing_outline, attribution=attribution,
        figure_metadata_by_stem=figure_metadata_by_stem,
        figure_outline=writing_outline,
    ) + saved_context

    used_fallback = False
    try:
        # Use the provided LLM client (matching report_model's endpoint),
        # falling back to supervisor's own client if not provided.
        client = report_llm_client or self.llm_client
        client.set_context(self.session_id, "report_generator")
        if force_fallback:
            logger.warning(
                "[%s] Building deterministic detailed-report fallback",
                self.session_id,
            )
            report_content = self._fallback_report(narrative, stats, language)
            used_fallback = True
        else:
            try:
                response = await self._budgeted_text_call(
                    client.chat,
                    model=report_model,
                    messages=[
                        {"role": "system", "content": self._get_report_system_prompt(language)},
                        {"role": "user", "content": report_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=NO_OUTPUT_TOKEN_CAP,
                    timeout=900.0,  # 15 min — large narratives need long generation time
                    prefer_stream=True,
                )
                report_content = response.content or ""
            except UpstreamExhaustedError:
                raise  # propagate to caller for retry/user-intervention
            except Exception:
                logger.exception(
                    f"[{self.session_id}] Report generation LLM call failed"
                )
                report_content = self._fallback_report(narrative, stats, language)
                used_fallback = True
    finally:
        # Restore supervisor context in case of subsequent LLM calls
        self._set_llm_context()

    return {
        "output_path": output_path, "workspace_path": workspace_path,
        "language": language, "_progress": _progress, "all_records": all_records,
        "stats": stats, "narrative": narrative, "figure_map": figure_map,
        "figure_metadata_by_stem": figure_metadata_by_stem, "table_list": table_list,
        "outline": outline, "attribution": attribution, "report_content": report_content,
        "used_fallback": used_fallback, "client": client,
        "saved_sections": saved_sections, "saved_context": saved_context,
        "reserved_names": reserved_names, "writing_figures": writing_figures,
        "writing_tables": writing_tables,
        "writing_outline": writing_outline,
        "saved_figure_captions": saved_figure_captions(analysis_state, figure_map, table_list, names=artifact_names),
    }

async def _request_unwritten_sections(
    self, ctx, analysis_state, sections, pending, report_model,
):
    """Hand the chapters the first pass never wrote back to the report writer.

    A report body can end before the outline does — the writer spends its
    output budget on private reasoning, or the request hits its wall-clock
    deadline — and the tail of the outline is then simply absent. Downstream
    that absence was invisible: ``align_sections_to_outline`` keeps a skipped
    section only when it owns a figure, so a text-only gate and a direction
    whose figures were all attributed elsewhere both vanished, taking their
    analysis rounds with them.

    Asking the writer to finish the chapters it planned is that agent's own
    work handed back to it, not a host gate over the science. It is therefore
    strictly best effort: one request, every failure swallowed, and whatever
    is still absent afterwards is disclosed rather than enforced. "Swallowed"
    has to cover the whole pass — building the second prompt and parsing an
    arbitrary continuation body included — because a recovery path that can
    raise turns "a report missing its tail chapters", which is publishable,
    into no report at all. The one exception is an exhausted upstream, which
    the workflow's own gate must still see so the user can Retry or Skip.
    """
    language = ctx["language"]
    client = ctx["client"]
    written = [
        str(section.get("id") or "")
        for section in sections
        if str(section.get("content") or "").strip()
    ]

    try:
        continuation = (
            "## Continuation of an unfinished report\n"
            "An earlier pass of this same report ended before it wrote every "
            "planned section. These sections are already written and MUST NOT be "
            "repeated, rewritten or referred to as pending: "
            f"{', '.join(written) or 'none'}.\n"
            "Write ONLY the sections listed below, in the same "
            "`## [section_id] Title` format, and emit nothing else — no preamble, "
            "no closing remarks, no sections outside that list.\n\n"
        ) + self._build_report_prompt(
            ctx.get("narrative", ""),
            ctx.get("stats", {}),
            analysis_state,
            ctx.get("writing_figures", ctx["figure_map"]),
            language,
            table_list=ctx.get("writing_tables", ctx.get("table_list")),
            outline=list(pending),
            attribution=ctx["attribution"],
            figure_metadata_by_stem=ctx["figure_metadata_by_stem"],
            # Ownership is decided by round over the WHOLE report. Handing the
            # allocation only the pending sections leaves every figure of an
            # already-written chapter unclaimed, and the prompt then offers it to
            # this pass as a free candidate — inside the block it calls BINDING.
            figure_outline=ctx.get("writing_outline", ctx["outline"]),
        ) + ctx.get("saved_context", "")
        client.set_context(self.session_id, "report_generator")
        response = await self._budgeted_text_call(
            client.chat,
            model=report_model,
            messages=[
                {"role": "system", "content": self._get_report_system_prompt(language)},
                {"role": "user", "content": continuation},
            ],
            temperature=0.3,
            max_tokens=NO_OUTPUT_TOKEN_CAP,
            timeout=900.0,
            prefer_stream=True,
        )
        content = response.content or ""
        parsed = self._parse_report_sections(
            content, language, outline=ctx["outline"]
        )
    except UpstreamExhaustedError:
        # The sibling first-pass call re-raises this for the same reason: the
        # workflow HITL gate is where the user can retry, or change the report
        # model, and swallowing it here spends that chance silently.
        raise
    except Exception:
        logger.exception(
            f"[{self.session_id}] Could not ask the report writer to complete "
            f"the missing sections; publishing the report as written"
        )
        return sections
    finally:
        self._set_llm_context()

    wanted = {osec.id for osec in pending}
    recovered = {
        str(section.get("id") or ""): section
        for section in parsed
        if str(section.get("id") or "") in wanted
        and str(section.get("content") or "").strip()
    }
    if not recovered:
        logger.warning(
            f"[{self.session_id}] Report writer returned no usable section on "
            f"the completion request; {len(wanted)} section(s) stay unwritten"
        )
        return sections

    completed = len(recovered)
    merged = [
        recovered.pop(str(section.get("id") or ""), section)
        for section in sections
    ]
    merged.extend(recovered.values())
    logger.info(
        f"[{self.session_id}] Report writer completed {completed} of "
        f"{len(wanted)} unwritten section(s)"
    )
    return merged

async def parse_and_place_report(
    self, ctx, analysis_state, workspace_dir, chat_messages,
    progress_callback, report_model,
):
    _progress = ctx["_progress"]
    report_content = ctx["report_content"]
    language = ctx["language"]
    outline = ctx["outline"]
    used_fallback = ctx["used_fallback"]
    attribution = ctx["attribution"]
    figure_map = ctx["figure_map"]
    figure_metadata_by_stem = ctx["figure_metadata_by_stem"]
    client = ctx["client"]
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
    saved_sections = ctx.get("saved_sections", [])
    if saved_sections:
        sections = merge_reusable_sections(sections, saved_sections)

    # For a normal report this is populated by the placement pass below. A
    # degraded fallback keeps its fixed three-section body, but that must not
    # make visual coverage disappear from the report semantics: every
    # ordinary result section in the intended outline is absent from that
    # fallback body and therefore has no in-section figure. Explicit
    # text-only sections (integration gates and closing syntheses) are not
    # included.
    figureless_sections: List[str] = []
    # Planned sections the published body does not carry. A degraded fallback
    # is already reported as such, so it leaves this empty rather than naming
    # every chapter of an outline it was never written against.
    missing_sections: List[str] = []

    # The fallback body is a degraded step summary with its own fixed
    # sections; forcing it through an outline it was never written against
    # would only scramble it. Its figures fall through to the supplementary
    # section, which is the honest presentation for a failed report.
    if used_fallback and saved_sections:
        saved_ids = {s["id"] for s in saved_sections}
        ordered_saved = sorted(saved_sections, key=lambda s: next(
            i for i, item in enumerate(outline) if item.id == s["id"]
        ))
        sections = [s for s in sections if s.get("id") not in saved_ids] + ordered_saved
    if not used_fallback:
        # The alignment below keeps a section the model skipped only when it
        # owns a figure, so a body that stopped early silently loses whole
        # chapters — and with them their analysis rounds. Give the writer one
        # chance to finish what it planned before anything is dropped.
        pending = [
            osec for osec in outline
            if osec.id not in {
                str(section.get("id") or "")
                for section in sections
                if str(section.get("content") or "").strip()
            }
        ]
        if pending and not used_fallback:
            logger.warning(
                f"[{self.session_id}] Report body is missing {len(pending)} "
                f"planned section(s): "
                f"{', '.join(osec.id for osec in pending)} — asking the report "
                f"writer to complete them"
            )
            sections = await _request_unwritten_sections(
                self, ctx, analysis_state, sections, pending, report_model,
            )
            if saved_sections:
                sections = merge_reusable_sections(
                    sections, saved_sections,
                )

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

        # Read before placement, while section content is still pure prose: a
        # section carrying nothing but appended figure markers has no narrative
        # and is exactly the gap being recorded. What remains unwritten is
        # disclosed to the PI and the reader; it never blocks publication.
        missing_sections = missing_outline_section_ids(sections, outline)
        if missing_sections:
            logger.warning(
                f"[{self.session_id}] Report published without "
                f"{len(missing_sections)} planned section(s): "
                f"{', '.join(missing_sections)}"
            )

        # Placement is enforced in code, not left to the model's compliance:
        # it relocates cross-round references, drops repeats, and appends
        # figures the model never cited to the section owning their round.
        # ``figure_map`` is collected from supervision execution order first,
        # then the persisted PublicationFigure sequence, with legacy discovery
        # last. Preserve that analysis order; filesystem mtimes change when a
        # session is restored or report assets are copied.
        figure_order = {
            name: float(index) for index, name in enumerate(figure_map)
        }
        protected_figures = reserved_artifact_names(saved_sections, figure_map)["FIGURE"]
        placement = enforce_figure_placement(
            sections, outline, attribution, figure_map.keys(),
            figure_order=figure_order,
            protected_sections={s["id"] for s in saved_sections},
            figure_captions={
                name: (
                    figure_metadata_by_stem.get(Path(name).stem, {}).get("description")
                    or figure_metadata_by_stem.get(Path(name).stem, {}).get("interpretation")
                    or ""
                )
                for name in figure_map
                if name not in protected_figures
            },
        )
        if placement.unplaced:
            logger.info(
                f"[{self.session_id}] {len(placement.unplaced)} figures had no "
                f"owning section and fall through to Supplementary Figures"
            )
        figureless_sections = list(placement.figureless_sections)
    else:
        figureless_sections = [
            section.id for section in outline if section.expects_figures
        ]

    # Build chat sections if chat_messages provided
    if chat_messages:
        try:
            from backend.services.report.chat_sections import build_chat_sections_async
            chat_sections = await build_chat_sections_async(
                chat_messages,
                figures_dir=str(Path(workspace_dir) / "figures") if workspace_dir else "",
                tables_dir=str(Path(workspace_dir) / "tables") if workspace_dir else "",
                llm_client=client,
                progress_callback=progress_callback,
                model=report_model,
                context_budget_callback=self._provider.budget_text_dispatch,
                context_budget_ack_callback=(
                    self._provider.acknowledge_text_dispatch
                ),
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

    ctx.update(
        sections=sections, figureless_sections=figureless_sections,
        missing_sections=missing_sections,
        abstract=abstract, conclusions=conclusions, filtered_sections=filtered_sections,
    )
