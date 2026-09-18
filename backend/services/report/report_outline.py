"""Compute the section outline of the final report from the analysis state.

The outline replaces a fixed template with one derived from what actually ran.
Two ordering principles were previously mixed in a single document — the opening
chapters were organised by *analysis type* (QC, clustering, differential
expression) while the later ones followed *round order*. Every figure then had a
home under both principles and nothing arbitrated between them, so figures drifted
into chapters describing a stage they had no part in.

Here a single axis carries the report: the round. A round is simultaneously the
unit of time and the unit of argument (each innovation round pursues one research
direction), so ordering by round is both chronological and argumentative. Method
description moves to a Methods chapter that owns no result figures.

Integration rounds are placed by what follows them, not by what they are. An
integration round whose batch is followed by further innovation work is a
*gate*: it explains why the next direction was chosen, so it belongs inline,
between the directions it connects, and stays short — its conclusions are
provisional and later rounds may revise them. The final integration round has no
successor and closes the argument, so it becomes a full chapter.

Keep this generic: no dataset-, tissue-, disease-, or organism-specific rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "OutlineSection",
    "build_report_outline",
    "GATE_SECTION_WORD_LIMIT",
    "FIGURE_POLICY_PREFER",
    "FIGURE_POLICY_TEXT_ONLY",
]

# A gate section is a hinge between two directions, not a findings chapter.
# Its conclusions are provisional — the rounds that follow may revise them — so
# giving it chapter-sized prose invites readers to bank superseded results.
GATE_SECTION_WORD_LIMIT = 200

# Figure expectation is separate from round ownership. Integration sections can
# own a round for ordering/provenance while remaining textual syntheses: an
# integration round often produces no new plot, and that is not a report defect.
FIGURE_POLICY_PREFER = "prefer"
FIGURE_POLICY_TEXT_ONLY = "text_only"


@dataclass(frozen=True)
class OutlineSection:
    """One planned section of the report.

    Attributes:
        id: Stable section id used in the ``## [id] Title`` marker.
        kind: Role of the section, for downstream formatting decisions.
        default_title: Fallback title when the model emits a placeholder.
        rounds: Round numbers whose figures belong to this section. Empty means
            the section owns no result figures at all.
        analysis_type: ``"general"`` or ``"specific"``, for the HTML badge.
        guidance: One-line description handed to the report model.
        figure_policy: ``"prefer"`` for ordinary result sections, or
            ``"text_only"`` for explicit synthesis/transition sections that do
            not require a figure even when they own a workflow round.
    """

    id: str
    kind: str
    default_title: str
    rounds: Tuple[int, ...] = ()
    analysis_type: str = "general"
    guidance: str = ""
    figure_policy: str = FIGURE_POLICY_PREFER
    result_keys: Tuple[str, ...] = ()

    @property
    def owns_figures(self) -> bool:
        return bool(self.rounds)

    @property
    def expects_figures(self) -> bool:
        """Whether an empty section should be reported as a coverage gap."""
        return self.owns_figures and self.figure_policy != FIGURE_POLICY_TEXT_ONLY

    @property
    def is_text_only(self) -> bool:
        return self.figure_policy == FIGURE_POLICY_TEXT_ONLY

    @property
    def is_brief(self) -> bool:
        return self.kind == "integration_gate"


def report_result_identity(result: Any) -> str:
    """Use the workflow identity, including for records written before it existed."""
    from backend.workflow.direction_coverage import build_result_identity_key, normalize_direction_key

    return getattr(result, "result_identity_key", "") or build_result_identity_key(
        getattr(result, "lineage_id", "") or "main",
        getattr(result, "phase", "") or "exploration",
        int(result.round_number),
        getattr(result, "direction_key", "") or normalize_direction_key(
            getattr(result, "direction", "") or ""
        ),
    )


def _phase_of(rr: Any) -> str:
    return (getattr(rr, "phase", None) or "exploration").lower()


def _sorted_rounds(state: Any) -> List[Any]:
    results = [
        rr for rr in (getattr(state, "round_results", None) or [])
        if getattr(rr, "round_number", None) is not None
    ]
    return sorted(results, key=lambda rr: int(rr.round_number))


def _legacy_outline(state: Any) -> List[OutlineSection]:
    """Type-organised outline, used when no innovation round ran.

    With a single pass through the data there is only one QC step, one
    clustering step and one differential-expression step, so organising by
    analysis type is unambiguous and matches how a single-pass pipeline reads.
    """
    rounds = tuple(int(rr.round_number) for rr in _sorted_rounds(state))
    return [
        OutlineSection(
            "abstract", "abstract", "Abstract",
            guidance="Research summary (background, methods, key findings, conclusions)",
        ),
        OutlineSection(
            "analysis_overview", "methods", "Analysis Methodology Overview",
            guidance=(
                "Analysis methodology overview (study design, data processing pipeline, "
                "computational methods, key parameters)"
            ),
        ),
        OutlineSection(
            "data_qc", "findings", "Data Overview and Quality Control", rounds,
            guidance="Data overview and quality control",
        ),
        OutlineSection(
            "clustering_annotation", "findings", "Clustering and Cell Type Annotation", rounds,
            guidance=(
                "Clustering and cell type annotation (include PI checkpoint guidance "
                "if available)"
            ),
        ),
        OutlineSection(
            "differential_expression", "findings", "Differential Expression Analysis", rounds,
            analysis_type="specific",
            guidance=(
                "Differential expression analysis (include PI checkpoint guidance "
                "if available)"
            ),
        ),
        OutlineSection(
            "mechanism_candidates", "mechanisms", "Data-Driven Mechanism Candidates",
            guidance=(
                "Data-driven candidate mechanisms (use hypothesis wording for low/medium "
                "confidence; never present as confirmed; include missing evidence)"
            ),
        ),
        OutlineSection(
            "key_findings", "findings", "Key Biological Findings", rounds,
            analysis_type="specific",
            guidance="Key biological findings",
        ),
        OutlineSection(
            "conclusions", "conclusions", "Conclusions and Future Directions",
            guidance=(
                "Conclusions and future directions, ending with exactly one "
                "**Limitations** subsection that consolidates evidence qualifications"
            ),
        ),
    ]


def build_report_outline(state: Any) -> List[OutlineSection]:
    """Derive the report outline from the rounds that actually ran.

    Saved PI chapters use the round-based outline, including single-pass runs.
    Older single-pass sessions without saved chapters keep their type-organised
    outline. Global sections provide methodology and cross-round context.
    """
    rounds = _sorted_rounds(state)
    innovation = [rr for rr in rounds if _phase_of(rr) == "innovation"]
    if not innovation and not any(getattr(rr, "report_chapter", None) for rr in rounds):
        return _legacy_outline(state)

    exploration_rounds = tuple(
        int(rr.round_number) for rr in rounds if _phase_of(rr) == "exploration"
    )

    sections: List[OutlineSection] = [
        OutlineSection(
            "abstract", "abstract", "Abstract",
            guidance="Research summary (background, methods, key findings, conclusions)",
        ),
        OutlineSection(
            "methods", "methods", "Methods",
            guidance=(
                "Study design, data processing, quality control thresholds, clustering and "
                "annotation procedure, statistical methods and key parameters. Describe the "
                "procedure only — results, and any figure, belong to the sections below"
            ),
        ),
    ]

    if exploration_rounds:
        sections.append(OutlineSection(
            "exploration_findings", "exploration", "Phase 1 Exploration Findings",
            exploration_rounds, analysis_type="specific",
            result_keys=tuple(report_result_identity(rr) for rr in rounds if _phase_of(rr) == "exploration"),
            guidance=(
                "What the unbiased first pass found: composition, cell states and quality "
                "characteristics of the data, plus the research directions it motivated"
            ),
        ))

    # Innovation and integration rounds, in round order. An integration round is
    # a gate when innovation work follows it, and the closing chapter otherwise.
    last_innovation_round = int(innovation[-1].round_number) if innovation else 0
    last_integration = next((rr for rr in reversed(rounds) if _phase_of(rr) == "integration"), None)
    direction_index = 0
    gate_index = 0
    final_integration_seen = False

    for rr in rounds:
        phase = _phase_of(rr)
        rn = int(rr.round_number)
        if phase == "innovation":
            direction_index += 1
            direction = (getattr(rr, "direction", "") or "").strip()
            short = direction[:80] if direction else f"Direction {direction_index}"
            sections.append(OutlineSection(
                f"direction_{direction_index}", "direction",
                f"Innovation Round {rn}", (rn,), analysis_type="specific",
                result_keys=(report_result_identity(rr),),
                guidance=(
                    f"Innovation round {rn} — research direction: {short}. State the "
                    f"hypothesis, the targeted analyses run against it, the findings and "
                    f"their biological significance. This section MUST be comprehensive "
                    f"and include ALL findings from this research direction"
                ),
            ))
        elif phase == "integration":
            if rn < last_innovation_round or rr is not last_integration:
                gate_index += 1
                sections.append(OutlineSection(
                    f"integration_gate_{gate_index}", "integration_gate",
                    f"Integration Cycle {gate_index}", (rn,),
                    figure_policy=FIGURE_POLICY_TEXT_ONLY,
                    result_keys=(report_result_identity(rr),),
                    guidance=(
                        f"Integration cycle {gate_index} (round {rn}) — a hinge between "
                        f"directions, not a findings chapter. Answer exactly two questions: "
                        f"what the directions completed so far agree, complement or "
                        f"contradict on, and which new direction that motivated. Keep it "
                        f"under {GATE_SECTION_WORD_LIMIT} words and label the conclusions "
                        f"as provisional — later rounds may revise them"
                    ),
                ))
            else:
                final_integration_seen = True
                sections.append(OutlineSection(
                    "integrated_findings", "integration_final",
                    "Integrated Findings Across All Phases", (rn,), analysis_type="specific",
                    figure_policy=FIGURE_POLICY_TEXT_ONLY,
                    result_keys=(report_result_identity(rr),),
                    guidance=(
                        "Closing cross-direction synthesis: how the directions converge, "
                        "which mechanisms they jointly support, where they remain in "
                        "tension, and what the combined evidence does and does not "
                        "establish"
                    ),
                ))

    if not final_integration_seen:
        # Innovation work ran but no closing integration round exists (the run
        # ended on a direction). The report still needs somewhere to draw the
        # threads together; it owns no figures of its own.
        sections.append(OutlineSection(
            "integrated_findings", "integration_final",
            "Integrated Findings Across All Phases", (), analysis_type="specific",
            figure_policy=FIGURE_POLICY_TEXT_ONLY,
            guidance=(
                "Cross-phase synthesis: connections between the directions pursued, and "
                "how the innovation findings validate or extend the exploration results"
            ),
        ))

    sections.append(OutlineSection(
        "mechanism_candidates", "mechanisms", "Data-Driven Mechanism Candidates",
        guidance=(
            "Data-driven candidate mechanisms (use hypothesis wording for low/medium "
            "confidence; never present as confirmed; include missing evidence)"
        ),
    ))
    sections.append(OutlineSection(
        "conclusions", "conclusions", "Conclusions and Future Directions",
        guidance=(
            "Conclusions and future directions, ending with exactly one "
            "**Limitations** subsection that consolidates evidence qualifications"
        ),
    ))
    return sections


def outline_section_titles(outline: Sequence[OutlineSection]) -> Dict[str, str]:
    """Map section id → default title, for placeholder repair at parse time."""
    return {sec.id: sec.default_title for sec in outline}


def section_for_round(
    outline: Sequence[OutlineSection], round_number: int
) -> Optional[OutlineSection]:
    """Return the section that owns figures from ``round_number``, if any."""
    for sec in outline:
        if round_number in sec.rounds:
            return sec
    return None


def align_sections_to_outline(
    sections: List[Dict[str, Any]],
    outline: Sequence[OutlineSection],
    require_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Reorder parsed sections into outline order and fill in missing ones.

    The reading order of the report is a property of the analysis, not of the
    order the model happened to emit its sections in, so it is imposed here
    rather than hoped for. Sections the model invented keep their relative
    order and follow the planned ones, which is where an unplanned aside
    belongs.

    ``require_ids`` names sections that must exist even if the model skipped
    them — used for sections that own figures, so those figures have somewhere
    to land instead of falling through to a supplementary dump.
    """
    planned_order = {sec.id: idx for idx, sec in enumerate(outline)}
    by_id: Dict[str, Dict[str, Any]] = {}
    extras: List[Dict[str, Any]] = []
    for sec in sections:
        sid = sec.get("id")
        if sid in planned_order and sid not in by_id:
            by_id[sid] = sec
        else:
            extras.append(sec)

    required = set(require_ids or ())
    ordered: List[Dict[str, Any]] = []
    for osec in outline:
        found = by_id.get(osec.id)
        if found is not None:
            ordered.append(found)
        elif osec.id in required:
            ordered.append({
                "id": osec.id,
                "title": osec.default_title,
                "content": "",
                "analysis_type": osec.analysis_type,
                "description": osec.guidance,
                "figures": [],
                "tables": [],
            })
    return ordered + extras


def missing_outline_section_ids(
    sections: Sequence[Dict[str, Any]],
    outline: Sequence[OutlineSection],
) -> List[str]:
    """Planned sections the report body does not carry, in outline order.

    A section is missing when it was never emitted, and equally when it was
    materialised without prose — a chapter heading over nothing states no
    result. Both cases used to leave no trace: a planned section owning no
    figure is dropped by ``align_sections_to_outline``, and a whole analysis
    round then disappeared from the report with only the model's own
    compliance standing between the reader and a silent gap.

    This is an observation, never a verdict. Callers disclose what it returns
    and publish the report regardless; whether an absent chapter matters to
    the science is the PI's call, not this function's.
    """
    written = {
        str(section.get("id") or "")
        for section in (sections or [])
        if str(section.get("content") or "").strip()
    }
    return [osec.id for osec in outline if osec.id not in written]
