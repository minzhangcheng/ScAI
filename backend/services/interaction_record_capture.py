"""Pure helpers that build and merge user-interaction records from AnalysisState.

No I/O, no LLM calls. Capture points (routes/workflow) call these, then persist
the state as usual.
"""
from typing import Any, Dict, List, Optional

from ..models.schemas import (
    AnalysisPhase, AnalysisState, SelectionEntry, DirectionSelectionRecord,
    AnnotationRecord, AnnotationClusterEntry,
)


def get_or_open_direction_record(
    state: AnalysisState,
    *,
    round_number: int,
    phase: str,
    integration_cycle: Optional[int] = None,
) -> DirectionSelectionRecord:
    for rec in state.direction_selection_records:
        if (
            rec.event_type == "open"
            and rec.round_number == round_number
            and rec.phase == phase
            and rec.integration_cycle == integration_cycle
        ):
            return rec
    rec = DirectionSelectionRecord(
        round_number=round_number, phase=phase, integration_cycle=integration_cycle,
    )
    state.direction_selection_records.append(rec)
    return rec


def find_open_pi_decide_record(state: AnalysisState) -> Optional[DirectionSelectionRecord]:
    for rec in reversed(state.direction_selection_records):
        if (
            rec.event_type == "open"
            and rec.phase == "exploration"
            and rec.let_pi_decide
            and rec.integration_cycle is None
        ):
            return rec
    return None


def map_selections(
    directions: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> List[SelectionEntry]:
    entries: List[SelectionEntry] = []
    for d in directions:
        text = (d.get("direction") or "").strip()
        idx = next(
            (i for i, c in enumerate(candidates)
             if (c.get("direction") or "").strip() == text),
            None,
        )
        is_custom = bool(d.get("is_custom")) if "is_custom" in d else (idx is None)
        entries.append(SelectionEntry(
            direction=text, notes=d.get("notes", "") or "",
            is_custom=is_custom, candidate_index=idx,
        ))
    return entries


def populate_direction_record_from_state(state: AnalysisState, rec: DirectionSelectionRecord, event_type: str) -> None:
    if state.analysis_phase == AnalysisPhase.INTEGRATION:
        rec.candidates = list(state.integration_gate_candidates)
        rec.literature_evidence = list(state.integration_gate_literature_evidence)
    else:
        report = state.exploration_report
        rec.candidates = list(
            getattr(report, "research_direction_candidates", []) if report else []
        )
        rec.literature_evidence = list(
            getattr(report, "direction_literature_evidence", []) if report else []
        )
    rec.event_type = event_type


def resolve_integration_gate(state, event_type: str):
    rec = get_or_open_direction_record(
        state,
        round_number=state.current_round,
        phase="integration",
        integration_cycle=state.integration_cycle,
    )
    populate_direction_record_from_state(state, rec, event_type)
    state.integration_gate_candidates = []
    state.integration_gate_literature_evidence = []
    return rec


def build_annotation_record(
    state: AnalysisState, *, auto_confirmed: bool,
) -> AnnotationRecord:
    ann = state.annotation_state
    clusters: List[AnnotationClusterEntry] = []
    if ann:
        for ca in ann.cluster_annotations:
            clusters.append(AnnotationClusterEntry(
                cluster_id=ca.cluster_id,
                original_cell_type=ca.cell_type,
                final_cell_type=(ca.user_override or ca.cell_type),
                user_override=bool(ca.user_override),
                confidence=ca.confidence,
                key_markers=list(ca.key_markers),
                group_proportions=dict(ca.group_proportions),
                alternative_cell_types=list(ca.alternative_cell_types),
                low_confidence_reason=ca.low_confidence_reason,
            ))
    return AnnotationRecord(
        organism=ann.organism if ann else "",
        tissue=ann.tissue if ann else "",
        tissue_summary=ann.tissue_summary if ann else "",
        proportion_dimension=ann.proportion_dimension if ann else "group",
        proportion_group_labels=list(ann.proportion_group_labels) if ann else [],
        clusters=clusters,
        supplementary_requests=list(state.annotation_supplementary_requests),
        auto_confirmed=auto_confirmed,
    )
