from dataclasses import dataclass
from typing import Iterable, Optional, Set

from ..models.schemas import AnalysisState, RoundResult


@dataclass(frozen=True)
class DirectionCoverageStatus:
    selected_keys: Set[str]
    completed_keys: Set[str]
    missing_keys: Set[str]
    complete: bool


def normalize_direction_key(direction: str) -> str:
    return " ".join((direction or "").strip().lower().split())


def current_lineage_id(state: AnalysisState) -> str:
    return state.restore_branch_id or "main"


def build_result_identity_key(lineage_id: str, phase: str, round_number: int, direction_key: str) -> str:
    return f"{lineage_id}:{phase}:{round_number}:{direction_key}"


def has_substantive_findings(result: RoundResult) -> bool:
    if result.round_status in ("failed", "skipped"):
        return False
    finding_groups = (
        result.pathology_findings,
        result.bioinformatics_findings,
        result.biochemistry_findings,
        result.literature_findings,
        result.data_driven_mechanism_programs,
    )
    return any(bool(group) for group in finding_groups)


def normalize_legacy_round_result(result: RoundResult) -> RoundResult:
    result = result.model_copy(deep=True)
    if not result.lineage_id:
        result.lineage_id = "main"
    if not result.direction_key:
        result.direction_key = normalize_direction_key(result.direction)
    if not result.result_identity_key:
        result.result_identity_key = build_result_identity_key(
            result.lineage_id,
            result.phase,
            result.round_number,
            result.direction_key,
        )
    if result.evidence_status == "empty" and has_substantive_findings(result):
        result.evidence_status = "substantive"
    if result.evidence_status == "failed" and result.round_status == "completed":
        result.round_status = "partial"
    return result


def selected_direction_keys(state: AnalysisState) -> Set[str]:
    return {
        key
        for key in (normalize_direction_key(d.get("direction", "")) for d in state.user_chosen_directions)
        if key
    }


def completed_innovation_direction_keys(state: AnalysisState, lineage_id: Optional[str] = None) -> Set[str]:
    target_lineage = lineage_id or current_lineage_id(state)
    completed: Set[str] = set()
    for raw in state.round_results:
        result = normalize_legacy_round_result(raw)
        if result.lineage_id != target_lineage:
            continue
        if result.phase != "innovation":
            continue
        if not result.direction_key:
            continue
        if not result.pi_summary.strip():
            continue
        if result.round_status != "completed":
            continue
        if result.evidence_status != "substantive":
            continue
        if not has_substantive_findings(result):
            continue
        completed.add(result.direction_key)
    return completed


def direction_coverage_status(state: AnalysisState, lineage_id: Optional[str] = None) -> DirectionCoverageStatus:
    selected = selected_direction_keys(state)
    completed = completed_innovation_direction_keys(state, lineage_id=lineage_id)
    missing = selected - completed
    return DirectionCoverageStatus(
        selected_keys=selected,
        completed_keys=completed,
        missing_keys=missing,
        complete=bool(selected) and not missing,
    )


def count_innovation_rounds_done(state: AnalysisState, lineage_id: Optional[str] = None) -> int:
    """Count innovation rounds with substantive direction coverage.

    Uses the same predicate as completed_innovation_direction_keys but
    counts individual matching rounds (not deduplicated by direction key),
    so one direction with multiple substantive rounds contributes >1.
    """
    target_lineage = lineage_id or current_lineage_id(state)
    count = 0
    for raw in state.round_results:
        result = normalize_legacy_round_result(raw)
        if result.lineage_id != target_lineage:
            continue
        if result.phase != "innovation":
            continue
        if not result.direction_key:
            continue
        if not result.pi_summary.strip():
            continue
        if result.round_status != "completed":
            continue
        if result.evidence_status != "substantive":
            continue
        if not has_substantive_findings(result):
            continue
        count += 1
    return count


def clone_round_results_for_branch(
    results: list[RoundResult],
    *,
    branch_id: str,
    branch_label: Optional[str],
    source_snapshot_id: str,
    source_lineage_id: str = "main",
) -> list[RoundResult]:
    """Create an explicit branch lineage view that inherits restored ancestors.

    Main-line results are preserved. Branch clones let branch coverage/reporting
    read only the branch lineage without losing valid work completed before the
    restore point.
    """
    output: list[RoundResult] = []
    seen_keys: set[str] = set()
    for raw in results:
        original = normalize_legacy_round_result(raw)
        output.append(original)
        seen_keys.add(original.result_identity_key)

        if original.lineage_id == branch_id:
            continue
        if original.lineage_id != source_lineage_id:
            continue
        clone = original.model_copy(deep=True)
        clone.lineage_id = branch_id
        clone.branch_id = branch_id
        clone.branch_label = branch_label
        clone.source_snapshot_id = source_snapshot_id
        clone.restored_from_snapshot_id = source_snapshot_id
        clone.result_identity_key = build_result_identity_key(
            clone.lineage_id,
            clone.phase,
            clone.round_number,
            clone.direction_key,
        )
        if clone.result_identity_key not in seen_keys:
            output.append(clone)
            seen_keys.add(clone.result_identity_key)
    return output
