from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Set

from ..models.schemas import AnalysisPhase, AnalysisState, RoundResult
from ..utils.direction_payload import normalize_direction_text, resolve_direction_context
from ..utils.text_coverage import uncovered_items


@dataclass(frozen=True)
class DirectionCoverageStatus:
    selected_keys: Set[str]
    completed_keys: Set[str]
    missing_keys: Set[str]
    complete: bool


def normalize_direction_key(direction: str) -> str:
    # One definition of "the same direction" for both the coverage ledger and
    # the payload lookup; they must not drift apart.
    return normalize_direction_text(direction)


def current_lineage_id(state: AnalysisState) -> str:
    return state.restore_branch_id or "main"


def build_result_identity_key(lineage_id: str, phase: str, round_number: int, direction_key: str) -> str:
    return f"{lineage_id}:{phase}:{round_number}:{direction_key}"


def has_substantive_findings(result: RoundResult) -> bool:
    if result.round_status in ("failed", "skipped"):
        return False
    # Integration rounds do not run the four domain agents.  Their evidence is
    # the cross-direction report itself (and, secondarily, the scored candidate
    # records), so applying the ordinary finding buckets makes every successful
    # integration round look empty.
    if result.phase == "integration":
        return bool(
            (result.integration_report_summary or "").strip()
            or result.integration_candidates
        )
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
    integration_has_evidence = bool(
        result.phase == "integration"
        and (
            (result.integration_report_summary or "").strip()
            or result.integration_candidates
        )
    )
    # 0.38.9 routed integration records through ordinary research-round
    # completion derivation.  Because those nodes intentionally have no
    # bioinformatics_result/literature_result, a real report was persisted as
    # skipped/failed.  This signature is unambiguous: an integration node that
    # raised never wrote a RoundResult at all.
    if result.evidence_schema_version is None and integration_has_evidence and (
        result.round_status in ("failed", "skipped")
        or result.evidence_status in ("empty", "failed")
    ):
        result.round_status = "completed"
        result.evidence_status = "substantive"
        result.direction_coverage_eligible = None
    elif (
        result.evidence_schema_version is None
        and result.evidence_status == "empty"
        and has_substantive_findings(result)
    ):
        result.evidence_status = "substantive"
    if (
        result.evidence_schema_version is None
        and result.evidence_status == "failed"
        and result.round_status == "completed"
    ):
        result.round_status = "partial"
    return result


def plan_text_for_coverage(plan: Any) -> str:
    """Flatten a plan into the prose a coverage check can search.

    Includes every step field rather than titles alone: which specific things a
    step examines is stated in its input and rationale, so a title-only view
    reports a genericised plan as fully covering its direction.
    """
    if plan is None:
        return ""
    parts: List[str] = [str(getattr(plan, "overview", "") or "")]
    for step in (getattr(plan, "steps", None) or []):
        for attr in ("title", "method", "input", "expected_output", "rationale"):
            parts.append(str(getattr(step, attr, "") or ""))
    parts.extend(str(p) for p in (getattr(plan, "key_decision_points", None) or []))
    return "\n".join(part for part in parts if part)


def uncovered_direction_requirements(
    state: AnalysisState,
    plan: Any,
    direction: str = "",
) -> List[str]:
    """Requirements of the active direction that the plan does not address.

    Coverage tracking elsewhere in this module answers "did a round run for
    this direction". It does not answer "did the plan address what the
    direction was selected to resolve", so a round can be recorded as covering
    a direction while never testing anything the direction asked about.

    Reads the resolved payload rather than the exploration candidate alone: a
    candidate exists only for the first directions taken out of the exploration
    report, so keying on it left this silent for most of a long run.
    """
    context = resolve_direction_context(state, direction)
    if not context.requirements:
        return []
    return uncovered_items(context.requirements, plan_text_for_coverage(plan))


def selected_direction_keys(
    state: AnalysisState,
    directions: Optional[Iterable[dict]] = None,
) -> Set[str]:
    source = state.user_chosen_directions if directions is None else directions
    return {
        key
        for key in (normalize_direction_key(d.get("direction", "")) for d in source)
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
        coverage_eligible = result.direction_coverage_eligible
        if coverage_eligible is None:
            # Backward compatibility for results persisted before the explicit
            # coverage field existed: retain the former strict interpretation.
            coverage_eligible = result.round_status == "completed"
        if not coverage_eligible:
            continue
        if result.evidence_status != "substantive":
            continue
        if not has_substantive_findings(result):
            continue
        completed.add(result.direction_key)
    return completed


def attempted_innovation_direction_keys(
    state: AnalysisState,
    lineage_id: Optional[str] = None,
    *,
    after_round: Optional[int] = None,
) -> Set[str]:
    """Return directions that reached a canonical innovation RoundResult.

    Attempted coverage is deliberately broader than completed coverage. Once a
    direction produced a terminal round result (including partial/failed), the
    workflow must not launch unbounded directionless rounds trying to make the
    same batch complete. Reporting records the gap for PI review and proceeds.
    """
    target_lineage = lineage_id or current_lineage_id(state)
    attempted: Set[str] = set()
    for raw in state.round_results:
        result = normalize_legacy_round_result(raw)
        if result.lineage_id != target_lineage:
            continue
        if result.phase != "innovation":
            continue
        if after_round is not None and result.round_number <= after_round:
            continue
        if result.direction_key:
            attempted.add(result.direction_key)
    return attempted


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


def direction_coverage_status_for_batch(
    state: AnalysisState,
    batch: Iterable[dict],
    lineage_id: Optional[str] = None,
) -> DirectionCoverageStatus:
    selected = selected_direction_keys(state, directions=list(batch))
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
        coverage_eligible = result.direction_coverage_eligible
        if coverage_eligible is None:
            coverage_eligible = result.round_status == "completed"
        if not coverage_eligible:
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


def at_integration_gate(state: AnalysisState) -> bool:
    """True when parked at the integration gate awaiting user direction selection."""
    return (
        getattr(state, "analysis_phase", None) == AnalysisPhase.INTEGRATION
        and bool(getattr(state, "integration_gate_candidates", None))
    )


def active_batch_incomplete(state: AnalysisState) -> bool:
    """True when an active direction batch exists and is not yet fully covered."""
    batch = getattr(state, "active_direction_batch", None) or []
    return bool(batch) and not direction_coverage_status_for_batch(state, batch).complete


def active_batch_complete(state: AnalysisState) -> bool:
    """True when an active direction batch exists and every direction is covered."""
    batch = getattr(state, "active_direction_batch", None) or []
    return bool(batch) and direction_coverage_status_for_batch(state, batch).complete


def active_batch_ready_for_integration(state: AnalysisState) -> bool:
    """True once every direction in the active batch has a terminal result.

    Integration must summarize partial attempts and their gaps instead of
    silently skipping the cross-direction stage. Results at or before the last
    integration boundary cannot trigger the same cycle again.
    """
    batch = getattr(state, "active_direction_batch", None) or []
    if not batch:
        return False
    selected = selected_direction_keys(state, directions=batch)
    attempted = attempted_innovation_direction_keys(
        state,
        after_round=int(getattr(state, "last_integrated_round", 0) or 0),
    )
    return bool(selected) and selected.issubset(attempted)


def active_batch_has_unattempted_directions(state: AnalysisState) -> bool:
    """True while the active batch still contains a direction never executed."""
    batch = getattr(state, "active_direction_batch", None) or []
    if not batch:
        return False
    selected = selected_direction_keys(state, directions=batch)
    attempted = attempted_innovation_direction_keys(
        state,
        after_round=int(getattr(state, "last_integrated_round", 0) or 0),
    )
    return bool(selected - attempted)


def awaiting_integration_direction(state: AnalysisState) -> bool:
    """True when an integration-gate auto-skip handed the decision to the PI
    (PI_DECIDE_SIGNAL) but pi_planning has not yet consumed it. Routing must
    bypass the max_rounds stop in this state so the flow can reach pi_planning
    for integration direction selection."""
    from ..constants import PI_DECIDE_SIGNAL
    return (
        getattr(state, "integration_cycle", 0) > 0
        and getattr(state, "user_chosen_direction", None) == PI_DECIDE_SIGNAL
    )
