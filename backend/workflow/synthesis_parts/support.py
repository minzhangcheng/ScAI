"""PI synthesis, user input wait, and convergence check."""
import asyncio
import json
import logging
import math
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...constants import (
    CONTINUE_SIGNAL,
    GATE_DIRECTION_SELECTION,
    GATE_INTEGRATION_SELECTION,
    GATE_ROUND_WAIT,
    LlmFailureAction,
    PI_DECIDE_SIGNAL,
    StateSyncChangeType,
    WaitInputReason,
)
from ...models.schemas import (
    AgentType,
    AnalysisPhase,
    AnalysisState,
    PathologyCoverageRecord,
    RoundResult,
    TaskStatus,
)
from ...services.interaction_record_capture import get_or_open_direction_record
from ...utils.direction_evidence import build_fallback_direction_evidence
from ...utils.llm_client import UpstreamExhaustedError
from ..direction_coverage import (
    at_integration_gate,
    build_result_identity_key,
    count_innovation_rounds_done,
    current_lineage_id,
    direction_coverage_status,
    has_substantive_findings,
    normalize_direction_key,
    normalize_legacy_round_result,
)
from ..helpers import wait_with_skip_time
from ..provenance import consume_restore_provenance, restore_provenance_id

if TYPE_CHECKING:
    from ..state import WorkflowState
else:
    WorkflowState = dict

logger = logging.getLogger(__name__)


def _bounded_coverage_json(value: Any, *, depth: int = 0) -> Any:
    """Convert one diagnostic value to a small JSON-safe representation."""
    if value is None or isinstance(value, (bool, int, str)):
        return value if not isinstance(value, str) else value[:1000]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if depth >= 2:
        return str(value)[:1000]
    if isinstance(value, Mapping):
        return {
            str(key)[:100]: _bounded_coverage_json(item, depth=depth + 1)
            for key, item in list(value.items())[:20]
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _bounded_coverage_json(item, depth=depth + 1)
            for item in list(value)[:20]
        ]
    return str(value)[:1000]


def extract_pathology_coverage(
    pathology_result: Any,
) -> List[PathologyCoverageRecord]:
    """Retain bounded per-image tile validation from a pathology result.

    This is a host observation: malformed or absent coverage is represented as
    an empty/errored record and never changes round completion policy.
    """
    if not isinstance(pathology_result, dict):
        return []
    raw_details = pathology_result.get("detailed_results")
    if not isinstance(raw_details, list):
        return []

    def optional_nonnegative_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            return max(0, int(value))
        except (TypeError, ValueError, OverflowError):
            return None

    coverage: List[PathologyCoverageRecord] = []
    for raw in raw_details[:100]:
        if not isinstance(raw, dict):
            continue
        validation = raw.get("tile_validation")
        validation = validation if isinstance(validation, dict) else {}
        expected = optional_nonnegative_int(validation.get("expected_tiles"))
        valid = optional_nonnegative_int(validation.get("valid_tiles"))

        ratio: Optional[float]
        raw_ratio = validation.get("coverage_ratio")
        try:
            ratio = float(raw_ratio) if raw_ratio is not None else None
        except (TypeError, ValueError, OverflowError):
            ratio = None
        if ratio is None or not math.isfinite(ratio):
            ratio = (
                valid / expected
                if expected and valid is not None
                else 0.0 if expected == 0 and valid is not None
                else None
            )
        if ratio is not None:
            ratio = max(0.0, min(1.0, ratio))

        status_raw = (
            validation.get("status")
            or raw.get("status")
            or (
                "skipped" if raw.get("skipped")
                else "failed" if raw.get("error")
                else "completed"
            )
        )
        status = str(status_raw).strip().lower()
        if status not in {"completed", "partial", "error", "failed", "skipped"}:
            status = "error"

        raw_batches = validation.get("invalid_batches")
        invalid_batches: List[Dict[str, Any]] = []
        if isinstance(raw_batches, (list, tuple)):
            for item in list(raw_batches)[:20]:
                if isinstance(item, Mapping):
                    invalid_batches.append({
                        str(key)[:100]: _bounded_coverage_json(value)
                        for key, value in list(item.items())[:20]
                    })

        coverage.append(PathologyCoverageRecord(
            image_id=str(raw.get("image_id") or "")[:200],
            image_type=str(raw.get("image_type") or "")[:200],
            status=status,
            expected_tiles=expected,
            valid_tiles=valid,
            coverage_ratio=ratio,
            invalid_batches=invalid_batches,
            error=str(raw.get("error") or validation.get("error") or "")[:1000],
        ))
    return coverage


def build_disabled_pathology_warning(files: list[dict], pathology_enabled: bool) -> str:
    """Return a modality-boundary warning when image files exist but pathology is disabled."""
    if pathology_enabled:
        return ""
    # Only genuine pathology-modality inputs matter here. Images belonging to
    # other modalities (e.g. a biochemistry panel PNG) or generic PI-chat
    # uploads are not pathology data and must not trigger this warning. The
    # canonical modality signal is the section: data_files carry it under
    # "section", section-based pending uploads under "type".
    has_pathology_input = any(
        (f.get("section") or f.get("type") or "").lower() == "pathology"
        for f in files
    )
    if not has_pathology_input:
        return ""
    return (
        "Pathology image inputs are present while the pathology agent is disabled. "
        "These images will be treated as non-pathology descriptive evidence unless pathology analysis is enabled."
    )


def classify_exploration_failure(bio) -> str:
    """Map the internal bioinformatics result to a STABLE reason code for the UI.

    Never leak raw internal status strings (e.g. 'max_steps_reached',
    'completed_with_fallback') to the frontend — the dialog localizes off this
    code. Returns one of: 'analysis_error', 'did_not_converge'."""
    if isinstance(bio, dict) and bio.get("error"):
        return "analysis_error"
    return "did_not_converge"


FAILED_AGENT_STATUSES = frozenset({
    "failed",
    "error",
    "timeout",
    "max_steps_reached",
})

SKIPPED_AGENT_STATUSES = frozenset({
    "skipped",
    "not_run",
})

PARTIAL_AGENT_STATUSES = frozenset({
    "partial",
    "completed_with_fallback",
})

CORE_ROUND_AGENT_RESULT_KEYS = (
    "bioinformatics_result",
    "literature_result",
)

OPTIONAL_ROUND_AGENT_RESULT_KEYS = (
    "pathology_result",
    "biochemistry_result",
)

CORE_FAILED_NODES = frozenset({
    "bioinformatics_analysis",
    "literature_research",
})


def _agent_result_status(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    if result.get("error"):
        return "failed"
    return str(result.get("status") or "").strip().lower()


def _statuses_for_keys(
    state: Dict[str, Any],
    keys: tuple[str, ...],
    *,
    missing_status: str = "",
) -> list[str]:
    statuses = []
    for key in keys:
        result = state.get(key)
        if isinstance(result, dict) and result:
            statuses.append(_agent_result_status(result))
        elif missing_status:
            statuses.append(missing_status)
    return statuses


def derive_round_completion_status(state: Dict[str, Any], result: RoundResult) -> RoundResult:
    """Populate RoundResult completion fields from actual agent execution state.

    Core bioinformatics/literature failures block direction completion. Optional
    pathology/biochemistry skipped/no-data results do not block completion when
    core evidence exists, but optional hard failures downgrade the round to
    partial so the report can disclose degraded modality coverage.
    """
    failed_node = state.get("failed_node")
    # This assignment is the provenance boundary for report-quality consumers.
    # Older persisted RoundResult objects contain the same field defaults but no
    # evidence that those defaults were ever derived from node execution.
    result.evidence_schema_version = 1

    # Integration is a first-class evidence-producing phase, but it deliberately
    # does not execute the ordinary bioinformatics/literature nodes.  Judge it by
    # its own output instead of manufacturing two ``not_run`` core-agent states.
    if result.phase == "integration":
        has_integration_evidence = bool(
            (result.integration_report_summary or "").strip()
            or result.integration_candidates
        )
        if failed_node == "integration_synthesis":
            result.round_status = "partial" if has_integration_evidence else "failed"
            result.evidence_status = (
                "substantive" if has_integration_evidence else "failed"
            )
            result.direction_coverage_eligible = None
            return result
        result.round_status = "completed" if has_integration_evidence else "partial"
        result.evidence_status = (
            "substantive" if has_integration_evidence else "empty"
        )
        # Direction coverage is an innovation-batch concept. Integration has a
        # valid completion/evidence verdict of its own but is not applicable to
        # that ledger.
        result.direction_coverage_eligible = None
        return result

    has_evidence = has_substantive_findings(result)
    has_core_evidence = bool(
        result.bioinformatics_findings
        or result.literature_findings
        or result.data_driven_mechanism_programs
    )

    if failed_node in CORE_FAILED_NODES:
        result.round_status = "partial" if has_evidence else "skipped"
        result.evidence_status = "substantive" if has_evidence else "failed"
        result.direction_coverage_eligible = False
        return result

    # Core agents are required for a completed research round. During
    # stop-and-summarize, the graph snapshot legitimately lacks results for
    # agents that had not run yet; absence is therefore an explicit not_run,
    # never evidence that the round completed successfully.
    core_statuses = _statuses_for_keys(
        state,
        CORE_ROUND_AGENT_RESULT_KEYS,
        missing_status="not_run",
    )
    optional_statuses = _statuses_for_keys(state, OPTIONAL_ROUND_AGENT_RESULT_KEYS)

    has_core_failed = any(status in FAILED_AGENT_STATUSES for status in core_statuses)
    has_core_skipped = any(status in SKIPPED_AGENT_STATUSES for status in core_statuses)
    has_core_partial = any(status in PARTIAL_AGENT_STATUSES for status in core_statuses)
    has_optional_failed = any(status in FAILED_AGENT_STATUSES for status in optional_statuses)
    has_optional_partial = any(status in PARTIAL_AGENT_STATUSES for status in optional_statuses)

    if has_core_failed:
        result.round_status = "partial" if has_evidence else "failed"
        result.evidence_status = "substantive" if has_evidence else "failed"
        result.direction_coverage_eligible = False
        return result

    if has_core_skipped:
        result.round_status = "partial" if has_evidence else "skipped"
        result.evidence_status = "substantive" if has_evidence else "failed"
        result.direction_coverage_eligible = False
        return result

    if has_core_partial:
        result.round_status = "partial"
        result.evidence_status = "substantive" if has_evidence else "empty"
        result.direction_coverage_eligible = False
        return result

    if has_optional_failed or has_optional_partial:
        result.round_status = "partial"
        result.evidence_status = "substantive" if has_evidence else "empty"
        # Optional modality degradation must remain visible in round_status,
        # but it must not keep a direction batch permanently incomplete when
        # both core agents completed and substantive evidence exists.
        result.direction_coverage_eligible = has_core_evidence
        return result

    result.round_status = "completed"
    result.evidence_status = "substantive" if has_evidence else "empty"
    result.direction_coverage_eligible = has_evidence
    return result


def _upsert_round_result(
    results: list[RoundResult],
    result: RoundResult,
    *,
    lineage_id: str,
    branch_id: Optional[str],
    branch_label: Optional[str],
    restored_from_snapshot_id: Optional[str],
) -> None:
    # Keep the caller's object synchronized with the canonical record. The
    # normalizer returns a deep copy, but synthesis/integration continue to use
    # their local ``RoundResult`` when building the following websocket payload.
    # Provenance must therefore be visible on both objects, not only in the
    # list entry.
    caller_result = result
    result.lineage_id = lineage_id or "main"
    result.branch_id = branch_id
    result.branch_label = branch_label
    result.direction_key = normalize_direction_key(result.direction)
    result.result_identity_key = build_result_identity_key(
        result.lineage_id,
        result.phase,
        result.round_number,
        result.direction_key,
    )
    result = normalize_legacy_round_result(result)

    # Restore provenance is immutable lineage for one logical result identity.
    # A retry/review replacement of that identity may update its content, but
    # must not erase the origin the first successful writer established.
    existing_origin = None
    other_origins: set[str] = set()
    normalized: list[RoundResult] = []
    identity_indexes: dict[str, int] = {}
    for raw in results:
        existing = normalize_legacy_round_result(raw)
        key = existing.result_identity_key
        if (
            key == result.result_identity_key
            and isinstance(existing.restored_from_snapshot_id, str)
            and existing.restored_from_snapshot_id
        ):
            existing_origin = existing.restored_from_snapshot_id
        elif (
            isinstance(existing.restored_from_snapshot_id, str)
            and existing.restored_from_snapshot_id
        ):
            other_origins.add(existing.restored_from_snapshot_id)
        if key in identity_indexes:
            # A retry written before identity-aware upsert may have appended a
            # second legacy record. Keep the later retained value in the first
            # stable slot so API/WS ordering does not jump on migration.
            normalized[identity_indexes[key]] = existing
            continue
        identity_indexes[key] = len(normalized)
        normalized.append(existing)
    provenance = restored_from_snapshot_id or existing_origin or None
    # A stale caller may carry the one-shot origin after another logical result
    # has already claimed it. Do not move that origin to this identity (or make
    # two canonical results claim the same snapshot); the established target
    # lineage, when present, remains authoritative.
    if (
        isinstance(restored_from_snapshot_id, str)
        and restored_from_snapshot_id
        and restored_from_snapshot_id in other_origins
        and restored_from_snapshot_id != existing_origin
    ):
        provenance = existing_origin or None
    result.restored_from_snapshot_id = provenance
    caller_result.restored_from_snapshot_id = provenance
    target_index = identity_indexes.get(result.result_identity_key)
    if target_index is None:
        normalized.append(result)
    else:
        normalized[target_index] = result
    results[:] = normalized


def _should_force_next_round(analysis_state: AnalysisState) -> bool:
    """Return True when workflow must honor pending work instead of stopping on convergence.

    Priority order:
    1. Pending integration round
    2. Real human text input (not synthetic continue markers)
    3. Newly uploaded files
    4. Remaining multi-direction queue entries, except for the stale final
       queue entry left after completed multi-direction analysis
    """
    if getattr(analysis_state, "pending_integration", False):
        return True
    has_real_text = bool(
        analysis_state.pending_human_input
        and analysis_state.pending_human_input.strip() not in ("", CONTINUE_SIGNAL, PI_DECIDE_SIGNAL)
    )
    has_uploaded_files = bool(analysis_state.pending_uploaded_files)
    has_remaining_directions = bool(
        analysis_state.directions_queue
        and any(d.get("direction", "").strip() for d in analysis_state.directions_queue)
    )
    if has_remaining_directions and _multi_direction_analysis_completed(analysis_state):
        has_remaining_directions = False
    return has_real_text or has_uploaded_files or has_remaining_directions


def _consume_signal_then_clear(event: asyncio.Event) -> None:
    event.clear()


def _multi_direction_analysis_completed(analysis_state) -> bool:
    """True once every direction in the active batch has been attempted."""
    from ..direction_coverage import active_batch_ready_for_integration
    return active_batch_ready_for_integration(analysis_state)



def _should_build_innovation_segment(analysis_state, round_result) -> bool:
    """Whether to build a detailed-report segment for the just-completed round.

    Keys off round_result.phase (captured before the exploration→innovation
    transition) rather than analysis_state.analysis_phase, which is already
    flipped to INNOVATION at the end of the exploration round and would
    otherwise spawn a spurious 'innovation_round_1' segment for round 1.

    Deduplicates by phase to stay idempotent across node retries.
    """
    if round_result.phase != "innovation":
        return False
    phase = f"innovation_round_{analysis_state.current_round}"
    return not any(
        s.phase == phase for s in analysis_state.detailed_report_segments
    )


def _direction_result_is_coverage_eligible(result: RoundResult) -> bool:
    """Apply the direction ledger's compatibility rule to legacy results."""
    if result.direction_coverage_eligible is not None:
        return result.direction_coverage_eligible
    return (
        result.round_status == "completed"
        and result.evidence_status == "substantive"
        and has_substantive_findings(result)
    )


def _direction_result_evidence_gaps(
    result: Optional[RoundResult],
) -> list[str]:
    """Build bounded, evidence-derived gaps for one integration input."""
    if result is None:
        return ["No canonical innovation-round result was recorded."]

    gaps: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in gaps and len(gaps) < 12:
            gaps.append(text[:500])

    if result.round_status != "completed":
        add(f"The direction attempt ended with round status '{result.round_status}'.")
    if result.evidence_status != "substantive":
        add(f"The retained evidence status is '{result.evidence_status}'.")
    if not _direction_result_is_coverage_eligible(result):
        add("The attempt did not satisfy direction-coverage eligibility.")
    if result.bioinformatics_evidence_verified is False:
        add(
            "One or more bioinformatics findings lack a current successful "
            "execution receipt."
        )

    for program in result.data_driven_mechanism_programs or []:
        missing = (
            program.get("missing_evidence", [])
            if isinstance(program, dict)
            else getattr(program, "missing_evidence", [])
        )
        if not isinstance(missing, (list, tuple, set)):
            missing = [missing]
        for item in missing or []:
            add(f"Mechanism evidence gap: {item}")
    for note in result.evidence_notes or []:
        add(note)
    return gaps


def _build_directions_with_results_for_correlation(analysis_state: AnalysisState) -> list[dict]:
    """Build integration inputs from every attempted active-batch direction."""
    target_lineage = current_lineage_id(analysis_state)
    selected_directions = (
        analysis_state.active_direction_batch
        or analysis_state.user_chosen_directions
        or []
    )
    selected_keys = {
        normalize_direction_key(item.get("direction", ""))
        for item in selected_directions
        if normalize_direction_key(item.get("direction", ""))
    }
    latest_results: Dict[str, RoundResult] = {}
    for raw in analysis_state.round_results:
        result = normalize_legacy_round_result(raw)
        if (
            result.lineage_id != target_lineage
            or result.phase != "innovation"
            or result.direction_key not in selected_keys
        ):
            continue
        previous = latest_results.get(result.direction_key)
        if previous is None or result.round_number >= previous.round_number:
            latest_results[result.direction_key] = result

    directions_with_results: list[dict] = []
    seen: set[str] = set()
    for chosen_dir in selected_directions:
        key = normalize_direction_key(chosen_dir.get("direction", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        result = latest_results.get(key)
        finding_groups = (
            result.pathology_findings,
            result.bioinformatics_findings,
            result.biochemistry_findings,
            result.literature_findings,
        ) if result is not None else ()
        available_findings = [
            str(finding)
            for group in finding_groups
            for finding in group
            if str(finding).strip()
        ][:20]
        evidence_gaps = _direction_result_evidence_gaps(result)
        if result is not None:
            for review in analysis_state.post_execution_reviews or []:
                if (
                    not isinstance(review, dict)
                    or int(review.get("round_number") or 0) != result.round_number
                ):
                    continue
                remaining = (review.get("assessment") or {}).get("remaining_gaps") or []
                if not isinstance(remaining, (list, tuple, set)):
                    remaining = [remaining]
                for gap in remaining:
                    text = str(gap or "").strip()
                    if text and text not in evidence_gaps and len(evidence_gaps) < 12:
                        evidence_gaps.append(f"Post-execution gap: {text}"[:500])
        directions_with_results.append({
            "direction": chosen_dir.get("direction", ""),
            "notes": chosen_dir.get("notes", ""),
            "round_number": result.round_number if result else None,
            "round_summary": result.pi_summary if result else "",
            "round_status": result.round_status if result else "not_attempted",
            "evidence_status": result.evidence_status if result else "empty",
            "direction_coverage_eligible": (
                _direction_result_is_coverage_eligible(result) if result else False
            ),
            "available_findings": available_findings,
            "evidence_gaps": evidence_gaps,
        })
    return directions_with_results


def _integration_inputs_are_partial(directions_with_results: list[dict]) -> bool:
    """Whether source evidence requires a partial integration verdict."""
    if not directions_with_results:
        return True
    return any(
        item.get("round_status") != "completed"
        or item.get("evidence_status") != "substantive"
        or item.get("direction_coverage_eligible") is not True
        for item in directions_with_results
    )


def build_round_summary_payload(
    round_result,
    analysis_state,
    effective_max_rounds: int,
    data_requests: dict | None = None,
    cross_modal_candidates: list | None = None,
    *,
    status: str = "waiting_input",
    wait_policy: str = "manual",
) -> dict:
    """Build the round_summary WebSocket payload from a RoundResult.

    Shared between _pi_synthesis (exploration/innovation) and
    _integration_synthesis (integration) so the shape stays consistent.
    """
    return {
        "round": round_result.round_number,
        "max_rounds": effective_max_rounds,
        "summary": round_result.pi_summary,
        "pathology_findings": round_result.pathology_findings,
        "pathology_coverage": [
            item.model_dump(mode="json")
            if hasattr(item, "model_dump") else dict(item)
            for item in (round_result.pathology_coverage or [])
        ],
        "bioinformatics_findings": round_result.bioinformatics_findings,
        "biochemistry_findings": round_result.biochemistry_findings,
        "literature_findings": round_result.literature_findings,
        "findings": [
            *round_result.pathology_findings,
            *round_result.bioinformatics_findings,
            *round_result.biochemistry_findings,
            *round_result.literature_findings,
        ],
        "cross_modal_candidates": cross_modal_candidates or [],
        "data_requests": data_requests or {"pathology": [], "biochemistry": [], "other": []},
        "status": status,
        "wait_policy": wait_policy,
        "phase": round_result.phase,
        "integration_candidates": round_result.integration_candidates,
        "integration_report_summary": round_result.integration_report_summary,
        "round_status": round_result.round_status,
        "evidence_status": round_result.evidence_status,
        "direction_coverage_eligible": round_result.direction_coverage_eligible,
        "bioinformatics_evidence_verified": round_result.bioinformatics_evidence_verified,
        "evidence_gate_degraded": round_result.evidence_gate_degraded,
        "evidence_execution_ids": list(round_result.evidence_execution_ids or []),
        "evidence_notes": list(round_result.evidence_notes or []),
        "evidence_gate_findings": list(round_result.evidence_gate_findings or []),
        "lineage_id": round_result.lineage_id,
        "branch_id": round_result.branch_id,
        "branch_label": round_result.branch_label,
        "restored_from_snapshot_id": round_result.restored_from_snapshot_id,
        "direction": round_result.direction,
        "direction_key": round_result.direction_key,
        "result_identity_key": round_result.result_identity_key,
        "evidence_schema_version": round_result.evidence_schema_version,
    }
