"""Which decision a snapshot precedes, and which snapshot to delete first.

Part B spec 3.3 and 4.7.

``round_boundary_gate`` is the single owner of the predicates
``_wait_for_user_input`` used to spell inline. The node calls it and derives its
two booleans from the answer, so the writer and the node cannot classify the
same boundary differently -- which they would eventually, since the node's
version was two separate expressions forty lines apart.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from backend.constants import (
    GATE_ANNOTATION_CONFIRMATION,
    GATE_DIRECTION_SELECTION,
    GATE_INTEGRATION_SELECTION,
    GATE_ROUND_WAIT,
    SEMANTIC_GATE_ACTION_NEXT_NODES,
)

#: What a restored session's status has to be for the gate it is parked on.
#: The annotation gate is the only one the workflow parks on
#: ``WAITING_ANNOTATION``; the other three are round-boundary waits.
_RESTORED_STATUS: dict[str, str] = {
    GATE_ANNOTATION_CONFIRMATION: "waiting_annotation",
    GATE_DIRECTION_SELECTION: "waiting_input",
    GATE_INTEGRATION_SELECTION: "waiting_input",
    GATE_ROUND_WAIT: "waiting_input",
}

#: Deletion priority, decided by the project owner on 2026-09-08 and recorded
#: in each manifest at capture. Rank 1 is the most valuable and is deleted
#: last. The rank follows the *decision*, not the round: the annotation
#: decision is the earliest one an analysis makes and everything downstream
#: depends on its labels, so it keeps rank 1 even when a re-run reaches it in a
#: later round.
_RETENTION_RANK: dict[str, int] = {
    GATE_ANNOTATION_CONFIRMATION: 1,
    GATE_DIRECTION_SELECTION: 2,
    GATE_INTEGRATION_SELECTION: 3,
    GATE_ROUND_WAIT: 4,
}

DECISION_GATES: frozenset[str] = frozenset(_RETENTION_RANK)


def round_boundary_gate(analysis_state: Any, workflow_state: Mapping | None = None) -> str:
    """The decision ``_wait_for_user_input`` is about to present.

    The three tests are the node's own, in the node's order:

    * an integration gate is either declared by the workflow state or implied
      by the phase plus the candidates the integration node published;
    * round 1 in the innovation phase is the post-Stage-2 direction gate;
    * everything else is the ordinary round-acceptance wait.

    ``at_integration_gate`` is imported lazily because it lives in the workflow
    package: a service module that imported it at module scope would invert the
    layering and give ``phases.py`` a cycle to walk back through.
    """
    from backend.models.schemas import AnalysisPhase
    from backend.workflow.direction_coverage import at_integration_gate

    declared = (workflow_state or {}).get("semantic_gate")
    if declared == GATE_INTEGRATION_SELECTION or at_integration_gate(analysis_state):
        return GATE_INTEGRATION_SELECTION
    if (
        getattr(analysis_state, "current_round", 0) == 1
        and getattr(analysis_state, "analysis_phase", None) == AnalysisPhase.INNOVATION
    ):
        return GATE_DIRECTION_SELECTION
    return GATE_ROUND_WAIT


def next_node_for(gate: str) -> str:
    """The node a restored session continues into once the decision is made."""
    node = SEMANTIC_GATE_ACTION_NEXT_NODES.get(str(gate))
    if node is None:
        raise ValueError(f"not a decision gate: {gate!r}")
    return node


def restored_status_for(gate: str) -> str:
    """The status a session restored from this gate opens in."""
    status = _RESTORED_STATUS.get(str(gate))
    if status is None:
        raise ValueError(f"not a decision gate: {gate!r}")
    return status


def retention_rank(gate: str) -> int:
    """How valuable this decision's snapshot is: 1 keeps longest, 4 goes first."""
    rank = _RETENTION_RANK.get(str(gate))
    if rank is None:
        raise ValueError(f"not a decision gate: {gate!r}")
    return rank


def _epoch(value: Any) -> float:
    """Seconds since the epoch for an ISO timestamp, or 0.0 when unreadable.

    Zero is the conservative answer: within a rank the *later* snapshot is
    deleted first, so a record whose timestamp cannot be read sorts as the
    oldest and is deleted last rather than first.
    """
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return 0.0


def deletion_sort_key(record: Mapping[str, Any]) -> tuple:
    """Order records so that the one to delete first compares smallest.

    Rank descending (4 before 1), then later before earlier, with the id as a
    final tiebreak so two records taken in the same second still order
    deterministically across processes.
    """
    return (
        -int(record.get("retention_rank") or 0),
        -_epoch(record.get("created_at")),
        -int(record.get("round") or 0),
        -int(record.get("runtime_revision") or 0),
        str(record.get("snapshot_id") or ""),
    )


def in_deletion_order(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The same records, ordered by which to delete first."""
    return sorted(records, key=deletion_sort_key)


def with_deletion_order(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Annotate each record with its 1-based position in the deletion order.

    Computed here rather than in the frontend, the CLI and the prune
    separately: three consumers deriving the same order from the same rule is
    three chances for them to disagree about which snapshot goes first.
    """
    ordered = in_deletion_order(records)
    positions = {
        str(record.get("snapshot_id")): index
        for index, record in enumerate(ordered, start=1)
    }
    return [
        {**record, "deletion_order": positions[str(record.get("snapshot_id"))]}
        for record in records
    ]


def mark_superseded(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flag every snapshot that a later capture of the same decision replaced.

    A re-run or a supplementary analysis reaches the same round and gate again
    and produces a second snapshot at a higher revision. Both stay restorable;
    the earlier one is only marked, because deleting it automatically would
    discard the state the user is most likely trying to get back to.
    """
    newest: dict[tuple[int, str], int] = {}
    for record in records:
        key = (int(record.get("round") or 0), str(record.get("gate") or ""))
        revision = int(record.get("runtime_revision") or 0)
        if revision > newest.get(key, -1):
            newest[key] = revision
    result = []
    for record in records:
        key = (int(record.get("round") or 0), str(record.get("gate") or ""))
        revision = int(record.get("runtime_revision") or 0)
        result.append({**record, "superseded": revision < newest[key]})
    return result
