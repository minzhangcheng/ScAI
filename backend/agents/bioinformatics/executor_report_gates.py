"""Statistical and evidence-gate input construction."""
from __future__ import annotations

from typing import Any, Dict, List


def gate_score_definitions(state, *, run_gates) -> List[Any]:
    """Keep current/pre-tracking scores and older evidence-valid records."""
    carried: List[Any] = []
    for score in (state.score_definitions or []):
        if getattr(score, "round_number", 0) in (0, state.current_round):
            carried.append(score)
            continue
        if run_gates(
            None,
            score_definitions=[score],
            scoring_plans=state.scoring_plans,
        ).passed:
            carried.append(score)
    return carried


def gate_samples_per_group(state) -> Dict[str, int]:
    numbered: Dict[Any, set] = {}
    for data_file in (state.data_files or []):
        if getattr(data_file, "sample_number", None) is None:
            continue
        numbered.setdefault(data_file.group_id, set()).add(data_file.sample_number)

    detected: Dict[Any, int] = {}
    for record in (getattr(state, "detected_sample_counts", None) or []):
        group_id = getattr(record, "group_id", None)
        count = int(getattr(record, "detected_count", 0) or 0)
        if group_id is None or count < 1:
            continue
        prior = detected.get(group_id)
        detected[group_id] = count if prior is None else min(prior, count)

    return {
        group.name: (
            detected[group.id]
            if group.id in detected
            else max(group.sample_count, len(numbered.get(group.id, ())))
        )
        for group in (state.treatment_groups or [])
        if getattr(group, "sample_count", None) is not None
    }


def evidence_gate_record(
    *,
    gate: str,
    detail: str,
    severity: str,
    rule_id: str = "",
    subject: str = "",
    rejection_count: int = 0,
    rejection_limit: int = 0,
    forced_release: bool = False,
):
    return {
        "gate": str(gate),
        "rule_id": str(rule_id),
        "detail": str(detail),
        "subject": str(subject),
        "severity": str(severity),
        "rejection_count": max(0, int(rejection_count)),
        "rejection_limit": max(0, int(rejection_limit)),
        "forced_release": bool(forced_release),
    }
