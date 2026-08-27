"""Invariant helpers for score and mechanism registration."""
from __future__ import annotations


def eligible_execution_ids(code_history, *, round_number: int, successful_executions):
    return {
        execution.id
        for execution in successful_executions(
            code_history, round_number=round_number
        )
    }


def mechanism_execution_id_error(programs, eligible_ids: set[str]) -> str | None:
    invalid_ids = sorted(
        {
            evidence_id
            for program in programs
            for evidence_id in program.evidence_execution_ids
            if evidence_id not in eligible_ids
        }
    )
    if not invalid_ids:
        return None
    return (
        "Mechanism programs cite failed, contradictory, or unknown execution IDs: "
        f"{invalid_ids}. Available IDs: {sorted(eligible_ids)[-20:]}"
    )
