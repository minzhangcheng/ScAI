"""Host-owned lookup of execution receipts for artifact provenance."""
from __future__ import annotations

from typing import Any, Optional


def lookup_execution_receipt(
    state: Any,
    execution_id: Any,
    *,
    current_receipts: Any = (),
) -> Optional[Any]:
    """Return the exact retained receipt, or None when the ID is absent/unknown."""
    target = str(execution_id or "").strip()
    if not target:
        return None
    for execution in [
        *(current_receipts or []),
        *(getattr(state, "code_history", None) or []),
    ]:
        if str(getattr(execution, "id", "") or "") == target:
            return execution
    return None


def artifact_provenance(
    state: Any,
    source_execution_id: Any,
    *,
    fallback_lineage: str,
    fallback_round: int,
    fallback_plan_id: Optional[str],
    current_receipts: Any = (),
) -> tuple[dict[str, Any], Optional[str]]:
    """Return provenance fields copied from the cited receipt.

    The second element is a bounded technical error when an explicit source
    execution ID does not resolve.  A missing ID is the legacy fallback case.
    """
    explicit = str(source_execution_id or "").strip()
    if not explicit:
        return (
            {
                "lineage_id": fallback_lineage,
                "round_number": int(fallback_round or 0),
                "plan_id": fallback_plan_id,
                "analysis_generation": int(
                    (getattr(state, "sandbox_state", None) or {}).get(
                        "analysis_generation", 0
                    )
                    or 0
                ),
                "cohort_generation": int(
                    (
                        (getattr(state, "sandbox_state", None) or {}).get(
                            "cohort_identity_v1", {}
                        )
                        or {}
                    ).get("cohort_generation", 0)
                    or 0
                ),
                "cohort_digest": (
                    str(
                        (
                            (getattr(state, "sandbox_state", None) or {}).get(
                                "cohort_identity_v1", {}
                            )
                            or {}
                        ).get("cohort_digest", "")
                        or ""
                    )
                    or None
                ),
                "legacy_provenance": True,
            },
            None,
        )
    receipt = lookup_execution_receipt(
        state, explicit, current_receipts=current_receipts
    )
    if receipt is None:
        return {}, "cited source_execution_id does not resolve to a retained receipt"
    from ...services.execution_evidence import execution_is_current_receipt

    receipt_lineage = getattr(receipt, "lineage_id", None) or "main"
    if not execution_is_current_receipt(
        receipt, lineage_id=receipt_lineage,
    ):
        return {}, "cited source_execution_id is not a successful evidence receipt"
    return (
        {
            "lineage_id": receipt_lineage,
            "round_number": int(getattr(receipt, "round_number", 0) or 0),
            "plan_id": str(getattr(receipt, "plan_id", "") or "") or None,
            "analysis_generation": int(
                getattr(receipt, "analysis_generation", 0) or 0
            ),
            "cohort_generation": int(
                getattr(receipt, "cohort_generation", 0) or 0
            ),
            "cohort_digest": str(getattr(receipt, "cohort_digest", "") or "") or None,
            "legacy_provenance": False,
        },
        None,
    )
