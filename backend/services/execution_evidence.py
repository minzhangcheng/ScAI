"""Execution guidance and objective receipt eligibility."""
from __future__ import annotations

import re
from typing import Any, Optional


PLAN_LINKED_RECEIPT_STATUSES = frozenset({"candidate", "confirmed", "bound"})


GENE_INDEX_ALIGNMENT_GUIDANCE = (
    "After mapping feature IDs to gene symbols, first check index.is_unique. "
    "Collapse duplicate raw-count rows with groupby(level=0).sum(); for already "
    "normalized/log-scale values use a scientifically justified aggregation "
    "such as mean. Align matrices explicitly with DataFrame.align(join='inner', "
    "axis=0), then assert that both indexes are unique and equal before building "
    "or applying a positional boolean mask. Do not use set intersection plus "
    ".loc on an index that may contain duplicate symbols."
)

LIANA_RUNTIME_GUIDANCE = (
    "For installed LIANA 1.8.x, use `import liana as li` and call "
    "`li.mt.rank_aggregate(adata, groupby='<annotation column>', "
    "resource_name='<resource>', key_added='liana_res', n_jobs=1)`. Use "
    "`mouseconsensus` for mouse and `consensus` for human unless an approved "
    "plan names a validated custom resource. Do not call legacy/nonexistent "
    "`singlecellpipe` or `run_liana`; inspect "
    "`liana.method.rank_aggregate` before changing the API."
)


_FAILED_EVIDENCE_OUTPUT_RE = re.compile(
    r"(?:\bvalidation\s+failed\b|\bcritical\s*:\s*[^\n]*(?:failed|no\s+\w+\s+found)|"
    r"\bmatched\s+in\s+bulk\s*:\s*\[\s*\]|\bexecution\s+failed\b)",
    re.IGNORECASE,
)


def execution_supports_evidence(execution: Any) -> bool:
    """Return whether an execution contains technically citable output."""
    if execution is None:
        return False
    if not str(getattr(execution, "code", "") or "").strip():
        return False
    if getattr(execution, "error", None):
        return False
    if not bool(getattr(execution, "evidence_eligible", True)):
        return False
    return not _FAILED_EVIDENCE_OUTPUT_RE.search(
        str(getattr(execution, "output", "") or "")
    )


def execution_is_current_receipt(
    execution: Any,
    *,
    round_number: Optional[int] = None,
    plan_id: Optional[str] = None,
    lineage_id: Optional[str] = None,
    analysis_generation: Optional[int] = None,
    cohort_digest: Optional[str] = None,
    cohort_generation: Optional[int] = None,
) -> bool:
    """Validate only Host-owned execution lifecycle and identity fields."""
    if getattr(execution, "evidence_schema_version", None) != 2:
        return False
    if getattr(execution, "execution_status", "succeeded") != "succeeded":
        return False
    if not execution_supports_evidence(execution):
        return False
    if round_number is not None and int(
        getattr(execution, "round_number", 0) or 0
    ) != int(round_number):
        return False
    if plan_id is not None and str(
        getattr(execution, "plan_id", "") or ""
    ) != str(plan_id):
        return False
    if lineage_id is not None and (
        getattr(execution, "lineage_id", None) or "main"
    ) != lineage_id:
        return False
    if analysis_generation is not None and int(
        getattr(execution, "analysis_generation", 0) or 0
    ) != int(analysis_generation):
        return False
    if cohort_digest is not None and str(
        getattr(execution, "cohort_digest", "") or ""
    ) != str(cohort_digest):
        return False
    if cohort_generation is not None and int(
        getattr(execution, "cohort_generation", 0) or 0
    ) != int(cohort_generation):
        return False
    return True
