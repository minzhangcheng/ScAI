"""Pure, sandbox-free helpers for the annotation validation round.

Deterministic logic (idempotency id, trigger gating, main-object invariant
comparison, result sanitizing) lives here so it is unit-testable without an LLM
or a live Jupyter kernel. Agent-coupled logic lives on the agents; orchestration
lives in AnnotationMixin.
"""
import hashlib
import json
from pathlib import Path as _Path
from typing import Any, Dict, List, Tuple


def compute_validation_attempt_id(
    current_round: int,
    tissue: str,
    organism: str,
    cluster_key: str,
    cluster_markers: Dict[str, Any],
) -> str:
    """Deterministic id for one pending annotation attempt.

    Stable across websocket reconnect / snapshot restore / callback re-entry for
    the same annotation; changes when round, tissue, organism, cluster key, or
    marker content changes. Marker order is normalized.
    """
    normalized = {
        str(cid): sorted(str(g) for g in (genes or []))
        for cid, genes in sorted((cluster_markers or {}).items(), key=lambda kv: str(kv[0]))
    }
    payload = json.dumps(
        {
            "round": int(current_round),
            "tissue": (tissue or "").strip().lower(),
            "organism": (organism or "").strip().lower(),
            "cluster_key": cluster_key or "",
            "markers": normalized,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_UNCERTAINTY_KEYWORDS = (
    "quality", "ambient", "doublet", "damaged", "mixed", "cycling",
    "contamination", "technical", "qc", "state", "stress", "ambiguous", "lineage",
)
# Triage bucket keys that are NOT risk signals (populated on normal runs).
# Substrings chosen to avoid matching "unresolved" (which contains "resolved").
_NON_RISK_TRIAGE_KEYS = ("covered", "matched", "supported", "hits", "known", "confident")


def should_run_validation(
    annotations: Dict[str, Any],
    triage: Dict[str, Any],
) -> bool:
    """Decide whether the validation executor should run for this annotation.

    Triggers if any non-empty triage RISK bucket exists (excluding non-risk
    buckets like `covered`), or any annotation is < high confidence, has
    alternatives, or has an uncertainty keyword in its label / low-confidence
    reason.
    """
    for key, value in (triage or {}).items():
        if not value:
            continue
        kl = str(key).lower()
        if any(nr in kl for nr in _NON_RISK_TRIAGE_KEYS):
            continue
        return True

    for info in (annotations or {}).values():
        if not isinstance(info, dict):
            continue
        if str(info.get("confidence", "")).strip().lower() in ("low", "medium"):
            return True
        if info.get("alternative_cell_types"):
            return True
        haystack = f"{info.get('cell_type', '')} {info.get('low_confidence_reason', '')}".lower()
        if any(kw in haystack for kw in _UNCERTAINTY_KEYWORDS):
            return True

    return False


MAIN_OBJECT_SIGNATURE_KEYS = (
    "id", "n_obs", "n_vars", "obs_columns", "cluster_counts",
    "layers", "x_id", "x_sum", "x_nnz", "obs_hash", "var_hash", "layers_fingerprint",
)
_SET_KEYS = ("obs_columns", "layers")


def signatures_match(
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Compare two main-object signatures.

    Returns (ok, changed_keys). A missing/empty `after` is unsafe (cannot prove
    the object is intact) -> (False, all keys). Set-valued keys compare order-
    insensitively; floats compare with a small relative tolerance.
    """
    if not before or not after:
        return (False, list(MAIN_OBJECT_SIGNATURE_KEYS))

    changed: List[str] = []
    for key in MAIN_OBJECT_SIGNATURE_KEYS:
        if key not in before or key not in after:
            changed.append(key)
            continue
        b, a = before[key], after[key]
        if key in _SET_KEYS:
            if set(b or []) != set(a or []):
                changed.append(key)
        elif key in ("x_sum",):
            denom = max(abs(b), abs(a), 1.0)
            if abs(float(b) - float(a)) / denom > 1e-6:
                changed.append(key)
        elif b != a:
            changed.append(key)
    return (len(changed) == 0, changed)


_MAX_SUMMARY = 4000
_MAX_ARTIFACTS = 30
_MAX_WARNINGS = 30
_MAX_FIELD = 2000


def _truncate(value: Any, limit: int = _MAX_FIELD) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…[truncated]"
    if isinstance(value, list):
        return [_truncate(v, limit) for v in value[:20]]
    if isinstance(value, dict):
        return {k: _truncate(v, limit) for k, v in value.items()}
    return value


def sanitize_validation_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Bound a validation result for safe storage/transport and enforce the
    text-first rule: EVERY figure artifact must carry a non-empty `summary`;
    any figure missing one downgrades status to 'partial' with a warning naming
    it. Returns a new dict."""
    if not isinstance(result, dict):
        return {"status": "failed", "summary": "Validation returned no structured result.",
                "cluster_results": {}, "suggested_annotation_changes": {}, "artifacts": [],
                "warnings": ["non-dict validation result"]}

    out: Dict[str, Any] = dict(result)
    out["summary"] = (str(out.get("summary", ""))[:_MAX_SUMMARY]) or "Validation completed."
    out["cluster_results"] = {
        str(cid): _truncate(cr) for cid, cr in list((out.get("cluster_results") or {}).items())[:200]
        if isinstance(cr, dict)
    }
    out["suggested_annotation_changes"] = _truncate(out.get("suggested_annotation_changes") or {})
    out["artifacts"] = [_truncate(a) for a in list(out.get("artifacts") or [])[:_MAX_ARTIFACTS]]
    warnings = list(out.get("warnings") or [])[:_MAX_WARNINGS]

    figures_missing_summary = [
        str(a.get("name", a.get("path", "?")))
        for a in out["artifacts"]
        if isinstance(a, dict) and a.get("type") == "figure" and not str(a.get("summary", "")).strip()
    ]
    if figures_missing_summary and out.get("status") == "completed":
        out["status"] = "partial"
        warnings.append(
            "Figure artifacts missing a text summary (incomplete): " + ", ".join(figures_missing_summary))
    out["warnings"] = warnings
    out.setdefault("status", "completed")
    out.setdefault("suggested_annotation_changes", {})
    return out


def resolve_artifact_path(raw_path: str, workspace_dir: str) -> str:
    """Resolve a validation artifact path to an absolute path the /api/file route
    can serve (it keys ownership on a 'sessions/<sid>/' segment under workspace).

    - Absolute path already inside `workspace_dir` -> kept as-is.
    - Stray absolute path (e.g. a repo path that merely contains '/workspace/')
      -> rerouted to `<workspace_dir>/annotation_validation/<basename>`.
    - Relative path -> joined under `workspace_dir`, stripping a leading
      'workspace/' segment if present.
    """
    ws = _Path(str(workspace_dir))
    p = _Path(str(raw_path or ""))
    if p.is_absolute():
        try:
            p.relative_to(ws)
            return str(p)
        except ValueError:
            return str(ws / "annotation_validation" / p.name)
    parts = list(p.parts)
    if parts and parts[0] == "workspace":
        parts = parts[1:]
    return str(ws / _Path(*parts)) if parts else str(ws / "annotation_validation")
