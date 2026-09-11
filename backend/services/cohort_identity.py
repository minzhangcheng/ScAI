"""Host-owned identity for the canonical analysis cohort."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional


_COHORT_MARKER = "__SCAI_COHORT_IDENTITY_V1__:"


def cohort_probe_code() -> str:
    """Return trusted kernel code that emits only a digest, never observation IDs."""
    return r'''
import hashlib as _scai_cohort_hashlib
import json as _scai_cohort_json

_scai_cohort_obj = globals().get("adata")
_scai_cohort_payload = {"present": False}
if _scai_cohort_obj is not None and hasattr(_scai_cohort_obj, "obs_names") and hasattr(_scai_cohort_obj, "var_names"):
    _scai_cohort_digest = _scai_cohort_hashlib.sha256()
    for _scai_cohort_axis, _scai_cohort_names in ((b"obs", _scai_cohort_obj.obs_names), (b"var", _scai_cohort_obj.var_names)):
        _scai_cohort_digest.update(_scai_cohort_axis)
        for _scai_cohort_name in _scai_cohort_names:
            _scai_cohort_value = str(_scai_cohort_name).encode("utf-8", errors="surrogatepass")
            _scai_cohort_digest.update(len(_scai_cohort_value).to_bytes(8, "big"))
            _scai_cohort_digest.update(_scai_cohort_value)
    _scai_cohort_payload = {
        "present": True,
        "canonical_object_name": "adata",
        "cohort_digest": _scai_cohort_digest.hexdigest(),
        "n_obs": int(_scai_cohort_obj.n_obs),
        "n_vars": int(_scai_cohort_obj.n_vars),
    }
print("__SCAI_COHORT_IDENTITY_V1__:" + _scai_cohort_json.dumps(_scai_cohort_payload, sort_keys=True))
del _scai_cohort_hashlib, _scai_cohort_json, _scai_cohort_obj, _scai_cohort_payload
if "_scai_cohort_digest" in globals(): del _scai_cohort_digest
if "_scai_cohort_axis" in globals(): del _scai_cohort_axis
if "_scai_cohort_names" in globals(): del _scai_cohort_names
if "_scai_cohort_name" in globals(): del _scai_cohort_name
if "_scai_cohort_value" in globals(): del _scai_cohort_value
'''


def parse_cohort_probe(output: str) -> Optional[Dict[str, Any]]:
    for line in reversed(str(output or "").splitlines()):
        if not line.startswith(_COHORT_MARKER):
            continue
        try:
            payload = json.loads(line[len(_COHORT_MARKER):])
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        if not payload.get("present"):
            return {"present": False}
        digest = str(payload.get("cohort_digest") or "")
        if len(digest) != 64:
            return None
        return {
            "present": True,
            "canonical_object_name": "adata",
            "cohort_digest": digest,
            "n_obs": max(0, int(payload.get("n_obs") or 0)),
            "n_vars": max(0, int(payload.get("n_vars") or 0)),
        }
    return None


async def probe_canonical_cohort(sandbox) -> Optional[Dict[str, Any]]:
    result = await sandbox.execute(
        cohort_probe_code(),
        records_history=False,
        skip_safety_check=True,
    )
    if getattr(result, "error", None):
        return None
    return parse_cohort_probe(getattr(result, "output", ""))


def active_cohort_identity(state) -> Dict[str, Any]:
    sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
    identity = sandbox_state.get("cohort_identity_v1")
    return dict(identity) if isinstance(identity, dict) else {}


def _annotate_prior_cohort_artifacts(state, active: Dict[str, Any]) -> int:
    """Attach a cohort warning without changing artifact eligibility.

    The compatibility name remains because this helper is part of the existing
    transition path. Cohort identity still advances for evidence provenance,
    while figures and tables remain visible in results and reports.
    """
    active_digest = str(active.get("cohort_digest") or "")
    active_generation = int(active.get("cohort_generation", 0) or 0)
    active_lineage = getattr(state, "restore_branch_id", None) or "main"
    annotated = 0
    warning = (
        "Warning: the canonical AnnData cohort changed after this artifact was "
        "generated; the artifact remains included and should be interpreted "
        "against its recorded cohort identity."
    )
    for collection_name in ("publication_figures", "publication_tables"):
        for artifact in getattr(state, collection_name, None) or []:
            if (getattr(artifact, "lineage_id", None) or "main") != active_lineage:
                continue
            artifact_digest = str(getattr(artifact, "cohort_digest", None) or "")
            artifact_generation = int(
                getattr(artifact, "cohort_generation", 0) or 0
            )
            if (
                artifact_digest == active_digest
                and artifact_generation == active_generation
            ):
                continue
            provenance = dict(
                getattr(artifact, "expression_provenance", None) or {}
            )
            provenance["cohort_change_warning"] = warning
            artifact.expression_provenance = provenance
            description = str(getattr(artifact, "description", "") or "")
            if warning not in description:
                artifact.description = (description + "\n\n" + warning).strip()
            annotated += 1
    return annotated


def record_cohort_transition(
    state,
    before: Optional[Dict[str, Any]],
    after: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Persist active cohort identity and annotate older artifacts on change."""
    sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
    stored = active_cohort_identity(state)
    baseline = stored or (before if before and before.get("present") else {})
    after_present = bool(after and after.get("present"))
    after_digest = str((after or {}).get("cohort_digest") or "")
    baseline_digest = str(baseline.get("cohort_digest") or "")

    if not after_present:
        if baseline_digest:
            generation = max(1, int(baseline.get("cohort_generation") or 1)) + 1
            identity = {
                "present": False,
                "canonical_object_name": "adata",
                "cohort_generation": generation,
                "cohort_digest": None,
                "n_obs": 0,
                "n_vars": 0,
                "lineage_id": getattr(state, "restore_branch_id", None) or "main",
                "round_number": int(getattr(state, "current_round", 0) or 0),
                "parent_cohort_digest": baseline_digest,
            }
            sandbox_state["cohort_identity_v1"] = identity
            sandbox_state["cohort_transition_notice"] = {
                "changed": True,
                "reason": "canonical_object_missing",
                "identity": identity,
            }
            state.sandbox_state = sandbox_state
            _annotate_prior_cohort_artifacts(state, identity)
            return identity
        return stored

    changed = not baseline_digest or baseline_digest != after_digest
    previous_generation = int(baseline.get("cohort_generation") or 0)
    generation = previous_generation + 1 if changed else max(1, previous_generation)
    identity = {
        **dict(after or {}),
        "cohort_generation": generation,
        "lineage_id": getattr(state, "restore_branch_id", None) or "main",
        "round_number": int(getattr(state, "current_round", 0) or 0),
        "parent_cohort_digest": baseline_digest or None,
    }
    sandbox_state["cohort_identity_v1"] = identity
    sandbox_state["cohort_generation"] = generation
    if changed:
        sandbox_state["cohort_transition_notice"] = {
            "changed": True,
            "reason": "canonical_cohort_changed",
            "identity": identity,
        }
    state.sandbox_state = sandbox_state
    if changed and baseline_digest:
        _annotate_prior_cohort_artifacts(state, identity)
    return identity
