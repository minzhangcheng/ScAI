"""
PI Agent - Principal Investigator that coordinates all agents
"""
from typing import Dict, Any, List, Optional
import logging

from ...utils.direction_payload import normalize_direction_text
from ...utils.formatting import as_list

# Configure logger
logger = logging.getLogger(__name__)
from ...models.schemas import AnalysisState
from ...constants import CONTINUE_SIGNAL, PI_DECIDE_SIGNAL

# Synthetic control sentinels that bypass the stop/continue LLM classifier and
# deterministically continue the workflow → (response text, reason) per signal.
_CONTROL_SIGNAL_RESULTS = {
    CONTINUE_SIGNAL: ("Continuing to next round of analysis.", "User requested to continue"),
    PI_DECIDE_SIGNAL: ("Proceeding with PI-selected research directions.", "PI to autonomously select directions"),
}

# Generic heuristics for the brief-gallery selection fallback. No dataset-,
# disease-, tissue-, or species-specific names.
# Must stay a superset of the result types `infer_figure_type` can emit, minus
# the catch-all: a type discovery classifies but this tuple omits is silently
# unreachable through the deterministic fallback.
_BRIEF_PREFERRED_TYPES = ("umap", "volcano", "violin", "dotplot", "heatmap", "barplot")
_BRIEF_QC_MARKERS = (
    "qc", "pca", "elbow", "hvg", "doublet", "silhouette", "variance", "metrics",
)

_DIRECTION_CANDIDATE_LIST_FIELDS = (
    "alternative_hypotheses",
    "artifact_risks",
    "contradictory_or_missing_evidence",
    "key_genes",
    "key_cell_types",
    "key_pathways",
    "key_uncertainties",
    "suggested_literature_topics",
    "suggested_methods",
    "suggested_validation_analysis",
    "suggested_wet_lab_validation",
    "prerequisite_dependencies",
    "risk_flags",
)


def _normalize_supporting_data(value: Any) -> List[Dict[str, str]]:
    """Normalize supporting evidence to the documented list-of-objects shape."""
    normalized = []
    for item in as_list(value):
        if isinstance(item, dict):
            evidence = str(item.get("evidence") or "").strip()
            if not evidence:
                continue
            strength = str(item.get("strength") or "low").strip().lower()
            if strength not in {"high", "medium", "low"}:
                strength = "low"
            normalized.append({
                "source": str(item.get("source") or "unspecified").strip(),
                "evidence": evidence,
                "strength": strength,
            })
            continue
        evidence = str(item).strip()
        if evidence:
            normalized.append({
                "source": "unspecified",
                "evidence": evidence,
                "strength": "low",
            })
    return normalized


def _normalize_direction_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize unstable LLM field shapes without discarding their content."""
    normalized = dict(candidate)
    normalized["supporting_data"] = _normalize_supporting_data(
        normalized.get("supporting_data")
    )
    for field in _DIRECTION_CANDIDATE_LIST_FIELDS:
        normalized[field] = as_list(normalized.get(field))

    validation_plan = normalized.get("validation_plan")
    if not isinstance(validation_plan, dict):
        validation_plan = {}
    else:
        validation_plan = dict(validation_plan)
    validation_plan["next_analysis"] = as_list(
        validation_plan.get("next_analysis")
        or normalized.get("suggested_validation_analysis")
    )
    validation_plan["wet_lab_validation"] = as_list(
        validation_plan.get("wet_lab_validation")
        or normalized.get("suggested_wet_lab_validation")
    )
    validation_plan["decision_criteria"] = as_list(
        validation_plan.get("decision_criteria")
        or normalized.get("decision_criteria")
    )
    normalized.pop("decision_criteria", None)
    normalized["validation_plan"] = validation_plan
    return normalized


def _coerce_priority_rank(value: Any) -> Optional[int]:
    """The rank as an int, or None when it is not one.

    Models return this as "1" about as often as 1, and ``isinstance(True, int)``
    is True, so a naive int check reads a boolean as rank 1.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("+-").isdigit():
            return int(text)
    return None


def _restore_unscored_candidates(
    scored: List[Dict[str, Any]], raw_candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Put back every candidate the scoring step did not return.

    Scoring ranks; it has no power to remove. An omission is unrecoverable --
    nothing downstream rereads ``raw_candidates``, and the integration node reads
    an empty scored list as the scientific statement that no direction is worth
    another round -- so an omitted candidate is restored here and the scoring
    step gets to say what it thinks through ``priority_rank`` instead.

    Restored candidates keep their original order and rank after every scored
    one. Deterministic and domain-neutral: it compares identity, never subject
    matter.
    """
    kept_ids = set()
    kept_directions = set()
    ranks: List[int] = []
    for candidate in scored:
        if not isinstance(candidate, dict):
            continue
        direction_id = str(candidate.get("direction_id", "")).strip()
        if direction_id:
            kept_ids.add(direction_id)
        direction = normalize_direction_text(candidate.get("direction"))
        if direction:
            kept_directions.add(direction)
        rank = _coerce_priority_rank(candidate.get("priority_rank"))
        if rank is None:
            candidate.pop("priority_rank", None)
        else:
            candidate["priority_rank"] = rank
            ranks.append(rank)

    # Ranked entries without a usable rank still have to come before restored
    # ones, so the floor is the number of them, not zero.
    next_rank = max([len(scored)] + ranks) + 1

    restored = list(scored)
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        direction_id = str(raw.get("direction_id", "")).strip()
        direction = normalize_direction_text(raw.get("direction"))
        if direction_id and direction_id in kept_ids:
            continue
        if direction and direction in kept_directions:
            continue
        recovered = _normalize_direction_candidate(dict(raw))
        recovered["priority_rank"] = next_rank
        next_rank += 1
        flags = recovered.get("risk_flags")
        if not isinstance(flags, list):
            flags = []
        if "unscored_by_gate" not in flags:
            flags.append("unscored_by_gate")
        recovered["risk_flags"] = flags
        logger.info(
            "[PI] Candidate %r came back unscored; passing it on ranked last",
            (str(raw.get("direction", ""))[:120] or direction_id or "unnamed"),
        )
        restored.append(recovered)
    return restored


def _normalize_integration_scored_candidates(
    scored: List[Any], raw_candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge score-only responses onto their raw candidates, then normalize."""
    raw_by_id = {}
    raw_by_direction = {}
    duplicate_ids = set()
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        direction_id = str(candidate.get("direction_id", ""))
        direction = str(candidate.get("direction", "")).strip()
        if direction_id and direction_id not in duplicate_ids:
            if direction_id in raw_by_id:
                raw_by_id.pop(direction_id, None)
                duplicate_ids.add(direction_id)
            else:
                raw_by_id[direction_id] = candidate
        if direction:
            raw_by_direction.setdefault(direction, candidate)
    normalized = []
    for candidate in scored:
        if not isinstance(candidate, dict):
            continue
        raw = raw_by_id.get(str(candidate.get("direction_id", "")))
        if raw is None:
            raw = raw_by_direction.get(str(candidate.get("direction", "")).strip())
        if raw is None:
            # The scorer is asked to SCORE the candidates it was given. A scored
            # entry matching none of them is a new direction the scoring step
            # invented -- previously `dict(raw or {})` accepted it and it became a
            # real candidate, which is how an empty proposal could still end up
            # producing directions to investigate.
            label = str(candidate.get("direction", ""))[:120] or "unnamed"
            logger.warning("[PI] Scorer returned an unmatched candidate %r; dropping it", label)
            continue
        merged = dict(raw)
        for key, value in candidate.items():
            if value not in (None, "", [], {}):
                merged[key] = value
        # A name-based fallback means the scorer returned a drifted ID. Keep the
        # canonical raw identity so evidence/dependency lookups remain stable.
        if raw is not None and raw.get("direction_id"):
            merged["direction_id"] = raw["direction_id"]
        normalized.append(_normalize_direction_candidate(merged))
    return normalized


def _bounded_brief_fallback(
    all_figures: list[dict], limit: int = 20
) -> list[str]:
    """Deterministic, bounded fallback when PI curation fails.

    Never returns the full candidate list. Returns only confident "good"
    candidates: publication figures, then likely result figures whose inferred
    type is a known result type (umap/volcano/dotplot/heatmap/barplot), dropping
    generic QC/technical diagnostics. Ambiguous ``type == "other"`` figures are
    NOT included -- per the design, when no good candidates exist this returns an
    empty list and the reporting layer keeps its own publication-figure
    fallback. Preserves input order within each priority bucket.
    """
    def _is_qc(name: str) -> bool:
        s = name.lower()
        return any(marker in s for marker in _BRIEF_QC_MARKERS)

    pub = [f["name"] for f in all_figures if f.get("is_publication")]
    preferred = [
        f["name"] for f in all_figures
        if not f.get("is_publication")
        and f.get("type") in _BRIEF_PREFERRED_TYPES
        and not _is_qc(f["name"])
    ]

    ordered: list[str] = []
    for name in pub + preferred:
        if name not in ordered:
            ordered.append(name)
        if len(ordered) >= limit:
            break
    return ordered


def _prior_label(hit) -> str:
    """Name the hit, folding in the synonyms that fired on the same genes.

    The kernel collapses those into one entry so they cannot fill the top-k;
    they still belong in the prompt as alternative readings of one signal.
    """
    name = str(hit.get("cell_type", "?"))
    also = [str(a) for a in (hit.get("also_matched") or [])]
    if not also:
        return name
    shown = " / ".join([name] + also[:2])
    return shown + (f" (+{len(also) - 2} more)" if len(also) > 2 else "")


def _tissue_note(hit) -> str:
    """Say whether the cell type is annotated in the tissue under analysis.

    Deliberately not a list of organ names: a cell type may be annotated in
    dozens, and showing an arbitrary few misrepresents it (an earlier version
    labelled CD8 T cells "Kidney"). Absent tissue data yields no note at all,
    which is the case for any table written before the organ column was carried.
    """
    match = hit.get("tissue_match")
    n = hit.get("n_tissues") or 0
    if match is None:
        return f", tissue: annotated in {n} organs" if n > 3 else ""
    if match:
        return ", tissue: matches"
    return f", tissue: other organs only ({n})" if n else ", tissue: other organs only"


def _format_database_priors_block(database_priors) -> str:
    if not database_priors:
        return ""
    lines = []
    for cid, hits in database_priors.items():
        parts = []
        for h in hits:
            ov = ", ".join(h.get("overlap", []))
            parts.append(f"{_prior_label(h)}: padj={h.get('p_adj', 1.0):.2g} "
                         f"(overlap: {ov}{_tissue_note(h)})")
        if parts:
            lines.append(f"Cluster {cid} \u2192 " + "; ".join(parts))
    if not lines:
        return ""
    return ("\n\n=== DATABASE PRIORS (offline ORA, supportive evidence) ===\n"
            "These are quantitative priors from offline marker databases; literature remains primary. "
            "If a prior contradicts the literature, favor biological plausibility and the literature.\n"
            "The databases span all organs and mix granularities, so a cell type from an unrelated "
            "tissue, or a subtype far finer than the data supports, can still score well on a couple "
            "of shared genes. Weigh the overlap size and the tissue note, not the ranking alone; "
            "'tissue: other organs only' is a reason to discount a hit, and a cell type annotated "
            "across many organs carries no tissue evidence either way.\n"
            + "\n".join(lines) + "\n")


def _build_default_bio_tasks(state: AnalysisState, detailed: bool = False) -> list:
    """Build default bioinformatics task list based on available sequencing data types."""
    has_bulk = state.has_bulk_rnaseq()
    has_scrna = state.has_scrnaseq()
    tasks = []
    if has_scrna:
        if detailed:
            tasks.append(
                "Execute standard scRNA-seq analysis: QC, normalization, clustering, "
                "and annotation; use measured design facts when the PI selects any "
                "condition analysis and claim level"
            )
        else:
            tasks.append("Execute standard scRNA-seq/snRNA-seq analysis workflow")
    if has_bulk:
        tasks.append(
            "Review the measured bulk RNA-seq design facts and execute the analysis "
            "and claim level selected by the PI"
        )
        if has_scrna:
            tasks.append(
                "If the PI selects a cross-modal question, compare the bulk and "
                "single-cell evidence using methods and claim levels the PI approves."
            )
    if not tasks:
        tasks.append("Execute standard sequencing analysis workflow")
    return tasks
