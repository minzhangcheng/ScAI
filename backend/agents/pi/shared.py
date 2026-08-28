"""
PI Agent - Principal Investigator that coordinates all agents
"""
from typing import Dict, Any, List
import logging

from ...utils.formatting import as_list

# Configure logger
logger = logging.getLogger(__name__)
from ...models.schemas import AnalysisState
from ...constants import CONTINUE_SIGNAL, PI_DECIDE_SIGNAL
from ...services.sample_design import (
    BULK_RNASEQ_MODALITY,
    SCRNA_MODALITY,
    has_replicated_condition_design,
)

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
        merged = dict(raw or {})
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
    scrna_replicated = has_replicated_condition_design(state, SCRNA_MODALITY)
    bulk_replicated = has_replicated_condition_design(
        state, BULK_RNASEQ_MODALITY,
    )
    tasks = []
    if has_scrna:
        if detailed:
            if scrna_replicated:
                tasks.append(
                    "Execute standard scRNA-seq analysis: QC, normalization, clustering, annotation, "
                    "then sample-level pseudobulk condition DEG + GO/KEGG for all supported cell types"
                )
            else:
                tasks.append(
                    "Execute standard scRNA-seq analysis: QC, normalization, clustering and annotation; "
                    "keep condition comparisons descriptive because each condition lacks independent replication"
                )
        else:
            tasks.append("Execute standard scRNA-seq/snRNA-seq analysis workflow")
    if has_bulk:
        tasks.append(
            "Perform bulk RNA-seq DEG analysis using biological samples as the unit"
            if bulk_replicated else
            "Inspect bulk RNA-seq descriptively; sample-to-group membership is unresolved, so do not report condition-level p/FDR"
        )
        if has_scrna:
            if scrna_replicated and bulk_replicated:
                tasks.append(
                    "Cross-validate bulk DEGs with sample-level scRNA-seq pseudobulk DEGs, "
                    "visualize concordance and cell-type-specific expression, and assess "
                    "cell-type deconvolution where appropriate."
                )
            elif bulk_replicated:
                tasks.append(
                    "Use replicated bulk DEG/pathway results as the inferential layer; "
                    "localize supported signals descriptively to single-cell cell types/states, "
                    "without inferential scRNA pseudobulk or cell-level condition p-values."
                )
            elif scrna_replicated:
                tasks.append(
                    "Use replicated single-cell sample-pseudobulk for condition inference; "
                    "treat bulk observations as descriptive until its design is verified."
                )
    if not tasks:
        tasks.append("Execute standard sequencing analysis workflow")
    return tasks
