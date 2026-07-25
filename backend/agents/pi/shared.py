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

# Synthetic control sentinels that bypass the stop/continue LLM classifier and
# deterministically continue the workflow → (response text, reason) per signal.
_CONTROL_SIGNAL_RESULTS = {
    CONTINUE_SIGNAL: ("Continuing to next round of analysis.", "User requested to continue"),
    PI_DECIDE_SIGNAL: ("Proceeding with PI-selected research directions.", "PI to autonomously select directions"),
}

# Generic heuristics for the brief-gallery selection fallback. No dataset-,
# disease-, tissue-, or species-specific names.
_BRIEF_PREFERRED_TYPES = ("umap", "volcano", "dotplot", "heatmap", "barplot")
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


def _format_database_priors_block(database_priors) -> str:
    if not database_priors:
        return ""
    lines = []
    for cid, hits in database_priors.items():
        parts = []
        for h in hits:
            ov = ", ".join(h.get("overlap", []))
            parts.append(f"{h.get('cell_type','?')}: padj={h.get('p_adj', 1.0):.2g} (overlap: {ov})")
        if parts:
            lines.append(f"Cluster {cid} \u2192 " + "; ".join(parts))
    if not lines:
        return ""
    return ("\n\n=== DATABASE PRIORS (offline ORA, supportive evidence) ===\n"
            "These are quantitative priors from offline marker databases; literature remains primary. "
            "If a prior contradicts the literature, favor biological plausibility and the literature.\n"
            + "\n".join(lines) + "\n")


def _build_default_bio_tasks(state: AnalysisState, detailed: bool = False) -> list:
    """Build default bioinformatics task list based on available sequencing data types."""
    has_bulk = state.has_bulk_rnaseq()
    has_scrna = state.has_scrnaseq()
    tasks = []
    if has_scrna:
        if detailed:
            tasks.append(
                "Execute standard scRNA-seq analysis: QC, normalization, clustering, annotation, "
                "then condition-level DEG + GO/KEGG for ALL clusters, cell communication analysis"
            )
        else:
            tasks.append("Execute standard scRNA-seq/snRNA-seq analysis workflow")
    if has_bulk:
        tasks.append("Perform bulk RNA-seq DEG analysis using pydeseq2 (fallback: scipy.stats.ttest_ind + FDR correction if pydeseq2 unavailable)")
        if has_scrna:
            tasks.append(
                "Cross-validate bulk DEGs with scRNA-seq pseudobulk DEGs: "
                "compute set intersection, generate Venn diagram (from matplotlib_venn import venn2), "
                "and visualize overlapping DEGs' cell-type-specific expression. "
                "CRITICAL: Perform cell-type deconvolution of bulk RNA-seq data using scRNA-seq as a reference "
                "(prioritize scipy.optimize.nnls or sklearn Non-negative Least Squares; if unavailable, try scaden) to estimate cell-type proportion changes."
            )
    if not tasks:
        tasks.append("Execute standard sequencing analysis workflow")
    return tasks


