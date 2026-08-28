"""Annotation fallback, context, proportion, and HITL payload helpers."""
from __future__ import annotations


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )

@staticmethod
def _build_fallback_annotations(
    cluster_markers: Dict[str, Any], error: Exception
) -> Dict[str, Any]:
    """Build fallback annotations when annotation service fails."""
    fallback_annotations = {}
    for cid in cluster_markers:
        fallback_annotations[str(cid)] = {
            "cell_type": "Unknown",
            "confidence": "low",
            "key_markers": (
                list(cluster_markers[cid])[:10]
                if isinstance(cluster_markers[cid], list)
                else []
            ),
            "evidence": "",
            "low_confidence_reason": f"Annotation service failed: {str(error)}",
        }
    return {"annotations": fallback_annotations, "tissue_summary": ""}

def _build_annotation_context_report(self, state: AnalysisState, graph_state: Optional["WorkflowState"]) -> str:
    """Build a comprehensive markdown summary of current analysis context for cell annotation."""
    parts = ["### Analysis Context Report\n"]

    # 1. Experimental Design
    parts.append("#### Experimental Design")
    parts.append(state.build_experimental_design_summary())
    parts.append("")

    # 2. Biochemistry & Pathology findings (from graph state if available)
    if graph_state:
        biochem = graph_state.get("biochemistry_result")
        if biochem and biochem.get("findings"):
            parts.append("#### Biochemistry Findings")
            for f in biochem["findings"]:
                parts.append(f"- {f}")
            parts.append("")

        patho = graph_state.get("pathology_result")
        if patho and patho.get("findings"):
            parts.append("#### Pathology Findings")
            for f in patho["findings"]:
                parts.append(f"- {f}")
            parts.append("")

    # 3. Previous Rounds' Key Findings
    if state.round_results and len(state.round_results) > 0:
        parts.append("#### Previous Analysis Highlights")
        last_round = state.round_results[-1]
        if last_round.pi_summary:
            parts.append(last_round.pi_summary)
        parts.append("")

    # 4. Literature Searches (current round)
    lit_steps = [step for step in state.analysis_log if step.step_type == "search_literature"]
    if lit_steps:
        parts.append("#### Recent Literature Research")
        for step in lit_steps[-3:]:  # Last 3 searches
            topic = step.literature_topic or "Unknown topic"
            result = step.literature_result or "No findings recorded"
            parts.append(f"**Topic:** {topic}")
            parts.append(f"**Findings:** {result[:300]}...\n")

    return "\n".join(parts)

@staticmethod
def _build_cluster_annotations(annotations: Dict) -> List[ClusterAnnotation]:
    """Build ClusterAnnotation list from annotation dict."""
    return [
        ClusterAnnotation(
            cluster_id=cid,
            cell_type=info.get("cell_type", "Unknown"),
            subtypes=info.get("subtypes", []),
            key_markers=info.get("key_markers", []),
            confidence=info.get("confidence", "low"),
            evidence=info.get("evidence", ""),
            n_cells=max(0, int(info.get("n_cells", 0) or 0)),
            n_cells_known=(
                bool(info.get("n_cells_known"))
                if "n_cells_known" in info
                else "n_cells" in info and info.get("n_cells") is not None
            ),
            n_cells_source=(
                str(info.get("n_cells_source") or "unknown")
                if "n_cells_source" in info
                else (
                    "annotation_payload"
                    if "n_cells" in info and info.get("n_cells") is not None
                    else "unknown"
                )
            ),
            group_proportions=info.get("group_proportions", {}) or {},
            alternative_cell_types=info.get("alternative_cell_types", []),
            low_confidence_reason=info.get("low_confidence_reason", ""),
            requires_reclustering=info.get("requires_reclustering", False),
            reclustering_reason=info.get("reclustering_reason", ""),
            suggested_reclustering_scope=info.get("suggested_reclustering_scope"),
        )
        for cid, info in annotations.items()
        if isinstance(info, dict)
    ]

@staticmethod
def _apply_group_proportions(
    cluster_anns: List[ClusterAnnotation],
    proportion_meta: Optional[Dict[str, Any]],
) -> None:
    proportions = (proportion_meta or {}).get("proportions") or {}
    counts = (proportion_meta or {}).get("counts") or {}
    for ca in cluster_anns:
        ca.group_proportions = dict(proportions.get(str(ca.cluster_id), {}) or {})
        count = counts.get(str(ca.cluster_id))
        if count is not None:
            ca.n_cells = max(0, int(count))
            ca.n_cells_known = True
            ca.n_cells_source = "cluster_group_proportions"

@staticmethod
def _annotations_from_cluster_annotations(cluster_annotations) -> Dict[str, Any]:
    """Rebuild an annotations dict from a saved ClusterAnnotation list (the
    inverse of _build_cluster_annotations), preserving validation revisions."""
    out: Dict[str, Any] = {}
    for ca in (cluster_annotations or []):
        out[ca.cluster_id] = {
            "cell_type": ca.user_override or ca.cell_type,
            "subtypes": list(ca.subtypes),
            "key_markers": list(ca.key_markers),
            "confidence": ca.confidence,
            "evidence": ca.evidence,
            "n_cells": ca.n_cells,
            "n_cells_known": ca.n_cells_known,
            "n_cells_source": ca.n_cells_source,
            "group_proportions": dict(ca.group_proportions or {}),
            "alternative_cell_types": list(ca.alternative_cell_types),
            "low_confidence_reason": ca.low_confidence_reason,
            "requires_reclustering": ca.requires_reclustering,
            "reclustering_reason": ca.reclustering_reason,
            "suggested_reclustering_scope": ca.suggested_reclustering_scope,
        }
    return out

@staticmethod
def _empty_validation_meta(status: str, summary: str = "", attempt_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "validation_attempt_id": attempt_id,
        "validation_status": status,
        "validation_plan": None,
        "validation_result": None,
        "validation_summary": summary or None,
        "validation_text_summary": None,
        "reclustering_recommendations": None,
    }
@staticmethod
def _build_validation_text_summary(val_result: Dict[str, Any]) -> str:
    """Compact, vision-free per-cluster verdict summary."""
    lines = [val_result.get("summary", "")]
    for cid, cr in (val_result.get("cluster_results") or {}).items():
        if isinstance(cr, dict):
            lines.append(f"- Cluster {cid}: {cr.get('support_level','?')}; "
                         f"action={cr.get('suggested_action','')}; qc={cr.get('qc_flags', [])}")
    return "\n".join(l for l in lines if l)

@staticmethod
def _build_annotation_confirmation_payload(
    state: AnalysisState, cluster_anns: List[ClusterAnnotation], tissue: str, organism: str,
    tissue_summary: str, analysis_summary: str,
    report_html_url: Optional[str], detailed_report_url: Optional[str],
) -> Dict[str, Any]:
    """Build the annotation_confirmation payload, including optional
    validation and proportion fields read from the pending annotation state."""
    vstate = state.annotation_state
    return {
        "annotations": [a.model_dump(mode="json") for a in cluster_anns],
        "tissue": tissue,
        "organism": organism,
        "tissue_summary": tissue_summary,
        "analysis_summary": analysis_summary,
        "report_html_url": report_html_url,
        "detailed_report_url": detailed_report_url,
        "proportion_group_labels": getattr(vstate, "proportion_group_labels", []),
        "proportion_dimension": getattr(vstate, "proportion_dimension", "group"),
        "validation_status": getattr(vstate, "validation_status", None),
        "validation_summary": getattr(vstate, "validation_summary", None),
        "validation_text_summary": getattr(vstate, "validation_text_summary", None),
        "validation_plan": getattr(vstate, "validation_plan", None),
        "validation_result": getattr(vstate, "validation_result", None),
        "reclustering_recommendations": getattr(vstate, "reclustering_recommendations", None),
    }
