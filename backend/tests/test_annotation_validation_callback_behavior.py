"""Test that the _annotate_clusters_callback wires validation, preserves meta, and
uses the real payload builder (not an inline dict)."""

import pytest
from types import SimpleNamespace
from backend.models.schemas import AnnotationConfirmationState, ClusterAnnotation


def _meta():
    return {
        "validation_attempt_id": "aid",
        "validation_status": "completed",
        "validation_plan": {"tasks": [{"task_id": "t"}]},
        "validation_result": {
            "status": "completed",
            "artifacts": [{"type": "figure", "name": "f", "path": "p.jpg"}],
        },
        "validation_summary": "Cluster 7 reclassified.",
        "validation_text_summary": "- Cluster 7: contradicted",
        "reclustering_recommendations": [
            {
                "cluster_id": "7",
                "requires_reclustering": True,
                "reclustering_reason": "bimodal",
                "suggested_reclustering_scope": None,
            }
        ],
    }


def test_annotation_state_constructed_with_validation_meta():
    """The AnnotationConfirmationState the callback builds must carry validation fields."""
    meta = _meta()
    st = AnnotationConfirmationState(
        cluster_annotations=[
            ClusterAnnotation(cluster_id="7", cell_type="Technical/Contamination")
        ],
        tissue="liver",
        organism="mouse",
        pending=True,
        **meta,
    )
    assert st.validation_status == "completed"
    assert st.validation_summary == "Cluster 7 reclassified."
    assert st.reclustering_recommendations[0]["cluster_id"] == "7"


def test_real_payload_builder_includes_validation_fields():
    """Exercise the REAL production helper, not a hand-written mirror."""
    from backend.workflow.annotation import AnnotationMixin

    state = SimpleNamespace(
        annotation_state=AnnotationConfirmationState(
            tissue="liver",
            organism="mouse",
            proportion_group_labels=["Control", "NASH"],
            proportion_dimension="group",
            **_meta(),
        )
    )
    cluster_anns = [
        ClusterAnnotation(
            cluster_id="7",
            cell_type="Technical/Contamination",
            group_proportions={"Control": 12.5, "NASH": 8.0},
        )
    ]
    payload = AnnotationMixin._build_annotation_confirmation_payload(
        state, cluster_anns, "liver", "mouse", "tsum", "asum", None, None
    )
    for key in (
        "validation_status",
        "validation_summary",
        "validation_text_summary",
        "validation_plan",
        "validation_result",
        "reclustering_recommendations",
        "proportion_group_labels",
        "proportion_dimension",
    ):
        assert key in payload, f"payload missing {key}"
    assert payload["validation_summary"] == "Cluster 7 reclassified."
    assert payload["validation_result"]["artifacts"][0]["type"] == "figure"
    assert payload["reclustering_recommendations"][0]["cluster_id"] == "7"
    assert payload["proportion_group_labels"] == ["Control", "NASH"]
    assert payload["proportion_dimension"] == "group"
    assert payload["annotations"][0]["group_proportions"] == {"Control": 12.5, "NASH": 8.0}


def test_callback_wires_validation_round_and_payload_builder():
    """Source-inspection guard: the callback invokes validation spread + payload builder."""
    import inspect

    from backend.workflow.annotation import AnnotationMixin

    src = inspect.getsource(AnnotationMixin._annotate_clusters_callback)

    # validation round invoked before building cluster annotations
    assert src.index("_run_annotation_validation_round") < src.index(
        "_build_cluster_annotations(annotations)"
    )

    # validation_meta spread into BOTH state constructions (initial + supplementary)
    assert src.count("**validation_meta") >= 2

    # supplementary branch re-runs validation with force=True
    assert "force=True" in src

    # payload built via the real (tested) helper
    assert "_build_annotation_confirmation_payload" in src


def test_callback_wires_group_proportion_computation_before_confirmation_state():
    import inspect

    from backend.workflow.annotation import AnnotationMixin

    src = inspect.getsource(AnnotationMixin._annotate_clusters_callback)

    assert "compute_cluster_group_proportions" in src
    assert "_apply_group_proportions" in src
    assert src.index("compute_cluster_group_proportions") < src.index("AnnotationConfirmationState(")
    assert "proportion_group_labels=proportion_meta.get(\"group_labels\", [])" in src
    assert "proportion_dimension=proportion_meta.get(\"dimension\", \"group\")" in src
    assert src.count("compute_cluster_group_proportions") >= 2
