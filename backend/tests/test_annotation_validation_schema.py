from backend.models.schemas import ClusterAnnotation, AnnotationConfirmationState
from backend.workflow.annotation import AnnotationMixin


def test_cluster_annotation_reclustering_defaults():
    ca = ClusterAnnotation(cluster_id="5", cell_type="Stromal")
    assert ca.requires_reclustering is False
    assert ca.reclustering_reason == ""
    assert ca.suggested_reclustering_scope is None


def test_annotation_state_validation_fields_default():
    st = AnnotationConfirmationState()
    assert st.validation_status is None
    assert st.validation_attempt_id is None
    assert st.validation_plan is None
    assert st.validation_result is None
    assert st.validation_summary is None
    assert st.validation_text_summary is None
    assert st.reclustering_recommendations is None


def test_build_cluster_annotations_maps_reclustering_fields():
    annotations = {
        "3": {
            "cell_type": "Hepatocyte (stress)", "confidence": "medium",
            "requires_reclustering": True, "reclustering_reason": "bimodal lineage",
            "suggested_reclustering_scope": "targeted_subcluster",
        }
    }
    out = AnnotationMixin._build_cluster_annotations(annotations)
    assert out[0].requires_reclustering is True
    assert out[0].reclustering_reason == "bimodal lineage"
    assert out[0].suggested_reclustering_scope == "targeted_subcluster"
