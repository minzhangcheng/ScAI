"""Test that the _annotate_clusters_callback wires validation, preserves meta, and
uses the real payload builder (not an inline dict)."""

import pytest


@pytest.mark.asyncio
async def test_confirmed_annotation_requires_same_partition_without_overwriting_history():
    from types import SimpleNamespace
    from backend.models.schemas import AnnotationConfirmationState, ClusterAnnotation
    from backend.workflow.annotation import AnnotationMixin
    confirmed = AnnotationConfirmationState(
        confirmed=True, annotation_source={"partition_digest": "final"},
        cluster_annotations=[ClusterAnnotation(cluster_id="0", cell_type="TypeA")],
    )
    host = AnnotationMixin()
    host.session_id = "source-test"
    host._current_analysis_state = SimpleNamespace(annotation_state=confirmed)
    same = await host._annotate_clusters_callback(
        {"0": ["GeneA"]}, "tissue", "human", annotation_source={"partition_digest": "final"},
    )
    assert same["already_confirmed"] is True
    changed = await host._annotate_clusters_callback(
        {"0": ["GeneA"]}, "tissue", "human", annotation_source={"partition_digest": "different"},
    )
    assert changed["status"] == "error"
    assert host._current_analysis_state.annotation_state is confirmed
    assert confirmed.cluster_annotations[0].cell_type == "TypeA"
