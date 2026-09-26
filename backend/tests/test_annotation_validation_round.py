import pytest
from types import SimpleNamespace
from backend.workflow.annotation import AnnotationMixin


class _Host(AnnotationMixin):
    def __init__(self, pi, bio, flag=True):
        self.session_id = "s"; self.pi_agent = pi; self.bioinformatics_agent = bio
        self._enable = flag
        self.sent = []

    def _validation_enabled(self):
        return self._enable

    async def _send_message(self, *a, **k):
        self.sent.append(a)


def _state(ann=None, rnd=1):
    return SimpleNamespace(annotation_state=ann, current_round=rnd)


@pytest.mark.asyncio
async def test_validation_uses_annotation_source_for_pi_and_executor():
    from unittest.mock import AsyncMock
    source = {"adata_var": "adata_merged", "cluster_key": "leiden", "partition_digest": "final"}
    metadata = {"var": "adata_merged", "cluster_key": "leiden", "partition_digest": "final"}
    bio = SimpleNamespace(
        resolve_analysis_source_metadata=AsyncMock(return_value=metadata),
        introspect_validation_metadata=AsyncMock(side_effect=AssertionError("must not rediscover")),
        validate_annotations_with_plan=AsyncMock(return_value={"status": "partial", "cluster_results": {}, "execution_observations": ["retained check"]}),
    )
    pi = SimpleNamespace(
        generate_annotation_validation_questions=AsyncMock(return_value={"tasks": ["inspect cluster 0"]}),
        revise_annotations_after_validation=AsyncMock(return_value={"annotations": {"0": {"cell_type": "X", "confidence": "low"}}}),
    )
    host = _Host(pi, bio)
    _, meta = await host._run_annotation_validation_round(
        _state(), {"0": {"cell_type": "X", "confidence": "low"}}, {"0": ["A"]},
        "tissue", "species", annotation_source=source,
    )
    bio.resolve_analysis_source_metadata.assert_awaited_once_with(source)
    assert pi.generate_annotation_validation_questions.await_args.kwargs["metadata_context"] == metadata
    assert bio.validate_annotations_with_plan.await_args.kwargs["context"] == metadata
    assert meta["validation_result"]["execution_observations"] == ["retained check"]
