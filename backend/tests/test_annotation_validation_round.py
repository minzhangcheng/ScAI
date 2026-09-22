import pytest
from types import SimpleNamespace
from backend.workflow.annotation import AnnotationMixin
from backend.models.schemas import AnnotationConfirmationState
from backend.workflow.annotation_validation import compute_validation_attempt_id


class _Host(AnnotationMixin):
    def __init__(self, pi, bio, flag=True):
        self.session_id = "s"; self.pi_agent = pi; self.bioinformatics_agent = bio
        self._enable = flag
        self.sent = []

    def _validation_enabled(self):
        return self._enable

    async def _send_message(self, *a, **k):
        self.sent.append(a)


def _amock(rv):
    async def _f(*a, **k):
        return rv
    return _f


def _state(ann=None, rnd=1):
    return SimpleNamespace(annotation_state=ann, current_round=rnd)


@pytest.mark.asyncio
async def test_disabled_returns_skipped():
    host = _Host(SimpleNamespace(), SimpleNamespace(), flag=False)
    anns = {"0": {"cell_type": "X", "confidence": "low"}}
    out, meta = await host._run_annotation_validation_round(
        _state(), anns, {"0": ["A"]}, "liver", "mouse", triage={}, database_priors={}, focused_evidence={})
    assert out == anns and meta["validation_status"] == "skipped"


@pytest.mark.asyncio
async def test_skips_when_no_triggers():
    bio = SimpleNamespace(introspect_validation_metadata=_amock({"var": "adata", "cluster_key": "leiden"}))
    host = _Host(SimpleNamespace(), bio)
    anns = {"0": {"cell_type": "Hepatocyte", "confidence": "high"}}
    out, meta = await host._run_annotation_validation_round(
        _state(), anns, {"0": ["Alb"]}, "liver", "mouse", triage={}, database_priors={}, focused_evidence={})
    assert out == anns and meta["validation_status"] == "skipped"


@pytest.mark.asyncio
async def test_idempotent_reuse_on_same_attempt():
    bio = SimpleNamespace(introspect_validation_metadata=_amock({"var": "adata", "cluster_key": "leiden"}))
    host = _Host(SimpleNamespace(), bio)
    anns = {"0": {"cell_type": "X", "confidence": "low"}}
    aid = compute_validation_attempt_id(1, "liver", "mouse", "leiden", {"0": ["Alb"]})
    prev = AnnotationConfirmationState(validation_status="completed", validation_attempt_id=aid,
                                       validation_summary="prior")
    out, meta = await host._run_annotation_validation_round(
        _state(prev), anns, {"0": ["Alb"]}, "liver", "mouse", triage={}, database_priors={}, focused_evidence={})
    assert out == anns
    assert meta["validation_status"] == "completed" and meta["validation_summary"] == "prior"


@pytest.mark.asyncio
async def test_full_path_applies_revision_and_reclustering():
    bio = SimpleNamespace(
        introspect_validation_metadata=_amock({"var": "adata", "cluster_key": "leiden"}),
        validate_annotations_with_plan=_amock({"status": "completed", "summary": "ok",
            "cluster_results": {"7": {"support_level": "contradicted", "requires_reclustering": True,
                                      "reclustering_reason": "bimodal", "qc_flags": ["x"]}},
            "suggested_annotation_changes": {}, "artifacts": [], "warnings": []}))
    pi = SimpleNamespace(
        generate_annotation_validation_questions=_amock({"scope": "annotation_validation_only",
            "tasks": [{"task_id": "t", "cluster_ids": ["7"]}]}),
        revise_annotations_after_validation=_amock({
            "annotations": {"7": {"cell_type": "Technical/Contamination", "confidence": "low"}},
            "validation_summary_for_user": "Cluster 7 is technical."}))
    host = _Host(pi, bio)
    anns = {"7": {"cell_type": "Hepatic neurons", "confidence": "medium"}}
    out, meta = await host._run_annotation_validation_round(
        _state(), anns, {"7": ["Nrxn1"]}, "liver", "mouse",
        triage={"suspected_contamination": [{"cluster_id": "7"}]}, database_priors={}, focused_evidence={})
    assert out["7"]["cell_type"] == "Technical/Contamination"
    assert out["7"]["requires_reclustering"] is True
    assert meta["validation_status"] == "completed"
    assert meta["validation_summary"] == "Cluster 7 is technical."
    assert meta["reclustering_recommendations"][0]["cluster_id"] == "7"
    assert meta["validation_attempt_id"]


@pytest.mark.asyncio
async def test_executor_failure_records_failed_status():
    bio = SimpleNamespace(
        introspect_validation_metadata=_amock({"var": "adata", "cluster_key": "leiden"}),
        validate_annotations_with_plan=_amock({"status": "failed", "summary": "boom",
            "cluster_results": {}, "warnings": ["boom"]}))
    pi = SimpleNamespace(
        generate_annotation_validation_questions=_amock({"tasks": [{"task_id": "t", "cluster_ids": ["7"]}]}),
        revise_annotations_after_validation=_amock({"annotations": {"7": {"cell_type": "X"}},
                                                    "validation_summary_for_user": "kept"}))
    host = _Host(pi, bio)
    anns = {"7": {"cell_type": "X", "confidence": "low"}}
    out, meta = await host._run_annotation_validation_round(
        _state(), anns, {"7": ["A"]}, "liver", "mouse", triage={}, database_priors={}, focused_evidence={})
    assert meta["validation_status"] == "failed" and "7" in out


@pytest.mark.asyncio
async def test_stray_absolute_artifact_path_is_rerouted(monkeypatch):
    import backend.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "get_session_workspace",
                        lambda sid: __import__("pathlib").Path("/data/sessions/s/workspace"),
                        raising=False)
    bio = SimpleNamespace(
        introspect_validation_metadata=_amock({"var": "adata", "cluster_key": "leiden"}),
        validate_annotations_with_plan=_amock({"status": "completed", "summary": "ok",
            "cluster_results": {"0": {"qc_flags": ["x"]}}, "suggested_annotation_changes": {},
            "artifacts": [{"type": "figure", "name": "m", "summary": "s",
                           "path": "/workspace/cmz/ScAI/scai/data/sessions/s/workspace/annotation_validation/m.jpg"}],
            "warnings": []}))
    pi = SimpleNamespace(
        generate_annotation_validation_questions=_amock({"tasks": [{"task_id": "t", "cluster_ids": ["0"]}]}),
        revise_annotations_after_validation=_amock({"annotations": {"0": {"cell_type": "X"}},
                                                    "validation_summary_for_user": "done"}))
    host = _Host(pi, bio)
    out, meta = await host._run_annotation_validation_round(
        _state(), {"0": {"cell_type": "X", "confidence": "low"}}, {"0": ["A"]}, "liver", "mouse",
        triage={}, database_priors={}, focused_evidence={})
    assert meta["validation_result"]["artifacts"][0]["path"] == \
        "/data/sessions/s/workspace/annotation_validation/m.jpg"


@pytest.mark.asyncio
async def test_reuse_returns_previously_revised_annotations():
    from backend.models.schemas import ClusterAnnotation
    from backend.workflow.annotation_validation import compute_validation_attempt_id
    bio = SimpleNamespace(introspect_validation_metadata=_amock({"var": "adata", "cluster_key": "leiden"}))
    host = _Host(SimpleNamespace(), bio)
    aid = compute_validation_attempt_id(1, "liver", "mouse", "leiden", {"7": ["Nrxn1"]})
    prev = AnnotationConfirmationState(
        validation_status="completed", validation_attempt_id=aid, validation_summary="reclassified",
        cluster_annotations=[ClusterAnnotation(
            cluster_id="7", cell_type="Technical/Contamination", confidence="low",
            requires_reclustering=True, reclustering_reason="bimodal")])
    prelim = {"7": {"cell_type": "Hepatic neurons", "confidence": "medium"}}
    out, meta = await host._run_annotation_validation_round(
        _state(prev), prelim, {"7": ["Nrxn1"]}, "liver", "mouse",
        triage={}, database_priors={}, focused_evidence={})
    assert out["7"]["cell_type"] == "Technical/Contamination"
    assert out["7"]["requires_reclustering"] is True
    assert meta["validation_status"] == "completed" and meta["validation_summary"] == "reclassified"


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
