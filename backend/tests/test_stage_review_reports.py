"""Decision-time reviews retain scope, versioned media, and session ownership."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from backend.models.schemas import AnalysisState, ClusterAnnotation, CodeExecution, RoundResult
from backend.routes.reports_parts import html_views
from backend.services.report import stage_reviews
from backend.services.report.round_chapters import round_artifact_catalog
from backend.tests.test_round_report_chapters import _state, _chapter


def _pi(ids):
    return SimpleNamespace(generate_stage_review_chapter=AsyncMock(return_value={
        "stage_artifact_ids": ids,
        "report_chapter": {"title": "Review", "blocks": [
            {"text": "Recorded methods and results.", "artifact_ids": ids},
        ], "conclusions": "Descriptive evidence."},
    }))


@pytest.mark.asyncio
async def test_reviews_freeze_independently_of_final_round_and_live_artifacts(tmp_path, monkeypatch):
    state = _state(tmp_path)
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    result = RoundResult(round_number=1, report_chapter=_chapter(state))
    state.round_results = [result]
    final_chapter = result.report_chapter.model_dump_json()
    pi = _pi(["baseline"])
    url1 = await stage_reviews.build_stage_review(state, pi, "stage1", annotations=[
        ClusterAnnotation(cluster_id="0", cell_type="Candidate <type>", key_markers=["M1"]),
    ])
    first = state.stage_review_reports[0]
    frozen = first.html
    assert 'data:image/' in frozen
    assert 'Candidate &lt;type&gt;' in frozen
    assert 'not accepted labels' in frozen
    assert "Original caption for late_result" in frozen
    pi = _pi(["late_result"])
    url2 = await stage_reviews.build_stage_review(state, pi, "stage2", round_result=result)
    second = state.stage_review_reports[-1]
    assert url1 != url2
    assert "Original caption for baseline" in second.html
    assert "Original caption for late_result" in second.html
    await stage_reviews.build_stage_review(state, pi, "stage2", round_result=result)
    assert state.stage_review_reports[-1].revision == 2
    assert result.report_chapter.model_dump_json() == final_chapter
    assert not state.detailed_report_segments
    for image in tmp_path.glob('*.jpg'):
        image.unlink()
    from backend.utils.blob_store import externalize_blobs, inline_blobs
    dumped = externalize_blobs(state.model_dump(mode="json"), tmp_path / "blobs")
    assert dumped["stage_review_reports"][0]["html"].startswith("@scai-blob:")
    loaded = AnalysisState.model_validate(inline_blobs(dumped, tmp_path / "blobs"))
    monkeypatch.setattr(html_views, "require_session_ownership", AsyncMock(return_value=loaded))
    request = SimpleNamespace(state=SimpleNamespace(user_id="u"))
    response = await html_views.get_stage_review_report_html(state.id, first.id, request)
    assert response.body.decode() == frozen
    assert response.headers["cache-control"] == "no-store"
    with pytest.raises(HTTPException) as error:
        await html_views.get_stage_review_report_html(state.id, first.id, request, branch_id="other")
    assert error.value.status_code == 404
    loaded.id = "child-session"
    assert stage_reviews.latest_stage_review_url(loaded, "stage1") is None
    with pytest.raises(HTTPException):
        await html_views.get_stage_review_report_html(loaded.id, first.id, request)


@pytest.mark.asyncio
async def test_annotation_gate_publishes_frozen_versions_before_both_decisions(tmp_path, monkeypatch):
    import asyncio
    from collections import deque
    from backend.workflow import annotation
    from backend.workflow.annotation_parts import callback

    state = _state(tmp_path)
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    wf = annotation.AnnotationMixin()
    wf.session_id = state.id
    wf.pi_agent = _pi(["baseline"])
    wf.pi_agent.review_cluster_annotations = AsyncMock(return_value={})
    receipt = CodeExecution(
        id="baseline-execution", code="measure()", round_number=1, plan_id="stage1-plan",
        plan_step_ids=[1], plan_binding_status="bound", evidence_schema_version=2,
        execution_status="succeeded", executor_run_kind="ordinary",
    )
    # An earlier attempt with identical code must not replace the actual source ID.
    state.code_history = [receipt.model_copy(update={"id": "earlier-attempt"})]
    for figure in state.publication_figures:
        figure.provenance_schema_version = 2
        figure.plan_id = "stage1-plan"
        figure.plan_step_id = 1
        figure.source_execution_id = f"{figure.id}-execution"
        figure.source_execution_validated = True
        figure.source_execution_run_kind = "ordinary"
        figure.producer_run_kind = "ordinary"
    assert round_artifact_catalog(state) == []
    wf.bioinformatics_agent = SimpleNamespace(
        sandbox=object(), compute_cluster_group_proportions=AsyncMock(return_value={}),
        code_history=[receipt, receipt],
    )
    preview = AsyncMock(side_effect=[
        {"image": "data:image/png;base64,first", "note": "First candidate preview"},
        {"image": "data:image/png;base64,second", "note": "Revised candidate preview"},
    ])
    monkeypatch.setattr(
        "backend.agents.bioinformatics.annotation_preview.render_candidate_annotation_preview", preview,
    )
    annotations = {"0": {"cell_type": "Candidate", "key_markers": ["M1"]}}
    wf._run_annotation_validation_round = AsyncMock(return_value=(annotations, {
        "validation_summary": "Review the uncertain cluster assignment.",
    }))
    wf._current_graph_state = {"bioinformatics_result": {"findings": ["Recorded result"]}}
    wf._build_and_save_detailed_segment = AsyncMock()
    wf._annotation_event = asyncio.Event()
    wf._supplementary_queue = deque([("Validate candidates", "annotation")])
    async def supplement(*args, **kwargs):
        wf.bioinformatics_agent.code_history.append(receipt.model_copy(update={"id": "late_result-execution"}))
        wf.pi_agent.generate_stage_review_chapter.return_value = {
            "stage_artifact_ids": ["baseline", "late_result"],
            "report_chapter": {"title": "Updated review", "blocks": [
                {"text": "Additional evidence", "artifact_ids": ["baseline", "late_result"]},
            ]},
        }

    wf._run_supplementary_analysis = AsyncMock(side_effect=supplement)
    wf._send_state_sync = AsyncMock()
    wf._set_status = lambda st, status: setattr(st, "status", status)
    published = []

    async def send(kind, content, agent):
        if kind == "annotation_confirmation":
            latest = state.stage_review_reports[-1]
            assert "Review the uncertain cluster assignment." in latest.html
            assert content["report_html_url"].endswith(f"/{latest.id}/html")
            assert state.annotation_state.pending
            published.append(content["report_html_url"])

    wf._send_message = send
    monkeypatch.setattr(annotation, "wait_with_skip_time", AsyncMock(side_effect=[True, False]))
    callback.bind_facade(annotation)
    monkeypatch.setattr(callback, "_capture_annotation_snapshot", AsyncMock())
    await callback.review_and_confirm_annotations(wf, {
        "state": state, "result": {}, "annotations": annotations,
    }, {"0": ["M1"]}, "", "", annotation_source={"adata_var": "chosen", "cluster_key": "groups"})
    assert preview.await_count == 2
    assert preview.call_args.args[1] == {"adata_var": "chosen", "cluster_key": "groups"}
    assert "First candidate preview" in state.stage_review_reports[0].html
    assert "Revised candidate preview" in state.stage_review_reports[1].html
    assert "Revised candidate preview" not in state.stage_review_reports[0].html
    assert all(a.artifact_id in {"baseline", "late_result"}
               for r in state.stage_review_reports for a in r.chapter.artifacts)
    assert len(published) == 2
    assert published[0] != published[1]
    assert [r.revision for r in state.stage_review_reports] == [1, 2]
    assert state.to_api_dict()["annotation_report_url"] == published[-1]
    assert state.annotation_state.confirmed
    assert {a.artifact_id for a in state.stage_review_reports[0].chapter.artifacts} == {"baseline", "late_result"}
    assert {a.artifact_id for a in state.stage_review_reports[1].chapter.artifacts} == {"baseline", "late_result"}
    assert [r.id for r in state.code_history] == ["earlier-attempt", "baseline-execution", "late_result-execution"]
