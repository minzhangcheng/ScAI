"""Decision-time reviews retain scope, versioned media, and session ownership."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from backend.agents.pi.reports import ReportsMixin
from backend.models.schemas import AnalysisState, ClusterAnnotation, RoundResult
from backend.routes.reports_parts import html_views
from backend.services.report import stage_reviews
from backend.services.report.round_chapters import chapter_prompt, round_artifact_catalog
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
    assert "Original caption for late_result" not in frozen
    pi = _pi(["late_result"])
    url2 = await stage_reviews.build_stage_review(state, pi, "stage2", round_result=result)
    second = state.stage_review_reports[-1]
    assert url1 != url2
    assert "Original caption for baseline" not in second.html
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
async def test_failure_never_reuses_previous_or_cumulative_report(tmp_path, monkeypatch):
    state = _state(tmp_path)
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    await stage_reviews.build_stage_review(state, _pi(["baseline"]), "stage2")
    old = state.stage_review_reports[-1].html
    pi = SimpleNamespace(generate_stage_review_chapter=AsyncMock(side_effect=RuntimeError("unavailable")))
    await stage_reviews.build_stage_review(state, pi, "stage2", summary="Retained finding")
    partial = state.stage_review_reports[-1]
    assert partial.status == "partial"
    assert partial.chapter.artifacts == []
    assert "Retained finding" in partial.html
    monkeypatch.setattr(stage_reviews, "render_detailed_report_document",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("render failed")))
    assert await stage_reviews.build_stage_review(state, _pi([]), "stage2") is None
    assert state.stage_review_reports[-1].status == "failed"
    assert state.stage_review_reports[0].html == old
    state.exploration_report_html_path = "stale.html"
    state.detailed_report_html_path = "cumulative.html"
    assert state.to_api_dict()["exploration_report_url"] is None


@pytest.mark.asyncio
async def test_pi_receives_stage_scope_and_late_annotation_role(tmp_path):
    import json
    state = _state(tmp_path)
    state.publication_figures[0].report_role = "foundation_annotation"
    pi = ReportsMixin()
    pi.refresh_runtime_config_if_needed = lambda: None
    pi.chat_without_history = AsyncMock(return_value='{"stage_artifact_ids": []}')
    pi._safe_parse_json = json.loads
    catalog = round_artifact_catalog(state)
    await pi.generate_stage_review_chapter(state, "stage2", catalog, {})
    prompt = pi.chat_without_history.call_args.args[0]
    assert '"report_role": "foundation_annotation"' in prompt
    assert "New downstream" in prompt
    assert "not creation time" in prompt
    final_prompt = chapter_prompt(catalog, state=state)
    assert "even when rendered during Stage 2" in final_prompt
    assert "do not move every UMAP" in final_prompt


@pytest.mark.asyncio
async def test_fabricated_stage_ids_produce_disclosed_partial_review(tmp_path, monkeypatch):
    state = _state(tmp_path)
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    await stage_reviews.build_stage_review(state, _pi(["nonexistent"]), "stage2")
    assert state.stage_review_reports[-1].status == "partial"
    assert state.stage_review_reports[-1].chapter.artifacts == []


def test_upstream_failure_clears_current_link_and_restored_urls(tmp_path):
    from backend.services.round_snapshots.restore import _strip_unavailable_report_urls
    state = _state(tmp_path)
    stage_reviews.record_stage_review_failure(state, "stage2")
    assert stage_reviews.latest_stage_review_url(state, "stage2") is None
    assert _strip_unavailable_report_urls({
        "html_url": f"/api/session/{state.id}/stage-review-report/version/html",
    })["html_url"] is None


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
    wf.bioinformatics_agent = SimpleNamespace(
        sandbox=object(), compute_cluster_group_proportions=AsyncMock(return_value={}),
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
    wf._run_supplementary_analysis = AsyncMock()
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


@pytest.mark.asyncio
@pytest.mark.parametrize("stage,field,endpoint", [
    ("stage1", "annotation_report_html_path", "annotation-report"),
    ("stage2", "exploration_report_html_path", "exploration-report"),
])
async def test_legacy_download_only_before_any_new_stage_review(tmp_path, monkeypatch, stage, field, endpoint):
    state = _state(tmp_path)
    path = tmp_path / f"{stage}.html"
    path.write_text("<!DOCTYPE html><html>Legacy</html>")
    setattr(state, field, str(path))
    api_field = "annotation_report_url" if stage == "stage1" else "exploration_report_url"
    assert state.to_api_dict()[api_field] == f"/api/session/{state.id}/{endpoint}/html"
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    await stage_reviews.build_stage_review(state, _pi([]), stage)
    assert "/stage-review-report/" in state.to_api_dict()[api_field]
    stage_reviews.record_stage_review_failure(state, stage)
    assert state.to_api_dict()[api_field] is None
