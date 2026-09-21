"""Decision-time reviews retain scope, versioned media, and session ownership."""
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from backend.agents.pi.reports import ReportsMixin
from backend.models.schemas import AnalysisState, ClusterAnnotation, CodeExecution, RoundResult
from backend.routes.reports_parts import html_views
from backend.services.report import stage_reviews
from backend.services.report.round_chapters import chapter_prompt, round_artifact_catalog
from backend.tests.test_round_report_chapters import _state, _chapter
from backend.tests.figure_image_helpers import write_valid_image
from backend.utils.image_utils import compress_image_to_base64


def _pi(ids):
    return SimpleNamespace(generate_stage_review_chapter=AsyncMock(return_value={
        "stage_artifact_ids": ids,
        "report_chapter": {"title": "Review", "blocks": [
            {"text": "Recorded methods and results.", "artifact_ids": ids},
        ], "conclusions": "Descriptive evidence."},
    }))


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["stage1", "stage2"])
@pytest.mark.parametrize("select_both", [False, True])
@pytest.mark.parametrize("corrupt_selected", [False, True])
async def test_review_preserves_all_same_named_versions(
    tmp_path, monkeypatch, stage, select_both, corrupt_selected,
):
    state = _state(tmp_path)
    old = state.publication_figures[1]
    current_path = Path(old.png_path)
    archived_path = tmp_path / ".revisions" / "old" / current_path.name
    archived_path.parent.mkdir(parents=True)
    current_path.rename(archived_path)
    old.png_path = str(archived_path)
    payload = write_valid_image(current_path, color=(220, 10, 10), min_bytes=2200)
    new = old.model_copy(update={
        "id": "new-baseline", "png_path": str(current_path),
        "render_sha256": hashlib.sha256(payload).hexdigest(),
        "description": "Selected updated baseline.",
    })
    state.publication_figures = [old, new]
    selected = [old.id, new.id] if select_both else [new.id]
    old_image = compress_image_to_base64(archived_path, max_width=1000, quality=82)
    new_image = compress_image_to_base64(current_path, max_width=1000, quality=82)
    assert old_image != new_image
    if corrupt_selected:
        # A valid older version must never substitute for corrupted selected bytes.
        current_path.write_bytes(archived_path.read_bytes())
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    await stage_reviews.build_stage_review(state, _pi(selected), stage)
    report = state.stage_review_reports[-1]
    assert report.status == "ready"
    assert {a.artifact_id for a in report.chapter.artifacts} == {old.id, new.id}
    assert old_image in report.html
    assert (new_image in report.html) is not corrupt_selected
    assert ("Recorded artifact unavailable:" in report.html) is corrupt_selected


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
async def test_failure_retains_all_artifacts_without_reusing_previous_report(tmp_path, monkeypatch):
    state = _state(tmp_path)
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    await stage_reviews.build_stage_review(state, _pi(["baseline"]), "stage2")
    old = state.stage_review_reports[-1].html
    pi = SimpleNamespace(generate_stage_review_chapter=AsyncMock(side_effect=RuntimeError("unavailable")))
    await stage_reviews.build_stage_review(state, pi, "stage2", summary="Retained finding")
    partial = state.stage_review_reports[-1]
    assert partial.status == "partial"
    assert {a.artifact_id for a in partial.chapter.artifacts} == {"baseline", "late_result"}
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
    pi.chat_without_history = AsyncMock(return_value='{"report_chapter": {}}')
    pi._safe_parse_json = json.loads
    catalog = round_artifact_catalog(state)
    await pi.generate_stage_review_chapter(state, "stage2", catalog, {})
    prompt = pi.chat_without_history.call_args.args[0]
    assert '"report_role": "foundation_annotation"' in prompt
    assert "the existing Stage 1 results" in prompt
    assert "regardless of creation time" in prompt
    final_prompt = chapter_prompt(catalog, state=state)
    assert "even when rendered during Stage 2" in final_prompt
    assert "do not move every UMAP" in final_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["stage1", "stage2"])
async def test_stage_contract_survives_large_context_compaction(tmp_path, stage):
    from backend.utils.context_budget import ContextBudget, compact_messages

    state = _state(tmp_path)
    pi = ReportsMixin()
    pi.refresh_runtime_config_if_needed = lambda: None
    pi._safe_parse_json = json.loads
    pi.chat_without_history = AsyncMock(return_value='{"report_chapter": {}}')
    await pi.generate_stage_review_chapter(state, stage, round_artifact_catalog(state), {
        "analysis_log": [{"code": "x" * 800_000}],
        "summary": "Retained summary",
        "full_round_chapter": {"conclusions": "Retained limitations"},
    })
    prompt = pi.chat_without_history.call_args.args[0]
    context = json.loads(prompt.split("Recorded review context:\n", 1)[1])
    assert context["analysis_log"]["omitted_chars"] > 700_000
    assert context["summary"] == "Retained summary"
    assert context["full_round_chapter"]["conclusions"] == "Retained limitations"
    messages, report = compact_messages([
        {"role": "system", "content": "Review recorded evidence."},
        {"role": "user", "content": prompt},
    ], ContextBudget(context_window=8192, completion_reserve=2048))
    # Also exercise shared fallback truncation when the remaining evidence grows.
    messages, report = compact_messages([
        messages[0], {"role": "user", "content": prompt + " evidence" * 100_000},
    ], ContextBudget(context_window=8192, completion_reserve=2048))
    assert any(action.kind == "fallback_truncate" for action in report.actions)
    sent = messages[1]["content"]
    assert "report_chapter" in sent
    assert f"Stage {1 if stage == 'stage1' else 2}: describe" in sent
    assert '"artifact_ids": ["exact catalog ID"]' in sent
    assert "Use every catalog artifact once" in sent


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_selection", [None, [], ["nonexistent"], ["late_result"]])
async def test_stage_selection_cannot_hide_existing_results(tmp_path, monkeypatch, legacy_selection):
    state = _state(tmp_path)
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    pi = ReportsMixin()
    pi.refresh_runtime_config_if_needed = lambda: None
    pi._safe_parse_json = json.loads
    raw = {
        "report_chapter": {"title": "Exploration", "blocks": [
            {"text": "Recorded finding", "artifact_ids": ["late_result"]},
        ]},
    }
    if legacy_selection is not None:
        raw["stage_artifact_ids"] = legacy_selection
    pi.chat_without_history = AsyncMock(return_value=json.dumps(raw))
    await stage_reviews.build_stage_review(state, pi, "stage2")
    assert pi.chat_without_history.await_count == 1
    report = state.stage_review_reports[-1]
    assert report.status == "ready"
    assert {a.artifact_id for a in report.chapter.artifacts} == {"late_result", "baseline"}
    assert "Original caption for late_result" in report.html
    assert "Original caption for baseline" in report.html


@pytest.mark.asyncio
async def test_stage_chapter_repair_is_bounded_and_preserves_all_results(tmp_path, monkeypatch):
    state = _state(tmp_path)
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    pi = ReportsMixin()
    pi.refresh_runtime_config_if_needed = lambda: None
    pi._safe_parse_json = json.loads
    pi.chat_without_history = AsyncMock(return_value='{}')
    await stage_reviews.build_stage_review(state, pi, "stage2", summary="Retained summary")
    assert pi.chat_without_history.await_count == 2
    assert "complete corrected JSON" in pi.chat_without_history.call_args.args[0]
    report = state.stage_review_reports[-1]
    assert report.status == "partial"
    assert {a.artifact_id for a in report.chapter.artifacts} == {"baseline", "late_result"}
    assert "Retained summary" in report.html
    assert report.html.count('<img ') == 2


@pytest.mark.asyncio
async def test_fabricated_reference_does_not_remove_real_results(tmp_path, monkeypatch):
    state = _state(tmp_path)
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    await stage_reviews.build_stage_review(state, _pi(["nonexistent"]), "stage2")
    report = state.stage_review_reports[-1]
    assert report.status == "partial"
    assert {a.artifact_id for a in report.chapter.artifacts} == {"baseline", "late_result"}
    assert report.html.count('<img ') == 2


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


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["stage1", "stage2"])
async def test_decision_review_discloses_unapproved_and_superseded_results(tmp_path, monkeypatch, stage):
    from backend.models.schemas import PublicationTable
    state = _state(tmp_path)
    older, newer = state.publication_figures
    older.review_status = "revision_requested"
    older.pi_feedback = "Revise the axis labels."
    older.unresolved_must_fix = ["Clarify the units"]
    older.is_current = False
    older.invalidation_reason = "Superseded by a later render"
    newer.review_status = "pending"
    newer.logical_figure_id = older.logical_figure_id = "same-result"
    newer.revision_of_id = newer.supersedes_id = older.id
    state.publication_tables = [PublicationTable(
        id="table", name="Measurements", description="All measured values",
        csv_path=str(tmp_path / "table.csv"), columns=["value"],
        preview_data=[{"value": 42}], round_number=1, review_status="pending",
    )]
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    await stage_reviews.build_stage_review(state, _pi([]), stage)
    report = state.stage_review_reports[-1]
    assert {a.artifact_id for a in report.chapter.artifacts} == {older.id, newer.id, "table"}
    assert report.html.count('<img ') == 2
    for text in ("revision_requested", "pending", "Clarify the units", "Revise the axis labels.",
                 "Later versions", "Superseded by a later render", "All measured values", "42"):
        assert text in report.html
    assert round_artifact_catalog(state) == []
