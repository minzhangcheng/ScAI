"""Single-round exports retain their settled content and branch ownership."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from backend.models.schemas import AnalysisPhase, AnalysisState, DetailedReportSegment, FigureReviewStatus, RoundResult
from backend.routes.reports_parts import html_views
from backend.routes.serialization import serialize_round_results
from backend.tests.test_pi_synthesis_cross_modal import _make_wf_for_synthesis
from backend.tests.test_round_report_chapters import _chapter, _state
from backend.workflow.synthesis import _should_build_innovation_segment, _reusable_innovation_segment_id


@pytest.mark.asyncio
async def test_round_ready_before_feedback_and_download_survives_reload(monkeypatch, tmp_path):
    state = _state(tmp_path)
    state.analysis_phase = AnalysisPhase.INNOVATION
    state.current_round = 2
    state.current_round_direction = "Measured response"
    for figure in state.publication_figures:
        figure.round_number = 2
    result = RoundResult(round_number=2, pi_summary="Summary", report_chapter=_chapter(state))
    # Human review retains a pending revision omitted from the publication chapter.
    rejected = state.publication_figures[0].model_copy(update={
        "id": "pending-revision", "review_status": FigureReviewStatus.PENDING, "pi_feedback": "Needs review",
        "revision_of_id": state.publication_figures[0].id,
        "description": "PENDING REVIEW RESULT",
    })
    state.publication_figures.append(rejected)
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    monkeypatch.setattr("backend.config.get_session_reports", lambda _: tmp_path / "reports")
    wf = _make_wf_for_synthesis(state)
    del wf._build_and_save_detailed_segment
    wf._compact_checkpoints = AsyncMock()
    wf.pi_agent = MagicMock()
    wf.pi_agent.synthesize_round = AsyncMock(return_value=result)
    wf.pi_agent.build_cross_modal_synthesis = AsyncMock(return_value=None)
    wf.pi_agent.identify_data_needs = AsyncMock(return_value=[])
    await wf._pi_synthesis({"analysis_state": state, "bioinformatics_result": {"findings": ["Observed"]}})
    payload = next(c.args[1] for c in wf._send_message.call_args_list if c.args[0] == "round_summary")
    segment_id = payload["report_segment_id"]
    assert segment_id
    assert state.round_results[0].report_segment_id == segment_id
    assert serialize_round_results(state.round_results)[0]["report_segment_id"] == segment_id

    from backend.utils.blob_store import externalize_blobs, inline_blobs
    saved_payload = externalize_blobs(state.model_dump(mode="json"), tmp_path / "blobs")
    loaded = AnalysisState.model_validate(inline_blobs(saved_payload, tmp_path / "blobs"))
    loaded.detailed_report_segments.append(DetailedReportSegment(
        phase="later", title="Later round", content_html="UNRELATED ROUND", round_number=3,
    ))
    # The download uses the frozen HTML, including embedded images, even after
    # workspace outputs disappear or later metadata changes.
    for figure in loaded.publication_figures:
        Path(figure.png_path).unlink(missing_ok=True)
    owner = AsyncMock(return_value=loaded)
    monkeypatch.setattr(html_views, "require_session_ownership", owner)
    response = await html_views.get_round_report_html(
        loaded.id, segment_id, SimpleNamespace(state=SimpleNamespace(user_id="u")),
    )
    owner.assert_awaited_once_with(session_id=loaded.id, user_id="u")
    html = response.body.decode()
    assert "Round 2 Analysis Report" in html
    assert html.index("First establish the baseline") < html.index("Then interpret the result")
    assert "Original caption for baseline." in html
    assert "data:image/" in html
    assert "UNRELATED ROUND" not in html
    assert "PENDING REVIEW RESULT" in html
    assert html.count("<img ") == 3
    assert "Needs review" in html
    saved = next(s for s in loaded.detailed_report_segments if s.id == segment_id)
    assert "PENDING REVIEW RESULT" not in saved.content_html
    assert "PENDING REVIEW RESULT" in saved.decision_content_html
    assert 'filename="round_2_report.html"' in response.headers["content-disposition"]


@pytest.mark.asyncio
async def test_round_download_requires_matching_branch_and_saved_segment(monkeypatch):
    segment = DetailedReportSegment(id="saved", phase="innovation_round_2", title="Round", content_html="MAIN", round_number=2)
    state = AnalysisState(user_id="u", restore_branch_id="child", detailed_report_segments=[segment], round_results=[
        RoundResult(round_number=2, phase="innovation", lineage_id="main", report_segment_id=segment.id),
        RoundResult(round_number=2, phase="innovation", lineage_id="child"),
    ])
    monkeypatch.setattr(html_views, "require_session_ownership", AsyncMock(return_value=state))
    request = SimpleNamespace(state=SimpleNamespace(user_id="u"))
    with pytest.raises(HTTPException) as error:
        await html_views.get_round_report_html(state.id, segment.id, request)
    assert error.value.status_code == 404
    response = await html_views.get_round_report_html(state.id, segment.id, request, branch_id="main")
    assert "MAIN" in response.body.decode()
    state.detailed_report_segments.clear()
    with pytest.raises(HTTPException) as error:
        await html_views.get_round_report_html(state.id, segment.id, request, branch_id="main")
    assert error.value.status_code == 404
    monkeypatch.setattr(html_views, "require_session_ownership", AsyncMock(side_effect=HTTPException(403)))
    with pytest.raises(HTTPException) as error:
        await html_views.get_round_report_html(state.id, segment.id, request, branch_id="main")
    assert error.value.status_code == 403


def test_retry_reuses_identity_bound_report_without_crossing_branches():
    key = "main:innovation:2:direction"
    segment = DetailedReportSegment(phase="innovation_round_2", title="Round", result_identity_key=key)
    state = AnalysisState(user_id="u", current_round=2, detailed_report_segments=[segment])
    retry = RoundResult(round_number=2, phase="innovation", result_identity_key=key)
    assert not _should_build_innovation_segment(state, retry)
    assert retry.report_segment_id is None
    assert _reusable_innovation_segment_id(state, retry) == segment.id
    other = RoundResult(round_number=2, phase="innovation", result_identity_key="child:innovation:2:direction")
    assert _should_build_innovation_segment(state, other)
    assert other.report_segment_id is None
