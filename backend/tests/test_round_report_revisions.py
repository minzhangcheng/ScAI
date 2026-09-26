"""Report ownership, revisions, and supplementary download regressions."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.models.schemas import AnalysisState, RoundResult, DetailedReportSegment
from backend.models.schema_parts.round_report import RoundReportChapter, RoundReportBlock
from backend.workflow.synthesis import _should_build_innovation_segment, _upsert_round_result
from backend.tests.test_round_report_download import _make_wf_for_synthesis
from backend.routes.reports_parts import html_views


def chapter(text):
    return RoundReportChapter(title=text, blocks=[RoundReportBlock(text=text)])


@pytest.mark.asyncio
async def test_revision_refreshes_stable_download_and_retains_supplements(monkeypatch, tmp_path):
    state = AnalysisState(user_id="u", current_round=2)
    result = RoundResult(round_number=2, phase="innovation", direction="direction", result_identity_key="main:innovation:2:direction", report_chapter=chapter("ORIGINAL"))
    state.round_results = [result]
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    monkeypatch.setattr("backend.config.get_session_reports", lambda _: tmp_path / "reports")
    wf = _make_wf_for_synthesis(state)
    del wf._build_and_save_detailed_segment
    await wf._build_and_save_detailed_segment(state, "innovation_round_2", "Round two", round_result=result)
    original_id = result.report_segment_id
    supplement = DetailedReportSegment(phase="supplementary_round_0", title="Supplement", content_html="Supplementary result", round_number=2)
    state.detailed_report_segments.append(supplement)
    result.report_supplement_ids = [supplement.id]
    assert not _should_build_innovation_segment(state, result)

    revised = _upsert_round_result(
        state.round_results,
        RoundResult(round_number=2, phase="innovation", direction="direction", report_chapter=chapter("REVISED")),
        lineage_id="main", branch_id=None, branch_label=None, restored_from_snapshot_id=None,
    )
    assert _should_build_innovation_segment(state, revised)
    await wf._build_and_save_detailed_segment(state, "innovation_round_2", "Round two", round_result=revised)
    assert revised.report_segment_id == original_id
    assert len(state.detailed_report_segments) == 2
    assert not _should_build_innovation_segment(state, revised)
    loaded = AnalysisState.model_validate_json(state.model_dump_json())
    monkeypatch.setattr(html_views, "require_session_ownership", AsyncMock(return_value=loaded))
    response = await html_views.get_round_report_html(state.id, original_id, SimpleNamespace(state=SimpleNamespace(user_id="u")))
    assert "REVISED" in response.body.decode()
    assert "ORIGINAL" not in response.body.decode()
    assert "Supplementary result" in response.body.decode()
