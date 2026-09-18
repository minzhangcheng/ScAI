"""Report ownership, revisions, and supplementary download regressions."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.models.schemas import AnalysisState, RoundResult, DetailedReportSegment
from backend.models.schema_parts.round_report import RoundReportChapter, RoundReportBlock
from backend.services.report.round_chapters import reusable_sections
from backend.services.report.report_outline import build_report_outline
from backend.workflow.synthesis import _should_build_innovation_segment, _upsert_round_result
from backend.tests.test_pi_synthesis_cross_modal import _make_wf_for_synthesis
from backend.routes.reports_parts import html_views


def chapter(text):
    return RoundReportChapter(title=text, blocks=[RoundReportBlock(text=text)])


def test_same_number_innovation_and_integration_keep_their_own_chapters():
    state = AnalysisState(user_id="u", round_results=[
        RoundResult(round_number=2, phase="innovation", direction="First direction", report_chapter=chapter("INNOVATION FINDINGS")),
        RoundResult(round_number=2, phase="integration", report_chapter=chapter("INTEGRATION SYNTHESIS")),
        RoundResult(round_number=3, phase="innovation", direction="Second direction", report_chapter=chapter("NEXT FINDINGS")),
    ])
    sections = {s["id"]: s for s in reusable_sections(state, build_report_outline(state), {}, [])}
    assert "INNOVATION FINDINGS" in sections["direction_1"]["content"], sections
    assert "INTEGRATION SYNTHESIS" in sections["integration_gate_1"]["content"]
    assert "INTEGRATION SYNTHESIS" not in sections["direction_1"]["content"]


def test_revised_round_chapter_must_not_reuse_stale_download():
    key = "main:innovation:2:direction"
    previous = DetailedReportSegment(phase="innovation_round_2", title="Old findings", content_html="OLD CHAPTER", result_identity_key=key)
    state = AnalysisState(user_id="u", current_round=2, detailed_report_segments=[previous])
    revised = RoundResult(round_number=2, phase="innovation", result_identity_key=key, report_chapter=chapter("REVISED CHAPTER"))
    assert _should_build_innovation_segment(state, revised), revised.report_segment_id


@pytest.mark.asyncio
async def test_supplementary_results_reach_the_round_download(monkeypatch, tmp_path):
    from backend.workflow.core_parts import supplementary

    state = AnalysisState(user_id="u", current_round=2)
    result = RoundResult(round_number=2, phase="innovation", result_identity_key="main:innovation:2:direction", report_chapter=chapter("ORIGINAL FINDINGS"))
    state.round_results = [result]
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    monkeypatch.setattr("backend.config.get_session_reports", lambda _: tmp_path / "reports")
    monkeypatch.setattr(supplementary, "_persist_request_map", AsyncMock())
    monkeypatch.setattr("backend.workflow.adata_checkpoint_hooks.capture_and_persist_boundary", AsyncMock(return_value=SimpleNamespace(available=True)))
    wf = _make_wf_for_synthesis(state)
    del wf._build_and_save_detailed_segment
    wf._current_graph_state = {"analysis_state": state}
    wf.pi_agent = MagicMock()
    wf.pi_agent.plan_supplementary_analysis = AsyncMock(return_value={"literature_topics": ["Reassess the evidence"]})
    wf.literature_agent = SimpleNamespace(process=AsyncMock(return_value={"report": "SUPPLEMENTARY NEW FINDING"}))
    await wf._build_and_save_detailed_segment(state, "innovation_round_2", "Round two", round_result=result)
    await wf._run_supplementary_analysis(state, "Check new evidence", "round")
    assert any("SUPPLEMENTARY NEW FINDING" in s.content_html for s in state.detailed_report_segments)
    loaded = AnalysisState.model_validate_json(state.model_dump_json())
    assert len(loaded.round_results[0].report_supplement_ids) == 1
    loaded.detailed_report_segments.append(DetailedReportSegment(
        phase="supplementary_round_foreign", title="Other branch", round_number=2,
        content_html="FOREIGN BRANCH FINDING",
    ))
    monkeypatch.setattr(html_views, "require_session_ownership", AsyncMock(return_value=loaded))
    response = await html_views.get_round_report_html(state.id, result.report_segment_id, SimpleNamespace(state=SimpleNamespace(user_id="u")))
    html = response.body.decode()
    assert html.index("ORIGINAL FINDINGS") < html.index("SUPPLEMENTARY NEW FINDING")
    assert "FOREIGN BRANCH FINDING" not in html


def test_same_phase_and_number_keeps_direction_and_branch_identity():
    results = [
        RoundResult(round_number=2, phase="innovation", direction="A", lineage_id="main", report_chapter=chapter("A MAIN")),
        RoundResult(round_number=2, phase="innovation", direction="B", lineage_id="main", report_chapter=chapter("B MAIN")),
        RoundResult(round_number=2, phase="innovation", direction="A", lineage_id="child", report_chapter=chapter("A CHILD")),
    ]
    state = AnalysisState(user_id="u", round_results=results)
    sections = reusable_sections(state, build_report_outline(state), {}, [])
    assert [s["content"] for s in sections] == ["A MAIN", "B MAIN", "A CHILD"]


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
