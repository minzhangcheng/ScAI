"""Regressions for supplementary reports and immutable chapter composition."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.supervisor_agent import SupervisorAgent
from backend.agents.supervisor import report_content, report_prompt
from backend.models.schemas import AnalysisState, DetailedReportSegment, RoundResult
from backend.services.report.round_chapters import (
    assemble_round_chapter, chapter_markdown, reusable_sections, round_artifact_catalog,
    saved_figure_captions,
)
from backend.services.report.report_outline import build_report_outline
from backend.services.report.variants_parts.detailed import (
    generate_detailed_report_segment_html, render_detailed_report_document,
    _deduplicate_detailed_figure_blocks,
)
from backend.tests.test_round_report_chapters import _state, _chapter
from backend.tests.test_pi_synthesis_cross_modal import _make_wf_for_synthesis


def test_unstructured_media_is_neutralized_without_losing_prose(tmp_path):
    state = _state(tmp_path)
    chapter = assemble_round_chapter({
        "title": "Evidence",
        "blocks": [{"text": "First finding. ![Extra image](image.jpg) Still useful.",
                    "artifact_ids": ["baseline"]},
                   {"text": "Second finding. <img src='image.jpg'> Still retained.",
                    "artifact_ids": ["late_result"]}],
        "conclusions": "Qualified conclusion. [figure: baseline.jpg] [TABLE_CAPTION: Injected]",
    }, round_artifact_catalog(state), "Summary", 1)
    assert chapter.source == "pi"
    assert "First finding." in chapter.blocks[0].text
    assert "Second finding." in chapter.blocks[1].text
    assert "Still retained." in chapter.blocks[1].text
    assert chapter.conclusions == "Qualified conclusion.  "
    assert chapter.assembly_notes
    rendered = chapter_markdown(chapter, {"baseline": "baseline.jpg", "late_result": "late_result.jpg"})
    assert rendered.count("[FIGURE:") == 2
    assert "[figure:" not in rendered and "<img" not in rendered and "![" not in rendered


def test_saved_and_supplementary_figures_are_embedded_once(monkeypatch, tmp_path):
    state = _state(tmp_path)
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    result = RoundResult(round_number=1, report_chapter=_chapter(state))
    primary = DetailedReportSegment(phase="exploration_complete", title="Round", round_number=1,
        content_html=generate_detailed_report_segment_html(state, "exploration_complete", "Round", round_result=result))
    supplementary = DetailedReportSegment(phase="supplementary_round_0", title="Supplement", round_number=1,
        content_html=generate_detailed_report_segment_html(state, "supplementary_round_0", "Supplement"))
    assert supplementary.content_html.count('<img ') == 2
    state.round_results = [result]
    state.detailed_report_segments = [primary, supplementary]
    for html in (render_detailed_report_document(state),
                 render_detailed_report_document(state, segment=primary, supplements=[supplementary])):
        assert html.count('<img ') == 2
        assert html.count("Original caption for baseline.") == 1
        assert html.index("First establish") < html.index('data-figure-key="baseline"') < html.index("Then interpret")


def test_dedup_removes_whole_nested_figure_block():
    block = '<div class="figure-item" data-figure-key="x"><img src="x"><div role="note">Disclosure</div></div>'
    assert _deduplicate_detailed_figure_blocks(block + '<p>Next</p>' + block) == block + '<p>Next</p>'


@pytest.mark.asyncio
async def test_annotation_supplement_is_linked_after_reload_and_first_synthesis(monkeypatch, tmp_path):
    from backend.workflow.core_parts import supplementary
    from backend.routes.reports_parts import html_views

    state = AnalysisState(user_id="u", current_round=1)
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    monkeypatch.setattr("backend.config.get_session_reports", lambda _: tmp_path / "reports")
    monkeypatch.setattr(supplementary, "_persist_request_map", AsyncMock())
    monkeypatch.setattr("backend.workflow.adata_checkpoint_hooks.capture_and_persist_boundary",
                        AsyncMock(return_value=SimpleNamespace(available=True)))
    wf = _make_wf_for_synthesis(state)
    del wf._build_and_save_detailed_segment
    wf._current_graph_state = {"analysis_state": state}
    wf.pi_agent = MagicMock()
    wf.pi_agent.plan_supplementary_analysis = AsyncMock(return_value={"literature_topics": ["Review evidence"]})
    wf.literature_agent = SimpleNamespace(process=AsyncMock(return_value={"report": "ANNOTATION SUPPLEMENT"}))
    await wf._run_supplementary_analysis(state, "Review evidence", "annotation")
    loaded = AnalysisState.model_validate_json(state.model_dump_json())
    assert loaded.detailed_report_segments[0].supplement_owner_key
    loaded.detailed_report_segments.append(DetailedReportSegment(
        phase="supplementary_annotation_foreign", title="Foreign", content_html="FOREIGN SUPPLEMENT",
        round_number=1, supplement_owner_key="other:exploration:1:",
    ))
    result = RoundResult(round_number=1, pi_summary="Completed exploration")
    loaded.round_results = [result]
    await wf._build_and_save_detailed_segment(loaded, "exploration_complete", "Round one", round_result=result)
    assert result.report_supplement_ids == [loaded.detailed_report_segments[0].id]
    monkeypatch.setattr(html_views, "require_session_ownership", AsyncMock(return_value=loaded))
    response = await html_views.get_round_report_html(loaded.id, result.report_segment_id,
        SimpleNamespace(state=SimpleNamespace(user_id="u")))
    assert "ANNOTATION SUPPLEMENT" in response.body.decode()
    assert "FOREIGN SUPPLEMENT" not in response.body.decode()


@pytest.mark.asyncio
async def test_fallback_keeps_saved_chapters_without_empty_outline_sections(tmp_path):
    state = _state(tmp_path)
    state.round_results = [RoundResult(round_number=1, report_chapter=_chapter(state)),
                           RoundResult(round_number=2, phase="innovation", direction="Legacy direction")]
    outline = build_report_outline(state)
    figures = {Path(f.png_path).name: Path(f.png_path) for f in state.publication_figures}
    saved = reusable_sections(state, outline, figures, [])
    fallback = [{"id": "analysis_summary", "title": "Retained steps", "content": "Degraded summary."}]
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.session_id = state.id
    agent._validate_report_content = lambda _: True
    agent._parse_report_sections = lambda *a, **kw: fallback
    report_prompt.bind_facade(report_content)
    ctx = {"_progress": AsyncMock(), "report_content": "Fallback", "language": "en",
           "outline": outline, "used_fallback": True, "attribution": {n: 2 for n in figures},
           "figure_map": figures, "figure_metadata_by_stem": {}, "client": SimpleNamespace(),
           "saved_sections": saved}
    await report_prompt.parse_and_place_report(agent, ctx, state, str(tmp_path), None, None, "model")
    assert [s["id"] for s in ctx["sections"]] == ["analysis_summary", "exploration_findings"]
    assert ctx["missing_sections"] == []
    assert ctx["sections"][1]["content"] == saved[0]["content"]


def test_exploration_only_guidance_does_not_request_unexecuted_directions(tmp_path):
    state = _state(tmp_path)
    state.round_results = [RoundResult(round_number=1, report_chapter=_chapter(state))]
    closing = next(s for s in build_report_outline(state) if s.id == "integrated_findings")
    assert closing.default_title == "Exploration Synthesis"
    assert "No innovation directions were executed" in closing.guidance
    assert "directions pursued" not in closing.guidance


@pytest.mark.parametrize("caption", ["", "Frozen caption."])
def test_live_drift_disclosures_survive_frozen_caption(monkeypatch, tmp_path, caption):
    state = _state(tmp_path)
    record = state.publication_figures[1]
    record.description = caption
    chapter = _chapter(state)
    record.expression_provenance = {"cohort_change_warning": "COHORT CHANGED", "representation_change_warning": "REPRESENTATION CHANGED"}
    record.description = "Changed live description. COHORT CHANGED"
    result = RoundResult(round_number=1, report_chapter=chapter)
    state.round_results = [result]
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    round_html = generate_detailed_report_segment_html(state, "exploration_complete", "Round", round_result=result)
    figures = {Path(f.png_path).name: Path(f.png_path) for f in state.publication_figures}
    saved = reusable_sections(state, build_report_outline(state), figures, [])
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.session_id = state.id
    out = tmp_path / "report"
    out.mkdir()
    path = agent._generate_html_report(
        {"title": "Study", "figures": [], "tables": [], "_analysis_state": state,
         "round_chapter_captions": saved_figure_captions(state, figures, [])},
        saved, "Abstract", "Conclusions", [], figures, {}, "en", out, tmp_path)
    for html in (round_html, path.read_text()):
        assert "COHORT CHANGED" in html and "REPRESENTATION CHANGED" in html
        assert "Changed live description." not in html
        if caption:
            assert caption in html


@pytest.mark.asyncio
async def test_badges_match_live_fallback_and_regenerated_reports(tmp_path):
    from backend.services.report.generator import DetailedReportGenerator
    from backend.services.report.regenerator import generate_html_report

    sections = [{"id": sid, "title": sid, "content": "Evidence", "analysis_type": "specific"}
                for sid in ("exploration_findings", "direction_1", "integrated_findings")]
    report = {"title": "Study", "sections": sections}
    (tmp_path / "report.json").write_text(json.dumps(report))
    regenerated = generate_html_report(tmp_path).read_text()
    generator = DetailedReportGenerator(object(), "pi", "bio", str(tmp_path / "fallback"))
    fallback = (await generator._generate_html_report(report, "en")).read_text()
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.session_id = "s"
    (tmp_path / "live").mkdir()
    live = agent._generate_html_report(report, sections, "", "", [], {}, {}, "en", tmp_path / "live", None).read_text()
    for html in (live, fallback, regenerated):
        for label in ("General / Exploration", "Specific / Innovation", "Integration"):
            assert html.count(f'>{label}</span>') == 2


def test_retry_lookup_is_pure_and_legacy_id_is_available():
    from backend.workflow.synthesis import _should_build_innovation_segment, _reusable_innovation_segment_id
    result = RoundResult(round_number=2, phase="innovation")
    segment = DetailedReportSegment(phase="innovation_round_2", title="Old", round_number=2)
    state = AnalysisState(user_id="u", current_round=2, round_results=[result], detailed_report_segments=[segment])
    assert not _should_build_innovation_segment(state, result)
    assert result.report_segment_id is None
    assert _reusable_innovation_segment_id(state, result) == segment.id


def test_legacy_filename_keys_deduplicate_with_gallery_stems():
    original = '<div class="figure-item" data-figure-key="baseline.jpg"><img src="x"></div>'
    gallery = '<div class="figure-item" data-figure-key="baseline"><img src="x"></div>'
    assert _deduplicate_detailed_figure_blocks(original + gallery) == original


def test_download_discloses_changes_after_chapter_was_saved(monkeypatch, tmp_path):
    state = _state(tmp_path)
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    result = RoundResult(round_number=1, report_chapter=_chapter(state))
    segment = DetailedReportSegment(phase="exploration_complete", title="Round", round_number=1,
        content_html=generate_detailed_report_segment_html(state, "exploration_complete", "Round", round_result=result))
    result.report_segment_id = segment.id
    state.round_results = [result]
    state.publication_figures[0].expression_provenance = {"cohort_change_warning": "LATER COHORT CHANGE"}
    html = render_detailed_report_document(state, segment=segment)
    assert "LATER COHORT CHANGE" in html
    assert "LATER COHORT CHANGE" not in segment.content_html


def test_empty_chapter_skips_unrelated_image_validation(monkeypatch, tmp_path):
    from backend.services.report.variants_parts import detailed
    from backend.models.schema_parts.round_report import RoundReportChapter, RoundReportBlock
    state = _state(tmp_path)
    monkeypatch.setattr(detailed, "resolved_record_figure_candidates", MagicMock(side_effect=AssertionError("Unrelated image scan")))
    result = RoundResult(round_number=1, report_chapter=RoundReportChapter(title="Text only", blocks=[RoundReportBlock(text="Synthesis")]))
    assert "Synthesis" in generate_detailed_report_segment_html(state, "integration_cycle_1", "Integration", round_result=result)


def test_same_named_distinct_renders_are_not_deduplicated():
    first = '<div class="figure-item" data-figure-key="baseline"><img src="original"></div>'
    revised = '<div class="figure-item" data-figure-key="baseline"><img src="revised"></div>'
    assert _deduplicate_detailed_figure_blocks(first + revised) == first + revised
