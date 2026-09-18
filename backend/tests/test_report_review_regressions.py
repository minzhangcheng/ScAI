"""Regression coverage for saved-chapter review findings and legacy loading."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.agents.pi.synthesis import SynthesisMixin
from backend.agents.supervisor import report_content, report_presentation, report_prompt
from backend.agents.supervisor_agent import SupervisorAgent
from backend.models.schemas import AnalysisPhase, AnalysisState, DetailedReportSegment, RoundResult
from backend.models.schema_parts.round_report import RoundReportBlock, RoundReportChapter
from backend.services.report.report_outline import build_report_outline
from backend.services.report.round_chapters import (
    merge_reusable_sections, reusable_sections, round_artifact_catalog, round_segment_index,
)
from backend.tests.figure_image_helpers import write_valid_image
from backend.tests.test_pi_synthesis_cross_modal import _make_wf_for_synthesis
from backend.tests.test_round_report_chapters import _chapter, _state
from backend.workflow.synthesis import _should_build_innovation_segment


def chapter(text="Retained findings"):
    return RoundReportChapter(title="Findings", blocks=[RoundReportBlock(text=text)])


def test_real_load_api_keeps_round_download_identifier():
    state = AnalysisState(user_id="u", round_results=[RoundResult(round_number=2, report_segment_id="saved")])
    loaded = AnalysisState.model_validate_json(state.model_dump_json())
    assert loaded.to_api_dict()["round_results"][0]["report_segment_id"] == "saved"
    assert AnalysisState(user_id="u", round_results=[RoundResult(round_number=1)]).to_api_dict()["round_results"][0]["report_segment_id"] is None


@pytest.mark.asyncio
async def test_legacy_segment_adopted_once_without_duplicate(monkeypatch, tmp_path):
    old = DetailedReportSegment(phase="innovation_round_2", round_number=2, title="Old", content_html="OLD")
    result = RoundResult(round_number=2, phase="innovation", report_chapter=chapter("NEW"))
    state = AnalysisState(user_id="u", current_round=2, detailed_report_segments=[old], round_results=[result])
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    monkeypatch.setattr("backend.config.get_session_reports", lambda _: tmp_path / "reports")
    wf = _make_wf_for_synthesis(state)
    del wf._build_and_save_detailed_segment
    assert _should_build_innovation_segment(state, result)
    await wf._build_and_save_detailed_segment(state, old.phase, "New", round_result=result)
    assert len(state.detailed_report_segments) == 1
    assert result.report_segment_id == old.id
    assert state.detailed_report_segments[0].result_identity_key
    assert state.detailed_report_segments[0].chapter_sha256
    assert "OLD" not in Path(state.detailed_report_html_path).read_text()
    assert not _should_build_innovation_segment(state, result)


@pytest.mark.parametrize("ambiguity", ["branch", "direction", "segments"])
def test_legacy_migration_does_not_guess_ambiguous_ownership(ambiguity):
    result = RoundResult(round_number=2, phase="innovation", direction="A", report_chapter=chapter())
    old = DetailedReportSegment(phase="innovation_round_2", round_number=2, title="Legacy")
    state = AnalysisState(user_id="u", round_results=[result], detailed_report_segments=[old])
    if ambiguity == "branch":
        state.restore_branch_id = "child"
        result.lineage_id = "child"
    elif ambiguity == "direction":
        state.round_results.append(RoundResult(round_number=2, phase="innovation", direction="B"))
    else:
        state.detailed_report_segments.append(old.model_copy(update={"id": "second"}))
    assert round_segment_index(state, old.phase, result) is None


@pytest.mark.parametrize("saved_kind,other_kind,saved_file,other_file", [
    ("FIGURE", "TABLE", "same.jpg", "same.csv"),
    ("TABLE", "FIGURE", "same.csv", "same.jpg"),
])
def test_reservations_are_separate_for_figures_and_tables(saved_kind, other_kind, saved_file, other_file):
    saved = [{"id": "saved", "content": f"[{saved_kind}: {saved_file}]"}]
    other_marker = f"[{other_kind}: {other_file}]\n[{other_kind}_CAPTION: Keep this caption]"
    generated = [{"id": "global", "content": other_marker + f"\n[{saved_kind}: {saved_file}]"}]
    merged = merge_reusable_sections(generated, saved)
    assert other_marker in merged[0]["content"]
    assert f"[{saved_kind}: {saved_file}]" not in merged[0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ['{"summary": "unfinished', '{"report_chapter": {"title": "Missing summary"}}', '[FIGURE: guessed.jpg] ```python invalid'])
async def test_unparsed_synthesis_is_not_published_as_a_chapter(response):
    state = AnalysisState(user_id="u", current_round=1)
    pi = SynthesisMixin()
    pi.refresh_runtime_config_if_needed = lambda: None
    pi.reset_history = lambda: None
    pi.chat = AsyncMock(return_value=response)
    def parse(text, default):
        try:
            return json.loads(text)
        except ValueError:
            return default
    pi._safe_parse_json = parse
    result = await pi.synthesize_round(state)
    state.round_results = [result]
    sections = reusable_sections(state, build_report_outline(state), {}, [])
    assert response not in result.pi_summary
    assert response not in str(sections)
    assert "could not be parsed" in sections[0]["content"]
    assert result.report_chapter.source == "fallback"


def test_new_single_pass_reuses_exploration_and_legacy_outline_is_retained():
    state = AnalysisState(user_id="u", round_results=[RoundResult(round_number=1, report_chapter=chapter())])
    ids = [s.id for s in build_report_outline(state)]
    assert "exploration_findings" in ids and "data_qc" not in ids
    state.round_results[0].report_chapter = None
    assert "data_qc" in [s.id for s in build_report_outline(state)]


def test_multiple_integration_rounds_have_one_final_section():
    state = AnalysisState(user_id="u", round_results=[
        RoundResult(round_number=n, phase="integration", report_chapter=chapter()) for n in (1, 2, 3)
    ])
    outline = build_report_outline(state)
    assert len({s.id for s in outline}) == len(outline)
    assert [s.rounds for s in outline if s.kind == "integration_gate"] == [(1,), (2,)]
    assert next(s for s in outline if s.kind == "integration_final").rounds == (3,)


def test_registered_round_zero_baseline_belongs_to_first_exploration(tmp_path):
    state = _state(tmp_path)
    state.publication_figures[0].round_number = 0
    assert "late_result" in {a.artifact_id for a in round_artifact_catalog(state)}
    state.current_round = 2
    state.analysis_phase = AnalysisPhase.INNOVATION
    assert "late_result" not in {a.artifact_id for a in round_artifact_catalog(state)}


@pytest.mark.asyncio
async def test_replacement_cannot_reenter_saved_chapter_through_placement_or_html(tmp_path):
    state = _state(tmp_path)
    state.artifact_registry_version = 1
    state.round_results = [RoundResult(round_number=1, report_chapter=_chapter(state))]
    original = state.publication_figures[1]
    payload = write_valid_image(Path(original.png_path), color=(230, 10, 20), min_bytes=2200)
    replacement = original.model_copy(update={"id": "replacement", "description": "Replacement caption", "render_sha256": hashlib.sha256(payload).hexdigest()})
    original.is_current = False
    state.publication_figures.append(replacement)
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.session_id = state.id
    agent.refresh_runtime_config_if_needed = lambda: None
    agent.get_supervision_log = lambda: []
    agent.get_summary_stats = lambda: {"total_steps_monitored": 0, "steps_with_technical_issues": 0, "pi_escalations": 0}
    agent._build_analysis_narrative = lambda *a: "Recorded analyses."
    agent._get_report_system_prompt = lambda _: "Write a report."
    agent._set_llm_context = lambda: None
    outline = build_report_outline(state)
    generated = "\n".join(f"## [{s.id}] {s.default_title}\nGlobal context." for s in outline if s.id != "exploration_findings")
    agent._budgeted_text_call = AsyncMock(return_value=SimpleNamespace(content=generated))
    client = SimpleNamespace(set_context=lambda *a: None, chat=AsyncMock())
    report_prompt.bind_facade(report_content)
    ctx = await report_prompt.prepare_report_generation(agent, state, "model", str(tmp_path / "report"), str(tmp_path), None, client, False)
    assert "baseline.jpg" in ctx["figure_map"]  # The replacement is separately registered.
    assert "baseline.jpg" not in ctx["writing_figures"]
    assert "baseline.jpg" in ctx["reserved_names"]
    expected = ctx["saved_sections"][0]["content"]
    assert "Recorded figure unavailable" in expected
    agent._validate_report_content = lambda _: True
    await report_prompt.parse_and_place_report(agent, ctx, state, str(tmp_path), None, None, "model")
    saved = next(s for s in ctx["sections"] if s["id"] == "exploration_findings")
    assert saved["content"] == expected
    assert "[FIGURE: baseline.jpg]" not in saved["content"]
    report_presentation.bind_facade(report_content)
    await report_presentation.build_report_figures(agent, ctx, state)
    path = agent._generate_html_report(
        {"title": "Study", "figures": ctx["report_figures"], "tables": [], "_analysis_state": state,
         "round_chapter_captions": ctx["saved_figure_captions"]},
        ctx["filtered_sections"], ctx["abstract"], ctx["conclusions"],
        ctx["report_figures"], ctx["figure_map"], ctx["stats"], "en", tmp_path / "report", tmp_path,
    )
    html = path.read_text()
    saved_html = html.split('id="exploration_findings"', 1)[1].split('<div class="section"', 1)[0]
    assert 'alt="baseline.jpg"' not in saved_html
    assert 'alt="late_result.jpg"' in saved_html
    assert 'id="supplementary_figures"' in html


@pytest.mark.asyncio
async def test_integration_summary_announces_the_ready_report():
    from backend.tests.test_integration_synthesis_node import _Harness, _state_with_batch

    workflow = _Harness([])
    state = _state_with_batch()
    async def save_segment(_state, _phase, _title, graph_state=None, round_result=None):
        round_result.report_segment_id = "ready-integration-report"
    workflow._build_and_save_detailed_segment = save_segment
    await workflow._integration_synthesis({"analysis_state": state})
    payload = next(data for kind, data in workflow.sent if kind == "round_summary")
    assert payload["report_segment_id"] == "ready-integration-report"
