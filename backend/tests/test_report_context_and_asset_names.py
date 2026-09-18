"""Distinct registered renders and bounded saved-chapter writing context."""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.agents.supervisor import report_content, report_presentation, report_prompt
from backend.agents.supervisor_agent import SupervisorAgent
from backend.models.schemas import RoundResult
from backend.services.report.round_chapters import merge_reusable_sections, reuse_context
from backend.services.report.report_outline import build_report_outline
from backend.tests.figure_image_helpers import write_valid_image
from backend.tests.test_round_report_chapters import _state, _chapter


@pytest.mark.asyncio
async def test_distinct_same_stem_render_stays_inline_with_its_own_caption(tmp_path):
    state = _state(tmp_path)
    state.artifact_registry_version = 1
    state.round_results = [RoundResult(round_number=1, report_chapter=_chapter(state)),
                           RoundResult(round_number=2, phase="innovation", direction="Follow-up")]
    path = tmp_path / "baseline.png"
    payload = write_valid_image(path, color=(240, 10, 10), min_bytes=2200)
    new = state.publication_figures[1].model_copy(update={
        "id": "new-baseline", "name": "new_baseline", "png_path": str(path),
        "description": "New round two caption.", "round_number": 2,
        "render_sha256": hashlib.sha256(payload).hexdigest(),
    })
    state.publication_figures.append(new)
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.session_id = state.id
    agent.refresh_runtime_config_if_needed = lambda: None
    agent.get_supervision_log = lambda: []
    agent.get_summary_stats = lambda: {}
    agent._build_analysis_narrative = lambda *a: "Recorded observations."
    agent._get_report_system_prompt = lambda _: "Write a report."
    agent._resolve_context_window = lambda: 64_000
    agent._set_llm_context = lambda: None
    outline = build_report_outline(state)
    generated = "\n".join(f"## [{s.id}] {s.default_title}\nGlobal context." + (
        "\n[FIGURE: baseline.png]\n[FIGURE_CAPTION: Wrong model caption.]" if s.id == "direction_1" else ""
    ) for s in outline if s.id != "exploration_findings")
    agent._budgeted_text_call = AsyncMock(return_value=SimpleNamespace(content=generated))
    client = SimpleNamespace(set_context=lambda *a: None, chat=AsyncMock())
    report_prompt.bind_facade(report_content)
    ctx = await report_prompt.prepare_report_generation(agent, state, "model", str(tmp_path / "report"), str(tmp_path), None, client, False)
    assert "baseline.png" in ctx["writing_figures"]
    assert "baseline.jpg" not in ctx["writing_figures"]
    prompt = agent._budgeted_text_call.call_args.kwargs["messages"][1]["content"]
    line = next(line for line in prompt.splitlines() if line.startswith("- baseline.png"))
    assert "New round two caption." in line
    expected = copy.deepcopy(ctx["saved_sections"][0])
    agent._validate_report_content = lambda _: True
    await report_prompt.parse_and_place_report(agent, ctx, state, str(tmp_path), None, None, "model")
    direction = next(s for s in ctx["sections"] if s["id"] == "direction_1")
    assert "[FIGURE: baseline.png]" in direction["content"]
    assert "New round two caption." in direction["content"]
    assert next(s for s in ctx["sections"] if s["id"] == "exploration_findings")["content"] == expected["content"]
    report_presentation.bind_facade(report_content)
    await report_presentation.build_report_figures(agent, ctx, state)
    assert next(f for f in ctx["report_figures"] if f["filename"] == "baseline.png")["description"] == "New round two caption."
    result = agent._generate_html_report(
        {"title": "Study", "figures": ctx["report_figures"], "tables": [], "_analysis_state": state,
         "round_chapter_captions": ctx["saved_figure_captions"]},
        ctx["filtered_sections"], ctx["abstract"], ctx["conclusions"], ctx["report_figures"],
        ctx["figure_map"], ctx["stats"], "en", tmp_path / "report", tmp_path,
    ).read_text()
    direction_html = result.split('id="direction_1"', 1)[1].split('<div class="section"', 1)[0]
    assert 'alt="baseline.png"' in direction_html
    assert "New round two caption." in direction_html
    assert "Original caption for baseline." not in direction_html


def test_unknown_extension_alias_still_cannot_steal_saved_asset():
    saved = [{"id": "saved", "content": "[FIGURE: baseline.jpg]"}]
    generated = [{"id": "global", "content": "[figure: baseline.jpeg] [FIGURE: baseline.png]"}]
    merged = merge_reusable_sections(generated, saved, figure_names=["baseline.jpg", "baseline.png"])
    assert "baseline.jpeg" not in merged[0]["content"]
    assert "[FIGURE: baseline.png]" in merged[0]["content"]


@pytest.mark.parametrize("limit", [0, 64, 1500, 6000])
def test_budgeted_excerpts_never_mutate_or_reorder_saved_chapters(limit):
    sections = [{"id": f"round_{n}", "title": f"Round {n}",
                 "content": f"Question {n}. " + "Evidence. " * 10000 + f"Conclusion {n}."} for n in (1, 2, 3)]
    before = copy.deepcopy(sections)
    text = reuse_context(sections, max_chars=limit)
    assert len(text) <= limit
    assert sections == before
    if limit >= 1500:
        assert text.index("round_1") < text.index("round_2") < text.index("round_3")
        for n in (1, 2, 3):
            assert f"Question {n}." in text and f"Conclusion {n}." in text
        assert "excerpts" in text and "verbatim" in text


def test_budget_uses_configured_window_and_remaining_prompt_space():
    agent = SimpleNamespace(session_id="s", _resolve_context_window=lambda: 64_000,
                            _get_report_system_prompt=lambda _: "System")
    sections = [{"id": "r1", "title": "Findings", "content": "Evidence " * 100_000}]
    text, roomy = report_prompt._budget_saved_context(agent, sections, "Short instructions", "en")
    tight_text, tight = report_prompt._budget_saved_context(agent, sections, "X" * 160_000, "en")
    assert roomy["truncated"] and tight["truncated"]
    assert roomy["original_chars"] == len(reuse_context(sections))
    assert len(text) == roomy["sent_chars"] <= roomy["budget_chars"]
    assert len(tight_text) == tight["sent_chars"] <= tight["budget_chars"] < roomy["budget_chars"]
    small = [{"id": "r1", "title": "Findings", "content": "Short chapter."}]
    unchanged, observation = report_prompt._budget_saved_context(agent, small, "Short", "en")
    assert unchanged == reuse_context(small)
    assert not observation["truncated"]
