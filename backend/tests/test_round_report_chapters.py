"""Round synthesis owns narrative order; final reports reuse it without rewriting."""
import hashlib
import json
from unittest.mock import AsyncMock

import pytest

from backend.agents.pi.synthesis import SynthesisMixin
from backend.models.schemas import (
    AnalysisState, FigureReviewStatus,
    PublicationFigure,
)
from backend.services.report.round_chapters import (
    assemble_round_chapter, report_artifact_names,
    round_artifact_catalog,
)
from backend.tests.figure_image_helpers import write_valid_image
from backend.utils.llm_client import UpstreamExhaustedError


def _state(tmp_path):
    figures = []
    for index, name in enumerate(("late_result", "baseline")):
        path = tmp_path / f"{name}.jpg"
        payload = write_valid_image(path, color=(30 + index * 80, 90, 140), min_bytes=2200)
        figures.append(PublicationFigure(
            id=name, name=name, description=f"Original caption for {name}.",
            figure_type="other", png_path=str(path), pdf_path="",
            render_sha256=hashlib.sha256(payload).hexdigest(), pdf_sha256="2" * 64,
            review_status=FigureReviewStatus.APPROVED, round_number=1,
        ))
    return AnalysisState(user_id="u", name="Study", current_round=1, publication_figures=figures)


def _chapter(state):
    return assemble_round_chapter({
        "title": "Evidence in context",
        "blocks": [
            {"text": "First establish the baseline.", "artifact_ids": ["baseline"]},
            {"text": "Then interpret the result.", "artifact_ids": ["late_result"]},
        ],
        "conclusions": "The evidence is descriptive.",
    }, round_artifact_catalog(state), "Summary.", 1)


@pytest.mark.asyncio
async def test_pi_synthesis_composes_and_persists_chapter_in_one_call(tmp_path):
    state = _state(tmp_path)
    pi = SynthesisMixin()
    pi.refresh_runtime_config_if_needed = lambda: None
    pi.reset_history = lambda: None
    pi._safe_parse_json = lambda text, default: json.loads(text)
    pi._extract_next_steps = lambda data: []
    chapter = _chapter(state)
    pi.chat = AsyncMock(return_value=json.dumps({
        "summary": "Summary.", "report_chapter": chapter.model_dump(),
    }))
    result = await pi.synthesize_round(state, bioinformatics_result="Measured observations.")
    pi.chat.assert_awaited_once()
    prompt = pi.chat.call_args.args[0]
    assert "catalog order" in prompt
    assert "Original caption for baseline." in prompt
    state.round_results = [result]
    loaded = AnalysisState.model_validate_json(state.model_dump_json())
    assert loaded.round_results[0].report_chapter == chapter
    assert loaded.round_results[0].report_chapter.blocks[0].artifact_ids == ["baseline"]


@pytest.mark.asyncio
async def test_pi_synthesis_decodes_json_string_chapter_without_retry(tmp_path):
    state = _state(tmp_path)
    pi = SynthesisMixin()
    pi.refresh_runtime_config_if_needed = lambda: None
    pi.reset_history = lambda: None
    pi._safe_parse_json = lambda text, default: json.loads(text)
    pi._extract_next_steps = lambda data: []
    raw_chapter = {
        "title": "Recovered chapter",
        "blocks": [{"text": "Measured result.", "artifact_ids": ["baseline"]}],
    }
    pi.chat = AsyncMock(return_value=json.dumps({
        "summary": "Original summary.", "report_chapter": json.dumps(raw_chapter),
    }))

    result = await pi.synthesize_round(state)

    pi.chat.assert_awaited_once()
    assert result.pi_summary == "Original summary."
    assert result.report_chapter.source == "pi"
    assert result.report_chapter.title == "Recovered chapter"


@pytest.mark.asyncio
async def test_pi_synthesis_retries_invalid_chapter_with_changed_prompt(tmp_path):
    state = _state(tmp_path)
    pi = SynthesisMixin()
    pi.refresh_runtime_config_if_needed = lambda: None
    pi.reset_history = lambda: None
    pi._safe_parse_json = lambda text, default: json.loads(text)
    pi._extract_next_steps = lambda data: []
    pi.chat = AsyncMock(side_effect=[
        json.dumps({"summary": "Original summary.", "report_chapter": "{invalid"}),
        json.dumps({"report_chapter": {
            "title": "Repaired chapter",
            "blocks": [{"text": "Measured result.", "artifact_ids": ["baseline"]}],
        }}),
    ])

    result = await pi.synthesize_round(state)

    assert pi.chat.await_count == 2
    assert pi.chat.await_args_list[0].args[0] != pi.chat.await_args_list[1].args[0]
    assert "Do not encode report_chapter as a string" in pi.chat.await_args_list[1].args[0]
    assert result.pi_summary == "Original summary."
    assert result.report_chapter.source == "pi"
    assert result.report_chapter.title == "Repaired chapter"


@pytest.mark.asyncio
async def test_pi_synthesis_keeps_fallback_after_one_invalid_repair(tmp_path):
    state = _state(tmp_path)
    pi = SynthesisMixin()
    pi.refresh_runtime_config_if_needed = lambda: None
    pi.reset_history = lambda: None
    pi._safe_parse_json = lambda text, default: json.loads(text)
    pi._extract_next_steps = lambda data: []
    pi.chat = AsyncMock(side_effect=[
        json.dumps({"summary": "Original summary.", "report_chapter": "{invalid"}),
        json.dumps({"report_chapter": "{still invalid"}),
    ])

    result = await pi.synthesize_round(state)

    assert pi.chat.await_count == 2
    assert result.pi_summary == "Original summary."
    assert result.report_chapter.source == "fallback"
    assert result.report_chapter.title == "Round 1 Findings"


@pytest.mark.asyncio
async def test_pi_synthesis_keeps_original_summary_when_repair_call_fails(tmp_path):
    state = _state(tmp_path)
    pi = SynthesisMixin()
    pi.refresh_runtime_config_if_needed = lambda: None
    pi.reset_history = lambda: None
    pi._safe_parse_json = lambda text, default: json.loads(text)
    pi._extract_next_steps = lambda data: []
    pi.chat = AsyncMock(side_effect=[
        json.dumps({"summary": "Original summary.", "report_chapter": "{invalid"}),
        UpstreamExhaustedError("repair unavailable"),
    ])

    result = await pi.synthesize_round(state)

    assert pi.chat.await_count == 2
    assert result.pi_summary == "Original summary."
    assert result.report_chapter.source == "fallback"


def test_saved_reference_cannot_borrow_a_same_named_replacement(tmp_path):
    state = _state(tmp_path)
    original = state.publication_figures[1]
    replacement = tmp_path / "replacement" / "baseline.jpg"
    replacement.parent.mkdir()
    write_valid_image(replacement, color=(220, 10, 10), min_bytes=2200)
    names = report_artifact_names(state, {"baseline.jpg": replacement}, [])
    assert original.id not in names
