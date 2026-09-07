"""Focused contracts for non-verdict graphic-abstract review outcomes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, create_autospec
from datetime import datetime
import logging
import uuid

import pytest

from backend.agents.pi_agent import PIAgent
from backend.models.schemas import AnalysisState, GraphicAbstract
from backend.models.schema_parts.outputs import GraphicAbstract as OutputGraphicAbstract
from backend.workflow.reporting_parts.graphic_abstract import GraphicAbstractMixin
from backend.utils.llm_client import OpenAICompatibleClient


def _agent(*, vision=True, response=None, vision_side_effect=None):
    vision_call = create_autospec(
        OpenAICompatibleClient, instance=True
    ).chat_with_vision
    if response is not None:
        vision_call.return_value = SimpleNamespace(content=response)
    if vision_side_effect is not None:
        vision_call.side_effect = vision_side_effect
    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.2
    provider.get_vision_setting.return_value = vision
    provider.get_llm_client.return_value = SimpleNamespace(
        chat_with_vision=vision_call,
        set_context=lambda *_args: None,
    )
    agent = PIAgent(settings_provider=provider)
    agent.set_session("graphic-abstract-availability")
    agent._set_llm_context = lambda: None
    return agent, vision_call


def _state():
    return AnalysisState(user_id="u", name="Project", initial_instruction="Analyze")


@pytest.mark.asyncio
async def test_graphic_abstract_review_without_vision_is_an_explicit_non_verdict():
    agent, _vision = _agent(vision=False)

    result = await agent.review_graphic_abstract(_state(), "aW1n", [])

    assert result["review_unavailable"] is True
    assert result["quality_score"] is None
    assert result["feedback"]
    assert result["strengths"] == []
    assert result["issues"] == []
    assert result["suggestions"] == []
    assert result["must_fix"] == []


@pytest.mark.asyncio
async def test_graphic_abstract_review_exception_is_an_explicit_non_verdict():
    agent, _vision = _agent(
        vision_side_effect=RuntimeError("vision transport unavailable")
    )

    result = await agent.review_graphic_abstract(_state(), "aW1n", [])

    assert result["review_unavailable"] is True
    assert result["quality_score"] is None
    assert result["strengths"] == []
    assert result["issues"] == []
    assert result["suggestions"] == []
    assert result["must_fix"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    ["", "not-json", '{"feedback":"missing score"}', '{"quality_score":11,"feedback":"bad"}'],
)
async def test_malformed_graphic_abstract_review_is_not_scored(response):
    agent, _vision = _agent(response=response)

    result = await agent.review_graphic_abstract(_state(), "aW1n", [])

    assert result["review_unavailable"] is True
    assert result["quality_score"] is None
    assert result["strengths"] == []
    assert result["issues"] == []
    assert result["suggestions"] == []
    assert result["must_fix"] == []


@pytest.mark.asyncio
async def test_valid_graphic_abstract_review_keeps_feedback_and_score():
    agent, _vision = _agent(
        response=(
            '{"quality_score":8,"feedback":"clear",'
            '"strengths":["flow"],"issues":[],"suggestions":["spacing"],'
            '"must_fix":[]}'
        )
    )

    result = await agent.review_graphic_abstract(_state(), "aW1n", [])

    assert result["review_unavailable"] is False
    assert result["quality_score"] == 8
    assert result["feedback"] == "clear"
    assert result["strengths"] == ["flow"]


@pytest.mark.asyncio
async def test_selection_failure_returns_none_without_fabricated_index():
    agent, _vision = _agent(vision=False)

    assert await agent.select_best_graphic_abstract(_state(), ["a", "b"], []) is None


@pytest.mark.asyncio
async def test_unavailable_review_stops_optional_revision_and_selection():
    class Harness(GraphicAbstractMixin):
        session_id = "graphic-abstract-workflow"

    harness = Harness()
    harness._send_message = AsyncMock()
    harness.pi_agent = SimpleNamespace(
        generate_graphic_abstract_prompt=AsyncMock(return_value="prompt"),
        review_graphic_abstract=AsyncMock(return_value={
            "review_unavailable": True,
            "review_reason": "vision unavailable",
            "quality_score": None,
            "feedback": "vision unavailable",
            "suggestions": [],
            "must_fix": [],
        }),
        select_best_graphic_abstract=AsyncMock(),
    )
    generated = SimpleNamespace(
        image_base64="initial",
        prompt="prompt",
        revision_count=0,
        revision_history=[],
        all_candidates=[],
        selected_index=0,
        created_at=datetime.now(),
    )
    harness.painter_agent = SimpleNamespace(
        generate_graphic_abstract=AsyncMock(return_value=generated),
        revise_graphic_abstract=AsyncMock(),
    )

    state = {"analysis_state": _state()}
    result = await harness._generate_image_graphic_abstract(state)

    graphic = result["analysis_state"].graphic_abstract
    assert graphic.review_available is False
    assert graphic.selection_available is None
    harness.painter_agent.revise_graphic_abstract.assert_not_awaited()
    harness.pi_agent.select_best_graphic_abstract.assert_not_awaited()


@pytest.mark.asyncio
async def test_unavailable_selection_keeps_latest_candidate_without_verdict():
    class Harness(GraphicAbstractMixin):
        session_id = "graphic-abstract-selection"

    harness = Harness()
    harness._send_message = AsyncMock()
    review = {
        "review_unavailable": False,
        "review_reason": None,
        "quality_score": 8,
        "feedback": "clear",
        "suggestions": [],
        "must_fix": [],
    }
    select = AsyncMock(return_value=None)
    harness.pi_agent = SimpleNamespace(
        generate_graphic_abstract_prompt=AsyncMock(return_value="prompt"),
        review_graphic_abstract=AsyncMock(side_effect=[review, review]),
        select_best_graphic_abstract=select,
    )
    generated = SimpleNamespace(
        image_base64="initial",
        prompt="prompt",
        revision_count=0,
        revision_history=[],
        all_candidates=[],
        selected_index=0,
        created_at=datetime.now(),
    )
    harness.painter_agent = SimpleNamespace(
        generate_graphic_abstract=AsyncMock(return_value=generated),
        revise_graphic_abstract=AsyncMock(
            side_effect=["revision-1", "revision-2"]
        ),
    )

    result = await harness._generate_image_graphic_abstract(
        {"analysis_state": _state()}
    )

    graphic = result["analysis_state"].graphic_abstract
    assert graphic.image_base64 == "revision-2"
    assert graphic.selected_index == 2
    assert graphic.review_available is None
    assert graphic.selection_available is False
    select.assert_awaited_once()


@pytest.mark.asyncio
async def test_report_assembly_carries_graphic_abstract_availability_metadata(tmp_path):
    from backend.services.report.generator import DetailedReportGenerator
    from backend.services.report.generator_parts.staged import _assemble_report

    state = _state()
    state.graphic_abstract = GraphicAbstract(
        prompt="diagram",
        image_base64="",
        text_diagram="A -> B",
        review_available=False,
        selection_available=False,
    )
    generator = DetailedReportGenerator(
        llm_client=object(),
        pi_model="pi",
        bio_model="bio",
        output_dir=str(tmp_path / "report"),
    )
    generator._build_full_analysis_log = lambda _state: []
    generator._maybe_generate_reproducibility_bundle = lambda **_kwargs: None

    async def fake_html(_report, _language, state=None):
        path = generator.output_dir / "report.html"
        path.write_text("<html>report</html>", encoding="utf-8")
        return path

    generator._generate_html_report = fake_html
    bindings = SimpleNamespace(
        logger=logging.getLogger(__name__),
        write_interaction_record_files=lambda *_args, **_kwargs: [],
        build_report_followups=lambda *_args, **_kwargs: [],
        build_agent_models_appendix=lambda *_args, **_kwargs: {},
        build_cross_modal_appendix=lambda *_args, **_kwargs: {},
        build_direction_selection_history=lambda *_args, **_kwargs: [],
        LITERATURE_REPORT_FILENAME="literature.md",
        unreported_figure_names=lambda *_args, **_kwargs: [],
        figureless_outline_sections=lambda *_args, **_kwargs: [],
        build_quality_disclosures=lambda *_args, **_kwargs: {},
        build_report_partial_fields=lambda **_kwargs: (
            "complete", "Complete", None, []
        ),
        datetime=datetime,
        uuid=uuid,
        get_session_states=lambda _session_id: tmp_path / "states",
        write_literature_report=lambda *_args, **_kwargs: None,
    )

    report = await _assemble_report(
        generator,
        state=state,
        report_state=state,
        report_id="report-id",
        report_plan={"title": "Report"},
        sections=[],
        abstract="",
        conclusions="",
        existing_figures=[],
        language="en",
        force_fallback=False,
        bindings=bindings,
    )

    assert report["graphic_abstract"] == {
        "selected_index": 0,
        "revision_count": 0,
        "review_available": False,
        "selection_available": False,
        "is_text_mode": True,
    }


def test_old_graphic_abstract_payload_keeps_availability_unknown():
    payload = {"prompt": "p", "image_base64": "aW1n"}
    restored = OutputGraphicAbstract.model_validate(payload)

    assert restored.review_available is None
    assert restored.selection_available is None


def test_graphic_abstract_availability_survives_session_json_roundtrip():
    state = _state()
    state.graphic_abstract = GraphicAbstract(
        prompt="p",
        image_base64="aW1n",
        selected_index=1,
        review_available=False,
        selection_available=True,
    )

    restored = AnalysisState.model_validate_json(state.model_dump_json())

    assert restored.graphic_abstract is not None
    assert restored.graphic_abstract.review_available is False
    assert restored.graphic_abstract.selection_available is True
