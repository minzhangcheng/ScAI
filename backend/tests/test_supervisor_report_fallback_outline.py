"""The degraded fallback report keeps its own shape.

When the report model fails, _fallback_report emits a fixed three-section body
whose ids belong to the single-pass template. Forcing that through a
round-derived outline would drop or reshuffle it, so the outline passes are
skipped and its figures fall through to the supplementary section.
"""
import hashlib
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from backend.agents.supervisor_agent import SupervisorAgent
from backend.models.schemas import (
    AgentType,
    AnalysisState,
    FigureReviewStatus,
    PublicationFigure,
    RoundResult,
)
from backend.services.report.output_generation import resolve_current_report_dir

_T0 = datetime(2026, 1, 1, 12, 0, 0)


def _state_with_rounds():
    st = AnalysisState(session_id="t", user_id="u", initial_instruction="x")
    st.name = "proj"
    for n, phase, direction in (
        (1, "exploration", ""),
        (2, "innovation", "direction A"),
        (3, "integration", ""),
    ):
        st.round_results.append(RoundResult(
            round_number=n, phase=phase, direction=direction,
            timestamp=_T0 + timedelta(minutes=n * 10),
        ))
    return st


def _agent(chat_side_effect):
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.session_id = "t"
    agent.refresh_runtime_config_if_needed = MagicMock()
    agent.get_supervision_log = MagicMock(return_value=[])
    agent.get_summary_stats = MagicMock(return_value={
        "total_steps_monitored": 3,
        "steps_with_technical_issues": 0,
        "pi_escalations": 0,
    })
    agent._set_llm_context = MagicMock()
    agent._build_appendix = MagicMock(return_value={})
    client = MagicMock()
    client.set_context = MagicMock()
    client.chat = AsyncMock(side_effect=chat_side_effect)
    client.base_url = "http://x"
    # llm_client is a read-only property; the client is injected through the
    # report_llm_client argument instead, which takes precedence over it.
    agent._provider = MagicMock()
    return agent, client


def _state_with_complete_visual_coverage(tmp_path):
    state = _state_with_rounds()
    for name, round_number in (("exploration_panel", 1), ("direction_panel", 2)):
        path = tmp_path / "workspace" / "figures" / f"{name}.jpg"
        pdf_path = path.with_suffix(".pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.effect_noise((256, 256), 100).convert("RGB")
        image.save(path)
        image.save(pdf_path, "PDF")
        assert path.stat().st_size > 2000
        state.publication_figures.append(PublicationFigure(
            name=name,
            description=f"{name} description",
            figure_type="heatmap",
            png_path=str(path),
            pdf_path=str(pdf_path),
            render_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            pdf_sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
            review_status=FigureReviewStatus.APPROVED,
            round_number=round_number,
        ))
    return state


def _complete_body():
    """Every section the round-derived outline plans, so nothing is absent."""
    return (
        "## [abstract] Summary\nsummary text\n\n"
        "## [methods] How it was done\nprocedure\n\n"
        "## [exploration_findings] What the first pass found\ncomposition\n\n"
        "## [direction_1] What direction A showed\nbody\n\n"
        "## [integrated_findings] Cross-direction synthesis\ncombined interpretation\n\n"
        "## [mechanism_candidates] Candidate mechanisms\nhypotheses\n\n"
        "## [conclusions] Wrap up\nend matter\n"
    )


@pytest.mark.asyncio
async def test_fallback_report_keeps_its_own_section_order(tmp_path):
    agent, client = _agent(RuntimeError("model down"))
    out = tmp_path / "reports"
    ws = tmp_path / "workspace"
    (ws / "figures").mkdir(parents=True, exist_ok=True)

    report = await agent.generate_detailed_report(
        _state_with_rounds(), "model-x", str(out), workspace_dir=str(ws),
        report_llm_client=client,
    )

    ids = [s["id"] for s in report["sections"]]
    # abstract and conclusions are lifted out; the middle section survives
    # intact rather than being relocated to the end by outline alignment.
    assert ids == ["analysis_overview"]
    assert report["abstract"]
    assert report["conclusions"]
    assert report["status"] == "partial"
    assert report["partial_reason"] == "degraded_fallback"
    assert report["unreported_figures"] == []
    assert report["figureless_sections"] == [
        "exploration_findings", "direction_1",
    ]
    assert "integrated_findings" not in report["figureless_sections"]
    assert report["current_step"].startswith("Degraded fallback report")
    assert "figure coverage gaps" in report["current_step"]


@pytest.mark.asyncio
async def test_forced_fallback_never_calls_the_report_llm(tmp_path):
    agent, client = _agent(AssertionError("report LLM must not be called"))
    out = tmp_path / "reports"
    ws = tmp_path / "workspace"
    (ws / "figures").mkdir(parents=True, exist_ok=True)

    report = await agent.generate_detailed_report(
        _state_with_complete_visual_coverage(tmp_path),
        "model-x",
        str(out),
        workspace_dir=str(ws),
        report_llm_client=client,
        force_fallback=True,
    )

    client.chat.assert_not_awaited()
    assert report["status"] == "partial"
    assert report["partial_reason"] == "degraded_fallback"
    published = resolve_current_report_dir(out)
    assert (published / "report.json").is_file()
    assert (published / "report.html").is_file()


@pytest.mark.asyncio
async def test_successful_report_is_aligned_to_the_round_outline(tmp_path):
    body = (
        "## [conclusions] Wrap up\nend matter\n\n"
        "## [direction_1] What direction A showed\nbody\n\n"
        "## [abstract] Summary\nsummary text\n\n"
        "## [methods] How it was done\nprocedure\n"
    )
    # The body omits planned sections, so the writer is asked once to complete
    # them; here it answers with nothing and the gap is disclosed instead.
    agent, client = _agent([
        SimpleNamespace(content=body), SimpleNamespace(content=""),
    ])
    out = tmp_path / "reports"
    ws = tmp_path / "workspace"
    (ws / "figures").mkdir(parents=True, exist_ok=True)

    report = await agent.generate_detailed_report(
        _state_with_rounds(), "model-x", str(out), workspace_dir=str(ws),
        report_llm_client=client,
    )

    ids = [s["id"] for s in report["sections"]]
    # Emitted conclusions-first; the outline puts methods before the direction
    # and abstract/conclusions are extracted, so what remains is in plan order.
    assert ids == ["methods", "direction_1"]
    assert report["figureless_sections"] == [
        "exploration_findings", "direction_1",
    ]
    assert report["missing_sections"] == [
        "exploration_findings", "integrated_findings", "mechanism_candidates",
    ]
    assert report["status"] == "partial"
    # An absent chapter outranks a chapter that is merely figureless, but both
    # stay on the record.
    assert report["partial_reason"] == "section_coverage"
    assert "figure_coverage" in report["partial_reasons"]
    assert report["unreported_figures"] == []
    assert "figure coverage gaps" in report["current_step"]
    assert "direction_1" in report["current_step"]


@pytest.mark.asyncio
async def test_chat_sections_reuse_frozen_report_client_and_model(
    tmp_path,
    monkeypatch,
):
    body = (
        "## [abstract] Summary\nsummary text\n\n"
        "## [methods] How it was done\nprocedure\n\n"
        "## [conclusions] Wrap up\nend matter\n"
    )
    agent, client = _agent([SimpleNamespace(content=body)])
    agent._provider.get_model.return_value = "new-model-after-refresh"
    chat_builder = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "backend.services.report.chat_sections.build_chat_sections_async",
        chat_builder,
    )
    out = tmp_path / "reports"
    ws = tmp_path / "workspace"
    (ws / "figures").mkdir(parents=True, exist_ok=True)

    await agent.generate_detailed_report(
        _state_with_rounds(),
        "frozen-report-model",
        str(out),
        workspace_dir=str(ws),
        report_llm_client=client,
        chat_messages=[{"role": "user", "content": "question"}],
    )

    kwargs = chat_builder.await_args.kwargs
    assert kwargs["llm_client"] is client
    assert kwargs["model"] == "frozen-report-model"


@pytest.mark.asyncio
async def test_chat_sections_reuse_supervisor_fallback_client_when_none_passed(
    tmp_path,
    monkeypatch,
):
    body = (
        "## [abstract] Summary\nsummary text\n\n"
        "## [methods] How it was done\nprocedure\n\n"
        "## [conclusions] Wrap up\nend matter\n"
    )
    agent, supervisor_client = _agent([SimpleNamespace(content=body)])
    agent.agent_type = AgentType.SUPERVISOR
    pi_client = MagicMock(name="pi_client_must_not_be_used")
    agent._provider.get_llm_client.side_effect = lambda name: (
        supervisor_client if name == "supervisor" else pi_client
    )
    chat_builder = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "backend.services.report.chat_sections.build_chat_sections_async",
        chat_builder,
    )
    out = tmp_path / "reports"
    ws = tmp_path / "workspace"
    (ws / "figures").mkdir(parents=True, exist_ok=True)

    await agent.generate_detailed_report(
        _state_with_rounds(),
        "supervisor-endpoint-model",
        str(out),
        workspace_dir=str(ws),
        report_llm_client=None,
        chat_messages=[{"role": "user", "content": "question"}],
    )

    kwargs = chat_builder.await_args.kwargs
    assert kwargs["llm_client"] is supervisor_client
    assert kwargs["llm_client"] is not pi_client
    assert kwargs["model"] == "supervisor-endpoint-model"


@pytest.mark.asyncio
async def test_text_only_integration_does_not_downgrade_report_status(tmp_path):
    agent, client = _agent([SimpleNamespace(content=_complete_body())])
    out = tmp_path / "reports"
    ws = tmp_path / "workspace"
    (ws / "figures").mkdir(parents=True, exist_ok=True)

    state = _state_with_complete_visual_coverage(tmp_path)

    report = await agent.generate_detailed_report(
        state, "model-x", str(out), workspace_dir=str(ws),
        report_llm_client=client,
    )

    assert report["figureless_sections"] == []
    assert report["status"] == "completed"
    assert report["partial_reason"] is None
    assert report["unreported_figures"] == []
    assert report["current_step"] == "Complete"


@pytest.mark.asyncio
async def test_explicit_partial_report_flag_is_preserved_in_status(tmp_path):
    agent, client = _agent([SimpleNamespace(content=_complete_body())])
    out = tmp_path / "reports"
    ws = tmp_path / "workspace"
    (ws / "figures").mkdir(parents=True, exist_ok=True)
    state = _state_with_complete_visual_coverage(tmp_path)
    state.allow_partial_report = True
    progress_events = []

    async def progress_callback(payload):
        progress_events.append(payload)

    report = await agent.generate_detailed_report(
        state, "model-x", str(out), workspace_dir=str(ws),
        report_llm_client=client,
        progress_callback=progress_callback,
    )

    assert report["figureless_sections"] == []
    assert report["status"] == "partial"
    assert report["partial_reason"] == "approved_partial"
    assert report["unreported_figures"] == []
    assert report["current_step"] == "Partial report approved"
    assert progress_events[-1] == {
        "progress": 100,
        "step": "Partial report approved",
        "status": "partial",
    }
