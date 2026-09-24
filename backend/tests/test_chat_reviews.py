"""Chat review ownership, finite corrections, and workflow vision fallback."""
import asyncio
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.pi_agent import PIAgent
from backend.config import SystemConfig
from backend.models.schemas import AnalysisState, AnalysisStep, PublicationFigure, FigureReviewStatus
from backend.routes.chat_parts.delegation import _build_execution_context, _build_synthesis_prompt
from backend.routes.chat_parts.figures import _collect_session_figures
from backend.routes.chat_parts.reviews import ChatReviewCallbacks, run_reviewed_analysis
from backend.tests.figure_image_helpers import write_valid_image, valid_pdf_bytes
from backend.utils.settings_provider import AgentSettingsProvider


def bridge(vision=(True, True, True), supervisor=True):
    state = AnalysisState(id="chat-review-test", user_id="owner")
    provider = SimpleNamespace(
        get_vision_setting=lambda name: dict(zip(("pi", "bioinformatics", "supervisor"), vision))[name],
        get_model=lambda name: "model" if supervisor else "",
    )
    pi = SimpleNamespace(review_figure=AsyncMock(return_value={"approved": True}),
                         review_chat_execution=AsyncMock())
    callbacks = ChatReviewCallbacks(state, provider, pi, AsyncMock(), AsyncMock(), None)
    callbacks.bioinformatics_agent = SimpleNamespace()
    return state, callbacks


@pytest.mark.asyncio
@pytest.mark.parametrize("vision,expected", [
    ((True, True, True), "pi"),
    ((False, True, True), "bio"),
    ((False, False, True), "supervisor"),
    ((False, False, False), "unavailable"),
])
async def test_chat_reuses_workflow_vision_fallback(vision, expected):
    state, callbacks = bridge(vision)
    callbacks._bio_self_figure_review = AsyncMock(return_value={"reviewer": "bio"})
    callbacks._supervisor_figure_review = AsyncMock(return_value={"reviewer": "supervisor"})
    figure = PublicationFigure(name="plot", description="test", figure_type="other",
                               png_path="plot.jpg", pdf_path="plot.pdf", png_base64="image")
    result = await callbacks._figure_review_callback(figure)
    if expected == "pi":
        callbacks.pi_agent.review_figure.assert_awaited_once_with(state, figure, "image")
    elif expected == "bio":
        callbacks._bio_self_figure_review.assert_awaited_once()
        callbacks.pi_agent.review_figure.assert_not_awaited()
    elif expected == "supervisor":
        callbacks._supervisor_figure_review.assert_awaited_once()
    else:
        assert result["review_unavailable"] is True
        assert result["approved"] is False
        callbacks.pi_agent.review_figure.assert_not_awaited()
        callbacks._bio_self_figure_review.assert_not_awaited()
        callbacks._supervisor_figure_review.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_supervisor_is_not_a_vision_fallback():
    _, callbacks = bridge((False, False, True), supervisor=False)
    result = await callbacks._figure_review_callback(PublicationFigure(
        id="id", name="plot", figure_type="other", description="", revision_count=0,
        png_path="plot.jpg", pdf_path="plot.pdf",
    ))
    assert result["review_unavailable"]
    assert callbacks.supervisor_agent is None


@pytest.mark.asyncio
async def test_pi_vision_failure_is_unavailable_not_approval():
    _, callbacks = bridge()
    callbacks.pi_agent.review_figure.side_effect = RuntimeError("vision unavailable")
    result = await callbacks._figure_review_callback(PublicationFigure(
        id="id", name="plot", figure_type="other", description="", revision_count=0,
        png_base64="image", png_path="plot.jpg", pdf_path="plot.pdf",
    ))
    assert result["review_unavailable"] and not result["approved"]


@pytest.mark.asyncio
async def test_supervisor_feedback_and_failure_are_visible():
    _, callbacks = bridge((False, False, False))
    callbacks.ensure_agent_ready("supervisor")
    from backend.agents.supervisor_agent import SupervisorEvalResult
    feedback = SupervisorEvalResult(feedback="Inspect the threshold calculation")
    callbacks.supervisor_agent.evaluate_step = AsyncMock(return_value=feedback)
    step = AnalysisStep(step_number=1, step_type="execute_code", code="calculate()")
    result = await callbacks.supervise(step, [])
    assert result is feedback
    callbacks.supervisor_agent.evaluate_step.assert_awaited_once_with(step, [])
    callbacks.supervisor_agent.evaluate_step.side_effect = RuntimeError("upstream unavailable")
    result = await callbacks.supervise(step, [])
    assert result.feedback is None
    assert callbacks.observations[-1]["step_id"] == step.id
    assert callbacks.observations[-1]["review_unavailable"]


def test_failed_figure_attempt_is_in_pi_evidence():
    result = {"analysis_log": [{"id": "plot-attempt", "step_number": 3,
        "pub_figure_name": "empty", "pub_figure_code": "plt.figure()",
        "pub_figure_result": "No plotted data: 0 data artists across 0 axes"}]}
    evidence = _build_execution_context(result)
    assert "0 data artists across 0 axes" in evidence
    assert "plt.figure()" in evidence


def execution(number, used):
    return {"status": "completed", "steps_executed": used,
            "findings": [f"finding-{number}"], "attempt_execution_ids": [f"e-{number}"],
            "code_history": [{"id": f"e-{number}", "code": f"code_{number}()",
                              "output": "output", "error": None}], "analysis_log": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("decisions,used,expected_calls,expected_status", [
    (["revise", "accept"], [47, 5], 2, "accept"),
    (["revise", "revise", "revise"], [10, 10, 10], 3, "budget_exhausted"),
    (["revise"], [60], 1, "budget_exhausted"),
    (["limited"], [12], 1, "limited"),
    (["unavailable"], [12], 1, "unavailable"),
])
async def test_chat_corrections_preserve_evidence_and_share_budget(
    decisions, used, expected_calls, expected_status,
):
    state, callbacks = bridge()
    callbacks.pi_agent.review_chat_execution.side_effect = [
        {"decision": item, "rationale": "assessment", "revision_instruction": "Correct calculation",
         **({"review_unavailable": True} if item == "unavailable" else {})}
        for item in decisions
    ]
    agent = SimpleNamespace(run_autonomous_analysis=AsyncMock(
        side_effect=[execution(i, n) for i, n in enumerate(used)],
    ))
    saved = []
    async def persist():
        saved.append(state.model_dump_json())
    result = await run_reviewed_analysis(
        state=state, instruction="Compare observed programs without imposing an order",
        bio_agent=agent, reviews=callbacks, persist=persist,
        collect_figures=lambda scopes: [], execution_context=_build_execution_context,
    )
    calls = agent.run_autonomous_analysis.await_args_list
    assert len(calls) == expected_calls
    assert calls[0].kwargs["max_steps"] == 60
    if used[0] == 47 and expected_calls == 2:
        assert calls[1].kwargs["max_steps"] == 13
        assert "code_0()" in calls[1].args[1]
        assert "Correct calculation" in calls[1].args[1]
    assert len({call.kwargs["run_scope_id"] for call in calls}) == expected_calls
    assert all(call.kwargs["resume_from_checkpoint"] is False for call in calls)
    assert len(result["code_history"]) == expected_calls
    assert result["findings"] == [f"finding-{expected_calls-1}"]
    assert result["chat_review"]["status"] == expected_status
    assert result["status"] == "completed"  # PI opinion does not rewrite execution status.
    restored = AnalysisState.model_validate_json(saved[-1])
    assert len(restored.chat_analysis_reviews[0]["cycles"]) == expected_calls
    prompt = _build_synthesis_prompt("question", {}, {"bioinformatics": result})
    assert "PI Chat review" in prompt and expected_status in prompt


@pytest.mark.asyncio
async def test_cancellation_records_interruption_without_rerun():
    state, callbacks = bridge()
    callbacks.pi_agent.review_chat_execution.side_effect = asyncio.CancelledError()
    agent = SimpleNamespace(run_autonomous_analysis=AsyncMock(return_value=execution(0, 4)))
    persist = AsyncMock()
    with pytest.raises(asyncio.CancelledError):
        await run_reviewed_analysis(
            state=state, instruction="task", bio_agent=agent, reviews=callbacks,
            persist=persist, collect_figures=lambda scopes: [], execution_context=_build_execution_context,
        )
    assert state.chat_analysis_reviews[0]["status"] == "cancelled"
    agent.run_autonomous_analysis.assert_awaited_once()
    assert persist.await_count >= 2


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ['{}', 'not JSON', '{"decision":"revise","rationale":"fix"}'])
async def test_invalid_pi_review_is_explicitly_unavailable(reply):
    state = AnalysisState(id="parse-review", user_id="owner")
    agent = PIAgent(AgentSettingsProvider(state, SystemConfig(), global_client=None))
    agent.chat_without_history = AsyncMock(return_value=reply)
    decision = await agent.review_chat_execution("request", "evidence", 10, 1)
    assert decision["review_unavailable"] is True
    assert decision["decision"] == "unavailable"


def test_scoped_chat_figures_exclude_raw_stale_and_rejected_but_keep_unavailable(tmp_path):
    state = AnalysisState(id="scoped-figures", user_id="owner")
    for name, status, scope in [
        ("valid", FigureReviewStatus.APPROVED, "current"),
        ("unreviewed", FigureReviewStatus.PENDING, "current"),
        ("rejected", FigureReviewStatus.REVISION_REQUESTED, "current"),
        ("old", FigureReviewStatus.APPROVED, "old"),
    ]:
        png, pdf = tmp_path / f"{name}.jpg", tmp_path / f"{name}.pdf"
        payload = write_valid_image(png)
        pdf.write_bytes(valid_pdf_bytes())
        state.publication_figures.append(PublicationFigure(
            name=name, description="plot", figure_type="other",
            png_path=str(png), pdf_path=str(pdf), producer_run_scope_id=scope,
            review_status=status, review_unavailable=name == "unreviewed",
            render_sha256=hashlib.sha256(payload).hexdigest(),
            pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        ))
    write_valid_image(tmp_path / "blank_autosave.jpg", color=(255, 255, 255))
    figures = _collect_session_figures(state.id, state=state, run_scope_ids=["current"])
    assert {f["name"] for f in figures} == {"valid", "unreviewed"}
    assert next(f for f in figures if f["name"] == "unreviewed")["review_unavailable"]
    # A later raw overwrite cannot impersonate the reviewed artifact.
    write_valid_image(tmp_path / "valid.jpg", color=(255, 255, 255))
    assert [f["name"] for f in _collect_session_figures(
        state.id, state=state, run_scope_ids=["current"],
    )] == ["unreviewed"]
