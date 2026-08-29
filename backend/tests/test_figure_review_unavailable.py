"""Unavailable visual review is truthful, durable, and non-blocking."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.agents.pi.reports import ReportsMixin
from backend.models.schemas import PublicationFigure, TaskStatus
from backend.services.report.artifacts import is_report_eligible
from backend.tests.test_figure_revision_gate import BLOCKING, CLEAN, _run
from backend.workflow.callbacks import CallbacksMixin


def _figure() -> PublicationFigure:
    return PublicationFigure(
        name="result",
        description="Result",
        figure_type="other",
        png_path="/tmp/result.jpg",
        pdf_path="/tmp/result.pdf",
        png_base64="image",
    )


@pytest.mark.asyncio
async def test_no_vision_capable_reviewer_is_not_pi_approval():
    workflow = CallbacksMixin.__new__(CallbacksMixin)
    workflow.session_id = "figure-review-unavailable"
    workflow._vision_settings = {
        "pi": False,
        "bioinformatics": False,
        "supervisor": False,
    }
    workflow.supervisor_agent = None
    workflow.ensure_agent_ready = lambda _name: None
    workflow._send_message = AsyncMock()

    review = await workflow._figure_review_callback(_figure())

    assert review == {
        "approved": False,
        "quality_score": None,
        "review_unavailable": True,
        "feedback": (
            "Figure review unavailable: no vision-capable model in pipeline"
        ),
        "suggestions": [],
        "must_fix": [],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["provider_error", "invalid_response"])
async def test_failed_or_unparseable_vision_review_has_no_synthetic_score(mode):
    workflow = CallbacksMixin.__new__(CallbacksMixin)
    workflow.session_id = "figure-review-unavailable"
    if mode == "provider_error":
        chat = AsyncMock(side_effect=RuntimeError("vision provider failed"))
    else:
        chat = AsyncMock(return_value=SimpleNamespace(content="not JSON"))
    agent = SimpleNamespace(
        model="test-model",
        llm_client=SimpleNamespace(chat_with_vision=chat),
        _set_llm_context=lambda: None,
        _safe_parse_json=lambda _value, default=None: default,
    )

    review = await workflow._vision_figure_review(
        agent,
        _figure(),
        "review",
        "image",
        label="Test reviewer",
    )

    assert review["approved"] is False
    assert review["review_unavailable"] is True
    assert review["quality_score"] is None
    assert review["suggestions"] == []
    assert review["must_fix"] == []


@pytest.mark.asyncio
async def test_unavailable_review_finalizes_without_revision_or_task(tmp_path):
    review = {
        "approved": False,
        "quality_score": None,
        "review_unavailable": True,
        "feedback": "vision provider unavailable",
        "suggestions": [],
        "must_fix": ["opinion must not become a gate"],
    }
    agent, state, result = await _run(tmp_path, [CLEAN], [review])

    figure = state.publication_figures[0]
    assert figure.review_status.value == "pending"
    assert figure.review_unavailable is True
    assert figure.producer_status == "published"
    assert figure.revision_count == 0
    assert figure.unresolved_must_fix == []
    assert figure.revision_history == [{
        "round": 1,
        "candidate_id": f"{figure.id}:review:1",
        "feedback": "vision provider unavailable",
        "suggestions": [],
        "must_fix": [],
        "quality_score": None,
        "review_unavailable": True,
    }]
    assert state.figure_revision_tasks == []
    assert state.status == TaskStatus.PENDING
    assert agent.revision_calls == []
    assert "Visual review was unavailable" in result
    event = next(
        payload for kind, payload in agent.notifications
        if kind == "figure_review" and isinstance(payload, dict)
    )
    assert event["approved"] is False
    assert event["quality_score"] is None
    assert event["review_unavailable"] is True
    assert event["figure"]["review_unavailable"] is True
    assert is_report_eligible(figure) is True
    api_figure = state.to_api_dict()["publication_figures"][0]
    assert api_figure["review_status"] == "pending"
    assert api_figure["review_unavailable"] is True


@pytest.mark.asyncio
async def test_unavailable_review_does_not_waive_blocking_layout_defect(tmp_path):
    agent, state, _ = await _run(tmp_path, [BLOCKING], [{
        "approved": False,
        "quality_score": None,
        "review_unavailable": True,
        "feedback": "vision provider unavailable",
        "suggestions": [],
        "must_fix": [],
    }])

    figure = state.publication_figures[0]
    assert agent.revision_calls == []
    assert state.figure_revision_tasks == []
    assert is_report_eligible(figure) is False


@pytest.mark.asyncio
async def test_pi_integrated_report_sees_review_unavailable(tmp_path):
    _, state, _ = await _run(tmp_path, [CLEAN], [{
        "approved": False,
        "quality_score": None,
        "review_unavailable": True,
        "feedback": "vision provider unavailable",
        "suggestions": [],
        "must_fix": [],
    }])

    class Harness(ReportsMixin):
        session_id = "figure-review-unavailable"

        def __init__(self):
            self.prompt = ""

        async def chat(self, prompt, **_kwargs):
            self.prompt = prompt
            return "report"

    harness = Harness()
    await harness.generate_integrated_report(state)

    assert "[Review Unavailable]" in harness.prompt
