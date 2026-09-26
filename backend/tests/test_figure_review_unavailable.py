"""Unavailable visual review is truthful, durable, and non-blocking."""


import pytest

from backend.models.schemas import TaskStatus
from backend.services.report.artifacts import is_report_eligible
from backend.tests.test_figure_revision_gate import BLOCKING, CLEAN, _run


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
