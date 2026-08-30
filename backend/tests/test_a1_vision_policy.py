"""A1 vision capability policy tests.

These tests keep provider failures separate from the agent's durable vision
capability decision: a transient transport failure is retryable, while a
classified multimodal-shape rejection is the only permanent session update.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.pi.figures import FiguresMixin
from backend.utils.llm_client import BadRequestError, InternalServerError, NotFoundError


class _FigureHarness(FiguresMixin):
    def __init__(self, error):
        self.session_id = "a1-vision"
        self.model = "vision-model"
        self.system_prompt = "system"
        self._vision = True
        self._set_llm_context = MagicMock()
        self.llm_client = SimpleNamespace(chat_with_vision=AsyncMock(side_effect=error))
        self.chat = AsyncMock(return_value='{"approved": true, "quality_score": 8}')

    @property
    def has_vision(self):
        return self._vision

    @has_vision.setter
    def has_vision(self, value):
        self._vision = value

    def _safe_parse_json(self, value, default=None):
        return {"approved": True, "quality_score": 8}


def _state_and_figure():
    state = SimpleNamespace(
        name="A1",
        initial_instruction="test",
        treatment_groups=[],
        current_plan=None,
        build_experimental_design_summary=lambda: "",
    )
    figure = SimpleNamespace(
        name="figure",
        figure_type="plot",
        description="test",
        revision_count=0,
        revision_history=[],
        pi_feedback="",
        layout_violations=None,
        plan_step_id=None,
    )
    return state, figure


@pytest.mark.asyncio
async def test_transient_vision_failure_does_not_disable_next_call():
    error = InternalServerError(
        message="Error code: 500 - transient upstream failure",
        response=MagicMock(status_code=500),
        body={"error": {"message": "transient upstream failure"}},
    )
    agent = _FigureHarness(error)
    state, figure = _state_and_figure()

    await agent.review_figure(state, figure, "aW1hZ2U=")

    assert agent.has_vision is True
    assert agent.chat.await_count == 0

    # A structured Responses image-shape rejection is a protocol fallback,
    # not a permanent model-capability decision.  It must leave the next
    # vision attempt enabled as well.
    shape_error = BadRequestError(
        message="Error code: 400 - invalid image_url shape",
        response=MagicMock(status_code=400),
        body={
            "error": {
                "message": "image_url should be a valid dictionary",
                "param": "messages[0].content[1].image_url",
            }
        },
    )
    agent.llm_client.chat_with_vision = AsyncMock(side_effect=shape_error)
    await agent.review_figure(state, figure, "aW1hZ2U=")
    assert agent.has_vision is True

    route_error = NotFoundError(
        message="Error code: 404 - route not found",
        response=MagicMock(status_code=404),
        body={"error": {"message": "route not found"}},
    )
    agent.llm_client.chat_with_vision = AsyncMock(side_effect=route_error)
    await agent.review_figure(state, figure, "aW1hZ2U=")
    assert agent.has_vision is True


@pytest.mark.asyncio
async def test_true_vision_mismatch_disables_once():
    error = NotFoundError(
        message="Error code: 404 - vision modality is unavailable",
        response=MagicMock(status_code=404),
        body={"error": {"message": "vision modality is unavailable"}},
    )
    agent = _FigureHarness(error)
    state, figure = _state_and_figure()

    await agent.review_figure(state, figure, "aW1hZ2U=")
    await agent.review_figure(state, figure, "aW1hZ2U=")

    assert agent.has_vision is False
    assert agent.llm_client.chat_with_vision.await_count == 1
    assert agent.chat.await_count == 0
