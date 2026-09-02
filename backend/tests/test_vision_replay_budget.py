"""A vision stream cut short after its first event is replayed, not taken as a verdict.

The vision call sites do not lose their result silently — each degrades to a
recorded outcome (an ``review_unavailable`` verdict, a text-only fallback). But
degrading is the wrong answer to a transient interruption, and for figure review
it is expensive: PI's visual verdict is the precondition for a figure being
adopted, so one blip can cost a figure that a single replay would have kept.

The budget goes at the call sites rather than inside ``chat_with_vision``
because the client method is the layer these areas substitute wholesale in
tests; a budget there would vanish behind every fake client. Three sites are
deliberately left alone: pathology already routes through ``_call_llm_with_hitl``,
bioinformatics figure revision already owns a budget, and
``biochemistry_agent`` already retries twice and re-raises a non-payload
``UpstreamExhaustedError`` itself.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.pi_agent import PIAgent
from backend.models.schemas import AnalysisState, PublicationFigure
from backend.utils.llm_client import UpstreamExhaustedError

_GOOD_REVIEW = (
    '{"approved":true,"quality_score":8,"feedback":"ok",'
    '"strengths":[],"issues":[],"suggestions":[],"must_fix":[]}'
)


def _interrupted(detail="stream ended before a terminal response"):
    return UpstreamExhaustedError(
        detail,
        agent_name="transport",
        model="transport-model",
        failure_kind="stream_interrupted",
    )


def _payload_rejected():
    """A deterministic image-payload rejection: replay cannot help."""
    return UpstreamExhaustedError(
        "invalid image part",
        agent_name="transport",
        model="transport-model",
        failure_kind="upstream_exhausted",
    )


def _pi_agent(vision_side_effect):
    vision = AsyncMock(side_effect=vision_side_effect)
    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.2
    provider.get_vision_setting.return_value = True
    provider.get_llm_client.return_value = SimpleNamespace(
        chat_with_vision=vision,
        set_context=lambda *_args: None,
    )
    agent = PIAgent(settings_provider=provider)
    agent.set_session("vision-replay")
    agent._set_llm_context = lambda: None
    return agent, vision


def _state():
    return AnalysisState(user_id="u", name="P", initial_instruction="Analyze")


def _figure():
    return PublicationFigure(
        name="fig", description="d", figure_type="volcano",
        png_path="/tmp/f.png", pdf_path="/tmp/f.pdf",
    )


def _ok(content):
    return SimpleNamespace(content=content)


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    monkeypatch.setattr(
        "backend.utils.llm_client_parts.retry.asyncio.sleep", AsyncMock()
    )


@pytest.mark.asyncio
async def test_review_figure_recovers_from_a_replayed_interruption():
    agent, vision = _pi_agent([
        _interrupted(), _interrupted(), _ok(_GOOD_REVIEW),
    ])

    review = await agent.review_figure(_state(), _figure(), "aW1n")

    assert not review.get("review_unavailable")
    assert review["approved"] is True
    assert vision.await_count == 3


@pytest.mark.asyncio
async def test_review_figure_still_degrades_when_replays_are_exhausted():
    """Exhaustion must keep the existing contract: a recorded non-verdict."""
    agent, vision = _pi_agent([_interrupted()] * 4)

    review = await agent.review_figure(_state(), _figure(), "aW1n")

    assert review["review_unavailable"] is True
    assert vision.await_count == 4


@pytest.mark.asyncio
async def test_review_figure_does_not_replay_a_payload_rejection():
    """A deterministic rejection would cost three more image uploads for nothing."""
    agent, vision = _pi_agent([_payload_rejected()])

    review = await agent.review_figure(_state(), _figure(), "aW1n")

    assert review["review_unavailable"] is True
    assert vision.await_count == 1


@pytest.mark.asyncio
async def test_graphic_abstract_review_shares_the_same_budget():
    """The other three PI vision sites go through the same helper."""
    agent, vision = _pi_agent([
        _interrupted(), _ok('{"approved":true,"feedback":"fine","suggestions":[]}'),
    ])

    await agent.review_graphic_abstract(_state(), "aW1n", [])

    assert vision.await_count == 2


@pytest.mark.asyncio
async def test_vision_figure_review_callback_replays_before_reporting_unavailable():
    from backend.workflow.callbacks import CallbacksMixin

    class _Host(CallbacksMixin):
        def __init__(self):
            self.session_id = "s"

    agent, vision = _pi_agent([_interrupted(), _ok(_GOOD_REVIEW)])

    review = await _Host()._vision_figure_review(
        agent, _figure(), "prompt", "aW1n", label="PI",
    )

    assert not review.get("review_unavailable")
    assert vision.await_count == 2


@pytest.mark.asyncio
async def test_supervisor_vision_evaluation_replays_before_text_fallback():
    from backend.agents.supervisor_agent import SupervisorAgent
    from backend.models.schemas import AnalysisStep

    vision = AsyncMock(side_effect=[_interrupted(), _ok("vision observation")])
    chat = AsyncMock(return_value=_ok('{"issues":[],"severity":"none"}'))
    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.2
    provider.get_vision_setting.return_value = True
    tools = AsyncMock(return_value=SimpleNamespace(
        content='{"issues":[],"severity":"none"}', tool_calls=[],
    ))
    provider.get_llm_client.return_value = SimpleNamespace(
        chat_with_vision=vision, chat=chat, chat_with_tools=tools,
        set_context=lambda *_args: None,
    )
    agent = SupervisorAgent(settings_provider=provider)
    agent.session_id = "vision-replay"
    agent._set_llm_context = lambda: None
    agent.has_vision = True
    # Real compression needs a real PNG; the budget under test sits after it.
    agent._compress_base64_figure = lambda fig_b64: fig_b64

    await agent._llm_evaluate(
        AnalysisStep(step_number=1, step_type="execute_code", code="plot()"),
        ["aW1n"],
    )

    assert vision.await_count == 2
