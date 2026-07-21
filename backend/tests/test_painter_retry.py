"""Tests for PainterAgent's unified protocol-cascade + round-retry behavior."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agents.painter_agent import (
    PainterAgent,
    _GOOGLE_AI_STUDIO_BASE_URL,
    _MAX_RETRY_ROUNDS,
    _image_error_disposition,
    _is_fatal_image_error,
)
from backend.models.schemas import AgentType


def _make_painter(base_url: str = f"https://{_GOOGLE_AI_STUDIO_BASE_URL}/v1beta"):
    """Build a PainterAgent without running __init__.

    BaseAgent exposes model/system_prompt/llm_client/config as read-only
    properties backed by a settings provider, so we mock the provider.

    Default base_url is a non-DashScope / non-OpenRouter host, so the protocol
    order is [images, chat].
    """
    painter = object.__new__(PainterAgent)
    painter.agent_type = AgentType.PAINTER
    painter.session_id = "test-session"
    painter._vision_disabled_runtime = False

    llm_client = MagicMock()
    llm_client.base_url = base_url
    llm_client.set_context = MagicMock()
    llm_client.client = MagicMock()

    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_system_prompt.return_value = "draw"
    provider.get_temperature.return_value = 0.7
    provider.get_llm_client.return_value = llm_client
    # Match the cached revision so refresh_runtime_config_if_needed early-returns
    # (the fixture bypasses BaseAgent.__init__, which normally sets this).
    provider.get_config_revision.return_value = 1
    painter._provider = provider
    painter._last_config_revision = 1
    return painter


def _mock_protocols(painter, *, images, chat, dashscope):
    """Attach AsyncMock stand-ins for the three protocol attempt methods."""
    painter._images_api_attempt = AsyncMock(side_effect=images)
    painter._chat_image_attempt = AsyncMock(side_effect=chat)
    painter._generate_image_dashscope = AsyncMock(side_effect=dashscope)
    return painter


# --------------------------------------------------------------------------- #
# Protocol ordering
# --------------------------------------------------------------------------- #

def test_protocol_order_default_images_first():
    assert _make_painter()._protocol_order() == ["images", "chat"]


def test_protocol_order_openrouter_chat_first():
    p = _make_painter(base_url="https://openrouter.ai/api/v1")
    assert p._protocol_order() == ["chat", "images"]


def test_protocol_order_dashscope_native_first():
    p = _make_painter(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    assert p._protocol_order() == ["dashscope"]


# --------------------------------------------------------------------------- #
# Cascade behavior
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_first_protocol_success_returns_immediately():
    """Images API (first for default order) succeeds; no other protocol or sleep."""
    painter = _make_painter()
    _mock_protocols(painter, images=["IMG_OK"], chat=[Exception("nope")], dashscope=[Exception("nope")])

    with patch("backend.agents.painter_agent.asyncio.sleep", new=AsyncMock()) as sleep:
        result = await painter._generate_image("a prompt")

    assert result == "IMG_OK"
    assert painter._images_api_attempt.await_count == 1
    assert painter._chat_image_attempt.await_count == 0
    assert painter._generate_image_dashscope.await_count == 0
    assert sleep.await_count == 0


@pytest.mark.asyncio
async def test_switches_to_next_protocol_on_any_failure():
    """Non-fatal images failure -> 1s wait -> chat succeeds; dashscope untouched."""
    painter = _make_painter()
    _mock_protocols(
        painter,
        images=[Exception("Error code: 400 - bad request")],
        chat=["CHAT_OK"],
        dashscope=[Exception("nope")],
    )

    with patch("backend.agents.painter_agent.asyncio.sleep", new=AsyncMock()) as sleep:
        result = await painter._generate_image("a prompt")

    assert result == "CHAT_OK"
    assert painter._images_api_attempt.await_count == 1
    assert painter._chat_image_attempt.await_count == 1
    assert painter._generate_image_dashscope.await_count == 0
    # exactly one inter-protocol wait of 1s, no round backoff
    sleep.assert_awaited_once_with(1.0)


@pytest.mark.asyncio
async def test_401_disables_affected_protocol_and_tries_next():
    """401 is deterministic for one protocol but another protocol may work."""
    painter = _make_painter()
    _mock_protocols(
        painter,
        images=[Exception("Error code: 401 - unauthorized")],
        chat=["CHAT_OK"],
        dashscope=["DASH_OK"],
    )

    with patch("backend.agents.painter_agent.asyncio.sleep", new=AsyncMock()) as sleep:
        result = await painter._generate_image("a prompt")

    assert result == "CHAT_OK"
    assert painter._images_api_attempt.await_count == 1
    assert painter._chat_image_attempt.await_count == 1
    assert painter._generate_image_dashscope.await_count == 0
    sleep.assert_awaited_once_with(1.0)


@pytest.mark.asyncio
async def test_full_round_fails_then_succeeds_next_round_with_backoff():
    """All 3 protocols fail round 1, then images succeeds round 2 after a 2s backoff."""
    painter = _make_painter()
    _mock_protocols(
        painter,
        images=[Exception("500 round1"), "IMG_OK"],
        chat=[Exception("500 round1")],
        dashscope=[Exception("500 round1")],
    )

    with patch("backend.agents.painter_agent.asyncio.sleep", new=AsyncMock()) as sleep:
        result = await painter._generate_image("a prompt")

    assert result == "IMG_OK"
    assert painter._images_api_attempt.await_count == 2
    assert painter._chat_image_attempt.await_count == 1
    assert painter._generate_image_dashscope.await_count == 0
    # round 1: 1s before chat; then 2s round backoff
    durations = [c.args[0] for c in sleep.await_args_list]
    assert durations == [1.0, 2]


@pytest.mark.asyncio
async def test_all_rounds_exhausted_raises():
    """Persistent failures try every protocol for all rounds, then raise."""
    painter = _make_painter()
    _mock_protocols(
        painter,
        images=[Exception("Error code: 502")],
        chat=[Exception("Error code: 502")],
        dashscope=[Exception("Error code: 502")],
    )

    with patch("backend.agents.painter_agent.asyncio.sleep", new=AsyncMock()) as sleep:
        with pytest.raises(ValueError, match=f"failed after at most {_MAX_RETRY_ROUNDS} rounds"):
            await painter._generate_image("a prompt")

    assert painter._images_api_attempt.await_count == _MAX_RETRY_ROUNDS
    assert painter._chat_image_attempt.await_count == _MAX_RETRY_ROUNDS
    assert painter._generate_image_dashscope.await_count == 0
    # backoffs between rounds must include the doubling schedule
    durations = [c.args[0] for c in sleep.await_args_list]
    for expected in (2, 4, 8, 16):
        assert expected in durations


@pytest.mark.asyncio
async def test_timeout_switches_protocol():
    """A per-protocol timeout is treated as a switchable failure, not fatal."""
    import asyncio

    painter = _make_painter()
    real_sleep = asyncio.sleep

    async def _hang(*a, **k):
        await real_sleep(999)

    painter._images_api_attempt = _hang  # genuinely hangs -> trips wait_for timeout
    painter._chat_image_attempt = AsyncMock(side_effect=["CHAT_OK"])
    painter._generate_image_dashscope = AsyncMock(side_effect=[Exception("nope")])

    async def _fast_sleep(delay, *a, **k):
        # Collapse inter-protocol / backoff sleeps but keep wait_for's timer real
        await real_sleep(0)

    # Shrink the timeout so the hanging protocol trips it fast.
    with patch("backend.agents.painter_agent._PROTOCOL_TIMEOUT", 0.05), \
         patch("backend.agents.painter_agent.asyncio.sleep", new=_fast_sleep):
        result = await painter._generate_image("a prompt")

    assert result == "CHAT_OK"
    assert painter._chat_image_attempt.await_count == 1


# --------------------------------------------------------------------------- #
# Revision uses the images/edits protocol first (default order)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_revision_uses_images_edit_first_then_falls_back_to_chat():
    painter = _make_painter()
    painter._images_edit_attempt = AsyncMock(side_effect=[Exception("edits unsupported")])
    painter._chat_image_attempt = AsyncMock(side_effect=["REVISED_OK"])
    painter._generate_image_dashscope = AsyncMock(side_effect=[Exception("nope")])

    with patch("backend.agents.painter_agent.asyncio.sleep", new=AsyncMock()):
        result = await painter.revise_graphic_abstract(
            state=MagicMock(),
            current_image_base64="QUJD",
            original_prompt="orig",
            feedback="make it bluer",
            suggestions=["bluer"],
            must_fix=["fix legend"],
            revision_history=[{"round": 1, "feedback": "x", "suggestions": []}],
        )

    assert result == "REVISED_OK"
    assert painter._images_edit_attempt.await_count == 1
    assert painter._chat_image_attempt.await_count == 1
    # edits was called with the current image
    assert painter._images_edit_attempt.await_args.args[1] == "QUJD"


@pytest.mark.asyncio
async def test_revision_generic_403_is_protocol_scoped_and_falls_back_to_chat():
    """A gateway may forbid /images/edits while allowing chat image output."""
    painter = _make_painter()
    painter._images_edit_attempt = AsyncMock(side_effect=[
        Exception(
            "Error code: 403 - {'error': {'message': 'openai_error', "
            "'code': 'bad_response_status_code'}}"
        )
    ])
    painter._chat_image_attempt = AsyncMock(side_effect=["REVISED_VIA_CHAT"])
    painter._generate_image_dashscope = AsyncMock(side_effect=[Exception("nope")])

    with patch("backend.agents.painter_agent.asyncio.sleep", new=AsyncMock()):
        result = await painter.revise_graphic_abstract(
            state=MagicMock(),
            current_image_base64="QUJD",
            original_prompt="orig",
            feedback="fix spelling",
            suggestions=["fix spelling"],
            must_fix=["correct title"],
            revision_history=[{"round": 1, "feedback": "x", "suggestions": []}],
        )

    assert result == "REVISED_VIA_CHAT"
    assert painter._images_edit_attempt.await_count == 1
    assert painter._chat_image_attempt.await_count == 1


# --------------------------------------------------------------------------- #
# Fatal-error classifier
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("msg", [
    "request blocked by content_filter",
    "request blocked by content policy",
])
def test_is_fatal_image_error_true(msg):
    assert _is_fatal_image_error(Exception(msg)) is True


@pytest.mark.parametrize("msg", [
    "Error code: 401 - unauthorized",
    "Client error '401 Unauthorized' for url 'https://example.test'",
    "invalid api key provided",
    "Error code: 400 - bad request",
    "Error code: 403 - forbidden",
    "Error code: 403 - bad_response_status_code openai_error",
    "Error code: 404 - not found",
    "Error code: 422 - unprocessable",
    "Error code: 429 - rate limit",
    "Error code: 500 - server error",
    "connection timeout",
])
def test_is_fatal_image_error_false(msg):
    assert _is_fatal_image_error(Exception(msg)) is False


@pytest.mark.parametrize("msg", [
    "Error code: 401 - unauthorized",
    "Client error '401 Unauthorized' for url 'https://example.test'",
    "Error code: 403 - forbidden",
    "Error code: 404 - endpoint missing",
    "Error code: 422 - unsupported request",
])
def test_deterministic_protocol_errors_are_disabled(msg):
    assert _image_error_disposition(Exception(msg)) == "disable"


@pytest.mark.asyncio
async def test_revision_does_not_repeat_deterministic_403_or_401():
    painter = _make_painter()
    painter._images_edit_attempt = AsyncMock(
        side_effect=Exception("Error code: 403 - edits forbidden")
    )
    painter._chat_image_attempt = AsyncMock(
        side_effect=Exception("Client error '401 Unauthorized'")
    )
    painter._generate_image_dashscope = AsyncMock(
        side_effect=AssertionError("non-DashScope provider must not use native API")
    )

    with patch("backend.agents.painter_agent.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(ValueError, match="eligible protocols"):
            await painter.revise_graphic_abstract(
                state=MagicMock(),
                current_image_base64="QUJD",
                original_prompt="orig",
                feedback="fix spelling",
                suggestions=[],
                must_fix=[],
                revision_history=[{"round": 1}],
            )

    assert painter._images_edit_attempt.await_count == 1
    assert painter._chat_image_attempt.await_count == 1
    assert painter._generate_image_dashscope.await_count == 0


# --------------------------------------------------------------------------- #
# _images_api_attempt internals (b64 -> URL response_format fallback preserved)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_images_api_attempt_b64_fallback_preserved():
    """Within one attempt, a 4xx on b64_json falls back to the no-format call."""
    painter = _make_painter(base_url="https://api.openai.com/v1")

    good_item = MagicMock()
    good_item.b64_json = "FALLBACKDATA"
    good_response = MagicMock()
    good_response.data = [good_item]
    good_response.model_dump = MagicMock(return_value={"data": ["x"]})

    painter.llm_client.client.images.generate = AsyncMock(side_effect=[
        Exception("Error code: 422 - response_format unsupported"),
        good_response,
    ])

    result = await painter._images_api_attempt("a prompt")

    assert result == "FALLBACKDATA"
    assert painter.llm_client.client.images.generate.await_count == 2


@pytest.mark.asyncio
async def test_images_edit_generic_403_does_not_repeat_denied_endpoint():
    painter = _make_painter(base_url="https://gateway.example/v1")
    painter.llm_client.client.images.edit = AsyncMock(
        side_effect=Exception(
            "Error code: 403 - {'error': {'code': 'bad_response_status_code'}}"
        )
    )

    with pytest.raises(Exception, match="403"):
        await painter._images_edit_attempt("revise", "QUJD")

    assert painter.llm_client.client.images.edit.await_count == 1
