"""Tests for PainterAgent's unified protocol-cascade + round-retry behavior."""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agents.painter_agent import (
    PainterAgent,
    _GOOGLE_AI_STUDIO_BASE_URL,
    _MAX_RETRY_ROUNDS,
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
    llm_client.api_key = "test-key"
    llm_client.set_context = MagicMock()
    llm_client.client = MagicMock()

    @asynccontextmanager
    async def direct_request_scope():
        yield llm_client

    llm_client.direct_request_scope = direct_request_scope

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


# Protocol ordering


# Cascade behavior


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


# Revision uses the images/edits protocol first (default order)


# Fatal-error classifier


# _images_api_attempt internals (b64 -> URL response_format fallback preserved)
