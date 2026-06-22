"""Tests for PainterAgent Images API (Strategy 2) retry behavior."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agents.painter_agent import PainterAgent, _GOOGLE_AI_STUDIO_BASE_URL
from backend.models.schemas import AgentType


def _make_painter(base_url: str = f"https://{_GOOGLE_AI_STUDIO_BASE_URL}/v1beta"):
    """Build a PainterAgent without running __init__.

    BaseAgent exposes model/system_prompt/llm_client/config as read-only
    properties backed by a settings provider, so we mock the provider.

    Default base_url is Google AI Studio so Strategy 1 (chat modalities) is
    skipped and tests exercise Strategy 2 directly.
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


@pytest.mark.asyncio
async def test_images_api_retries_on_transient_then_succeeds():
    """A 5xx/transient failure is retried; a later success is returned."""
    painter = _make_painter()
    attempt = AsyncMock(side_effect=[
        Exception("Error code: 500 - server error"),
        Exception("Error code: 503 - unavailable"),
        "BASE64DATA",
    ])
    painter._images_api_attempt = attempt

    with patch("backend.agents.painter_agent.asyncio.sleep", new=AsyncMock()) as sleep:
        result = await painter._generate_image("a prompt")

    assert result == "BASE64DATA"
    assert attempt.await_count == 3
    assert sleep.await_count == 2  # backoff before attempts 2 and 3


@pytest.mark.asyncio
async def test_images_api_does_not_retry_on_client_error():
    """A 4xx client error breaks out immediately without retrying or sleeping."""
    painter = _make_painter()
    attempt = AsyncMock(side_effect=Exception("Error code: 400 - bad request"))
    painter._images_api_attempt = attempt

    with patch("backend.agents.painter_agent.asyncio.sleep", new=AsyncMock()) as sleep:
        with pytest.raises(ValueError, match="Image generation failed"):
            await painter._generate_image("a prompt")

    assert attempt.await_count == 1
    assert sleep.await_count == 0


@pytest.mark.asyncio
async def test_images_api_raises_after_all_retries_exhausted():
    """Persistent transient failures exhaust 3 attempts then raise."""
    painter = _make_painter()
    attempt = AsyncMock(side_effect=Exception("Error code: 502 - bad gateway"))
    painter._images_api_attempt = attempt

    with patch("backend.agents.painter_agent.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(ValueError, match="Images API: .*502"):
            await painter._generate_image("a prompt")

    assert attempt.await_count == 3


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
