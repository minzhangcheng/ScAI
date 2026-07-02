import pytest
from unittest.mock import MagicMock, AsyncMock
from backend.utils.llm_client import OpenAICompatibleClient


@pytest.mark.asyncio
async def test_emit_log_response_tolerates_magicmock_logger():
    c = OpenAICompatibleClient(api_key="test", base_url="http://test")
    c.llm_logger = MagicMock()
    await c._emit_log_response(interaction=MagicMock(), response_content="x", usage=None, duration_ms=1.0)


@pytest.mark.asyncio
async def test_emit_log_response_awaits_real_coroutine():
    c = OpenAICompatibleClient(api_key="test", base_url="http://test")
    c.llm_logger = MagicMock()
    c.llm_logger.log_response_async = AsyncMock()
    await c._emit_log_response(interaction=MagicMock(), response_content="x", usage=None, duration_ms=1.0)
    c.llm_logger.log_response_async.assert_awaited_once()
