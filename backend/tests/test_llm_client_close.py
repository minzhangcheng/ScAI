"""Tests: OpenAICompatibleClient.close() idempotency and lifecycle."""
import asyncio
import logging
import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.asyncio
async def test_client_close_idempotent():
    from backend.utils.llm_client import OpenAICompatibleClient
    c = OpenAICompatibleClient(api_key="test-key", base_url="http://x")
    assert hasattr(c, "_closed")
    assert c._closed is False
    await c.close()
    assert c._closed is True
    await c.close()
    assert c._closed is True


@pytest.mark.asyncio
async def test_client_close_idempotent_google_deep_research():
    from backend.utils.llm_client import GoogleDeepResearchClient
    c = GoogleDeepResearchClient(api_key="test-key")
    assert hasattr(c, "_closed")
    assert c._closed is False
    await c.close()
    assert c._closed is True
    await c.close()
    assert c._closed is True


@pytest.mark.asyncio
async def test_llm_manager_close_idempotent():
    from backend.utils.llm_client import LLMManager
    m = LLMManager(api_key="test-key", base_url="http://x")
    assert m.llm._closed is False
    await m.close()
    assert m.llm._closed is True
    await m.close()
    assert m.llm._closed is True


def test_client_del_warns_if_not_closed():
    import backend.utils.llm_client as llm_mod
    from backend.utils.llm_client import OpenAICompatibleClient
    with patch.object(llm_mod.logger, "warning") as mock_warning:
        c = OpenAICompatibleClient(api_key="test-key", base_url="http://x")
        c.__del__()
        mock_warning.assert_called_once()
        args, _ = mock_warning.call_args
        assert "leaked" in args[0]


def test_client_del_no_warning_if_closed():
    import backend.utils.llm_client as llm_mod
    from backend.utils.llm_client import OpenAICompatibleClient
    c = OpenAICompatibleClient(api_key="test-key", base_url="http://x")
    asyncio.run(c.close())
    with patch.object(llm_mod.logger, "warning") as mock_warning:
        c.__del__()
        mock_warning.assert_not_called()
