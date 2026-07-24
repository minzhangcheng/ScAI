"""Tests for _log_empty_content_diagnostic helper extracted from _extract_content_from_message."""
from backend.tests.split_facade import patch_facade
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import MagicMock, patch
from backend.utils.llm_client import OpenAICompatibleClient


@pytest.fixture
def client():
    return OpenAICompatibleClient(api_key="test", base_url="http://test")


def test_log_diagnostic_returns_completion_tokens(client):
    msg = MagicMock()
    msg.tool_calls = None
    msg.model_dump.return_value = {"content": None, "role": "assistant"}
    response = MagicMock()
    response.usage = MagicMock()
    response.usage.completion_tokens = 42
    response.usage.prompt_tokens = 10

    with patch_facade("backend.utils.llm_client", "logger"):
        tokens, has_tools = client._log_empty_content_diagnostic(msg, response)
    assert tokens == 42
    assert has_tools is False


def test_log_diagnostic_detects_tool_calls(client):
    msg = MagicMock()
    msg.tool_calls = [{"id": "tc1"}]
    msg.model_dump.return_value = {"content": None, "role": "assistant", "tool_calls": [{"id": "tc1"}]}
    response = MagicMock()
    response.usage = MagicMock()
    response.usage.completion_tokens = 50
    response.usage.prompt_tokens = 10

    with patch_facade("backend.utils.llm_client", "logger"):
        tokens, has_tools = client._log_empty_content_diagnostic(msg, response)
    assert tokens == 50
    assert has_tools is True


def test_log_diagnostic_returns_zero_when_no_usage(client):
    msg = MagicMock()
    response = MagicMock()
    response.usage = None

    with patch_facade("backend.utils.llm_client", "logger"):
        tokens, has_tools = client._log_empty_content_diagnostic(msg, response)
    assert tokens == 0
    assert has_tools is False
