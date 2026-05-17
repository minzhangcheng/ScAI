"""Tests for _extract_content_from_message reasoning field extraction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import MagicMock
from backend.utils.llm_client import OpenAICompatibleClient


@pytest.fixture
def client():
    return OpenAICompatibleClient(api_key="test", base_url="http://test")


def _make_message(content=None, **extra_attrs):
    msg = MagicMock()
    msg.content = content
    for k, v in extra_attrs.items():
        setattr(msg, k, v)
    msg.model_dump.return_value = {"content": content, **extra_attrs}
    return msg


def test_reasoning_key_extracted(client):
    msg = _make_message(content=None, reasoning="chain-of-thought text")
    assert client._extract_content_from_message(msg) == "chain-of-thought text"


def test_reasoning_content_key_extracted(client):
    msg = _make_message(content=None, reasoning_content="deepseek cot")
    assert client._extract_content_from_message(msg) == "deepseek cot"


def test_thinking_key_extracted(client):
    msg = _make_message(content=None, thinking="alternative provider cot")
    assert client._extract_content_from_message(msg) == "alternative provider cot"


def test_thought_key_extracted(client):
    msg = _make_message(content=None, thought="another provider cot")
    assert client._extract_content_from_message(msg) == "another provider cot"


def test_reasoning_list_content_extracted(client):
    msg = MagicMock()
    msg.content = None
    msg.reasoning = None
    msg.reasoning_content = None
    msg.thinking = None
    msg.thought = None
    msg.text = None
    msg.refusal = None
    msg.model_dump.return_value = {
        "content": None,
        "reasoning": [{"type": "text", "text": "list-based reasoning"}],
    }
    result = client._extract_content_from_message(msg)
    assert "list-based reasoning" in result


def test_content_takes_priority_over_reasoning(client):
    msg = _make_message(content="primary content", reasoning="should not be used")
    assert client._extract_content_from_message(msg) == "primary content"
