"""Reasoning fields must never be exposed as assistant answer content."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import logging
from backend.utils.llm_client import (
    EMPTY_CONTENT_PLACEHOLDER,
    OpenAICompatibleClient,
)


@pytest.fixture
def client():
    value = OpenAICompatibleClient(api_key="test", base_url="http://test")
    try:
        yield value
    finally:
        asyncio.run(value.close())


def _make_message(content=None, **extra_attrs):
    values = {
        "content": content,
        "reasoning": None,
        "reasoning_content": None,
        "thinking": None,
        "thought": None,
        "text": None,
        "refusal": None,
        "tool_calls": None,
        **extra_attrs,
    }
    msg = SimpleNamespace(**values)
    msg.model_dump = lambda: dict(values)
    return msg


def _response(completion_tokens):
    return SimpleNamespace(
        usage=SimpleNamespace(completion_tokens=completion_tokens),
    )


@pytest.mark.parametrize(
    ("field", "private_text"),
    [
        ("reasoning", "private reasoning fixture"),
        ("reasoning_content", "private reasoning-content fixture"),
        ("thinking", "private thinking fixture"),
        ("thought", "private thought fixture"),
    ],
)
def test_reasoning_fields_are_not_extracted(client, caplog, field, private_text):
    msg = _make_message(content=None, **{field: private_text})

    with caplog.at_level(logging.WARNING):
        result = client._extract_content_from_message(msg, _response(42))

    assert result == EMPTY_CONTENT_PLACEHOLDER
    assert private_text not in result
    assert private_text not in caplog.text


def test_reasoning_value_is_not_logged_in_empty_content_diagnostic(client, caplog):
    private_text = "private recovered reasoning"
    msg = _make_message(content=None, reasoning=private_text)

    with caplog.at_level(logging.WARNING):
        result = client._extract_content_from_message(msg, _response(42))

    assert result == EMPTY_CONTENT_PLACEHOLDER
    assert "empty content" in caplog.text.lower()
    assert private_text not in caplog.text


def test_model_dump_failure_does_not_log_private_exception_text(client, caplog):
    private_text = "private model-dump exception fixture"
    msg = _make_message(content=None, reasoning=private_text)

    def fail_dump():
        raise RuntimeError(private_text)

    msg.model_dump = fail_dump
    with caplog.at_level(logging.DEBUG):
        result = client._extract_content_from_message(msg, _response(7))

    assert result == EMPTY_CONTENT_PLACEHOLDER
    assert private_text not in caplog.text
    assert "type=RuntimeError" in caplog.text
