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


def test_reasoning_list_content_is_not_extracted(client):
    private_text = "private list reasoning fixture"
    msg = _make_message(
        content=None,
        reasoning=[{"type": "text", "text": private_text}],
    )
    result = client._extract_content_from_message(msg, _response(17))
    assert result == EMPTY_CONTENT_PLACEHOLDER
    assert private_text not in result


@pytest.mark.parametrize(
    "private_content",
    [
        [{"type": "thinking", "text": "private content-block fixture"}],
        {"reasoning_content": "private dict fixture"},
    ],
)
def test_reasoning_shaped_content_payload_is_not_stringified(
    client, private_content
):
    msg = _make_message(content=private_content)

    result = client._extract_content_from_message(msg, _response(11))

    assert result == EMPTY_CONTENT_PLACEHOLDER
    assert "private" not in result


def test_untyped_block_with_reasoning_key_does_not_expose_text(client):
    private_text = "private untyped reasoning fixture"
    msg = _make_message(
        content=[{"reasoning": "provider metadata", "text": private_text}]
    )

    result = client._extract_content_from_message(msg, _response(11))

    assert result == EMPTY_CONTENT_PLACEHOLDER
    assert private_text not in result


@pytest.mark.parametrize(
    "text_value",
    [
        {"text": "private dict message.text fixture"},
        [{"type": "thinking", "text": "private list message.text fixture"}],
    ],
)
def test_structured_message_text_is_not_an_answer_channel(client, text_value):
    msg = _make_message(content=None, text=text_value)

    result = client._extract_content_from_message(msg, _response(11))

    assert result == EMPTY_CONTENT_PLACEHOLDER
    assert "private" not in result


def test_refusal_and_tool_only_and_zero_token_contracts_remain_intact(client):
    refusal = _make_message(content=None, refusal="policy refusal")
    assert client._extract_content_from_message(refusal) == "[Refusal]: policy refusal"

    tool_only = _make_message(
        content=None,
        tool_calls=[{"id": "tool-1", "type": "function"}],
        reasoning="private tool reasoning",
    )
    assert client._extract_content_from_message(tool_only, _response(11)) == ""

    zero_token = _make_message(content=None, reasoning="private zero-token reasoning")
    assert client._extract_content_from_message(zero_token, _response(0)) == ""


def test_content_takes_priority_over_reasoning(client):
    msg = _make_message(content="primary content", reasoning="should not be used")
    assert client._extract_content_from_message(msg) == "primary content"


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


def test_true_empty_response_logs_field_presence_not_field_values(client, caplog):
    msg = _make_message(
        content=None,
        reasoning=None,
        reasoning_content=None,
        thinking=None,
        thought=None,
        text=None,
        refusal=None,
        tool_calls=None,
    )
    with caplog.at_level(logging.WARNING):
        result = client._extract_content_from_message(msg, _response(12))

    assert result == EMPTY_CONTENT_PLACEHOLDER
    assert "empty content" in caplog.text.lower()
    assert "values preview" not in caplog.text


def test_zero_token_empty_answer_remains_empty_for_retry_contract(client):
    msg = _make_message(content=None, reasoning="private zero-token fixture")

    assert client._extract_content_from_message(msg, _response(0)) == ""
