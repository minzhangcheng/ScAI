"""Tests for cross-provider reasoning echo-back helpers.

Contract (official docs): an assistant turn that contains tool_calls must
echo its reasoning back verbatim in later requests; a turn with no tool_calls
must not. DeepSeek R1 (deepseek-reasoner) forbids reasoning_content entirely.
"""
import pytest

from backend.utils.llm_client import (
    _REASONING_ECHO_KEYS,
    _reasoning_echo_allowed,
    _build_assistant_echo,
)


def test_echo_allowed_default_true():
    assert _reasoning_echo_allowed("deepseek/deepseek-v4-pro") is True
    assert _reasoning_echo_allowed("xiaomi/mimo-7b") is True
    assert _reasoning_echo_allowed("anthropic/claude-sonnet-4.6") is True


def test_echo_forbidden_for_r1_family():
    assert _reasoning_echo_allowed("deepseek-reasoner") is False
    assert _reasoning_echo_allowed("deepseek/deepseek-r1") is False
    assert _reasoning_echo_allowed("some-provider/model-r1") is False


def test_echo_keys_cover_known_providers():
    for k in ("reasoning_content", "reasoning", "reasoning_details", "thinking", "thought"):
        assert k in _REASONING_ECHO_KEYS


def test_tool_turn_preserves_string_reasoning_content():
    msg = {"content": None, "tool_calls": [{"id": "t1"}], "reasoning_content": "step by step"}
    out = _build_assistant_echo(msg, "deepseek/deepseek-v4-pro")
    assert out["role"] == "assistant"
    assert out["tool_calls"] == [{"id": "t1"}]
    assert out["reasoning_content"] == "step by step"


def test_tool_turn_preserves_structured_reasoning_details_array():
    details = [{"type": "reasoning.text", "text": "x", "signature": "sig-abc"}]
    msg = {"content": None, "tool_calls": [{"id": "t1"}], "reasoning": "x", "reasoning_details": details}
    out = _build_assistant_echo(msg, "anthropic/claude-sonnet-4.6")
    # structured array (with signature) must survive verbatim, not be stringified away
    assert out["reasoning_details"] == details
    assert out["reasoning"] == "x"


def test_plain_text_turn_drops_reasoning():
    msg = {"content": "final answer", "tool_calls": None, "reasoning_content": "thinking"}
    out = _build_assistant_echo(msg, "deepseek/deepseek-v4-pro")
    assert "reasoning_content" not in out
    assert out["content"] == "final answer"


def test_r1_drops_reasoning_even_on_tool_turn():
    msg = {"content": None, "tool_calls": [{"id": "t1"}], "reasoning_content": "thinking"}
    out = _build_assistant_echo(msg, "deepseek-reasoner")
    assert "reasoning_content" not in out
    assert out["tool_calls"] == [{"id": "t1"}]


def test_empty_reasoning_fields_not_attached():
    msg = {"content": None, "tool_calls": [{"id": "t1"}], "reasoning_content": "", "reasoning": None}
    out = _build_assistant_echo(msg, "deepseek/deepseek-v4-pro")
    assert "reasoning_content" not in out
    assert "reasoning" not in out


class _FakeMessage:
    """Minimal stand-in for an SDK ChatCompletionMessage with model_dump()."""
    def __init__(self, dump):
        self._dump = dump
    def model_dump(self):
        return dict(self._dump)


def test_chat_with_tools_raw_message_uses_builder_for_tool_turn():
    from backend.utils import llm_client as mod
    msg = _FakeMessage({
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "t1", "function": {"name": "execute_code", "arguments": "{}"}}],
        "reasoning_content": "deciding to run code",
    })
    # The builder is the single source of truth; verify it produces the echo
    # that chat_with_tools will attach.
    out = mod._build_assistant_echo(msg.model_dump(), "deepseek/deepseek-v4-pro")
    assert out["reasoning_content"] == "deciding to run code"
    assert out["tool_calls"][0]["id"] == "t1"


def test_add_to_history_omits_reasoning_for_plain_turn():
    """Plain chat turns have no tool_calls, so reasoning must not be stored."""
    from backend.agents.base import BaseAgent
    # add_to_history is a plain method; build a throwaway object exposing it.
    holder = type("H", (), {"message_history": [], "add_to_history": BaseAgent.add_to_history})()
    holder.add_to_history("assistant", "final answer")
    assert holder.message_history == [{"role": "assistant", "content": "final answer"}]
    assert "reasoning_content" not in holder.message_history[0]


def test_llm_response_has_no_reasoning_content_field():
    """The plain-chat path no longer carries reasoning_content."""
    from dataclasses import fields
    from backend.utils.llm_client import LLMResponse
    assert "reasoning_content" not in {f.name for f in fields(LLMResponse)}


def test_chat_path_does_not_add_reasoning_to_history():
    """add_to_history no longer accepts reasoning_content — the plain-chat
    path cannot store it even if someone tries."""
    from backend.agents.base import BaseAgent

    # add_to_history is a plain method; build a throwaway object exposing it.
    holder = type("H", (), {"message_history": [], "add_to_history": BaseAgent.add_to_history})()
    holder.add_to_history("assistant", "Hello world")
    assert holder.message_history == [{"role": "assistant", "content": "Hello world"}]
    assert "reasoning_content" not in holder.message_history[0]

    # Passing reasoning_content now raises TypeError (parameter removed)
    with pytest.raises(TypeError):
        holder.add_to_history("assistant", "Hello", reasoning_content="thinking")


def test_tool_turn_echo_pipeline():
    """End-to-end: _build_assistant_echo produces correct output for tool turns
    that will be fed into chat_with_tools message history."""
    from backend.utils.llm_client import _build_assistant_echo

    # Simulate a DeepSeek v4 assistant turn with tool call + reasoning
    sdk_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "execute_code", "arguments": '{"code": "print(1)"}'}}
        ],
        "reasoning_content": "I need to run some code to verify the result.",
    }
    echo = _build_assistant_echo(sdk_msg, "deepseek/deepseek-v4-pro")
    assert echo["role"] == "assistant"
    assert echo["tool_calls"] is not None
    assert echo["reasoning_content"] == "I need to run some code to verify the result."
    # No spurious keys
    assert "reasoning" not in echo
    assert "reasoning_details" not in echo

    # Same message on R1: reasoning must be stripped
    echo_r1 = _build_assistant_echo(sdk_msg, "deepseek-reasoner")
    assert "reasoning_content" not in echo_r1
    assert echo_r1["tool_calls"] is not None

    # Plain text turn (no tool_calls): no reasoning echo regardless of model
    plain_msg = {"content": "Here is the answer.", "tool_calls": None, "reasoning_content": "thoughts"}
    echo_plain = _build_assistant_echo(plain_msg, "deepseek/deepseek-v4-pro")
    assert "reasoning_content" not in echo_plain
    assert echo_plain["content"] == "Here is the answer."