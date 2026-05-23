"""Tests for _recover_stringified_tool_call helper.

Background: some upstream gateways (notably litellm proxying n1n.ai for
Claude / Gemini) occasionally lose the structured tool_calls field and put
the entire tool_use JSON into message.content as a string. This helper
recognizes and parses those stringified payloads so downstream agent
loops do not mistake them for plain-text responses.
"""
import asyncio
import pytest
from unittest.mock import MagicMock
from backend.utils.llm_client import OpenAICompatibleClient, _recover_stringified_tool_call


def test_recovers_n1n_style_tool_parameters():
    """n1n.ai's stringified form uses {"tool": ..., "parameters": ...}."""
    content = '{"tool":"report_findings","parameters":{"summary":"done","findings":[]}}'
    result = _recover_stringified_tool_call(content)
    assert result is not None
    name, args = result
    assert name == "report_findings"
    assert args == {"summary": "done", "findings": []}


def test_recovers_openai_style_name_arguments():
    """OpenAI-style stringified tool call uses {"name": ..., "arguments": ...}."""
    content = '{"name":"execute_code","arguments":{"code":"print(1)","description":"x","step_type":"technical"}}'
    result = _recover_stringified_tool_call(content)
    assert result is not None
    name, args = result
    assert name == "execute_code"
    assert args["code"] == "print(1)"


def test_recovers_anthropic_style_tool_use_input():
    """Anthropic-native form uses {"type":"tool_use","name":...,"input":...}."""
    content = '{"type":"tool_use","name":"consult_pi","input":{"question":"q","context":"c"}}'
    result = _recover_stringified_tool_call(content)
    assert result is not None
    name, args = result
    assert name == "consult_pi"
    assert args == {"question": "q", "context": "c"}


def test_recovers_from_markdown_fenced_json():
    """Some providers wrap the JSON in a ```json fence."""
    content = '```json\n{"tool":"report_findings","parameters":{"summary":"x"}}\n```'
    result = _recover_stringified_tool_call(content)
    assert result is not None
    name, args = result
    assert name == "report_findings"
    assert args == {"summary": "x"}


def test_recovers_from_fence_with_trailing_newline():
    """Trailing newline after closing fence must not prevent recovery."""
    content = '```json\n{"tool":"report_findings","parameters":{"summary":"x"}}\n```\n'
    result = _recover_stringified_tool_call(content)
    assert result is not None
    name, args = result
    assert name == "report_findings"
    assert args == {"summary": "x"}


def test_ignores_plain_text():
    """Plain text responses must not be misinterpreted as tool calls."""
    assert _recover_stringified_tool_call("I will now proceed to analyze...") is None
    assert _recover_stringified_tool_call("") is None
    assert _recover_stringified_tool_call("   ") is None


def test_ignores_invalid_json():
    """Strings that start like JSON but fail to parse must return None."""
    content = '{"tool":"report_findings","parameters":{"summary":"unterminated'
    assert _recover_stringified_tool_call(content) is None


def test_ignores_json_without_recognized_keys():
    """JSON objects that don't match any known tool-call shape return None."""
    content = '{"foo":"bar","baz":42}'
    assert _recover_stringified_tool_call(content) is None


def test_ignores_args_not_dict():
    """If arguments/parameters/input is not a dict, treat as not-a-tool-call."""
    content = '{"tool":"report_findings","parameters":"oops"}'
    assert _recover_stringified_tool_call(content) is None


def test_ignores_missing_name():
    """If the name/tool field is absent or empty, return None."""
    content = '{"parameters":{"summary":"x"}}'
    assert _recover_stringified_tool_call(content) is None
    content2 = '{"tool":"","parameters":{}}'
    assert _recover_stringified_tool_call(content2) is None


def _make_fake_response(content: str, tool_calls=None):
    """Build a fake openai SDK response with given content and optional tool_calls."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    message.model_dump = MagicMock(return_value={
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    })
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    response.model = "test-model"
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    return response


def test_chat_with_tools_recovers_stringified_call(monkeypatch):
    """When upstream returns no tool_calls but content is a stringified tool call,
    chat_with_tools must synthesize a ToolCall and clear content."""
    client = OpenAICompatibleClient(api_key="k", base_url="http://x", agent_name="t")

    stringified = '{"tool":"report_findings","parameters":{"summary":"done"}}'
    fake_response = _make_fake_response(stringified, tool_calls=None)

    async def fake_call(*args, **kwargs):
        return fake_response

    monkeypatch.setattr(client, "_call_with_temp_fallback", fake_call)

    result = asyncio.run(client.chat_with_tools(
        model="claude-sonnet-4.6",
        messages=[{"role": "user", "content": "go"}],
        tools=[{"type": "function", "function": {"name": "report_findings",
                                                      "parameters": {"type": "object"}}}],
    ))

    assert result.has_tool_calls, "recovery should have synthesized a ToolCall"
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.name == "report_findings"
    assert tc.arguments == {"summary": "done"}
    assert tc.id.startswith("recovered_")
    assert result.content is None, "content must be cleared after recovery"


def test_chat_with_tools_keeps_plain_text_when_not_a_tool_call(monkeypatch):
    """Genuine plain-text replies must not be misinterpreted as tool calls."""
    client = OpenAICompatibleClient(api_key="k", base_url="http://x", agent_name="t")

    plain = "I will now proceed with the analysis."
    fake_response = _make_fake_response(plain, tool_calls=None)

    async def fake_call(*args, **kwargs):
        return fake_response

    monkeypatch.setattr(client, "_call_with_temp_fallback", fake_call)

    result = asyncio.run(client.chat_with_tools(
        model="x", messages=[{"role": "user", "content": "go"}],
        tools=[{"type": "function", "function": {"name": "f",
                                                      "parameters": {"type": "object"}}}],
    ))

    assert not result.has_tool_calls
    assert result.content == plain


def test_chat_with_tools_native_tool_calls_unaffected(monkeypatch):
    """When upstream returns proper tool_calls, recovery path must not run."""
    client = OpenAICompatibleClient(api_key="k", base_url="http://x", agent_name="t")

    native_tc = MagicMock()
    native_tc.id = "call_native_1"
    native_tc.function.name = "execute_code"
    native_tc.function.arguments = '{"code":"print(1)"}'

    fake_response = _make_fake_response(content=None, tool_calls=[native_tc])

    async def fake_call(*args, **kwargs):
        return fake_response

    monkeypatch.setattr(client, "_call_with_temp_fallback", fake_call)

    result = asyncio.run(client.chat_with_tools(
        model="x", messages=[{"role": "user", "content": "go"}],
        tools=[{"type": "function", "function": {"name": "execute_code",
                                                      "parameters": {"type": "object"}}}],
    ))

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_native_1"
    assert not result.tool_calls[0].id.startswith("recovered_")
