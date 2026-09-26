from backend.utils.llm_client import _anthropic_messages_translate


from backend.utils.llm_client import _anthropic_restore_thinking


import pytest
from unittest.mock import AsyncMock, MagicMock
from backend.utils.llm_client import _AnthropicAdapter, LLMToolResponse


def _fake_anthropic(dump):
    resp = MagicMock()
    resp.model_dump.return_value = dump
    resp.model = "claude-x"
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=resp)
    return client


@pytest.mark.asyncio
async def test_anthropic_adapter_complete_with_tools_roundtrips_thinking():
    dump = {"content": [
        {"type": "thinking", "thinking": "r", "signature": "sig"},
        {"type": "tool_use", "id": "tu1", "name": "f", "input": {"a": 1}},
    ], "stop_reason": "tool_use", "usage": {"input_tokens": 1, "output_tokens": 2}}
    client = _fake_anthropic(dump)
    adapter = _AnthropicAdapter.__new__(_AnthropicAdapter)  # bypass __init__ (no real SDK)
    adapter.client = client
    adapter.agent_name = "t"
    r = await adapter.complete_with_tools(
        model="claude-x", messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
        tool_choice="auto", temperature=0.5, max_tokens=1000)
    assert isinstance(r, LLMToolResponse)
    assert r.tool_calls[0].name == "f"
    # thinking captured into raw_assistant_message.reasoning_details for echo
    assert r.raw_assistant_message["reasoning_details"][0]["signature"] == "sig"
    # max_tokens passed (Anthropic requires it)
    assert client.messages.create.call_args.kwargs["max_tokens"] == 1000


def test_translate_preserves_restored_thinking_before_tool_use():
    msgs = [{
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "tu1",
            "type": "function",
            "function": {"name": "f", "arguments": "{\"a\": 1}"},
        }],
        "reasoning_details": [{
            "type": "thinking",
            "thinking": "reason...",
            "signature": "sig123",
        }],
    }]

    restored = _anthropic_restore_thinking(msgs)
    _system, out = _anthropic_messages_translate(restored)

    assert out == [{
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "reason...", "signature": "sig123"},
            {"type": "tool_use", "id": "tu1", "name": "f", "input": {"a": 1}},
        ],
    }]
