from backend.utils.llm_client import (
    _anthropic_messages_translate,
    _anthropic_translate_tools,
    _anthropic_translate_tool_choice,
)


def test_system_extracted_and_text_passthrough():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    system, out = _anthropic_messages_translate(msgs)
    assert system == "sys"
    assert out == [{"role": "user", "content": "hi"}]


def test_assistant_tool_calls_become_tool_use_blocks():
    msgs = [{
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "t1", "type": "function",
                        "function": {"name": "f", "arguments": "{\"a\": 1}"}}],
    }]
    _system, out = _anthropic_messages_translate(msgs)
    block = out[0]["content"][0]
    assert block == {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}}


def test_tool_result_message_becomes_user_tool_result():
    msgs = [{"role": "tool", "tool_call_id": "t1", "content": "42"}]
    _system, out = _anthropic_messages_translate(msgs)
    assert out[0]["role"] == "user"
    assert out[0]["content"][0] == {"type": "tool_result", "tool_use_id": "t1", "content": "42"}


def test_translate_tools_to_input_schema():
    tools = [{"type": "function", "function": {"name": "f", "description": "d",
              "parameters": {"type": "object"}}}]
    out = _anthropic_translate_tools(tools)
    assert out == [{"name": "f", "description": "d", "input_schema": {"type": "object"}}]


def test_tool_choice_mapping():
    assert _anthropic_translate_tool_choice("auto") == {"type": "auto"}
    assert _anthropic_translate_tool_choice("required") == {"type": "any"}
    assert _anthropic_translate_tool_choice(
        {"type": "function", "function": {"name": "f"}}) == {"type": "tool", "name": "f"}
    # none -> None signals "omit tools entirely" (Anthropic has no 'none')
    assert _anthropic_translate_tool_choice("none") is None


from backend.utils.llm_client import _parse_anthropic_dump, _anthropic_restore_thinking, ToolCall


def test_parse_anthropic_text_and_usage():
    dump = {"content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 4, "output_tokens": 6}}
    text, tool_calls, thinking, usage = _parse_anthropic_dump(dump)
    assert text == "hello"
    assert tool_calls == []
    assert thinking is None
    assert usage == {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}


def test_parse_anthropic_tool_use():
    dump = {"content": [{"type": "tool_use", "id": "tu1", "name": "f", "input": {"a": 1}}],
            "stop_reason": "tool_use", "usage": {"input_tokens": 1, "output_tokens": 1}}
    text, tool_calls, thinking, usage = _parse_anthropic_dump(dump)
    assert text == ""
    assert isinstance(tool_calls[0], ToolCall)
    assert tool_calls[0].id == "tu1" and tool_calls[0].arguments == {"a": 1}


def test_parse_anthropic_thinking_captured():
    dump = {"content": [
        {"type": "thinking", "thinking": "reason...", "signature": "sig123"},
        {"type": "tool_use", "id": "tu1", "name": "f", "input": {}},
    ], "stop_reason": "tool_use", "usage": {}}
    text, tool_calls, thinking, usage = _parse_anthropic_dump(dump)
    assert thinking == [{"type": "thinking", "thinking": "reason...", "signature": "sig123"}]


def test_restore_thinking_roundtrip():
    # raw_assistant_message carried thinking blocks under reasoning_details;
    # restoring puts them back as leading content blocks (with signature).
    msgs = [{
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "tu1", "name": "f", "input": {}}],
        "reasoning_details": [{"type": "thinking", "thinking": "r", "signature": "sig"}],
    }]
    restored = _anthropic_restore_thinking(msgs)
    blocks = restored[0]["content"]
    assert blocks[0] == {"type": "thinking", "thinking": "r", "signature": "sig"}
    assert "reasoning_details" not in restored[0]


import pytest
from types import SimpleNamespace
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


@pytest.mark.asyncio
async def test_anthropic_adapter_none_tool_choice_omits_tools():
    dump = {"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn", "usage": {}}
    client = _fake_anthropic(dump)
    adapter = _AnthropicAdapter.__new__(_AnthropicAdapter)
    adapter.client = client
    adapter.agent_name = "t"
    await adapter.complete_with_tools(
        model="claude-x", messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
        tool_choice="none", temperature=0.5, max_tokens=1000)
    kwargs = client.messages.create.call_args.kwargs
    assert "tools" not in kwargs and "tool_choice" not in kwargs


@pytest.mark.asyncio
async def test_anthropic_adapter_complete_vision_passes_timeout():
    dump = {"content": [{"type": "text", "text": "vision"}], "stop_reason": "end_turn", "usage": {}}
    client = _fake_anthropic(dump)
    adapter = _AnthropicAdapter.__new__(_AnthropicAdapter)
    adapter.client = client
    adapter.agent_name = "t"

    await adapter.complete_vision(
        model="claude-x", text_prompt="describe", images=[],
        temperature=0.5, max_tokens=1000, timeout=42.0,
        system_prompt="PATHOLOGY ROLE")

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["timeout"] == 42.0
    assert "PATHOLOGY ROLE" in kwargs["system"]
    assert "respond exclusively in English" in kwargs["system"]


@pytest.mark.asyncio
async def test_anthropic_adapter_streaming_text_aggregates_usage():
    async def _stream():
        yield MagicMock(
            type="message_start",
            message=MagicMock(
                model="claude-x",
                usage=MagicMock(
                    input_tokens=3,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=2,
                ),
            ),
        )
        yield MagicMock(
            type="content_block_delta",
            delta=MagicMock(type="text_delta", text="Hel"),
        )
        yield MagicMock(
            type="content_block_delta",
            delta=MagicMock(type="text_delta", text="lo"),
        )
        yield MagicMock(type="message_delta", usage=MagicMock(output_tokens=2))
        yield MagicMock(type="message_stop")

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=_stream())
    adapter = _AnthropicAdapter.__new__(_AnthropicAdapter)
    adapter.client = client
    adapter.agent_name = "t"

    result = await adapter.complete(
        model="claude-x",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
        max_tokens=1000,
        stream=True,
    )

    assert result.content == "Hello"
    assert result.usage == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
        "cached_tokens": 2,
    }
    assert client.messages.create.call_args.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_anthropic_streaming_usage_preserves_missing_cache_telemetry():
    async def _stream():
        yield SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                model="claude-x",
                usage=SimpleNamespace(input_tokens=3),
            ),
        )
        yield SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="Hello"),
        )
        yield SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(output_tokens=2),
        )
        yield SimpleNamespace(type="message_stop")

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=_stream())
    adapter = _AnthropicAdapter.__new__(_AnthropicAdapter)
    adapter.client = client
    adapter.agent_name = "t"

    result = await adapter.complete(
        model="claude-x",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
        max_tokens=1000,
        stream=True,
    )

    assert result.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }


@pytest.mark.asyncio
async def test_anthropic_streaming_reasoning_only_max_tokens_is_incomplete():
    async def _stream():
        yield SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                model="claude-x",
                usage=SimpleNamespace(input_tokens=3),
            ),
        )
        yield SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="thinking_delta", thinking="reasoning"),
        )
        yield SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="max_tokens"),
            usage=SimpleNamespace(output_tokens=8),
        )
        yield SimpleNamespace(type="message_stop")

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=_stream())
    adapter = _AnthropicAdapter.__new__(_AnthropicAdapter)
    adapter.client = client
    adapter.agent_name = "t"

    result = await adapter.complete(
        model="claude-x",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
        max_tokens=8,
        stream=True,
    )

    assert result.content == ""
    assert result.status == "incomplete"
    assert result.incomplete_details == {"reason": "max_tokens"}


@pytest.mark.asyncio
async def test_anthropic_nonstream_reasoning_only_max_tokens_is_incomplete():
    dump = {
        "content": [{"type": "thinking", "thinking": "reasoning"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 3, "output_tokens": 8},
    }
    client = _fake_anthropic(dump)
    adapter = _AnthropicAdapter.__new__(_AnthropicAdapter)
    adapter.client = client
    adapter.agent_name = "t"

    result = await adapter.complete(
        model="claude-x",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
        max_tokens=8,
    )

    assert result.content == ""
    assert result.status == "incomplete"
    assert result.incomplete_details == {"reason": "max_tokens"}


@pytest.mark.asyncio
async def test_anthropic_midstream_eof_is_closed_and_not_accepted():
    from backend.utils.llm_client import StreamInterruptedError

    class ClosableStream:
        def __init__(self):
            self.events = iter([
                MagicMock(
                    type="message_start",
                    message=MagicMock(
                        model="claude-x",
                        usage=MagicMock(
                            input_tokens=1,
                            cache_creation_input_tokens=0,
                            cache_read_input_tokens=0,
                        ),
                    ),
                ),
                MagicMock(
                    type="content_block_delta",
                    delta=MagicMock(type="text_delta", text="partial"),
                ),
            ])
            self.close = AsyncMock()

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration as error:
                raise StopAsyncIteration from error

    stream = ClosableStream()
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=stream)
    adapter = _AnthropicAdapter.__new__(_AnthropicAdapter)
    adapter.client = client
    adapter.agent_name = "t"

    with pytest.raises(StreamInterruptedError, match="message_stop"):
        await adapter.complete(
            model="claude-x",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=1000,
            stream=True,
        )

    stream.close.assert_awaited_once()


from backend.utils.llm_client import _tool_calls_to_raw_assistant


def test_tool_calls_to_raw_assistant_serializes_tool_calls():
    raw = _tool_calls_to_raw_assistant(
        "use tool",
        [ToolCall(id="call_1", name="lookup", arguments={"gene": "TP53"})],
    )

    assert raw == {
        "role": "assistant",
        "content": "use tool",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"gene": "TP53"}'},
            }
        ],
    }


def test_tool_calls_to_raw_assistant_uses_none_for_empty_content_and_no_tools():
    raw = _tool_calls_to_raw_assistant("", [])
    assert raw == {"role": "assistant", "content": None, "tool_calls": None}


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
