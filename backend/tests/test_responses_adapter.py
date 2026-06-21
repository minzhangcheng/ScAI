from backend.utils.llm_client import LLMResponse


def test_llmresponse_citations_defaults_none():
    r = LLMResponse(content="hi", model="m", usage={}, raw_response=None)
    assert r.citations is None


def test_llmresponse_citations_settable():
    cites = [{"title": "T", "url": "http://x"}]
    r = LLMResponse(content="hi", model="m", usage={}, raw_response=None, citations=cites)
    assert r.citations == cites


from backend.utils.llm_client import (
    _responses_messages_to_input,
    _responses_translate_tools,
    _responses_translate_tool_choice,
)


def test_messages_to_input_extracts_system_as_instructions():
    msgs = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    instructions, items = _responses_messages_to_input(msgs)
    assert instructions == "be brief"
    assert items == [{"role": "user", "content": "hi"}]


def test_messages_to_input_no_system():
    instructions, items = _responses_messages_to_input([{"role": "user", "content": "hi"}])
    assert instructions is None
    assert items == [{"role": "user", "content": "hi"}]


def test_translate_tools_to_flat():
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]
    out = _responses_translate_tools(tools)
    assert out == [{"type": "function", "name": "f", "description": "d", "parameters": {"type": "object"}}]


def test_translate_tool_choice_passthrough():
    assert _responses_translate_tool_choice("auto") == "auto"
    assert _responses_translate_tool_choice("required") == "required"


def test_translate_tool_choice_specific_function():
    tc = {"type": "function", "function": {"name": "f"}}
    assert _responses_translate_tool_choice(tc) == {"type": "function", "name": "f"}


from backend.utils.llm_client import _parse_responses_dump, ToolCall


def test_parse_responses_text_and_citations():
    dump = {
        "output": [
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Hello.",
                 "annotations": [{"type": "url_citation", "url": "http://a", "title": "A"}]},
            ]},
        ],
        "usage": {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8},
    }
    text, tool_calls, citations, usage = _parse_responses_dump(dump)
    assert text == "Hello."
    assert tool_calls == []
    assert citations == [{"title": "A", "url": "http://a"}]
    assert usage == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}


def test_parse_responses_function_call():
    dump = {
        "output": [
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{\"a\": 1}"},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    text, tool_calls, citations, usage = _parse_responses_dump(dump)
    assert text == ""
    assert len(tool_calls) == 1
    assert isinstance(tool_calls[0], ToolCall)
    assert tool_calls[0].id == "c1" and tool_calls[0].name == "f"
    assert tool_calls[0].arguments == {"a": 1}


def test_parse_responses_bad_arguments_sets_parse_error():
    dump = {"output": [{"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{bad"}],
            "usage": {}}
    _, tool_calls, _, _ = _parse_responses_dump(dump)
    assert tool_calls[0].parse_error is not None
    assert tool_calls[0].arguments == {}


import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from backend.utils.llm_client import _ResponsesAdapter, LLMToolResponse


def _fake_openai_with_response(dump):
    resp_obj = MagicMock()
    resp_obj.model_dump.return_value = dump
    resp_obj.model = "gpt-x"
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock(return_value=resp_obj)
    return client


@pytest.mark.asyncio
async def test_responses_adapter_complete():
    dump = {"output": [{"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "hi"}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
    client = _fake_openai_with_response(dump)
    adapter = _ResponsesAdapter(client, agent_name="t")
    r = await adapter.complete(model="gpt-x", messages=[{"role": "user", "content": "hi"}],
                               temperature=0.5, max_tokens=100)
    assert r.content == "hi"
    assert r.usage["prompt_tokens"] == 1
    # instructions/input were passed
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["model"] == "gpt-x"
    assert kwargs["max_output_tokens"] == 100


@pytest.mark.asyncio
async def test_responses_adapter_with_tools_injects_web_search_for_dr():
    dump = {"output": [{"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}],
            "usage": {}}
    client = _fake_openai_with_response(dump)
    adapter = _ResponsesAdapter(client, agent_name="t")
    r = await adapter.complete_with_tools(
        model="o3-deep-research", messages=[{"role": "user", "content": "x"}],
        tools=[], tool_choice="auto", temperature=0.5, max_tokens=None)
    assert isinstance(r, LLMToolResponse)
    assert r.has_tool_calls
    sent_tools = client.responses.create.call_args.kwargs["tools"]
    assert {"type": "web_search_preview"} in sent_tools


@pytest.mark.asyncio
async def test_responses_adapter_stream_yields_text_deltas():
    async def _fake_stream():
        yield SimpleNamespace(type="response.output_text.delta", delta="Hel")
        yield SimpleNamespace(type="response.output_text.delta", delta="lo")
        yield SimpleNamespace(type="response.completed", delta=None)

    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock(return_value=_fake_stream())
    adapter = _ResponsesAdapter(client, agent_name="t")
    chunks = []
    async for c in adapter.stream(model="gpt-x", messages=[{"role": "user", "content": "hi"}],
                                  temperature=0.5, max_tokens=None):
        chunks.append(c)
    assert "".join(chunks) == "Hello"
