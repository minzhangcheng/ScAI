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


def test_messages_to_input_converts_assistant_tool_calls_to_responses_function_calls():
    msgs = [
        {"role": "system", "content": "be precise"},
        {"role": "user", "content": "run code"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "execute_code",
                        "arguments": '{"code": "print(1)"}',
                    },
                }
            ],
        },
    ]
    instructions, items = _responses_messages_to_input(msgs)
    assert instructions == "be precise"
    assert items == [
        {"role": "user", "content": "run code"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "execute_code",
            "arguments": '{"code": "print(1)"}',
        },
    ]
    assert all("tool_calls" not in item for item in items)


def test_messages_to_input_converts_tool_result_to_responses_function_call_output():
    msgs = [
        {"role": "user", "content": "continue"},
        {"role": "tool", "tool_call_id": "call_1", "content": "stdout: 1"},
    ]
    instructions, items = _responses_messages_to_input(msgs)
    assert instructions is None
    assert items == [
        {"role": "user", "content": "continue"},
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "stdout: 1",
        },
    ]


def test_messages_to_input_converts_bioinformatics_tool_history_without_raw_chat_tool_fields():
    msgs = [
        {"role": "system", "content": "You are a bioinformatics analyst."},
        {"role": "user", "content": "Analyze the dataset."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_QhuLHeAd3okoIPcQrr5fflbj",
                    "type": "function",
                    "function": {
                        "name": "execute_code",
                        "arguments": '{"code": "print(\'inventory\')", "description": "Inventory inputs"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_QhuLHeAd3okoIPcQrr5fflbj",
            "content": "Allowed count-like inputs:\\nfiltered_feature_bc_matrix.h5",
        },
        {"role": "user", "content": "Continue with QC."},
    ]
    instructions, items = _responses_messages_to_input(msgs)
    assert instructions == "You are a bioinformatics analyst."
    assert items == [
        {"role": "user", "content": "Analyze the dataset."},
        {
            "type": "function_call",
            "call_id": "call_QhuLHeAd3okoIPcQrr5fflbj",
            "name": "execute_code",
            "arguments": '{"code": "print(\'inventory\')", "description": "Inventory inputs"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_QhuLHeAd3okoIPcQrr5fflbj",
            "output": "Allowed count-like inputs:\\nfiltered_feature_bc_matrix.h5",
        },
        {"role": "user", "content": "Continue with QC."},
    ]
    assert all("tool_calls" not in item for item in items)
    assert all("tool_call_id" not in item for item in items)


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
async def test_responses_adapter_complete_sends_explicit_web_search_tool():
    dump = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "researched"}],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    }
    client = _fake_openai_with_response(dump)
    adapter = _ResponsesAdapter(client, agent_name="literature")

    result = await adapter.complete(
        model="o4-mini-deep-research",
        messages=[{"role": "user", "content": "research copper transport"}],
        temperature=0.3,
        max_tokens=None,
        timeout=1800.0,
        tools=[{"type": "web_search_preview"}],
    )

    assert result.content == "researched"
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["tools"] == [{"type": "web_search_preview"}]
    assert kwargs["timeout"] == 1800.0


@pytest.mark.asyncio
async def test_responses_adapter_complete_retries_explicit_web_search_tool_name():
    success_resp = MagicMock()
    success_resp.model_dump.return_value = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "researched"}],
            }
        ],
        "usage": {},
    }
    success_resp.model = "gpt-x"
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock(
        side_effect=[Exception("Unknown tool type: web_search_preview"), success_resp]
    )
    adapter = _ResponsesAdapter(client, agent_name="literature")

    result = await adapter.complete(
        model="o4-mini-deep-research",
        messages=[{"role": "user", "content": "research copper transport"}],
        temperature=0.3,
        max_tokens=None,
        tools=[{"type": "web_search_preview"}],
    )

    assert result.content == "researched"
    first_call = client.responses.create.call_args_list[0].kwargs
    second_call = client.responses.create.call_args_list[1].kwargs
    assert first_call["tools"] == [{"type": "web_search_preview"}]
    assert second_call["tools"] == [{"type": "web_search"}]


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
async def test_responses_adapter_complete_with_tools_sends_converted_tool_history():
    dump = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    }
    client = _fake_openai_with_response(dump)
    adapter = _ResponsesAdapter(client, agent_name="bioinformatics")
    messages = [
        {"role": "system", "content": "You are a bioinformatics analyst."},
        {"role": "user", "content": "Analyze the dataset."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "execute_code",
                        "arguments": '{"code": "print(1)"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "1"},
        {"role": "user", "content": "Continue."},
    ]
    result = await adapter.complete_with_tools(
        model="gpt-5.5",
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "description": "Execute Python code",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice="auto",
        temperature=0.2,
        max_tokens=4096,
    )
    assert result.content == "done"
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["instructions"] == "You are a bioinformatics analyst."
    assert kwargs["input"] == [
        {"role": "user", "content": "Analyze the dataset."},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "execute_code",
            "arguments": '{"code": "print(1)"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "1",
        },
        {"role": "user", "content": "Continue."},
    ]
    assert all("tool_calls" not in item for item in kwargs["input"])
    assert all("tool_call_id" not in item for item in kwargs["input"])


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


def test_messages_to_input_prevents_unknown_parameter_input_tool_calls_regression():
    messages = [
        {"role": "system", "content": "You are a professional bioinformatics analyst."},
        {"role": "user", "content": "Project context"},
        {"role": "user", "content": "Plan context"},
        {"role": "user", "content": "Execute stage 1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_QhuLHeAd3okoIPcQrr5fflbj",
                    "type": "function",
                    "function": {
                        "name": "execute_code",
                        "arguments": '{"code": "import pathlib\\nprint(\'Allowed count-like inputs:\')", "description": "Inventory count-matrix inputs"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_QhuLHeAd3okoIPcQrr5fflbj",
            "content": "Allowed count-like inputs:\\nfiltered_feature_bc_matrix.h5\\nExcluded loom inputs:\\nCon.loom",
        },
        {"role": "user", "content": "Continue analysis."},
    ]

    instructions, input_items = _responses_messages_to_input(messages)

    assert instructions == "You are a professional bioinformatics analyst."
    assert input_items[3] == {
        "type": "function_call",
        "call_id": "call_QhuLHeAd3okoIPcQrr5fflbj",
        "name": "execute_code",
        "arguments": '{"code": "import pathlib\\nprint(\'Allowed count-like inputs:\')", "description": "Inventory count-matrix inputs"}',
    }
    assert "tool_calls" not in input_items[3]
    assert input_items[4] == {
        "type": "function_call_output",
        "call_id": "call_QhuLHeAd3okoIPcQrr5fflbj",
        "output": "Allowed count-like inputs:\\nfiltered_feature_bc_matrix.h5\\nExcluded loom inputs:\\nCon.loom",
    }
    assert all(item.get("role") != "tool" for item in input_items)


def test_build_params_adds_background_for_deep_research():
    adapter = _ResponsesAdapter(MagicMock(), agent_name="t")
    params = adapter._build_params(
        "o4-mini-deep-research", [{"role": "user", "content": "x"}], 0.3, None,
        background_ok=True)
    assert params.get("background") is True


def test_build_params_no_background_for_normal_model():
    adapter = _ResponsesAdapter(MagicMock(), agent_name="t")
    params = adapter._build_params(
        "gpt-5.5", [{"role": "user", "content": "x"}], 0.3, None)
    assert "background" not in params


def test_build_params_no_background_when_streaming_deep_research():
    adapter = _ResponsesAdapter(MagicMock(), agent_name="t")
    params = adapter._build_params(
        "o4-mini-deep-research", [{"role": "user", "content": "x"}], 0.3, None,
        stream=True)
    assert "background" not in params


def test_build_params_no_background_without_background_ok():
    adapter = _ResponsesAdapter(MagicMock(), agent_name="t")
    params = adapter._build_params(
        "o4-mini-deep-research", [{"role": "user", "content": "x"}], 0.3, None)
    assert "background" not in params


def test_build_params_skips_background_when_endpoint_cached(monkeypatch):
    monkeypatch.setattr(
        "backend.utils.llm_client._NO_BACKGROUND_ENDPOINT", {"https://api.x/v1"})
    adapter = _ResponsesAdapter(MagicMock(), agent_name="t", base_url="https://api.x/v1")
    params = adapter._build_params(
        "o4-mini-deep-research", [{"role": "user", "content": "x"}], 0.3, None,
        background_ok=True)
    assert "background" not in params


@pytest.mark.asyncio
async def test_complete_with_tools_deep_research_no_background():
    dump = {"output": [{"type": "function_call", "call_id": "c1", "name": "f",
            "arguments": "{}"}], "usage": {}}
    client = _fake_openai_with_response(dump)
    adapter = _ResponsesAdapter(client, agent_name="t", base_url="https://api.x/v1")
    await adapter.complete_with_tools(
        model="o3-deep-research", messages=[{"role": "user", "content": "x"}],
        tools=[], tool_choice="auto", temperature=0.5, max_tokens=None)
    assert "background" not in client.responses.create.call_args.kwargs


@pytest.mark.asyncio
async def test_complete_vision_deep_research_no_background():
    dump = {"output": [{"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "hi"}]}], "usage": {}}
    client = _fake_openai_with_response(dump)
    adapter = _ResponsesAdapter(client, agent_name="t", base_url="https://api.x/v1")
    await adapter.complete_vision(
        model="o4-mini-deep-research", text_prompt="x", images=[],
        temperature=0.5, max_tokens=None)
    assert "background" not in client.responses.create.call_args.kwargs


def _mk_resp(status, resp_id="resp_1", dump=None, error=None):
    r = MagicMock()
    r.status = status
    r.id = resp_id
    if dump is not None:
        r.model_dump.return_value = dump
    if error is not None:
        r.error = error
    return r


@pytest.mark.asyncio
async def test_await_completion_returns_immediately_when_completed():
    done = _mk_resp("completed")
    adapter = _ResponsesAdapter(MagicMock(), agent_name="t")
    result = await adapter._await_completion(done)
    assert result is done


@pytest.mark.asyncio
async def test_await_completion_polls_until_completed(monkeypatch):
    monkeypatch.setattr("backend.utils.llm_client._DR_BACKGROUND_POLL_INTERVAL", 0)
    in_progress = _mk_resp("in_progress")
    done = _mk_resp("completed")
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.retrieve = AsyncMock(side_effect=[in_progress, done])
    adapter = _ResponsesAdapter(client, agent_name="t")
    start = _mk_resp("in_progress")
    result = await adapter._await_completion(start)
    assert result is done
    assert client.responses.retrieve.await_count == 2
    client.responses.retrieve.assert_awaited_with("resp_1")


@pytest.mark.asyncio
async def test_await_completion_raises_on_failed(monkeypatch):
    monkeypatch.setattr("backend.utils.llm_client._DR_BACKGROUND_POLL_INTERVAL", 0)
    failed = _mk_resp("failed", error="boom")
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.retrieve = AsyncMock(return_value=failed)
    adapter = _ResponsesAdapter(client, agent_name="t")
    start = _mk_resp("in_progress")
    with pytest.raises(RuntimeError):
        await adapter._await_completion(start)


@pytest.mark.asyncio
async def test_await_completion_tolerates_transient_poll_errors(monkeypatch):
    monkeypatch.setattr("backend.utils.llm_client._DR_BACKGROUND_POLL_INTERVAL", 0)
    done = _mk_resp("completed")
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.retrieve = AsyncMock(
        side_effect=[Exception("temporary network blip"), done])
    adapter = _ResponsesAdapter(client, agent_name="t")
    start = _mk_resp("in_progress")
    result = await adapter._await_completion(start)
    assert result is done
    assert client.responses.retrieve.await_count == 2


@pytest.mark.asyncio
async def test_await_completion_raises_after_repeated_poll_errors(monkeypatch):
    monkeypatch.setattr("backend.utils.llm_client._DR_BACKGROUND_POLL_INTERVAL", 0)
    monkeypatch.setattr("backend.utils.llm_client._DR_BACKGROUND_MAX_POLL_ERRORS", 3)
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.retrieve = AsyncMock(side_effect=Exception("down"))
    adapter = _ResponsesAdapter(client, agent_name="t")
    start = _mk_resp("in_progress")
    with pytest.raises(Exception):
        await adapter._await_completion(start)
    assert client.responses.retrieve.await_count == 3


@pytest.mark.asyncio
async def test_complete_deep_research_uses_background_polling(monkeypatch):
    monkeypatch.setattr("backend.utils.llm_client._DR_BACKGROUND_POLL_INTERVAL", 0)
    dump = {
        "output": [
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "report"}]}
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    submitted = _mk_resp("in_progress")
    submitted.model = "o4-mini-deep-research"
    completed = _mk_resp("completed", dump=dump)
    completed.model = "o4-mini-deep-research"
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock(return_value=submitted)
    client.responses.retrieve = AsyncMock(return_value=completed)
    adapter = _ResponsesAdapter(client, agent_name="literature")

    r = await adapter.complete(
        model="o4-mini-deep-research",
        messages=[{"role": "user", "content": "x"}],
        temperature=0.3,
        max_tokens=None,
        timeout=1800.0,
        tools=[{"type": "web_search_preview"}],
    )

    assert r.content == "report"
    assert client.responses.create.call_args.kwargs.get("background") is True
    client.responses.retrieve.assert_awaited_with("resp_1")


def test_background_polling_constants_exist():
    from backend.utils import llm_client as lc
    assert lc._DR_BACKGROUND_POLL_INTERVAL > 0
    assert lc._DR_BACKGROUND_MAX_WAIT >= 600
    assert lc._DR_BACKGROUND_MAX_POLL_ERRORS >= 1
    assert "failed" in lc._DR_BACKGROUND_TERMINAL_FAILED


def test_is_unsupported_background_error_detects():
    from backend.utils.llm_client import _is_unsupported_background_error
    assert _is_unsupported_background_error(Exception("Unknown parameter: 'background'")) is True
    assert _is_unsupported_background_error(Exception("background mode is not supported")) is True
    assert _is_unsupported_background_error(Exception("'background' is not allowed")) is True


def test_is_unsupported_background_error_ignores_unrelated():
    from backend.utils.llm_client import _is_unsupported_background_error
    assert _is_unsupported_background_error(Exception("rate limit exceeded")) is False
    # mentions background but is not a rejection of the parameter
    assert _is_unsupported_background_error(Exception("the background of the histology image")) is False


def test_no_background_endpoint_cache_exists():
    from backend.utils import llm_client as lc
    assert isinstance(lc._NO_BACKGROUND_ENDPOINT, set)


@pytest.mark.asyncio
async def test_create_fallback_strips_background_and_retries(monkeypatch):
    monkeypatch.setattr("backend.utils.llm_client._NO_BACKGROUND_ENDPOINT", set())
    import backend.utils.llm_client as lc
    success = MagicMock()
    success.model_dump.return_value = {"output": [], "usage": {}}
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock(
        side_effect=[Exception("Unknown parameter: 'background'"), success])
    adapter = _ResponsesAdapter(client, agent_name="t", base_url="https://api.x/v1")

    result = await adapter._create_responses_with_fallback(
        {"model": "o4-mini-deep-research", "input": [], "background": True})

    assert result is success
    assert client.responses.create.await_count == 2
    second_kwargs = client.responses.create.call_args_list[1].kwargs
    assert "background" not in second_kwargs
    assert "https://api.x/v1" in lc._NO_BACKGROUND_ENDPOINT


@pytest.mark.asyncio
async def test_create_fallback_does_not_strip_on_unrelated_error(monkeypatch):
    monkeypatch.setattr("backend.utils.llm_client._NO_BACKGROUND_ENDPOINT", set())
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock(side_effect=RuntimeError("upstream 500"))
    adapter = _ResponsesAdapter(client, agent_name="t", base_url="https://api.x/v1")

    with pytest.raises(RuntimeError):
        await adapter._create_responses_with_fallback(
            {"model": "o4-mini-deep-research", "input": [], "background": True})
    assert client.responses.create.await_count == 1
