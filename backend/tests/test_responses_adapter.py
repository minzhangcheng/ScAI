from backend.tests.split_facade import setattr_facade


from backend.utils.llm_client import _responses_messages_to_input


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


from backend.utils.llm_client import _parse_responses_dump


def test_parse_responses_bad_arguments_sets_parse_error():
    dump = {"output": [{"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{bad"}],
            "usage": {}}
    _, tool_calls, _, _ = _parse_responses_dump(dump)
    assert tool_calls[0].parse_error is not None
    assert tool_calls[0].arguments == {}


import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from backend.utils.llm_client import _ResponsesAdapter


@pytest.mark.asyncio
async def test_responses_adapter_incomplete_stream_preserves_terminal_metadata():
    incomplete = MagicMock()
    incomplete.model = "gpt-x"
    incomplete.model_dump.return_value = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "error": None,
        "output": [{"type": "reasoning"}],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 32768,
            "total_tokens": 32778,
        },
    }

    async def events():
        yield SimpleNamespace(type="response.incomplete", response=incomplete)

    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock(return_value=events())
    adapter = _ResponsesAdapter(client, agent_name="bioinformatics")

    result = await adapter.complete(
        model="gpt-x",
        messages=[{"role": "user", "content": "answer"}],
        temperature=0.2,
        max_tokens=32768,
        stream=True,
    )

    assert result.content == ""
    assert result.status == "incomplete"
    assert result.incomplete_details == {"reason": "max_output_tokens"}
    assert result.usage["completion_tokens"] == 32768


@pytest.mark.asyncio
async def test_create_fallback_strips_background_and_retries(monkeypatch):
    setattr_facade(monkeypatch, "backend.utils.llm_client", "_NO_BACKGROUND_ENDPOINT", set())
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
    setattr_facade(monkeypatch, "backend.utils.llm_client", "_NO_BACKGROUND_ENDPOINT", set())
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock(side_effect=RuntimeError("upstream 500"))
    adapter = _ResponsesAdapter(client, agent_name="t", base_url="https://api.x/v1")

    with pytest.raises(RuntimeError):
        await adapter._create_responses_with_fallback(
            {"model": "o4-mini-deep-research", "input": [], "background": True})
    assert client.responses.create.await_count == 1
