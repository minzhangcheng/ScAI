import asyncio
import json

import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock
from backend.utils.llm_client import OpenAICompatibleClient


@pytest_asyncio.fixture
async def client():
    value = OpenAICompatibleClient(api_key="test", base_url="http://test")
    try:
        yield value
    finally:
        await value.close()


@pytest.mark.asyncio
async def test_emit_log_response_tolerates_magicmock_logger(client):
    client.llm_logger = MagicMock()
    await client._emit_log_response(interaction=MagicMock(), response_content="x", usage=None, duration_ms=1.0)
    client.llm_logger.log_response.assert_called_once()


@pytest.mark.asyncio
async def test_emit_log_response_awaits_real_coroutine(client):
    client.llm_logger = MagicMock()
    client.llm_logger.log_response_async = AsyncMock()
    await client._emit_log_response(interaction=MagicMock(), response_content="x", usage=None, duration_ms=1.0)
    client.llm_logger.log_response_async.assert_awaited_once()
    client.llm_logger.log_response.assert_not_called()


@pytest.mark.asyncio
async def test_emit_log_response_falls_back_to_sync_only_logger(client):
    class SyncOnlyLogger:
        def __init__(self):
            self.calls = 0

        def log_response(self, **kwargs):
            self.calls += 1

    logger = SyncOnlyLogger()
    client.llm_logger = logger

    await client._emit_log_response(
        interaction=MagicMock(),
        response_content="x",
        usage=None,
        duration_ms=1.0,
    )

    assert logger.calls == 1


@pytest.mark.asyncio
async def test_emit_log_response_writes_one_daily_and_one_session_record(
    tmp_path, monkeypatch, client,
):
    from backend.utils import llm_logger as llm_logger_module

    daily_dir = tmp_path / "daily"
    session_dir = tmp_path / "session"
    monkeypatch.setattr(llm_logger_module, "LLM_LOG_DIR", daily_dir)
    monkeypatch.setattr(
        llm_logger_module,
        "get_session_logs",
        lambda session_id: session_dir,
    )

    async def inline_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    # The host test runner's thread-pool shutdown is unreliable under Python
    # 3.13; execute the same logging callable inline so this test verifies the
    # exact write count without testing asyncio's executor implementation.
    monkeypatch.setattr(llm_logger_module.asyncio, "to_thread", inline_to_thread)

    logger = object.__new__(llm_logger_module.LLMLogger)
    logger.enabled = True
    logger.log_to_console = False
    logger._async_lock = asyncio.Lock()
    interaction = logger.log_request(
        session_id="session-1",
        agent="test",
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        cache={"strategy": "automatic"},
    )
    client.llm_logger = logger

    await client._emit_log_response(
        interaction=interaction,
        response_content="world",
        usage=None,
        duration_ms=1.0,
    )

    daily_files = list(daily_dir.glob("*.jsonl"))
    assert len(daily_files) == 1
    assert daily_files[0].read_text().count("\n") == 1
    assert (session_dir / "llm.jsonl").read_text().count("\n") == 1
    daily_record = json.loads(daily_files[0].read_text())
    session_record = json.loads((session_dir / "llm.jsonl").read_text())
    assert "messages" not in daily_record
    assert daily_record["message_count"] == 1
    assert daily_record["cache"] == {"strategy": "automatic"}
    assert session_record["messages"][0]["content"] == "hello"
    assert session_record["cache"] == {"strategy": "automatic"}


def test_session_log_is_bounded_and_redacts_inline_images(tmp_path, monkeypatch):
    from backend.utils import llm_logger as llm_logger_module

    daily_dir = tmp_path / "daily"
    session_dir = tmp_path / "session"
    monkeypatch.setattr(llm_logger_module, "LLM_LOG_DIR", daily_dir)
    monkeypatch.setattr(llm_logger_module, "get_session_logs", lambda _sid: session_dir)

    logger = object.__new__(llm_logger_module.LLMLogger)
    logger.enabled = True
    logger.log_to_console = False
    logger._async_lock = asyncio.Lock()
    messages = [{"role": "system", "content": "rules"}]
    messages.extend(
        {"role": "user", "content": "x" * 50_000 + str(i)}
        for i in range(30)
    )
    messages.append({
        "role": "user",
        "content": [{
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + "A" * 100_000},
        }],
    })
    interaction = logger.log_request(
        session_id="bounded",
        agent="test",
        model="model",
        messages=messages,
    )
    logger.log_response(interaction, response_content="y" * 100_000)

    record = json.loads((session_dir / "llm.jsonl").read_text())
    assert record["message_count"] == len(messages)
    assert len(record["messages"]) == llm_logger_module._MAX_LOG_MESSAGES
    assert "inline image omitted" in str(record["messages"][-1])
    assert len(record["response_content"]) < 41_000
    assert (session_dir / "llm.jsonl").stat().st_size < 300_000


def test_compact_messages_omits_private_reasoning_everywhere():
    from backend.utils.llm_logger import _compact_messages

    private_values = {
        "top reasoning",
        "top reasoning content",
        "top reasoning details",
        "top thinking",
        "top thought",
        "thinking block",
        "redacted thinking block",
        "unknown reasoning block",
        "untyped reasoning block",
    }
    message = {
        "role": "assistant",
        "reasoning": "top reasoning",
        "reasoning_content": "top reasoning content",
        "reasoning_details": [{"text": "top reasoning details"}],
        "thinking": "top thinking",
        "thought": "top thought",
        "content": [
            {"type": "thinking", "text": "thinking block"},
            {"type": "redacted_thinking", "text": "redacted thinking block"},
            {"type": "vendor_reasoning_v9", "text": "unknown reasoning block"},
            {
                "reasoning_content": "provider metadata",
                "text": "untyped reasoning block",
            },
            {"type": "text", "text": "public answer"},
        ],
        "refusal": "public refusal",
        "tool_calls": [{"id": "tool-1", "type": "function"}],
    }

    compacted = _compact_messages([message])[0]
    rendered = json.dumps(compacted)

    for private_value in private_values:
        assert private_value not in rendered
    for key in (
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "thinking",
        "thought",
    ):
        assert key not in compacted
    assert "public answer" in rendered
    assert compacted["refusal"] == "public refusal"
    assert compacted["tool_calls"] == [{"id": "tool-1", "type": "function"}]
