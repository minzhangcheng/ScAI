import asyncio

import pytest
from unittest.mock import MagicMock, AsyncMock
from backend.utils.llm_client import OpenAICompatibleClient


@pytest.mark.asyncio
async def test_emit_log_response_tolerates_magicmock_logger():
    c = OpenAICompatibleClient(api_key="test", base_url="http://test")
    c.llm_logger = MagicMock()
    await c._emit_log_response(interaction=MagicMock(), response_content="x", usage=None, duration_ms=1.0)
    c.llm_logger.log_response.assert_called_once()


@pytest.mark.asyncio
async def test_emit_log_response_awaits_real_coroutine():
    c = OpenAICompatibleClient(api_key="test", base_url="http://test")
    c.llm_logger = MagicMock()
    c.llm_logger.log_response_async = AsyncMock()
    await c._emit_log_response(interaction=MagicMock(), response_content="x", usage=None, duration_ms=1.0)
    c.llm_logger.log_response_async.assert_awaited_once()
    c.llm_logger.log_response.assert_not_called()


@pytest.mark.asyncio
async def test_emit_log_response_falls_back_to_sync_only_logger():
    class SyncOnlyLogger:
        def __init__(self):
            self.calls = 0

        def log_response(self, **kwargs):
            self.calls += 1

    c = OpenAICompatibleClient(api_key="test", base_url="http://test")
    logger = SyncOnlyLogger()
    c.llm_logger = logger

    await c._emit_log_response(
        interaction=MagicMock(),
        response_content="x",
        usage=None,
        duration_ms=1.0,
    )

    assert logger.calls == 1


@pytest.mark.asyncio
async def test_emit_log_response_writes_one_daily_and_one_session_record(
    tmp_path, monkeypatch
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
    )
    client = OpenAICompatibleClient(api_key="test", base_url="http://test")
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
