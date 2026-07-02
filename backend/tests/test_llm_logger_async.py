"""Tests for async LLM logging (Task 7.3)"""
import pytest
from backend.utils.llm_logger import LLMInteraction


sample_interaction = LLMInteraction(
    timestamp="2026-07-02T12:00:00",
    session_id="test_async",
    agent="test_agent",
    model="test_model",
    interaction_type="chat",
    messages=[{"role": "user", "content": "hello"}],
    response_content="hi there",
    usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    duration_ms=100.0,
)


@pytest.mark.asyncio
async def test_log_interaction_is_awaitable_and_serialized(tmp_path, monkeypatch):
    from backend.utils import llm_logger as llm_logger_module
    monkeypatch.setattr(llm_logger_module, "LLM_LOG_DIR", tmp_path)
    await llm_logger_module.llm_logger.log_interaction_async(sample_interaction)
    files = list(tmp_path.glob("*.jsonl"))
    assert files
    assert files[0].read_text().count("\n") == 1
