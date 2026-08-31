from types import SimpleNamespace

import pytest

from backend.services.report.llm_calls import LLMCallsMixin


@pytest.mark.asyncio
async def test_report_tool_loop_prefers_responses_streaming():
    sentinel = object()

    class Harness(LLMCallsMixin):
        _session_id = "s"
        bio_model = "gpt-x"
        llm_client = SimpleNamespace(chat_with_tools=sentinel)

        def _set_llm_context(self, _agent_name):
            return None

        async def _budgeted_text_call(self, *args, **kwargs):
            self.captured = (args, kwargs)
            return "ok"

    harness = Harness()
    result = await harness._bio_chat_with_tools([
        {"role": "user", "content": "prepare report evidence"}
    ])

    assert result == "ok"
    args, kwargs = harness.captured
    assert args == ("bioinformatics", sentinel)
    assert kwargs["prefer_stream"] is True
