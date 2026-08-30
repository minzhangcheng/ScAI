"""C1 — verify chat_sections passes (model, messages=...) and reads LLMResponse.content."""
import pytest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Dict


@dataclass
class _FakeLLMResponse:
    content: str


class _FakeLLMClient:
    """Records every chat() call's positional + keyword args."""
    def __init__(self, content: str):
        self._content = content
        self.calls: List[Dict[str, Any]] = []

    async def chat(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        return _FakeLLMResponse(content=self._content)


@pytest.mark.asyncio
async def test_titles_generated_via_correct_chat_signature(tmp_path):
    from backend.services.report.chat_sections import build_chat_sections_async

    messages = [
        {"role": "user", "content": "What pathways are enriched in cluster 3?"},
        {"role": "assistant", "content": "Pathway A and B are enriched."},
        {"role": "user", "content": "Compare clusters 1 and 2."},
        {"role": "assistant", "content": "Cluster 1 shows X, cluster 2 shows Y."},
    ]
    fake_titles = '[{"index": 1, "title": "Pathway enrichment cluster 3"},' \
                  ' {"index": 2, "title": "Compare clusters 1 and 2"}]'
    client = _FakeLLMClient(content=fake_titles)

    sections = await build_chat_sections_async(
        messages,
        figures_dir=str(tmp_path),
        tables_dir=str(tmp_path),
        llm_client=client,
        model="google/gemini-3.1-flash-lite-preview",
    )

    # 1) called with positional model + keyword messages
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "google/gemini-3.1-flash-lite-preview"
    assert isinstance(call["messages"], list)
    assert call["messages"][0]["role"] == "user"

    # 2) titles extracted from LLMResponse.content (not dict.get)
    assert sections[0]["title"] == "Pathway enrichment cluster 3"
    assert sections[1]["title"] == "Compare clusters 1 and 2"


@pytest.mark.asyncio
async def test_falls_back_to_truncated_content_when_no_model(tmp_path):
    """When model is None we cannot call LLM — fall back to truncated user content."""
    from backend.services.report.chat_sections import build_chat_sections_async

    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
    ]
    sections = await build_chat_sections_async(
        messages,
        figures_dir=str(tmp_path),
        tables_dir=str(tmp_path),
        llm_client=None,
        model=None,
    )
    assert sections[0]["title"] == "Q1"


@pytest.mark.asyncio
async def test_title_generation_uses_context_budget_callbacks(tmp_path):
    from backend.services.report.chat_sections import build_chat_sections_async

    messages = [
        {"role": "user", "content": "Original question"},
        {"role": "assistant", "content": "Original answer"},
    ]
    client = _FakeLLMClient(content='[{"index": 1, "title": "Budgeted title"}]')
    dispatch = SimpleNamespace(
        messages=[{"role": "user", "content": "Compacted title prompt"}],
        max_tokens=123,
    )
    budget_calls = []
    acknowledgements = []

    async def budget(agent_name, request_messages):
        budget_calls.append((agent_name, request_messages))
        return dispatch

    async def acknowledge(agent_name, acknowledged_dispatch):
        acknowledgements.append((agent_name, acknowledged_dispatch))

    sections = await build_chat_sections_async(
        messages,
        figures_dir=str(tmp_path),
        tables_dir=str(tmp_path),
        llm_client=client,
        model="title-model",
        context_budget_callback=budget,
        context_budget_ack_callback=acknowledge,
    )

    assert budget_calls[0][0] == "supervisor"
    assert client.calls[0]["messages"] == dispatch.messages
    assert client.calls[0]["kwargs"]["max_tokens"] == 123
    assert acknowledgements == [("supervisor", dispatch)]
    assert sections[0]["title"] == "Budgeted title"
