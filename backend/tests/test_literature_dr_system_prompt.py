"""_deep_research_via_chat must inject agent system_prompt as system message."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock
from backend.agents.literature_agent import LiteratureAgent


def _make_agent():
    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_temperature.return_value = 0.3
    provider.get_system_prompt.return_value = "ROLE-SYSTEM-PROMPT"
    provider.get_literature_config.return_value = {}
    provider.get_config_revision.return_value = 0
    provider.clear_agent_client_cache = MagicMock()
    provider.budget_text_dispatch = AsyncMock(side_effect=(
        lambda _agent, messages, max_tokens=None, **_kwargs: SimpleNamespace(
            messages=messages,
            max_tokens=max_tokens,
            pi_observation_ids=(),
        )
    ))
    provider.acknowledge_text_dispatch = AsyncMock()
    client = MagicMock()
    client.set_context = MagicMock()
    resp = MagicMock()
    resp.content = "[1] Smith et al. (2023). Title. Journal. https://doi.org/x"
    client.chat = AsyncMock(return_value=resp)
    provider.get_llm_client.return_value = client
    agent = LiteratureAgent(
        settings_provider=provider,
        literature_mode="deep_research",
        deep_research_model="dr-model",
    )
    return agent, client


def test_dr_chat_includes_system_message():
    agent, client = _make_agent()
    asyncio.run(agent._deep_research_via_chat("some topic", state=None))
    dr_call = client.chat.call_args_list[0]
    messages = dr_call.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "ROLE-SYSTEM-PROMPT"
    assert messages[-1]["role"] == "user"


def test_deep_research_via_chat_requests_web_search_tool():
    agent, client = _make_agent()
    asyncio.run(agent._deep_research_via_chat("hepatic copper uptake", state=None))
    dr_call = client.chat.call_args_list[0]
    assert dr_call.kwargs["model"] == "dr-model"
    assert dr_call.kwargs["timeout"] == 1800.0
    assert dr_call.kwargs["tools"] == [{"type": "web_search_preview"}]


def test_findings_extract_max_chars_is_100k():
    assert LiteratureAgent._FINDINGS_EXTRACT_MAX_CHARS == 100000
