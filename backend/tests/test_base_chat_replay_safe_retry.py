"""Each text entrypoint owns one replay budget and acknowledges one dispatch."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.base import BaseAgent
from backend.models.schemas import AgentType, AnalysisState
from backend.utils.llm_client import UpstreamExhaustedError
from backend.utils.llm_client_parts.responses import LLMResponse
from backend.utils.settings_provider import BudgetedTextDispatch


class _ProbeAgent(BaseAgent):
    """Smallest concrete agent: ``process`` is the only abstract member."""

    async def process(self, state, message=None):  # pragma: no cover - unused
        return None


def _stream_interrupted(detail: str) -> UpstreamExhaustedError:
    return UpstreamExhaustedError(
        detail,
        agent_name="pi",
        model="test-model",
        llm_server="https://example.invalid",
        failure_kind="stream_interrupted",
    )


def _reply(content: str) -> LLMResponse:
    return LLMResponse(
        content=content, model="test-model", usage={}, raw_response=None
    )


def _make_agent(client_side_effect, agent_type=_ProbeAgent):
    client = MagicMock()
    client.chat = AsyncMock(side_effect=client_side_effect)
    client.base_url = "https://example.invalid"

    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_system_prompt.return_value = "system"
    provider.get_temperature.return_value = 0.3
    provider.get_config_revision.return_value = 1
    provider.get_llm_client.return_value = client
    provider.budget_text_dispatch = AsyncMock(
        side_effect=lambda label, messages, **kwargs: BudgetedTextDispatch(
            messages=[dict(message) for message in messages],
            report=None,
            max_tokens=None,
        )
    )
    provider.acknowledge_text_dispatch = AsyncMock()

    if agent_type is _ProbeAgent:
        agent = agent_type(AgentType.PI, provider, session_id="test-session")
    else:
        agent = agent_type(settings_provider=provider, sandbox=None)
        agent.session_id = "test-session"
        agent._notify = AsyncMock()
    return agent, client, provider


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Keep the 2s/4s/8s replay backoff out of the wall clock."""
    sleep = AsyncMock()
    monkeypatch.setattr(
        "backend.utils.llm_client_parts.retry.asyncio.sleep", sleep
    )
    return sleep


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["chat", "chat_without_history"])
async def test_chat_budgets_once_across_replays(entrypoint):
    agent, client, provider = _make_agent([
        _stream_interrupted("eof 1"),
        _stream_interrupted("eof 2"),
        _reply("recovered"),
    ])

    assert await getattr(agent, entrypoint)("plan round 1") == "recovered"
    assert client.chat.await_count == 3
    assert all(call == client.chat.await_args_list[0] for call in client.chat.await_args_list)
    assert provider.budget_text_dispatch.await_count == 1
    assert provider.acknowledge_text_dispatch.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["chat", "chat_without_history"])
async def test_chat_records_one_turn_after_replays(entrypoint):
    agent, _, _ = _make_agent([
        _stream_interrupted("eof 1"),
        _reply("recovered"),
    ])

    await getattr(agent, entrypoint)("plan round 1")

    expected = [
        {"role": "user", "content": "plan round 1"},
        {"role": "assistant", "content": "recovered"},
    ]
    assert agent.message_history == (expected if entrypoint == "chat" else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["chat", "chat_without_history"])
async def test_chat_escalates_to_the_gate_when_replays_are_exhausted(entrypoint):
    agent, client, _ = _make_agent([_stream_interrupted(f"eof {i}") for i in range(1, 6)])

    with pytest.raises(UpstreamExhaustedError) as exc_info:
        await getattr(agent, entrypoint)("plan round 1")

    assert exc_info.value.failure_kind == "stream_interrupted"
    assert "remained incomplete after 3 retries" in str(exc_info.value)
    assert client.chat.await_count == 4


@pytest.mark.asyncio
async def test_planning_spends_one_replay_budget_not_two():
    from backend.agents.bioinformatics_agent import BioinformaticsAgent

    agent, client, provider = _make_agent(
        _stream_interrupted("eof"), BioinformaticsAgent,
    )
    provider.get_vision_setting.return_value = False
    with pytest.raises(UpstreamExhaustedError):
        await agent.generate_analysis_plan(
            state=AnalysisState(user_id="owner"), pi_tasks=["QC"],
            overview="overview", phase="exploration",
        )
    assert client.chat.await_count == 4
    assert provider.budget_text_dispatch.await_count == 1
