"""``BaseAgent.chat`` replays a stream-interrupted request before the HITL gate.

A Responses stream that ends without a terminal event reaches callers as
``UpstreamExhaustedError(failure_kind="stream_interrupted")`` — classified since
0.51.8 at the one dispatch exit every protocol shares. ``workflow/retry.py``
catches that class *before* its generic handler and parks the run on the
LLM-failure gate, so an agent without a replay budget of its own escalates a
transient blip straight to a human. ``bioinformatics`` already absorbs it with
``retry_replay_safe_llm_call``; these tests pin the same budget onto the shared
``BaseAgent.chat`` path that PI uses.

The replay must wrap the client call only. ``_apply_context_budget`` writes
state (``record_observation`` persists under the session lock), so budgeting a
second time for one logical dispatch is not a no-op.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.base import BaseAgent
from backend.models.schemas import AgentType
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


def _make_agent(client_side_effect):
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

    agent = _ProbeAgent(AgentType.PI, provider, session_id="test-session")
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
async def test_chat_returns_the_reply_a_replay_recovers():
    agent, client, _ = _make_agent([
        _stream_interrupted("eof 1"),
        _stream_interrupted("eof 2"),
        _reply("recovered"),
    ])

    assert await agent.chat("plan round 1") == "recovered"
    assert client.chat.await_count == 3


@pytest.mark.asyncio
async def test_chat_replays_the_identical_request():
    agent, client, _ = _make_agent([
        _stream_interrupted("eof 1"),
        _reply("recovered"),
    ])

    await agent.chat("plan round 1")

    first = client.chat.await_args_list[0]
    assert all(call == first for call in client.chat.await_args_list)


@pytest.mark.asyncio
async def test_chat_budgets_once_across_replays():
    agent, _, provider = _make_agent([
        _stream_interrupted("eof 1"),
        _stream_interrupted("eof 2"),
        _reply("recovered"),
    ])

    await agent.chat("plan round 1")

    assert provider.budget_text_dispatch.await_count == 1
    assert provider.acknowledge_text_dispatch.await_count == 1


@pytest.mark.asyncio
async def test_chat_records_one_turn_after_replays():
    agent, _, _ = _make_agent([
        _stream_interrupted("eof 1"),
        _reply("recovered"),
    ])

    await agent.chat("plan round 1")

    assert agent.message_history == [
        {"role": "user", "content": "plan round 1"},
        {"role": "assistant", "content": "recovered"},
    ]


@pytest.mark.asyncio
async def test_chat_escalates_to_the_gate_when_replays_are_exhausted():
    agent, client, _ = _make_agent([_stream_interrupted(f"eof {i}") for i in range(1, 6)])

    with pytest.raises(UpstreamExhaustedError) as exc_info:
        await agent.chat("plan round 1")

    assert exc_info.value.failure_kind == "stream_interrupted"
    assert "remained incomplete after 3 retries" in str(exc_info.value)
    assert client.chat.await_count == 4


@pytest.mark.asyncio
async def test_chat_does_not_replay_an_unrelated_upstream_failure():
    agent, client, _ = _make_agent([
        UpstreamExhaustedError(
            "503 from upstream",
            agent_name="pi",
            model="test-model",
            failure_kind="upstream_exhausted",
        ),
        _reply("never reached"),
    ])

    with pytest.raises(UpstreamExhaustedError):
        await agent.chat("plan round 1")

    assert client.chat.await_count == 1


@pytest.mark.asyncio
async def test_chat_does_not_replay_a_non_llm_error():
    agent, client, _ = _make_agent([RuntimeError("boom"), _reply("never reached")])

    with pytest.raises(RuntimeError, match="boom"):
        await agent.chat("plan round 1")

    assert client.chat.await_count == 1
