"""``chat_without_history`` owns the replay budget for its 29 call sites.

``BaseAgent.chat`` already replays a stream-interrupted request rather than
escalating a transient blip to the LLM-failure gate. ``chat_without_history`` is
the larger surface — PI annotation, PI figures, literature, pathology and the
bioinformatics planner all reach the model through it — and carried no budget of
its own, so every one of those escalated on the first interruption.

Putting the budget inside the method, not at each call site, is what covers all
29 at once. It also means the one caller that already wrapped its own call
(``bioinformatics``'s ``_call_planning_llm``) must stop doing so: two nested
budgets would turn a persistently broken upstream into 4x4 = 16 paid attempts.
``on_retry`` exists so that caller keeps its user-visible notification while the
budget itself lives in exactly one place.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.base import BaseAgent
from backend.models.schemas import AgentType
from backend.utils.llm_client import UpstreamExhaustedError
from backend.utils.llm_client_parts.responses import LLMResponse
from backend.utils.settings_provider import BudgetedTextDispatch


class _ProbeAgent(BaseAgent):
    async def process(self, state, message=None):  # pragma: no cover - unused
        return None


def _interrupted(detail: str) -> UpstreamExhaustedError:
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
    sleep = AsyncMock()
    monkeypatch.setattr(
        "backend.utils.llm_client_parts.retry.asyncio.sleep", sleep
    )
    return sleep


@pytest.mark.asyncio
async def test_chat_without_history_returns_the_reply_a_replay_recovers():
    agent, client, _ = _make_agent([
        _interrupted("eof 1"), _interrupted("eof 2"), _reply("recovered"),
    ])

    assert await agent.chat_without_history("question") == "recovered"
    assert client.chat.await_count == 3


@pytest.mark.asyncio
async def test_chat_without_history_replays_the_identical_request():
    agent, client, _ = _make_agent([_interrupted("eof"), _reply("recovered")])

    await agent.chat_without_history("question")

    first = client.chat.await_args_list[0]
    assert all(call == first for call in client.chat.await_args_list)


@pytest.mark.asyncio
async def test_chat_without_history_budgets_once_across_replays():
    agent, _, provider = _make_agent([
        _interrupted("eof 1"), _interrupted("eof 2"), _reply("recovered"),
    ])

    await agent.chat_without_history("question")

    assert provider.budget_text_dispatch.await_count == 1
    assert provider.acknowledge_text_dispatch.await_count == 1


@pytest.mark.asyncio
async def test_chat_without_history_keeps_history_untouched_across_replays():
    """The method's defining property must survive the new retry."""
    agent, _, _ = _make_agent([_interrupted("eof"), _reply("recovered")])
    agent.add_to_history("user", "prior")

    await agent.chat_without_history("question")

    assert agent.message_history == [{"role": "user", "content": "prior"}]


@pytest.mark.asyncio
async def test_chat_without_history_escalates_when_replays_are_exhausted():
    agent, client, _ = _make_agent([_interrupted(f"eof {i}") for i in range(5)])

    with pytest.raises(UpstreamExhaustedError) as exc_info:
        await agent.chat_without_history("question")

    assert exc_info.value.failure_kind == "stream_interrupted"
    assert "remained incomplete after 3 retries" in str(exc_info.value)
    assert client.chat.await_count == 4


@pytest.mark.asyncio
async def test_chat_without_history_does_not_replay_an_unrelated_error():
    agent, client, _ = _make_agent([RuntimeError("boom"), _reply("unreached")])

    with pytest.raises(RuntimeError, match="boom"):
        await agent.chat_without_history("question")

    assert client.chat.await_count == 1


@pytest.mark.asyncio
async def test_chat_without_history_reports_each_replay_to_the_caller():
    """The passthrough that lets planning keep its status message."""
    seen = []

    async def on_retry(attempt, max_retries, delay, error):
        seen.append((attempt, max_retries, delay, type(error).__name__))

    agent, _, _ = _make_agent([
        _interrupted("eof 1"), _interrupted("eof 2"), _reply("recovered"),
    ])

    await agent.chat_without_history("question", on_retry=on_retry)

    assert [row[0] for row in seen] == [1, 2]
    assert [row[1] for row in seen] == [3, 3]
    assert [row[2] for row in seen] == [2.0, 4.0]


# --------------------------------------------------------------------------
# The budget must exist exactly once on the one path that already had its own
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_planning_spends_one_replay_budget_not_two():
    """_call_planning_llm must stop wrapping now that the method carries it.

    Two nested budgets multiply: 4 outer attempts x 4 inner = 16 paid calls
    against an upstream that is persistently broken, each one able to run for
    minutes.
    """
    from backend.agents.bioinformatics_agent import BioinformaticsAgent
    from backend.models.schemas import AnalysisState

    client = MagicMock()
    client.chat = AsyncMock(side_effect=_interrupted("eof"))
    client.base_url = "https://example.invalid"

    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_system_prompt.return_value = "system"
    provider.get_temperature.return_value = 0.3
    provider.get_config_revision.return_value = 1
    provider.get_vision_setting.return_value = False
    provider.get_llm_client.return_value = client
    provider.budget_text_dispatch = AsyncMock(
        side_effect=lambda label, messages, **kwargs: BudgetedTextDispatch(
            messages=[dict(m) for m in messages], report=None, max_tokens=None,
        )
    )
    provider.acknowledge_text_dispatch = AsyncMock()

    agent = BioinformaticsAgent(settings_provider=provider, sandbox=None)
    agent.session_id = "test-session"
    agent._notify = AsyncMock()

    with pytest.raises(UpstreamExhaustedError):
        await agent.generate_analysis_plan(
            state=AnalysisState(user_id="test-user"),
            pi_tasks=["QC"],
            overview="overview",
            phase="exploration",
        )

    assert client.chat.await_count == 4
