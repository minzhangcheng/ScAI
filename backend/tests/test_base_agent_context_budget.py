"""BaseAgent applies one shared budget and acknowledges PI context on success."""

from types import SimpleNamespace

import pytest

from backend.agents.base import BaseAgent
from backend.config import SystemConfig
from backend.models.schemas import AgentType, AnalysisState, ContextBudgetObservation
from backend.utils.context_budget import append_observation
from backend.utils.settings_provider import AgentSettingsProvider


class _Client:
    def __init__(self, *, error=None):
        self.error = error
        self.calls = []

    def set_context(self, *_args):
        return None

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content="answer")


class _Agent(BaseAgent):
    async def process(self, state, instruction, **kwargs):
        return {}


def _agent(state, client):
    provider = AgentSettingsProvider(state, SystemConfig(), global_client=None)
    provider.get_llm_client = lambda _agent: client
    return _Agent(AgentType.PI, provider)


def _pending(state):
    return append_observation(state, ContextBudgetObservation(
        agent="literature",
        measured_tokens=100,
        final_tokens=90,
        ceiling=80,
        requested_completion_reserve=20,
        effective_completion_reserve=20,
        actions=["best effort"],
        fits=False,
    ))


@pytest.mark.asyncio
async def test_small_context_sends_the_reduced_completion_maximum():
    state = AnalysisState(user_id="user-1")
    state.agent_server_configs["pi"] = {"context_window": 8_192}
    client = _Client()

    assert await _agent(state, client).chat("review") == "answer"
    assert client.calls[0]["max_tokens"] == 7_679


@pytest.mark.asyncio
async def test_successful_pi_dispatch_acknowledges_injected_observation():
    state = AnalysisState(user_id="user-1")
    observation = _pending(state)
    client = _Client()

    await _agent(state, client).chat("review")

    assert client.calls[0]["messages"][1]["content"].startswith(
        "Technical context observations"
    )
    assert observation.pi_delivered_at is not None


@pytest.mark.asyncio
async def test_failed_pi_dispatch_leaves_observation_pending():
    state = AnalysisState(user_id="user-1")
    observation = _pending(state)
    client = _Client(error=RuntimeError("upstream failed"))

    with pytest.raises(RuntimeError, match="upstream failed"):
        await _agent(state, client).chat("review")

    assert observation.pi_delivered_at is None
