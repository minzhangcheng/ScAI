"""BaseAgent applies one shared budget and acknowledges PI context on success."""

from types import SimpleNamespace

import pytest

from backend.agents.base import BaseAgent
from backend.config import SystemConfig
from backend.models.schemas import AgentType, AnalysisState, ContextBudgetObservation
from backend.utils.context_budget import append_observation
from backend.utils.llm_client_parts.prompt_cache import (
    PROMPT_CACHE_EPHEMERAL_TAIL_KEY,
    PROMPT_CACHE_STABLE_BOUNDARY_KEY,
    PromptCachePlan,
)
from backend.utils.settings_provider import AgentSettingsProvider


class _Client:
    def __init__(self, *, error=None, prompt_cache_plan=None):
        self.error = error
        self.calls = []
        self._prompt_cache_plan = prompt_cache_plan

    def set_context(self, *_args):
        return None

    def prompt_cache_plan_for_history(self, _model):
        return self._prompt_cache_plan

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content="answer")


class _Agent(BaseAgent):
    async def process(self, state, instruction, **kwargs):
        return {}


def _agent(state, client, agent_type=AgentType.PI):
    provider = AgentSettingsProvider(state, SystemConfig(), global_client=None)
    provider.get_llm_client = lambda _agent: client
    return _Agent(agent_type, provider)


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
    assert client.calls[0]["prefer_stream"] is True


@pytest.mark.asyncio
async def test_pi_history_chat_marks_the_persisted_user_for_explicit_cache():
    state = AnalysisState(user_id="user-1")
    client = _Client(prompt_cache_plan=PromptCachePlan())
    agent = _agent(state, client)

    await agent.chat("review")

    request = client.calls[0]
    assert request["prompt_cache_plan"] == PromptCachePlan()
    assert request["messages"][-1] == {
        "role": "user",
        "content": "review",
        PROMPT_CACHE_STABLE_BOUNDARY_KEY: True,
    }
    assert agent.message_history[0] == {"role": "user", "content": "review"}


@pytest.mark.asyncio
async def test_pi_history_chat_can_disable_cache_without_marking_the_user():
    state = AnalysisState(user_id="user-1")
    disabled = PromptCachePlan(
        mode="disabled", prefer_anthropic_messages=False
    )
    client = _Client(prompt_cache_plan=disabled)

    await _agent(state, client).chat("review")

    request = client.calls[0]
    assert request["prompt_cache_plan"] == disabled
    assert PROMPT_CACHE_STABLE_BOUNDARY_KEY not in request["messages"][-1]


@pytest.mark.asyncio
async def test_bioinformatics_standalone_chat_caches_only_the_stable_system():
    state = AnalysisState(user_id="user-1")
    client = _Client(prompt_cache_plan=PromptCachePlan())

    await _agent(
        state, client, AgentType.BIOINFORMATICS
    ).chat_without_history("build a plan")

    request = client.calls[0]
    assert request["prompt_cache_plan"] == PromptCachePlan()
    assert request["messages"][-1] == {
        "role": "user",
        "content": "build a plan",
        PROMPT_CACHE_EPHEMERAL_TAIL_KEY: True,
    }


@pytest.mark.asyncio
async def test_pi_standalone_chat_propagates_disabled_cache_without_marker():
    state = AnalysisState(user_id="user-1")
    disabled = PromptCachePlan(
        mode="disabled", prefer_anthropic_messages=False
    )
    client = _Client(prompt_cache_plan=disabled)

    await _agent(state, client).chat_without_history("review")

    request = client.calls[0]
    assert request["prompt_cache_plan"] == disabled
    assert PROMPT_CACHE_EPHEMERAL_TAIL_KEY not in request["messages"][-1]


@pytest.mark.asyncio
async def test_non_pi_history_chat_keeps_the_default_cache_policy():
    state = AnalysisState(user_id="user-1")
    client = _Client(prompt_cache_plan=PromptCachePlan())

    await _agent(state, client, AgentType.LITERATURE).chat("review")

    request = client.calls[0]
    assert "prompt_cache_plan" not in request
    assert PROMPT_CACHE_STABLE_BOUNDARY_KEY not in request["messages"][-1]


@pytest.mark.asyncio
async def test_pi_history_chat_keeps_default_policy_for_other_models():
    state = AnalysisState(user_id="user-1")
    client = _Client()

    await _agent(state, client).chat("review")

    request = client.calls[0]
    assert "prompt_cache_plan" not in request
    assert PROMPT_CACHE_STABLE_BOUNDARY_KEY not in request["messages"][-1]


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
