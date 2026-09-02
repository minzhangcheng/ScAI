"""PI's task-refinement calls escalate an exhausted upstream instead of degrading.

``plan_agent_tasks`` and ``plan_deep_research_tasks`` are the only two swallow
points on the PI planning path with no stated rationale and no user-visible
trace: both logged a warning and returned a value the caller cannot tell apart
from a real answer. ``plan_agent_tasks`` returns the unrefined task list, and
each of its three callers has already told the user "PI is refining ... tasks";
``plan_deep_research_tasks`` returns ``{"decompose": False, "tasks": topics}``,
byte-identical to the caller's own "do not decompose" default, so afterwards
nothing distinguishes a considered decision from a dead upstream.

Both sit directly under a ``@with_retry`` node, so escalation reaches the
LLM-failure gate. Everything else must still degrade: refinement is an
enrichment, and a malformed payload is not a reason to stop a round.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.pi_agent import PIAgent
from backend.constants import LlmFailureAction
from backend.models.schemas import AgentType, AnalysisState
from backend.utils.llm_client import UpstreamExhaustedError


def _exhausted():
    return UpstreamExhaustedError(
        "upstream exhausted",
        agent_name="pi",
        model="test-model",
        llm_server="https://example.invalid",
        failure_kind="stream_interrupted",
    )


def _pi_agent(chat_result):
    """``side_effect`` only for exceptions: a str side_effect is iterated,
    handing back one character per call."""
    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    provider.get_vision_setting.return_value = False
    agent = PIAgent(settings_provider=provider)
    agent.set_session("plan-escalation")
    if isinstance(chat_result, BaseException):
        agent.chat = AsyncMock(side_effect=chat_result)
    else:
        agent.chat = AsyncMock(return_value=chat_result)
    return agent


def _state():
    return AnalysisState(user_id="u", name="P", initial_instruction="Analyze")


_REFINED = '{"refined_tasks":["refined A","refined B"]}'
_DR_PLAN = '{"decompose":true,"tasks":["task A","task B"]}'


# ---------------------------------------------------------- plan_agent_tasks

@pytest.mark.asyncio
async def test_plan_agent_tasks_escalates_an_exhausted_upstream():
    agent = _pi_agent(_exhausted())

    with pytest.raises(UpstreamExhaustedError):
        await agent.plan_agent_tasks(
            agent_name="bioinformatics",
            original_tasks=["original"],
            completed_results={"literature": ["a finding"]},
            overall_strategy="strategy",
            research_objective="objective",
            state=_state(),
        )


@pytest.mark.asyncio
async def test_plan_agent_tasks_still_degrades_on_other_errors():
    agent = _pi_agent(RuntimeError("malformed"))

    tasks = await agent.plan_agent_tasks(
        agent_name="bioinformatics",
        original_tasks=["original"],
        completed_results={"literature": ["a finding"]},
        overall_strategy="strategy",
        research_objective="objective",
        state=_state(),
    )

    assert tasks == ["original"]


@pytest.mark.asyncio
async def test_plan_agent_tasks_returns_the_refinement_when_it_succeeds():
    agent = _pi_agent(_REFINED)

    tasks = await agent.plan_agent_tasks(
        agent_name="bioinformatics",
        original_tasks=["original"],
        completed_results={"literature": ["a finding"]},
        overall_strategy="strategy",
        research_objective="objective",
        state=_state(),
    )

    assert tasks == ["refined A", "refined B"]


# --------------------------------------------------- plan_deep_research_tasks

@pytest.mark.asyncio
async def test_plan_deep_research_tasks_escalates_an_exhausted_upstream():
    agent = _pi_agent(_exhausted())

    with pytest.raises(UpstreamExhaustedError):
        await agent.plan_deep_research_tasks(
            topics=["topic one", "topic two"],
            completed_results={},
            overall_strategy="strategy",
            research_objective="objective",
            state=_state(),
        )


@pytest.mark.asyncio
async def test_plan_deep_research_tasks_still_degrades_on_other_errors():
    """The degraded value is the caller's own default, so it must stay reachable
    for everything that is not an exhausted upstream."""
    agent = _pi_agent(RuntimeError("malformed"))

    plan = await agent.plan_deep_research_tasks(
        topics=["topic one", "topic two"],
        completed_results={},
        overall_strategy="strategy",
        research_objective="objective",
        state=_state(),
    )

    assert plan == {"decompose": False, "tasks": ["topic one", "topic two"]}


@pytest.mark.asyncio
async def test_plan_deep_research_tasks_returns_the_decomposition_when_it_succeeds():
    agent = _pi_agent(_DR_PLAN)

    plan = await agent.plan_deep_research_tasks(
        topics=["topic one", "topic two"],
        completed_results={},
        overall_strategy="strategy",
        research_objective="objective",
        state=_state(),
    )

    assert plan["decompose"] is True
    assert plan["tasks"] == ["task A", "task B"]


# ------------------------------------------ the gate the escalation reaches

@pytest.mark.asyncio
async def test_literature_node_gates_on_a_refinement_exhaustion():
    """End to end: the failure must reach @with_retry("literature_research")."""
    from backend.workflow.agents import AgentNodesMixin

    class _LitAgent:
        is_deep_research_available = False
        deep_research_model = ""

        async def process(self, state, topic, **kwargs):
            return {"findings": ["f"], "report": "r", "search_mode": "quick"}

        async def chat_without_history(self, prompt, temperature=0.2):
            return "[]"

        def resolve_model_label(self, search_mode):
            return "mock/lit"

        @property
        def model(self):
            return "mock/lit"

    class _Host(AgentNodesMixin):
        def __init__(self, pi_agent):
            self.literature_agent = _LitAgent()
            self.pi_agent = pi_agent
            self.session_id = "s1"
            self._wait_for_llm_intervention = AsyncMock(
                return_value=LlmFailureAction.SKIP
            )

        async def _send_message(self, *a, **k):
            return None

        def _build_previous_findings_context(self, state, who):
            return ""

        def _collect_current_round_results(self, state):
            return {"bioinformatics": ["a completed finding"]}

    host = _Host(_pi_agent(_exhausted()))
    state = {
        "analysis_state": _state(),
        "current_instructions": {"literature_topics": ["x"], "overview": "ov"},
        "pi_instructions": {"literature_topics": ["x"], "overview": "ov"},
    }

    await host._literature_research(state)

    assert host.pi_agent.chat.await_count >= 1
    assert host._wait_for_llm_intervention.await_count == 1
