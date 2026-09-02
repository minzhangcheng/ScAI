"""An exhausted upstream during mechanism extraction reaches the LLM-failure gate.

Mechanism-signal extraction sits three broad ``except Exception`` handlers deep:
``extract_mechanism_signals`` -> ``_persist_literature_mechanism_signals`` ->
the ``_literature_research`` body. Each logged a warning and returned, so an
exhausted upstream was neither retried nor shown to anyone — the round finished
"successfully" with the signals silently missing from
``_build_literature_audit_context``.

``literature_node.py`` already guards the same class one function away (the DR
coverage evaluation re-raises ``UpstreamExhaustedError`` so the dispatcher can
gate), and ``@with_retry("literature_research")`` sits directly above this path,
so the escalation machinery exists and only these three handlers defeated it.

Every other failure must still degrade exactly as before: extraction is
enrichment, and a malformed payload or a bug in the persistence helper is not a
reason to stop a round.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.constants import LlmFailureAction
from backend.models.schemas import (
    AgentType,
    AnalysisState,
    LiteratureMechanismSignal,
    LiteratureReport,
)
from backend.utils.llm_client import UpstreamExhaustedError
from backend.workflow.agents import AgentNodesMixin


def _exhausted() -> UpstreamExhaustedError:
    return UpstreamExhaustedError(
        "upstream exhausted",
        agent_name="literature",
        model="test-model",
        llm_server="https://example.invalid",
        failure_kind="stream_interrupted",
    )


# --------------------------------------------------------------------------
# Layer 1 — the agent method that owns the LLM call
# --------------------------------------------------------------------------

def _synthesis_agent(side_effect):
    from backend.agents.literature.synthesis import SynthesisMixin

    agent = SynthesisMixin.__new__(SynthesisMixin)
    agent.chat_without_history = AsyncMock(side_effect=side_effect)
    agent._safe_parse_json = MagicMock(return_value={"mechanism_signals": []})
    agent.session_id = "s1"
    return agent


@pytest.mark.asyncio
async def test_extract_mechanism_signals_propagates_exhausted_upstream():
    agent = _synthesis_agent(_exhausted())

    with pytest.raises(UpstreamExhaustedError):
        await agent.extract_mechanism_signals(
            report_text="body", references=[], observed_context="ctx",
        )


@pytest.mark.asyncio
async def test_extract_mechanism_signals_still_degrades_on_other_errors():
    agent = _synthesis_agent(RuntimeError("malformed"))

    assert await agent.extract_mechanism_signals(
        report_text="body", references=[], observed_context="ctx",
    ) is None


# --------------------------------------------------------------------------
# Layer 2 — the node's persistence helper
# --------------------------------------------------------------------------

class _PersistHost(AgentNodesMixin):
    def __init__(self):
        self.session_id = "s1"


def _state_with_report():
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    report = LiteratureReport(topic="t", report="body")
    state.literature_reports.append(report)
    return state, report


@pytest.mark.asyncio
async def test_persist_mechanism_signals_propagates_exhausted_upstream():
    state, report = _state_with_report()
    agent = MagicMock()
    agent.extract_mechanism_signals = AsyncMock(side_effect=_exhausted())

    with pytest.raises(UpstreamExhaustedError):
        await _PersistHost()._persist_literature_mechanism_signals(
            state, agent, report_ids=[report.id], observed_context="ctx",
        )


@pytest.mark.asyncio
async def test_persist_mechanism_signals_still_degrades_on_other_errors():
    state, report = _state_with_report()
    agent = MagicMock()
    agent.extract_mechanism_signals = AsyncMock(side_effect=RuntimeError("boom"))

    await _PersistHost()._persist_literature_mechanism_signals(
        state, agent, report_ids=[report.id], observed_context="ctx",
    )

    assert state.literature_reports[-1].mechanism_signals is None


# --------------------------------------------------------------------------
# Layer 3 — the node body, up to the gate @with_retry installs
# --------------------------------------------------------------------------

class _LitAgent:
    """Enough of LiteratureAgent for _literature_research to reach persistence."""

    is_deep_research_available = False
    deep_research_model = ""

    def __init__(self, report_id, extract_side_effect):
        self._report_id = report_id
        self.extract_mechanism_signals = AsyncMock(side_effect=extract_side_effect)

    async def process(self, state, topic, **kwargs):
        return {
            "findings": ["f"],
            "report": "report body",
            "report_id": self._report_id,
            "search_mode": "quick",
        }

    async def chat_without_history(self, prompt, temperature=0.2):
        return "[]"

    def resolve_model_label(self, search_mode):
        return self.model

    @property
    def model(self):
        return "mock/lit-model"


class _PIAgent:
    async def plan_deep_research_tasks(self, **kwargs):
        return {"decompose": False, "tasks": []}

    async def generate_round_research_brief(self, **kwargs):
        return "BRIEF"


class _NodeHost(AgentNodesMixin):
    """Minimal host; mirrors test_literature_dr_decompose._Host plus the gate."""

    def __init__(self, lit_agent):
        self.literature_agent = lit_agent
        self.pi_agent = _PIAgent()
        self.session_id = "s1"
        self._wait_for_llm_intervention = AsyncMock(
            return_value=LlmFailureAction.SKIP
        )

    async def _send_message(self, *a, **k):
        return None

    def _build_previous_findings_context(self, state, who):
        return ""

    def _collect_current_round_results(self, state):
        return {}


def _node_state(analysis_state):
    return {
        "analysis_state": analysis_state,
        "current_instructions": {"literature_topics": ["x"], "overview": ""},
        "pi_instructions": {"literature_topics": ["x"], "overview": ""},
    }


@pytest.mark.asyncio
async def test_literature_research_escalates_mechanism_exhaustion_to_the_gate():
    analysis_state, report = _state_with_report()
    host = _NodeHost(_LitAgent(report.id, _exhausted()))

    await host._literature_research(_node_state(analysis_state))

    # Without this the assertion below is satisfied by never reaching
    # extraction at all, which is exactly how this test would go vacuous.
    assert host.literature_agent.extract_mechanism_signals.await_count == 1
    assert host._wait_for_llm_intervention.await_count == 1
    error = host._wait_for_llm_intervention.await_args.args[0]
    assert isinstance(error, UpstreamExhaustedError)


@pytest.mark.asyncio
async def test_literature_research_does_not_gate_on_an_extraction_bug():
    analysis_state, report = _state_with_report()
    host = _NodeHost(_LitAgent(report.id, RuntimeError("boom")))

    out = await host._literature_research(_node_state(analysis_state))

    assert host.literature_agent.extract_mechanism_signals.await_count == 1
    host._wait_for_llm_intervention.assert_not_awaited()
    assert out["literature_result"]["findings"]
