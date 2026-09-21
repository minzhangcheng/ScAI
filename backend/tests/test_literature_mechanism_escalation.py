"""Mechanism extraction uses local intervention and preserves completed searches."""

import asyncio
import json
from types import SimpleNamespace
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


@pytest.fixture(autouse=True)
def isolated_extraction_persistence(monkeypatch):
    save = AsyncMock(return_value=True)
    monkeypatch.setattr("backend.services.session_store.async_save_session", save)
    return save


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
        self.process_calls = 0
        self.extract_mechanism_signals = AsyncMock(side_effect=extract_side_effect)

    async def process(self, state, topic, **kwargs):
        self.process_calls += 1
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
    plan_literature_search = AsyncMock()
    review_literature_search = AsyncMock()

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

    out = await host._literature_research(_node_state(analysis_state))

    # Without this the assertion below is satisfied by never reaching
    # extraction at all, which is exactly how this test would go vacuous.
    assert host.literature_agent.extract_mechanism_signals.await_count == 1
    assert host._wait_for_llm_intervention.await_count == 1
    error = host._wait_for_llm_intervention.await_args.args[0]
    assert isinstance(error, UpstreamExhaustedError)
    assert out["literature_result"]["findings"] == ["f"]
    assert out["failed_node"] is None
    assert "retained" in out["literature_result"]["source_provenance"][-1]


@pytest.mark.asyncio
async def test_mechanism_retry_does_not_repeat_completed_search():
    analysis_state, report = _state_with_report()
    host = _NodeHost(_LitAgent(report.id, [_exhausted(), []]))
    host._wait_for_llm_intervention.return_value = LlmFailureAction.RETRY

    out = await host._literature_research(_node_state(analysis_state))

    assert host.literature_agent.process_calls == 1
    assert host.literature_agent.extract_mechanism_signals.await_count == 2
    assert out["literature_result"]["findings"] == ["f"]


@pytest.mark.asyncio
async def test_mechanism_terminate_still_terminates():
    from backend.workflow.helpers import WorkflowTerminatedError

    analysis_state, report = _state_with_report()
    host = _NodeHost(_LitAgent(report.id, _exhausted()))
    host._wait_for_llm_intervention.return_value = LlmFailureAction.TERMINATE

    with pytest.raises(WorkflowTerminatedError):
        await host._literature_research(_node_state(analysis_state))

    assert host.literature_agent.process_calls == 1


@pytest.mark.asyncio
async def test_literature_research_does_not_gate_on_an_extraction_bug():
    analysis_state, report = _state_with_report()
    host = _NodeHost(_LitAgent(report.id, RuntimeError("boom")))

    out = await host._literature_research(_node_state(analysis_state))

    assert host.literature_agent.extract_mechanism_signals.await_count == 1
    host._wait_for_llm_intervention.assert_not_awaited()
    assert out["literature_result"]["findings"]


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_input", [None, "round", "topics", "findings"])
async def test_extraction_recovers_with_new_workflow_and_agent_without_research(
    isolated_extraction_persistence, tmp_path, changed_input,
):
    state, report = _state_with_report()
    host = _NodeHost(_LitAgent(report.id, _exhausted()))
    host._wait_for_llm_intervention.side_effect = asyncio.CancelledError()
    checkpoint_path = tmp_path / "session.json"

    async def persist(session_id, analysis, *, force=False):
        assert force is True
        checkpoint_path.write_text(analysis.model_dump_json(), encoding="utf-8")
        return True

    isolated_extraction_persistence.side_effect = persist
    with pytest.raises(asyncio.CancelledError):
        await host._literature_research(_node_state(state))
    assert host.literature_agent.process_calls == 1

    restored = AnalysisState.model_validate_json(checkpoint_path.read_text(encoding="utf-8"))
    recovery = restored.sandbox_state["literature_extraction_resume_v1"]
    assert recovery["phase"] == "pending"
    assert recovery["report_ids"] == [report.id]
    assert recovery["result"]["findings"] == ["f"]
    graph = _node_state(restored)
    if changed_input == "round":
        restored.current_round += 1
    elif changed_input == "topics":
        graph["pi_instructions"]["literature_topics"] = ["new topic"]
    elif changed_input == "findings":
        graph["bioinformatics_result"] = {"findings": ["new measured result"]}
    new_host = _NodeHost(_LitAgent(report.id, [[]]))
    new_host.literature_agent.deep_research_model = "new extraction configuration"
    out = await new_host._literature_research(graph)
    assert new_host.literature_agent.process_calls == (1 if changed_input else 0)
    new_host.literature_agent.extract_mechanism_signals.assert_awaited_once()
    assert out["literature_result"]["findings"] == ["f"]
    assert restored.sandbox_state["literature_extraction_resume_v1"]["phase"] == "completed"

    if changed_input is None:
        # Also cover interruption after the extraction settled but before the
        # graph committed its node result: neither paid stage should repeat.
        finished = AnalysisState.model_validate_json(checkpoint_path.read_text(encoding="utf-8"))
        last_host = _NodeHost(_LitAgent(report.id, AssertionError("must not extract twice")))
        replayed = await last_host._literature_research(_node_state(finished))
        assert last_host.literature_agent.process_calls == 0
        last_host.literature_agent.extract_mechanism_signals.assert_not_awaited()
        assert replayed["literature_result"] == out["literature_result"]

        # Changing topics also invalidates the completed aggregate cache.
        changed = _node_state(finished)
        changed["pi_instructions"]["literature_topics"] = ["different research question"]
        changed_host = _NodeHost(_LitAgent(report.id, [[]]))
        await changed_host._literature_research(changed)
        assert changed_host.literature_agent.process_calls == 1


@pytest.mark.asyncio
async def test_extraction_skip_disclosure_reaches_pi_synthesis_and_realtime_result():
    from backend.workflow import synthesis
    from backend.workflow.synthesis_parts import phases

    state, report = _state_with_report()
    host = _NodeHost(_LitAgent(report.id, _exhausted()))
    host._send_message = AsyncMock()
    result = await host._literature_research(_node_state(state))
    disclosure = result["literature_result"]["source_provenance"][-1]
    assert "Mechanism extraction was skipped" in disclosure
    sent = [call.args[1] for call in host._send_message.await_args_list if call.args[0] == "result"]
    assert disclosure in sent[-1]["source_provenance"]

    pi = SimpleNamespace(synthesize_round=AsyncMock(return_value=MagicMock(next_steps=[])))
    synthesis_host = SimpleNamespace(
        session_id=state.id, pi_agent=pi, _settings_provider=None,
        _send_message=AsyncMock(), _require_core_agent=AsyncMock(return_value=None),
    )
    phases.bind_facade(synthesis)
    await phases.synthesize_agent_results(synthesis_host, result)
    actual_input = json.loads(pi.synthesize_round.await_args.kwargs["literature_result"])
    assert disclosure in actual_input["source_provenance"]
    assert actual_input["findings"] == ["f"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["context", "save"])
async def test_extraction_recovery_helpers_do_not_repeat_or_block_search(
    isolated_extraction_persistence, failure,
):
    state, report = _state_with_report()
    host = _NodeHost(_LitAgent(report.id, [[]]))
    if failure == "context":
        host._build_observed_context_for_literature = MagicMock(
            side_effect=RuntimeError("context unavailable"),
        )
    else:
        isolated_extraction_persistence.side_effect = OSError("disk unavailable")
    result = await host._literature_research(_node_state(state))
    assert host.literature_agent.process_calls == 1
    host._wait_for_llm_intervention.assert_not_awaited()
    assert result["literature_result"]["findings"] == ["f"]
    if failure == "context":
        host.literature_agent.extract_mechanism_signals.assert_not_awaited()
        assert "context unavailable" in result["literature_result"]["source_provenance"][-1]
    else:
        host.literature_agent.extract_mechanism_signals.assert_awaited_once()
