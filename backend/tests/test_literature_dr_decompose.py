"""_literature_research issues N serial DR calls when PI decomposes."""
from unittest.mock import AsyncMock

import pytest

from backend.workflow.agents import AgentNodesMixin
from backend.workflow.helpers import cache_store
from backend.models.schemas import AnalysisState, AgentType


class _Host(AgentNodesMixin):
    """Minimal host wiring only what _literature_research touches."""
    def __init__(self, lit_agent, pi_agent):
        self.literature_agent = lit_agent
        self.pi_agent = pi_agent
        self.session_id = "s1"

    async def _send_message(self, *a, **k):
        return None

    def _build_previous_findings_context(self, state, who):
        return ""

    def _collect_current_round_results(self, state):
        return {}


class _LitAgent:
    is_deep_research_available = True
    deep_research_model = "openai/o4-mini-deep-research"

    def __init__(self):
        self.calls = []

    async def process(self, state, topic, **kwargs):
        self.calls.append({"topic": topic, "kwargs": kwargs})
        return {
            "findings": [f"finding for {topic[:30]}"],
            "report": f"report {topic[:30]}",
            "search_mode": "deep_research",
        }

    async def chat_without_history(self, prompt, temperature=0.2):
        """Simulate full coverage — no gaps."""
        return "[]"

    def resolve_model_label(self, search_mode):
        """Map search_mode to display label (mirrors LiteratureAgent)."""
        if search_mode == "gemini_deep_research":
            return "Google Deep Research"
        if search_mode == "deep_research":
            return self.deep_research_model or self.model
        return self.model

    @property
    def model(self):
        return "mock/lit-model"


class _PIAgent:
    def __init__(self, plan, brief="ROUND BRIEF"):
        self._plan = plan
        self._brief = brief
        self.called = False
        self.brief_calls = []

    async def plan_deep_research_tasks(self, **kwargs):
        self.called = True
        return self._plan

    async def generate_round_research_brief(self, **kwargs):
        self.brief_calls.append(kwargs)
        return self._brief


def _state():
    return AnalysisState(id="s1", name="t", initial_instruction="obj", species="human", user_id="u1")


@pytest.mark.asyncio
async def test_decompose_runs_one_call_per_task():
    lit = _LitAgent()
    pi = _PIAgent({"decompose": True, "tasks": ["TA", "TB"]})
    host = _Host(lit, pi)
    state = {
        "analysis_state": _state(),
        "current_instructions": {"literature_topics": ["x"], "overview": ""},
        "pi_instructions": {"literature_topics": ["x"], "overview": ""},
    }
    out = await host._literature_research(state)
    # one DR call per decomposed task (2 tasks → 2 calls)
    assert len(lit.calls) == 2
    # Each call should contain the task text (prefixed by species)
    assert "TA" in lit.calls[0]["topic"]
    assert "TB" in lit.calls[1]["topic"]
    # Combined findings should contain all 2
    assert len(out["literature_result"]["findings"]) == 2


@pytest.mark.asyncio
async def test_no_decompose_merges_into_single_call():
    lit = _LitAgent()
    pi = _PIAgent({"decompose": False, "tasks": ["x", "y"]})
    host = _Host(lit, pi)
    state = {
        "analysis_state": _state(),
        "current_instructions": {"literature_topics": ["x", "y"], "overview": ""},
        "pi_instructions": {"literature_topics": ["x", "y"], "overview": ""},
    }
    out = await host._literature_research(state)
    # merged path -> exactly one DR call (chat_without_history returns "[]"
    # so _evaluate_dr_coverage finds no gaps)
    assert len(lit.calls) == 1
    assert "comprehensive literature review" in lit.calls[0]["topic"]


@pytest.mark.asyncio
async def test_gemini_mode_short_circuits_to_merged():
    # Gemini Deep Research self-decomposes internally: the workflow must skip
    # the PI planning call and go straight to a single merged DR call, even if
    # PI would otherwise have asked to decompose.
    lit = _LitAgent()
    lit.literature_mode = "gemini_deep_research"
    pi = _PIAgent({"decompose": True, "tasks": ["TA", "TB"]})
    host = _Host(lit, pi)
    state = {
        "analysis_state": _state(),
        "current_instructions": {"literature_topics": ["x", "y"], "overview": ""},
        "pi_instructions": {"literature_topics": ["x", "y"], "overview": ""},
    }
    out = await host._literature_research(state)
    assert pi.called is False                # PI decomposition short-circuited
    assert len(lit.calls) == 1               # single merged DR call
    assert "comprehensive literature review" in lit.calls[0]["topic"]
    assert out["literature_result"]["findings"]


@pytest.mark.asyncio
async def test_decomposed_round_passes_pi_research_brief():
    lit = _LitAgent()
    pi = _PIAgent({"decompose": True, "tasks": ["TA", "TB"]}, brief="ROUND BRIEF")
    host = _Host(lit, pi)
    state = {
        "analysis_state": _state(),
        "current_instructions": {"literature_topics": ["x"], "overview": ""},
        "pi_instructions": {"literature_topics": ["x"], "overview": ""},
    }

    await host._literature_research(state)

    assert [c["kwargs"].get("research_brief") for c in lit.calls] == ["ROUND BRIEF", "ROUND BRIEF"]
    assert len(pi.brief_calls) == 2


@pytest.mark.asyncio
async def test_merged_round_passes_pi_research_brief():
    lit = _LitAgent()
    pi = _PIAgent({"decompose": False, "tasks": ["x", "y"]}, brief="MERGED BRIEF")
    host = _Host(lit, pi)
    state = {
        "analysis_state": _state(),
        "current_instructions": {"literature_topics": ["x", "y"], "overview": ""},
        "pi_instructions": {"literature_topics": ["x", "y"], "overview": ""},
    }

    await host._literature_research(state)

    assert len(lit.calls) == 1
    assert lit.calls[0]["kwargs"].get("research_brief") == "MERGED BRIEF"
    assert len(pi.brief_calls) == 1


@pytest.mark.asyncio
async def test_round_brief_empty_uses_safe_topic_without_previous_context():
    class _PreviousContextHost(_Host):
        def _build_previous_findings_context(self, state, who):
            return "PREVIOUS LITERATURE CONTEXT SHOULD NOT FALL BACK"

    lit = _LitAgent()
    pi = _PIAgent(
        {"decompose": True, "tasks": ["focused macrophage topic", "second topic"]},
        brief="   ",
    )
    host = _PreviousContextHost(lit, pi)
    state = {
        "analysis_state": _state(),
        "current_instructions": {"literature_topics": ["x"], "overview": ""},
        "pi_instructions": {"literature_topics": ["x"], "overview": ""},
    }

    await host._literature_research(state)

    assert lit.calls[0]["kwargs"].get("research_brief") is None
    assert "focused macrophage topic" in lit.calls[0]["topic"]
    assert "PREVIOUS LITERATURE CONTEXT" not in lit.calls[0]["topic"]
    assert "Focus on aspects NOT already covered" not in lit.calls[0]["topic"]
    assert "PREVIOUS LITERATURE CONTEXT" in pi.brief_calls[0]["topic"]


@pytest.mark.asyncio
async def test_pubmed_uses_deferred_round_topics_without_species_or_history_in_topic():
    lit = _LitAgent()
    lit.is_deep_research_available = False
    lit.literature_mode = "pubmed_llm"
    pi = _PIAgent({"decompose": False, "tasks": []})
    host = _Host(lit, pi)
    state = {
        "analysis_state": _state(),
        "pi_instructions": {
            "literature_topics": [],
            "overview": "",
            "deferred_literature_topics": [
                "Metallothionein induction during hepatic copper deficiency"
            ],
            "deferred_literature_overview": "",
        },
    }

    await host._literature_research(state)

    assert len(lit.calls) == 1
    assert lit.calls[0]["topic"] == (
        "Metallothionein induction during hepatic copper deficiency"
    )


@pytest.mark.asyncio
async def test_empty_topics_are_planned_instead_of_sliced_from_findings():
    lit = _LitAgent()
    lit.is_deep_research_available = False
    lit.literature_mode = "pubmed_llm"
    pi = _PIAgent({"decompose": False, "tasks": []})
    pi.plan_agent_tasks = AsyncMock(return_value=[
        "Mechanisms linking hepatic copper deficiency to metallothionein induction"
    ])
    host = _Host(lit, pi)
    host._collect_current_round_results = lambda _state: {
        "pathology_findings": ["A long pathology finding that must not become a query"]
    }
    state = {
        "analysis_state": _state(),
        "pi_instructions": {"literature_topics": [], "overview": ""},
        "pathology_result": {
            "findings": ["A long pathology finding that must not become a query"]
        },
    }

    await host._literature_research(state)

    assert pi.plan_agent_tasks.await_count == 1
    assert lit.calls[0]["topic"] == (
        "Mechanisms linking hepatic copper deficiency to metallothionein induction"
    )
    assert "single cell sequencing" not in lit.calls[0]["topic"].lower()


@pytest.mark.asyncio
async def test_empty_planned_topics_return_explicit_nonblocking_status():
    lit = _LitAgent()
    lit.is_deep_research_available = False
    lit.literature_mode = "pubmed_llm"
    pi = _PIAgent({"decompose": False, "tasks": []})
    pi.plan_agent_tasks = AsyncMock(return_value=[])
    host = _Host(lit, pi)
    host._collect_current_round_results = lambda _state: {
        "bioinformatics_findings": ["completed finding"]
    }
    state = {
        "analysis_state": _state(),
        "pi_instructions": {"literature_topics": [], "overview": ""},
    }

    result = await host._literature_research(state)

    literature_result = result["literature_result"]
    assert lit.calls == []
    assert literature_result["status"] == "no_focused_topic"
    assert "no literature search query was executed" in literature_result["reason"]
    assert literature_result["findings"] == []


@pytest.mark.asyncio
async def test_stale_pubmed_node_cache_is_reexecuted_for_current_query_policy():
    lit = _LitAgent()
    lit.is_deep_research_available = False
    lit.literature_mode = "pubmed_llm"
    lit.process = AsyncMock(return_value={
        "findings": ["new finding"],
        "report": "new report",
        "search_mode": "pubmed_llm",
    })
    pi = _PIAgent({"decompose": False, "tasks": []})
    host = _Host(lit, pi)
    analysis_state = _state()
    cache_store(analysis_state, AgentType.LITERATURE, {
        "topics": ["old topic"],
        "findings": ["old zero-hit result"],
        "reports": [],
        "search_modes": ["pubmed_llm"],
    })
    state = {
        "analysis_state": analysis_state,
        "pi_instructions": {"literature_topics": ["focused topic"], "overview": ""},
    }

    result = await host._literature_research(state)

    assert lit.process.await_count == 1
    assert result["literature_result"]["findings"] == ["new finding"]
    assert result["literature_result"]["query_policy_version"] == 2

    await host._literature_research(result)
    assert lit.process.await_count == 1
