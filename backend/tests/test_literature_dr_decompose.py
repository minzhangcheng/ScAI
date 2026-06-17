"""_literature_research issues N serial DR calls when PI decomposes."""
import pytest

from backend.workflow.agents import AgentNodesMixin
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
        self.calls.append(topic)
        return {"findings": [f"finding for {topic[:30]}"], "report": f"report {topic[:30]}",
                "search_mode": "deep_research"}

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
    def __init__(self, plan):
        self._plan = plan
        self.called = False

    async def plan_deep_research_tasks(self, **kwargs):
        self.called = True
        return self._plan


def _state():
    return AnalysisState(id="s1", name="t", initial_instruction="obj", species="human", user_id="u1")


@pytest.mark.asyncio
async def test_decompose_runs_one_call_per_task():
    lit = _LitAgent()
    pi = _PIAgent({"decompose": True, "tasks": ["TA", "TB", "TC"]})
    host = _Host(lit, pi)
    state = {
        "analysis_state": _state(),
        "current_instructions": {"literature_topics": ["x"], "overview": ""},
        "pi_instructions": {"literature_topics": ["x"], "overview": ""},
    }
    out = await host._literature_research(state)
    # one DR call per decomposed task (3 tasks → 3 calls)
    assert len(lit.calls) == 3
    # Each call should contain the task text (prefixed by species)
    assert "TA" in lit.calls[0]
    assert "TB" in lit.calls[1]
    assert "TC" in lit.calls[2]
    # Combined findings should contain all 3
    assert len(out["literature_result"]["findings"]) == 3


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
    assert "comprehensive literature review" in lit.calls[0]


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
    assert "comprehensive literature review" in lit.calls[0]
    assert out["literature_result"]["findings"]
