"""PI.plan_deep_research_tasks: complexity decision + task-list clamping."""
import pytest

from backend.agents.pi_agent import PIAgent
from backend.config._config import SystemConfig
from backend.models.schemas import AnalysisState
from backend.utils.settings_provider import AgentSettingsProvider


class _FakePI(PIAgent):
    """Bypass PIAgent.__init__; stub the LLM round-trip."""
    def __init__(self, response: str):
        state = AnalysisState(id="test", name="Test", user_id="u", initial_instruction="x")
        super().__init__(settings_provider=AgentSettingsProvider(
            state=state, config=SystemConfig(), global_client=None))
        self._response = response

    async def chat(self, prompt, temperature=0.3):
        return self._response


@pytest.mark.asyncio
async def test_decompose_true_returns_clamped_tasks():
    # 3 tasks returned, must clamp to 2
    resp = (
        '{"decompose": true, "tasks": '
        '["t1","t2","t3"]}'
    )
    pi = _FakePI(resp)
    out = await pi.plan_deep_research_tasks(
        topics=["original"], completed_results={}, overall_strategy="",
        research_objective="obj", state=None,
    )
    assert out["decompose"] is True
    assert out["tasks"] == ["t1", "t2"]


@pytest.mark.asyncio
async def test_decompose_false_keeps_simple():
    pi = _FakePI('{"decompose": false, "tasks": ["t1"]}')
    out = await pi.plan_deep_research_tasks(
        topics=["a", "b"], completed_results={}, overall_strategy="",
        research_objective="obj", state=None,
    )
    assert out["decompose"] is False


@pytest.mark.asyncio
async def test_bad_json_falls_back_to_no_decompose():
    pi = _FakePI("not json at all")
    out = await pi.plan_deep_research_tasks(
        topics=["a", "b"], completed_results={}, overall_strategy="",
        research_objective="obj", state=None,
    )
    assert out == {"decompose": False, "tasks": ["a", "b"]}


@pytest.mark.asyncio
async def test_decompose_true_but_one_task_is_not_complex():
    # decompose asked but only one usable task -> downgrade to non-decompose
    pi = _FakePI('{"decompose": true, "tasks": ["only-one"]}')
    out = await pi.plan_deep_research_tasks(
        topics=["a", "b"], completed_results={}, overall_strategy="",
        research_objective="obj", state=None,
    )
    assert out["decompose"] is False
    assert out["tasks"] == ["only-one"]