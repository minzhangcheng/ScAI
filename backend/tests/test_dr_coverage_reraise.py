"""E3: _evaluate_dr_coverage must re-raise UpstreamExhaustedError (not swallow → [])."""
import pytest

from backend.workflow.agents import AgentNodesMixin
from backend.utils.llm_client import UpstreamExhaustedError


class _Stub(AgentNodesMixin):
    """Minimal host exposing only what _evaluate_dr_coverage touches."""
    def __init__(self, agent):
        self.literature_agent = agent
        self.session_id = "test-session"


class _FakeAgent:
    def __init__(self, exc=None, response=None):
        self._exc = exc
        self._response = response

    async def chat_without_history(self, prompt, temperature=0.2):
        if self._exc is not None:
            raise self._exc
        return self._response


DR = {"report": "Deep research report body covering some topics."}
TOPICS = ["topic A", "topic B"]


@pytest.mark.asyncio
async def test_reraises_upstream_exhausted():
    stub = _Stub(_FakeAgent(exc=UpstreamExhaustedError("upstream down")))
    with pytest.raises(UpstreamExhaustedError):
        await stub._evaluate_dr_coverage(TOPICS, DR)


@pytest.mark.asyncio
async def test_generic_exception_returns_all_topics():
    stub = _Stub(_FakeAgent(exc=ValueError("bad parse")))
    result = await stub._evaluate_dr_coverage(TOPICS, DR)
    assert result == TOPICS


@pytest.mark.asyncio
async def test_valid_json_returns_gaps():
    stub = _Stub(_FakeAgent(response='["topic B"]'))
    result = await stub._evaluate_dr_coverage(TOPICS, DR)
    assert result == ["topic B"]


@pytest.mark.asyncio
async def test_empty_report_returns_all_topics():
    stub = _Stub(_FakeAgent(response="[]"))
    result = await stub._evaluate_dr_coverage(TOPICS, {"report": ""})
    assert result == TOPICS
