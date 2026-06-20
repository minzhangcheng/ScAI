import inspect
import types
import pytest
from unittest.mock import AsyncMock, MagicMock
from backend.agents.literature_agent import LiteratureAgent


def _mock_provider():
    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_llm_client.return_value = MagicMock()
    provider.get_temperature.return_value = 0.7
    provider.get_system_prompt.return_value = "test prompt"
    return provider


def _agent(mode):
    return LiteratureAgent(settings_provider=_mock_provider(), literature_mode=mode,
                           deep_research_model="dr-model")


def _fake_state(initial_instruction=""):
    # Minimal-but-correct stand-in. The only state fields any path these tests
    # reach actually touch: _finalize_result appends to .literature_reports, and
    # the PI/prompt code reads .initial_instruction (via getattr). No test here
    # reaches _build_research_context, so its fields are intentionally omitted.
    return types.SimpleNamespace(
        initial_instruction=initial_instruction,
        literature_reports=[],
    )


def _patch_dr_chat(agent, captured):
    # _deep_research_via_chat uses dr_client = self._deep_research_llm_client or self.llm_client.
    # Inject a stand-in DR client directly so the test does not depend on the
    # provider mock's property behavior. (_set_llm_context still touches the
    # provider-backed self.llm_client, which is a MagicMock and is safe.)
    async def fake_chat(**kwargs):
        captured["messages"] = kwargs["messages"]
        return types.SimpleNamespace(content="report body [1] Title http://x")
    agent._deep_research_llm_client = types.SimpleNamespace(chat=fake_chat)
    agent._extract_findings = AsyncMock(return_value=[])


@pytest.mark.asyncio
async def test_dr_chat_uses_brief_with_footer():
    agent = _agent("deep_research")
    captured = {}
    _patch_dr_chat(agent, captured)
    await agent._deep_research_via_chat("the topic", _fake_state("STUDY GOAL"),
                                        research_brief="FOCUSED BRIEF")
    assert captured["messages"][0]["role"] == "system"
    prompt = captured["messages"][-1]["content"]
    assert "FOCUSED BRIEF" in prompt
    assert LiteratureAgent._DEEP_RESEARCH_CITATION_FOOTER in prompt  # verbatim footer contract
    assert "STUDY GOAL" not in prompt               # no injection
    assert "Experimental background" not in prompt
    assert "molecular mechanisms" not in prompt


@pytest.mark.asyncio
async def test_dr_chat_fallback_is_slim():
    agent = _agent("deep_research")
    captured = {}
    _patch_dr_chat(agent, captured)
    await agent._deep_research_via_chat("my topic", _fake_state("STUDY GOAL"),
                                        research_brief=None)
    assert captured["messages"][0]["role"] == "system"
    prompt = captured["messages"][-1]["content"]
    assert "my topic" in prompt
    assert LiteratureAgent._DEEP_RESEARCH_CITATION_FOOTER in prompt  # verbatim footer contract
    assert "STUDY GOAL" not in prompt
    assert "Experimental background" not in prompt


@pytest.mark.asyncio
async def test_dr_chat_has_research_brief_param():
    sig = inspect.signature(LiteratureAgent._deep_research_via_chat)
    assert "research_brief" in sig.parameters


def _patch_gemini(agent, captured):
    fake_dr = MagicMock()
    async def fake_research(query):
        captured["query"] = query
        return {"report": "r", "sources": []}
    fake_dr.research = fake_research
    agent._deep_research = fake_dr
    agent._extract_findings = AsyncMock(return_value=[])


@pytest.mark.asyncio
async def test_gemini_uses_brief():
    agent = _agent("gemini_deep_research")
    captured = {}
    _patch_gemini(agent, captured)
    await agent._multi_round_gemini("the topic", _fake_state("STUDY GOAL"),
                                    research_brief="FOCUSED BRIEF")
    assert captured["query"] == "FOCUSED BRIEF"


@pytest.mark.asyncio
async def test_gemini_fallback_uses_topic_without_injection():
    agent = _agent("gemini_deep_research")
    captured = {}
    _patch_gemini(agent, captured)
    await agent._multi_round_gemini("just the topic", _fake_state("STUDY GOAL"),
                                    research_brief=None)
    assert captured["query"] == "just the topic"
    assert "STUDY GOAL" not in captured["query"]


def test_build_research_query_is_removed():
    assert not hasattr(LiteratureAgent, "_build_research_query")


@pytest.mark.asyncio
async def test_do_research_forwards_brief_to_gemini():
    agent = _agent("gemini_deep_research")
    agent._multi_round_gemini = AsyncMock(return_value={"report": "r"})
    await agent._do_research("t", _fake_state(), research_brief="BRIEF")
    assert agent._multi_round_gemini.call_args.kwargs["research_brief"] == "BRIEF"


@pytest.mark.asyncio
async def test_do_research_forwards_brief_to_dr_chat():
    agent = _agent("deep_research")
    agent._deep_research_via_chat = AsyncMock(return_value={"report": "r"})
    await agent._do_research("t", _fake_state(), research_brief="BRIEF")
    assert agent._deep_research_via_chat.call_args.kwargs["research_brief"] == "BRIEF"


@pytest.mark.asyncio
async def test_process_forwards_research_brief():
    agent = _agent("deep_research")
    agent._do_research = AsyncMock(return_value={"report": "r"})
    await agent.process(_fake_state(), "instruction", research_brief="BRIEF")
    assert agent._do_research.call_args.kwargs["research_brief"] == "BRIEF"


@pytest.mark.asyncio
async def test_do_research_pubmed_ignores_research_brief():
    # Guard: the PubMed branch must not receive research_brief (no DR brief plumbing there).
    agent = _agent("pubmed_llm")
    agent._multi_round_pubmed = AsyncMock(return_value={"report": "r"})
    await agent._do_research("t", _fake_state(), research_brief="BRIEF")
    assert "research_brief" not in agent._multi_round_pubmed.call_args.kwargs


@pytest.mark.asyncio
async def test_annotate_clusters_forwards_brief_to_do_research():
    agent = _agent("gemini_deep_research")
    agent._do_research = AsyncMock(return_value={"report": "lit", "references": []})
    await agent.annotate_clusters(
        cluster_markers={"0": ["Cd3d", "Cd3e"]},
        organism="mouse", tissue="liver",
        state=_fake_state(), skip_matching=True,
        pi_research_brief="ANNOTATION BRIEF",
    )
    assert agent._do_research.call_args.kwargs["research_brief"] == "ANNOTATION BRIEF"
