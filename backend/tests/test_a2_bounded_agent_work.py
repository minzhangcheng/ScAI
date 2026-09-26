"""Focused A2 RED tests for bounded search, image reuse, and tool refusal."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import hashlib

import pytest
from PIL import Image

from backend.agents.biochemistry_agent import BiochemistryAgent
from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.agents.literature.pubmed import PUBMED_QUERY_POLICY_VERSION, PubMedMixin
from backend.models.schemas import AgentType, DataFile


class _EmptyPubMedAgent(PubMedMixin):
    """Deterministic empty-result provider used to exercise the search budget."""

    max_search_rounds = 40
    search_wall_clock_seconds = 30
    max_articles_per_topic = 200
    max_sufficiency_evaluations = 2
    plateau_rounds = 2
    plateau_min_new_articles = 0

    def __init__(self):
        self.search_calls = 0
        self.reports = []

    async def _generate_pubmed_query(self, *_args):
        return "  exact empty query  "

    async def _validate_and_degrade_query(self, query, *_args, **kwargs):
        attempt_log = kwargs.pop("attempt_log", None)
        articles = await self._pubmed_search(query, **kwargs)
        if attempt_log is not None:
            attempt_log.append({
                "kind": "initial",
                "query": query,
                "result_count": len(articles),
            })
        return query, articles

    async def _pubmed_search(self, _query, **_kwargs):
        self.search_calls += 1
        return []

    async def _quick_search(self, topic, _state):
        return {"status": "completed", "report": f"Fallback: {topic}"}


@pytest.mark.asyncio
async def test_pubmed_each_budget_stops_and_exact_empty_result_is_cached():
    state = SimpleNamespace(literature_search_cache={}, literature_reports=[])
    agent = _EmptyPubMedAgent()

    first = await agent._multi_round_pubmed("  Empty Topic  ", state)

    assert first["search_meta"]["stop_reason"] == "no_pubmed_results"
    assert first["search_meta"]["fallback"] == "none"
    assert first["search_meta"]["rounds"] == 1
    assert first["search_meta"]["query_policy_version"] == PUBMED_QUERY_POLICY_VERSION
    assert first["search_meta"]["query_attempts"] == [{
        "kind": "initial",
        "query": "  exact empty query  ",
        "result_count": 0,
    }]
    assert agent.search_calls == 1

    calls = agent.search_calls
    second = await agent._multi_round_pubmed("empty topic", state)

    assert second["search_meta"]["stop_reason"] == "cache_reuse"
    assert second["search_meta"]["cache_hit"] is True
    assert agent.search_calls == calls


@pytest.mark.asyncio
async def test_biochemistry_sha_cache_requires_matching_terminal_usable_record(tmp_path):
    image_path = tmp_path / "biochem.png"
    Image.new("RGB", (12, 12), "white").save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()

    client = MagicMock()
    client.chat_with_vision = AsyncMock(
        return_value=SimpleNamespace(content="EXTRACTION_STATUS: usable\nfresh")
    )
    agent = BiochemistryAgent.__new__(BiochemistryAgent)
    agent.agent_type = AgentType.BIOCHEMISTRY
    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_system_prompt.return_value = ""
    provider.get_llm_client.return_value = client
    agent._provider = provider
    agent._set_llm_context = MagicMock()

    matching = DataFile(
        filename=image_path.name,
        file_path=str(image_path),
        section="biochemistry",
        data_type="image",
        extracted_data={
            "type": "visual_analysis_v2",
            "input_sha256": digest,
            "extraction_status": "usable",
            "terminal": True,
            "analysis": "cached analysis",
        },
    )
    assert await agent._analyze_single_image(matching, "review") == "cached analysis"
    assert client.chat_with_vision.await_count == 0

    mismatched = matching.model_copy(deep=True)
    mismatched.extracted_data["input_sha256"] = "0" * 64
    assert await agent._analyze_single_image(mismatched, "review") == "EXTRACTION_STATUS: usable\nfresh"
    assert client.chat_with_vision.await_count == 1

    nonterminal = matching.model_copy(deep=True)
    nonterminal.extracted_data["terminal"] = False
    assert await agent._analyze_single_image(nonterminal, "review") == "EXTRACTION_STATUS: usable\nfresh"
    assert client.chat_with_vision.await_count == 2
    assert all(call.kwargs["prefer_stream"] is True for call in client.chat_with_vision.await_args_list)


def test_required_tool_refusal_has_protocol_failure_kind():
    """A reachable model refusing a required tool is not transport exhaustion."""
    import backend.utils.llm_client as llm_client

    protocol_error = getattr(llm_client, "ToolProtocolError", None)
    assert protocol_error is not None, "ToolProtocolError must be part of the client contract"

    shell = SimpleNamespace(
        llm_client=SimpleNamespace(base_url="https://example.test/v1"),
        model="test-model",
    )
    bound = BioinformaticsAgent._raise_text_only_exhausted.__get__(shell, type(shell))

    with pytest.raises(protocol_error) as exc_info:
        bound(2)

    assert exc_info.value.failure_kind == "tool_protocol_failure"
