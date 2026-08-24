"""Focused A2 RED tests for bounded search, image reuse, and tool refusal."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import hashlib
import time

import pytest
from PIL import Image

from backend.agents.biochemistry_agent import BiochemistryAgent
from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.agents.literature.pubmed import PubMedMixin
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

    async def _generate_pubmed_query(self, *_args):
        return "  exact empty query  "

    async def _validate_and_degrade_query(self, query, *_args, **kwargs):
        return query, await self._pubmed_search(query, **kwargs)

    async def _pubmed_search(self, _query, **_kwargs):
        self.search_calls += 1
        return []

    async def _quick_search(self, topic, _state):
        return {"status": "completed", "report": f"Fallback: {topic}"}


@pytest.mark.asyncio
async def test_pubmed_each_budget_stops_and_exact_empty_result_is_cached():
    state = SimpleNamespace(literature_search_cache={})
    agent = _EmptyPubMedAgent()

    first = await agent._multi_round_pubmed("  Empty Topic  ", state)

    assert first["search_meta"]["stop_reason"] == "marginal_yield_plateau"
    assert first["search_meta"]["fallback"] == "llm_knowledge"
    assert first["search_meta"]["rounds"] == agent.plateau_rounds
    assert agent.search_calls == agent.plateau_rounds

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


@pytest.mark.asyncio
async def test_focused_pubmed_obeys_remaining_deadline_and_final_article_cap():
    class _Focused(PubMedMixin):
        max_search_rounds = 2
        max_articles_per_topic = 2
        search_wall_clock_seconds = 10

        def __init__(self):
            self.search_calls = []

        async def _validate_and_degrade_query(self, query, *_args, **_kwargs):
            return query, [{"pmid": "main", "title": "main"}]

        async def _pubmed_search(self, query, **_kwargs):
            self.search_calls.append(query)
            return [
                {"pmid": "focused-1", "title": "focused 1"},
                {"pmid": "focused-2", "title": "focused 2"},
                {"pmid": "focused-3", "title": "focused 3"},
            ]

        async def _judge_focused_relevance(self, *_args):
            return {"status": "strong"}

        async def _synthesize_pubmed_results(self, _topic, articles):
            return f"synthesized {len(articles)}"

        async def _finalize_result(self, _topic, report, references, _state, _rounds):
            return {"report": report, "references": references}

        @staticmethod
        def _dedup_by_pmid(items, key, drop_empty=False):
            seen = set()
            result = []
            for item in items:
                value = key(item)
                if drop_empty and not value:
                    continue
                if value in seen:
                    continue
                seen.add(value)
                result.append(item)
            return result

        @staticmethod
        def _focused_result_score(result):
            return ({"strong": 3, "limited": 2}.get(result.get("status"), 0), 0)

    agent = _Focused()
    triage = {
        "sufficient": False,
        "triage_complete": True,
        "triage_low_confidence": False,
        "rare_unresolved": [{
            "cluster_id": "c1",
            "candidate_celltype": "unknown",
            "synonyms": [],
            "top_markers": ["Marker1"],
        }],
        "suspected_contamination": [],
    }
    result = await agent._multi_round_pubmed(
        "topic",
        initial_query="q",
        evaluate_fn=AsyncMock(return_value=triage),
        organism="mouse",
        tissue="liver",
    )
    assert len(result["references"]) == agent.max_articles_per_topic
    assert result["report"] == "synthesized 2"
    assert result["search_meta"]["stop_reason"] == "article_budget"

    expired = await agent._focused_rare_search(
        triage["rare_unresolved"], "mouse", "liver", deadline=time.monotonic() - 1
    )
    assert expired["c1"]["exhaustion_reason"] == "wall_clock_budget"


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
