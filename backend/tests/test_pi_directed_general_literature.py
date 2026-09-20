"""Focused regression coverage for PI-directed general literature retrieval."""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.agents.literature.pubmed import (
    PUBMED_QUERY_POLICY_VERSION,
    _search_cache_key,
)
from backend.agents.literature_agent import LiteratureAgent
from backend.agents.pi_agent import PIAgent
from backend.models.schemas import AnalysisState
from backend.utils.llm_client import UpstreamExhaustedError
from backend.workflow.callbacks import CallbacksMixin


def _state():
    return AnalysisState(
        id="general-lit", name="study", initial_instruction="project question",
        species="species A", user_id="u1",
    )


class _PlanningPI(PIAgent):
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def refresh_runtime_config_if_needed(self):
        return None

    async def chat_without_history(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        return json.dumps(self.payload)


@pytest.mark.asyncio
async def test_pi_plan_rejects_non_string_or_duplicate_queries():
    malformed = _PlanningPI({
        "primary_query": "query",
        "alternative_queries": [" query "],
        "rationale": "duplicate",
    })
    with pytest.raises(ValueError, match="duplicate"):
        await malformed.plan_literature_search(topic="topic", context="context")

    non_string = _PlanningPI({
        "primary_query": {"query": "not executable"},
        "alternative_queries": [],
        "rationale": "bad shape",
    })
    with pytest.raises(ValueError, match="executable"):
        await non_string.plan_literature_search(topic="topic", context="context")


@pytest.mark.asyncio
async def test_pi_review_exposes_inventory_truncation_facts():
    pi = _PlanningPI({
        "search_exhausted": True,
        "evidence_strength": "sparse",
        "reasoning": "done",
        "refined_query": "",
        "missing_aspects": [],
    })
    articles = [
        {"pmid": str(i), "title": "t", "abstract": "a" * 2100}
        for i in range(61)
    ]

    await pi.review_literature_search(
        topic="topic", context="context", articles=articles,
        search_history=[{
            "round": 1,
            "primary_result_count": 0,
            "query_attempts": [
                {"kind": "initial", "query": "primary query", "result_count": 0},
                {
                    "kind": "seed", "query": "alternative query",
                    "result_count": None, "error": "seed service failed",
                },
            ],
        }],
        remaining_budget_seconds=10,
    )

    prompt = pi.prompts[0]
    assert '"source": 61' in prompt
    assert '"retained": 60' in prompt
    assert '"omitted": 1' in prompt
    assert '"abstract_truncated": true' in prompt
    assert '"primary_result_count": 0' in prompt
    assert '"query": "alternative query"' in prompt
    assert '"error": "seed service failed"' in prompt


class _GeneralLit(LiteratureAgent):
    def __init__(self):
        self.literature_mode = "pubmed_llm"
        self.reports = []
        self.max_search_rounds = 3
        self.search_wall_clock_seconds = 30
        self.max_articles_per_topic = 20
        self.max_sufficiency_evaluations = 3
        self.plateau_rounds = 4
        self.plateau_min_new_articles = 0
        self.seed_query_articles = 20
        self.post_triage_refinement_rounds = 2
        self.searched = []

    async def _validate_and_degrade_query(self, query, _topic, **kwargs):
        self.searched.append(query)
        attempts = kwargs.get("attempt_log")
        if attempts is not None:
            attempts.append({"kind": "initial", "query": query, "result_count": 0})
        return query, []

    async def _pubmed_search(self, query, **_kwargs):
        self.searched.append(query)
        if query == "broader query":
            return [{
                "pmid": "123", "title": "Retrieved paper", "abstract": "Full abstract",
                "year": "2025", "authors": "A", "journal": "J",
            }]
        return []

    async def _synthesize_pubmed_results(self, _topic, _articles, **_kwargs):
        return "source-backed synthesis"

    async def _extract_findings(self, _report):
        return ["finding"]

    def refresh_runtime_config_if_needed(self):
        return None


@pytest.mark.asyncio
async def test_general_plan_queries_and_true_zero_review_drive_followup():
    agent = _GeneralLit()
    state = _state()
    plan = AsyncMock(return_value={
        "primary_query": "strict query",
        "alternative_queries": ["seed query"],
        "rationale": "decomposed",
    })
    reviews = []

    async def review(**kwargs):
        reviews.append({**kwargs, "articles": list(kwargs["articles"])})
        if not kwargs["articles"]:
            return {
                "search_exhausted": False,
                "evidence_strength": "unassessed",
                "reasoning": "The zero result needs a broader query.",
                "refined_query": "broader query",
                "missing_aspects": ["terminology"],
            }
        return {
            "search_exhausted": True,
            "evidence_strength": "sparse",
            "reasoning": "One directly relevant source was retained.",
            "refined_query": "",
            "missing_aspects": [],
        }

    result = await agent._do_research(
        "evidence question",
        state,
        pi_plan_fn=plan,
        pi_review_fn=review,
        research_context="authoritative context",
    )

    assert plan.await_count == 1
    assert agent.searched[:2] == ["strict query", "seed query"]
    assert "broader query" in agent.searched
    assert reviews[0]["articles"] == []
    assert result["references"][0]["pmid"] == "123"
    assert result["search_meta"]["sufficiency_evaluations"] == 2


@pytest.mark.asyncio
async def test_source_inventory_is_persisted_before_synthesis_failure():
    agent = _GeneralLit()
    state = _state()

    async def first_hit(query, _topic, **_kwargs):
        return query, [{
            "pmid": "99", "title": "Raw title", "abstract": "Raw abstract",
            "year": "2024", "authors": "Author", "journal": "Journal",
        }]

    agent._validate_and_degrade_query = first_hit
    agent._synthesize_pubmed_results = AsyncMock(side_effect=RuntimeError("synthesis broke"))

    async def stop(_articles, _history):
        return {
            "search_exhausted": True,
            "evidence_strength": "sparse",
            "reasoning": "stop",
            "refined_query": "",
            "missing_aspects": [],
        }

    result = await agent._multi_round_pubmed(
        "topic", state, initial_query="query", evaluate_fn=stop,
        research_context="context", planned_queries=["query"],
    )

    assert result["status"] == "partial"
    assert result["references"][0]["abstract"] == "Raw abstract"
    assert result["retained_source_report_ids"] == [state.literature_reports[0].id]
    assert len(state.literature_reports) == 1
    assert state.literature_reports[0].references[0]["pmid"] == "99"
    checkpoint = next(iter(state.literature_search_cache.values()))["checkpoint"]
    assert checkpoint["articles"][0]["abstract"] == "Raw abstract"
    assert checkpoint["status"] == "partial"
    assert "result" not in next(iter(state.literature_search_cache.values()))


@pytest.mark.asyncio
async def test_partial_synthesis_is_retried_instead_of_reused_from_cache():
    agent = _GeneralLit()
    state = _state()

    async def first_hit(query, _topic, **kwargs):
        attempts = kwargs.get("attempt_log")
        if attempts is not None:
            attempts.append({"kind": "initial", "query": query, "result_count": 1})
        return query, [{"pmid": "98", "title": "t", "abstract": "retained"}]

    agent._validate_and_degrade_query = first_hit
    agent._synthesize_pubmed_results = AsyncMock(
        side_effect=[RuntimeError("synthesis broke"), "recovered synthesis"],
    )

    async def stop(_articles, _history):
        return {"search_exhausted": True, "evidence_strength": "sparse", "reasoning": "stop"}

    first = await agent._multi_round_pubmed(
        "topic", state, initial_query="query", evaluate_fn=stop,
    )
    second = await agent._multi_round_pubmed(
        "topic", state, initial_query="query", evaluate_fn=stop,
    )

    assert first["status"] == "partial"
    assert second["status"] == "completed"
    assert second["search_meta"]["stop_reason"] != "cache_reuse"
    assert agent._synthesize_pubmed_results.await_count == 2


@pytest.mark.asyncio
async def test_source_inventory_is_persisted_before_review_upstream_exhaustion():
    agent = _GeneralLit()
    state = _state()

    async def first_hit(query, _topic, **_kwargs):
        return query, [{"pmid": "88", "title": "t", "abstract": "retained"}]

    agent._validate_and_degrade_query = first_hit

    async def exhausted(_articles, _history):
        raise UpstreamExhaustedError(
            "review exhausted", agent_name="pi", model="test", llm_server="test",
        )

    with pytest.raises(UpstreamExhaustedError):
        await agent._multi_round_pubmed(
            "topic", state, initial_query="query", evaluate_fn=exhausted,
        )

    assert state.literature_reports[0].references[0]["abstract"] == "retained"
    cache_entry = next(iter(state.literature_search_cache.values()))
    assert "result" not in cache_entry
    assert cache_entry["checkpoint"]["status"] == "in_progress"
    assert cache_entry["checkpoint"]["articles"][0]["pmid"] == "88"


@pytest.mark.asyncio
async def test_true_zero_with_failed_pi_review_is_not_reported_as_search_exhausted():
    agent = _GeneralLit()
    state = _state()

    async def failed_review(_articles, _history):
        raise ValueError("unreadable PI review")

    result = await agent._multi_round_pubmed(
        "topic", state, initial_query="query", evaluate_fn=failed_review,
    )

    assert result["search_meta"]["stop_reason"] == "evaluation_failed"
    assert result["search_meta"]["evaluation_failed"] is True
    assert result["search_meta"]["search_exhausted"] is False
    assert result["search_meta"]["evidence_strength"] == "unassessed"
    assert result["status"] == "partial"
    cache_entry = next(iter(state.literature_search_cache.values()))
    assert cache_entry["checkpoint"]["status"] == "partial"
    assert "result" not in cache_entry


@pytest.mark.asyncio
async def test_failed_pi_review_is_retried_but_reviewed_true_zero_is_cached():
    agent = _GeneralLit()
    state = _state()
    review = AsyncMock(side_effect=[
        ValueError("unreadable PI review"),
        {
            "search_exhausted": True,
            "evidence_strength": "unassessed",
            "reasoning": "The executed queries returned zero results.",
            "refined_query": "",
            "missing_aspects": [],
        },
    ])

    first = await agent._multi_round_pubmed(
        "topic", state, initial_query="query", evaluate_fn=review,
    )
    second = await agent._multi_round_pubmed(
        "topic", state, initial_query="query", evaluate_fn=review,
    )
    third = await agent._multi_round_pubmed(
        "topic", state, initial_query="query", evaluate_fn=review,
    )

    assert first["status"] == "partial"
    assert second["status"] == "completed"
    assert second["search_meta"]["stop_reason"] == "no_pubmed_results"
    assert third["search_meta"]["stop_reason"] == "cache_reuse"
    assert review.await_count == 2


@pytest.mark.asyncio
async def test_expired_deadline_before_first_query_is_uncached_technical_partial():
    agent = _GeneralLit()
    state = _state()
    agent._validate_and_degrade_query = AsyncMock()

    result = await agent._multi_round_pubmed(
        "topic", state, initial_query="query",
        evaluate_fn=AsyncMock(), deadline_override=time.monotonic() - 1,
    )

    assert result["status"] == "partial"
    assert result["search_meta"]["stop_reason"] == "wall_clock_budget"
    assert result["search_rounds"] == 0
    assert "before any query completed" in result["report"]
    assert "executed queries" not in result["report"]
    assert "wall-clock budget" in result["source_processing_error"]
    agent._validate_and_degrade_query.assert_not_awaited()
    cache_entry = next(iter(state.literature_search_cache.values()))
    assert cache_entry["checkpoint"]["status"] == "partial"
    assert "result" not in cache_entry


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("second_failure", "stop_reason", "error_text"),
    [
        (asyncio.TimeoutError(), "wall_clock_budget", "wall-clock budget"),
        (RuntimeError("second query failed"), "pubmed_error", "second query failed"),
    ],
)
async def test_technical_search_stop_after_sources_retains_synthesis_without_cache(
    second_failure, stop_reason, error_text,
):
    agent = _GeneralLit()
    state = _state()

    async def first_hit(query, _topic, **kwargs):
        attempts = kwargs.get("attempt_log")
        if attempts is not None:
            attempts.append({"kind": "initial", "query": query, "result_count": 1})
        return query, [{"pmid": "source-1", "title": "t", "abstract": "retained"}]

    agent._validate_and_degrade_query = first_hit
    agent._pubmed_search = AsyncMock(side_effect=second_failure)

    async def continue_once(_articles, _history):
        return {
            "search_exhausted": False,
            "evidence_strength": "sparse",
            "reasoning": "Try one refinement.",
            "refined_query": "second query",
            "missing_aspects": ["refinement"],
        }

    result = await agent._multi_round_pubmed(
        "topic", state, initial_query="first query", evaluate_fn=continue_once,
    )

    assert result["status"] == "partial"
    assert result["report"] == "source-backed synthesis"
    assert result["references"][0]["pmid"] == "source-1"
    assert result["search_meta"]["stop_reason"] == stop_reason
    assert error_text in result["source_processing_error"]
    assert result["source_provenance"] == "pubmed_retrieved_partial"
    failed_attempt = result["search_meta"]["query_attempts"][-1]
    assert failed_attempt["kind"] == "refinement"
    assert failed_attempt["query"] == "second query"
    assert failed_attempt["result_count"] is None
    assert failed_attempt["error"]
    cache_entry = next(iter(state.literature_search_cache.values()))
    assert cache_entry["checkpoint"]["status"] == "partial"
    assert cache_entry["checkpoint"]["stop_reason"] == stop_reason
    assert cache_entry["checkpoint"]["source_processing_error"]
    assert cache_entry["checkpoint"]["articles"][0]["pmid"] == "source-1"
    assert "result" not in cache_entry


@pytest.mark.asyncio
@pytest.mark.parametrize("seed_outcome", ["hit", "error"])
async def test_pi_review_receives_exact_primary_and_seed_attempts(seed_outcome):
    agent = _GeneralLit()
    state = _state()
    captured = []

    async def seed_search(query, **_kwargs):
        agent.searched.append(query)
        if seed_outcome == "error":
            raise RuntimeError("seed service failed")
        return [{"pmid": "seed-1", "title": "seed", "abstract": "a"}]

    agent._pubmed_search = seed_search

    async def review(_articles, history):
        captured.append(history)
        return {
            "search_exhausted": True,
            "evidence_strength": "sparse" if seed_outcome == "hit" else "unassessed",
            "reasoning": "reviewed exact attempts",
            "refined_query": "",
            "missing_aspects": [],
        }

    await agent._multi_round_pubmed(
        "topic", state, initial_query="primary query",
        seed_queries=["alternative query"], evaluate_fn=review,
        planned_queries=["primary query", "alternative query"],
    )

    first_round = captured[0][0]
    assert first_round["primary_result_count"] == 0
    assert first_round["new_results"] == 0
    attempts = first_round["query_attempts"]
    assert attempts[0] == {
        "kind": "initial", "query": "primary query", "result_count": 0,
    }
    assert attempts[1]["kind"] == "seed"
    assert attempts[1]["query"] == "alternative query"
    if seed_outcome == "hit":
        assert attempts[1]["result_count"] == 1
        assert first_round["supplementary_new_results"] == 1
    else:
        assert attempts[1]["result_count"] is None
        assert attempts[1]["error"] == "seed service failed"
        assert first_round["supplementary_new_results"] == 0


@pytest.mark.asyncio
async def test_findings_extraction_failure_updates_same_source_report():
    agent = _GeneralLit()
    state = _state()

    async def first_hit(query, _topic, **_kwargs):
        return query, [{"pmid": "7", "title": "t", "abstract": "a"}]

    agent._validate_and_degrade_query = first_hit
    agent._extract_findings = AsyncMock(side_effect=RuntimeError("extract broke"))

    async def stop(_articles, _history):
        return {"search_exhausted": True, "evidence_strength": "sparse", "reasoning": "done"}

    result = await agent._multi_round_pubmed(
        "topic", state, initial_query="query", evaluate_fn=stop,
    )

    assert result["status"] == "partial"
    assert len(agent.reports) == 1
    assert len(state.literature_reports) == 1
    assert state.literature_reports[0].report == "source-backed synthesis"
    assert result["references"]


@pytest.mark.asyncio
async def test_gemini_extraction_failure_retains_external_source_provenance():
    agent = _GeneralLit()
    agent._deep_research = SimpleNamespace(research=AsyncMock(return_value={
        "report": "External source-backed synthesis",
        "sources": [{"title": "External source", "uri": "https://example.test/source"}],
    }))
    agent._lease_runtime_client = lambda _client: None
    agent._release_runtime_client = lambda _client: None
    agent._extract_findings = AsyncMock(side_effect=RuntimeError("extract broke"))

    result = await agent._multi_round_gemini("topic", _state())

    assert result["status"] == "partial"
    assert result["references"] == [
        {"title": "External source", "url": "https://example.test/source"}
    ]
    assert result["source_provenance"] == "gemini_deep_research_retrieved_partial"
    assert "PubMed" not in result["findings"][0]


def test_pi_guided_cache_identity_includes_policy_context_and_plan():
    common = {
        "topic": "topic", "query": "query", "organism": "species A",
        "tissue": "organ A", "pi_guided": True,
    }
    assert _search_cache_key(
        **common, query_policy_version=PUBMED_QUERY_POLICY_VERSION,
        research_context="context one", planned_queries=["query", "alternative A"],
    ) != _search_cache_key(
        **common, query_policy_version=PUBMED_QUERY_POLICY_VERSION - 1,
        research_context="context one", planned_queries=["query", "alternative A"],
    )
    assert _search_cache_key(
        **common, research_context="context one", planned_queries=["query", "alternative A"],
    ) != _search_cache_key(
        **common, research_context="context two", planned_queries=["query", "alternative B"],
    )


@pytest.mark.asyncio
async def test_non_source_fallback_is_labeled_knowledge_only():
    agent = _GeneralLit()
    agent._multi_round_pubmed = AsyncMock(side_effect=ValueError("planning failed"))
    agent._quick_search = AsyncMock(return_value={
        "status": "completed", "report": "model memory", "references": [],
        "findings": ["memory"],
    })

    result = await agent._do_research("topic", _state())

    assert result["source_provenance"] == "knowledge_only"
    assert result["source_failure"]["mode"] == "pubmed_llm"
    assert result["source_provenance"] == "knowledge_only"


@pytest.mark.asyncio
async def test_explicit_knowledge_only_process_contract_is_preserved_and_labeled():
    agent = _GeneralLit()
    agent._quick_search = AsyncMock(return_value={
        "status": "completed", "report": "model memory", "references": [],
        "findings": [],
    })

    result = await agent.process(_state(), "topic", use_deep_research=False)

    assert result["source_provenance"] == "knowledge_only"
    assert result["search_mode"] == "quick_search (explicit knowledge-only mode)"


@pytest.mark.asyncio
async def test_bioinformatics_callback_passes_per_call_pi_ownership_and_provenance():
    state = _state()
    literature = SimpleNamespace(
        refresh_runtime_config_if_needed=lambda: None,
        process=AsyncMock(return_value={
            "report": "retrieved report",
            "source_provenance": "pubmed_retrieved_partial",
        }),
    )
    pi = SimpleNamespace(
        plan_literature_search=AsyncMock(),
        review_literature_search=AsyncMock(),
    )
    host = SimpleNamespace(
        _current_analysis_state=state,
        literature_agent=literature,
        pi_agent=pi,
    )

    returned = await CallbacksMixin._literature_callback(host, "question")

    kwargs = literature.process.await_args.kwargs
    assert kwargs["pi_plan_fn"] is pi.plan_literature_search
    assert kwargs["pi_review_fn"] is pi.review_literature_search
    assert "project question" in kwargs["research_context"]
    assert "pubmed_retrieved_partial" in returned
