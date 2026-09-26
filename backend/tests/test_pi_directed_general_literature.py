"""Focused regression coverage for PI-directed general literature retrieval."""

from unittest.mock import AsyncMock

import pytest

from backend.agents.literature_agent import LiteratureAgent
from backend.models.schemas import AnalysisState


def _state():
    return AnalysisState(
        id="general-lit", name="study", initial_instruction="project question",
        species="species A", user_id="u1",
    )


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
