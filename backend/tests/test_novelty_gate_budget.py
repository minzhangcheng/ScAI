"""The novelty gate has to be able to finish before it is worth consulting.

Three things made it return "unknown_due_to_timeout" for every candidate:

1. The per-step bound was 30s, which is where the triage model's median latency
   sits. Roughly half of every step timed out on a healthy system.
2. ``matching_cache`` is computed once per candidate and is identical for all of
   that candidate's queries, but the per-query path re-triaged it every time --
   five to seven full LLM calls per candidate, each able to consume a whole step
   budget. That, not PubMed, is what exhausted the direction budget.
3. Candidates were evaluated one after another, so the gate's wall clock was the
   sum of every candidate's direction budget. That is what made a budget large
   enough to succeed unaffordable.

A verdict of "unknown" is not neutral: it is the answer the scoring step then
has to reason from.
"""
import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.agents.literature.direction_novelty import (
    NOVELTY_CANDIDATE_CONCURRENCY,
    NOVELTY_DIRECTION_TIMEOUT_SECONDS,
    NOVELTY_QUERY_TIMEOUT_SECONDS,
)
from backend.agents.literature_agent import LiteratureAgent


def _make_agent():
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    provider.get_llm_client.return_value = MagicMock()
    agent = LiteratureAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


def _candidates(n):
    return [
        {"direction_id": f"NEW-{i:02d}", "direction": f"Direction {i}",
         "key_genes": ["Gene1"], "key_pathways": ["Pathway1"]}
        for i in range(1, n + 1)
    ]


# --- 1. the bounds --------------------------------------------------------------


def test_step_bound_clears_the_observed_triage_latency_band():
    # Observed on a real run: p50 29.3s, p75 36.4s, p90 60.8s against a 30s
    # bound. A bound at the median is a coin flip per step.
    assert NOVELTY_QUERY_TIMEOUT_SECONDS >= 60.0


def test_direction_bound_leaves_room_for_more_than_one_step():
    assert NOVELTY_DIRECTION_TIMEOUT_SECONDS >= 4 * NOVELTY_QUERY_TIMEOUT_SECONDS


def test_the_minimum_attempt_path_can_still_complete_one_call():
    # At 10s/15s the cheap fallback timed out before any model could answer,
    # which made "cheap" mean "guaranteed unknown".
    assert LiteratureAgent.MIN_ATTEMPT_QUERY_TIMEOUT >= 30.0
    assert (
        LiteratureAgent.MIN_ATTEMPT_DIRECTION_TIMEOUT
        >= 2 * LiteratureAgent.MIN_ATTEMPT_QUERY_TIMEOUT
    )


# --- 2. the cache is triaged once per candidate --------------------------------


@pytest.mark.asyncio
async def test_cached_articles_are_triaged_once_per_candidate(monkeypatch):
    agent = _make_agent()
    triaged = []

    async def fake_triage(articles, candidate, query):
        triaged.append((candidate.get("direction_id"), query.category, len(articles)))
        return [], {"off_topic": False, "contradictory_ambiguous": False,
                    "species_tissue_qualification": "", "summary": ""}

    async def fake_search(query, max_results=20):
        return []

    monkeypatch.setattr(agent, "_triage_articles_with_llm", fake_triage)
    monkeypatch.setattr(agent, "_pubmed_search", fake_search)
    monkeypatch.setattr(agent, "_article_matches_candidate", lambda a, c: True)
    monkeypatch.setattr(
        agent, "_extract_session_cached_pubmed_articles",
        lambda state: [{"pmid": "1", "title": "t", "abstract": "a"}],
    )

    await agent.evaluate_direction_novelty_candidates(_candidates(1), state=None)

    cache_calls = [t for t in triaged if t[1] == "session_cache"]
    assert len(cache_calls) == 1, f"cache triaged {len(cache_calls)} times: {triaged}"


@pytest.mark.asyncio
async def test_the_per_query_path_is_not_handed_the_cache_again(monkeypatch):
    agent = _make_agent()
    seen = []

    async def fake_execute(query, candidate, matching_cache, timeout, deadline=None):
        seen.append(list(matching_cache))
        return [], {}, False, []

    monkeypatch.setattr(agent, "_execute_query_batch", fake_execute)
    monkeypatch.setattr(agent, "_article_matches_candidate", lambda a, c: True)
    monkeypatch.setattr(
        agent, "_extract_session_cached_pubmed_articles",
        lambda state: [{"pmid": "1", "title": "t", "abstract": "a"}],
    )

    async def fake_cache_triage(candidate, matching_cache, query_timeout_seconds, deadline):
        return [], {}, False, []

    monkeypatch.setattr(agent, "_triage_cached_articles_once", fake_cache_triage)

    await agent.evaluate_direction_novelty_candidates(_candidates(1), state=None)

    assert seen, "no query batch ran"
    assert all(cache == [] for cache in seen), "the cache was passed down per query again"


@pytest.mark.asyncio
async def test_a_cache_triage_timeout_is_recorded_once_not_per_query(monkeypatch):
    agent = _make_agent()

    async def slow_triage(articles, candidate, query):
        await asyncio.sleep(5)
        return [], {}

    async def fake_search(query, max_results=20):
        return []

    monkeypatch.setattr(agent, "_triage_articles_with_llm", slow_triage)
    monkeypatch.setattr(agent, "_pubmed_search", fake_search)
    monkeypatch.setattr(agent, "_article_matches_candidate", lambda a, c: True)
    monkeypatch.setattr(
        agent, "_extract_session_cached_pubmed_articles",
        lambda state: [{"pmid": "1", "title": "t", "abstract": "a"}],
    )

    results = await agent.evaluate_direction_novelty_candidates(
        _candidates(1), state=None, query_timeout_seconds=0.05,
        direction_timeout_seconds=2.0, tier3_query_timeout_seconds=0.05,
    )

    cache_timeouts = [
        lim for lim in results[0]["limitations"] if "Cache triage timeout" in str(lim)
    ]
    assert len(cache_timeouts) <= 1


# --- 3. candidates are evaluated concurrently ----------------------------------


def test_a_concurrency_bound_exists_and_is_finite():
    assert 1 < NOVELTY_CANDIDATE_CONCURRENCY <= 16


@pytest.mark.asyncio
async def test_candidates_do_not_wait_for_each_other(monkeypatch):
    agent = _make_agent()
    active = {"now": 0, "peak": 0}

    async def fake_single(candidate, cached, query_timeout, **kwargs):
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0.05)
        active["now"] -= 1
        return {"direction_id": candidate["direction_id"], "queries": ["q"]}

    monkeypatch.setattr(agent, "_evaluate_single_candidate", fake_single)
    monkeypatch.setattr(agent, "_extract_session_cached_pubmed_articles", lambda state: [])

    started = time.monotonic()
    results = await agent.evaluate_direction_novelty_candidates(_candidates(4), state=None)
    elapsed = time.monotonic() - started

    assert active["peak"] > 1, "candidates were evaluated one after another"
    assert elapsed < 0.2, f"serial-looking wall clock: {elapsed:.2f}s"
    assert [r["direction_id"] for r in results] == ["NEW-01", "NEW-02", "NEW-03", "NEW-04"]


@pytest.mark.asyncio
async def test_concurrency_stays_within_its_bound(monkeypatch):
    agent = _make_agent()
    active = {"now": 0, "peak": 0}

    async def fake_single(candidate, cached, query_timeout, **kwargs):
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0.02)
        active["now"] -= 1
        return {"direction_id": candidate["direction_id"], "queries": ["q"]}

    monkeypatch.setattr(agent, "_evaluate_single_candidate", fake_single)
    monkeypatch.setattr(agent, "_extract_session_cached_pubmed_articles", lambda state: [])

    await agent.evaluate_direction_novelty_candidates(
        _candidates(10), state=None, concurrency=3, max_full_evidence_candidates=10,
    )

    assert active["peak"] <= 3


@pytest.mark.asyncio
async def test_one_candidate_failing_does_not_take_the_others_down(monkeypatch):
    agent = _make_agent()

    async def fake_single(candidate, cached, query_timeout, **kwargs):
        if candidate["direction_id"] == "NEW-02":
            raise RuntimeError("gateway said no")
        return {"direction_id": candidate["direction_id"], "queries": ["q"]}

    async def fake_minimum(candidate, cached, **kwargs):
        return {"direction_id": candidate["direction_id"], "queries": [], "limitations": [],
                "timed_out": False, "novelty_status": "unknown"}

    monkeypatch.setattr(agent, "_evaluate_single_candidate", fake_single)
    monkeypatch.setattr(agent, "_evaluate_minimum_attempt_candidate", fake_minimum)
    monkeypatch.setattr(agent, "_extract_session_cached_pubmed_articles", lambda state: [])

    results = await agent.evaluate_direction_novelty_candidates(_candidates(3), state=None)

    assert [r["direction_id"] for r in results] == ["NEW-01", "NEW-02", "NEW-03"]
    assert results[1]["novelty_status"] == "unknown_due_to_failure"


@pytest.mark.asyncio
async def test_carried_candidates_are_not_starved_of_a_full_evaluation(monkeypatch):
    # Carried candidates are appended last. With the old cap of 4 they always
    # landed past it, so the feature that exists to give a direction a second
    # hearing guaranteed it a degraded one.
    agent = _make_agent()
    full = []

    async def fake_single(candidate, cached, query_timeout, **kwargs):
        full.append(candidate["direction_id"])
        return {"direction_id": candidate["direction_id"], "queries": ["q"]}

    monkeypatch.setattr(agent, "_evaluate_single_candidate", fake_single)
    monkeypatch.setattr(agent, "_extract_session_cached_pubmed_articles", lambda state: [])

    await agent.evaluate_direction_novelty_candidates(_candidates(8), state=None)

    assert len(full) == 8
