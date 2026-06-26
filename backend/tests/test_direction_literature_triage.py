import asyncio
import json
import time
from unittest.mock import MagicMock

import pytest

from backend.agents.literature_agent import LiteratureAgent
from backend.models.schemas import PubMedEvidenceItem


def _make_literature_agent():
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    provider.get_llm_client.return_value = MagicMock()
    agent = LiteratureAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


@pytest.mark.asyncio
async def test_direction_triage_uses_llm_and_not_hardcoded_example_terms(monkeypatch):
    agent = _make_literature_agent()
    searched = []
    prompts = []

    async def fake_search(query, max_results=20):
        searched.append(query)
        return [{"pmid": "1", "title": "STAT1 in kidney", "abstract": "STAT1 kidney inflammation", "year": "2024", "authors": "A", "journal": "J"}]

    async def fake_chat(prompt, temperature=0.3):
        prompts.append(prompt)
        return json.dumps({
            "items": [{"pmid": "1", "bucket": "analogous", "relevance": "medium", "why_it_matters": "Analogous inflammatory mechanism."}],
            "off_topic": False,
            "contradictory_ambiguous": False,
            "species_tissue_qualification": "cross_tissue_only",
            "summary": "LLM triaged evidence.",
        }), False

    monkeypatch.setattr(agent, "_pubmed_search", fake_search)
    monkeypatch.setattr(agent, "_chat_with_fallback", fake_chat)
    monkeypatch.setattr(agent, "_extract_session_cached_pubmed_articles", lambda state: [])

    result = await agent.evaluate_direction_novelty_candidates([
        {"direction_id": "dir_001", "direction": "Kidney STAT1 inflammation", "key_genes": ["Stat1"], "key_cell_types": ["Renal epithelial cell"], "key_pathways": ["inflammation"], "suggested_methods": ["single-cell RNA-seq"], "organism": "Mus musculus", "tissue": "kidney", "experimental_context": "renal injury"}
    ])

    assert prompts, "LLM triage must be called"
    assert all("copper" not in q.lower() for q in searched)
    assert all("hepatocyte" not in q.lower() for q in searched)
    assert result[0]["species_tissue_qualification"] == "cross_tissue_only"


@pytest.mark.asyncio
async def test_negative_validation_requires_axis_coverage(monkeypatch):
    agent = _make_literature_agent()

    async def fake_search(query, max_results=20):
        return []

    async def fake_chat(prompt, temperature=0.3):
        return json.dumps({"items": [], "off_topic": False, "contradictory_ambiguous": False, "summary": "No evidence."}), False

    monkeypatch.setattr(agent, "_pubmed_search", fake_search)
    monkeypatch.setattr(agent, "_chat_with_fallback", fake_chat)
    monkeypatch.setattr(agent, "_extract_session_cached_pubmed_articles", lambda state: [])

    result = await agent.evaluate_direction_novelty_candidates([
        {"direction_id": "dir_002", "direction": "Sparse direction", "key_genes": ["X"], "key_cell_types": ["Y"], "key_pathways": ["Z"], "suggested_methods": ["co-culture"], "organism": "Mus musculus", "tissue": "example tissue", "experimental_context": "model"}
    ])

    axes = result[0]["negative_validation_axes"]
    assert result[0]["negative_validation_confidence"] == "high"
    assert all(axes.values())


@pytest.mark.asyncio
async def test_negative_validation_high_when_all_axes_covered_and_no_evidence(monkeypatch):
    agent = _make_literature_agent()

    async def fake_search(query, max_results=20):
        return []

    async def fake_chat(prompt, temperature=0.3):
        return json.dumps({"items": [], "off_topic": False, "contradictory_ambiguous": False, "summary": "No evidence."}), False

    monkeypatch.setattr(agent, "_pubmed_search", fake_search)
    monkeypatch.setattr(agent, "_chat_with_fallback", fake_chat)
    monkeypatch.setattr(agent, "_extract_session_cached_pubmed_articles", lambda state: [])

    result = await agent.evaluate_direction_novelty_candidates([
        {"direction_id": "dir_002b", "direction": "Covered direction", "key_genes": ["Stat1"], "key_cell_types": ["Representative cell"], "key_pathways": ["interferon signaling"], "suggested_methods": ["co-culture"], "organism": "Mus musculus", "tissue": "example tissue", "experimental_context": "knockout model"}
    ])

    assert all(result[0]["negative_validation_axes"].values())
    assert result[0]["negative_validation_confidence"] == "high"


@pytest.mark.asyncio
async def test_timeout_returns_partial_evidence(monkeypatch):
    agent = _make_literature_agent()

    async def slow_search(query, max_results=20):
        await asyncio.sleep(1)
        return []

    monkeypatch.setattr(agent, "_pubmed_search", slow_search)
    monkeypatch.setattr(agent, "_extract_session_cached_pubmed_articles", lambda state: [])

    result = await agent.evaluate_direction_novelty_candidates([
        {"direction_id": "dir_003", "direction": "Timeout direction", "key_genes": ["A"], "key_cell_types": ["B"], "key_pathways": ["C"], "suggested_methods": ["D"], "organism": "Mus musculus", "tissue": "example tissue", "experimental_context": "model"}
    ], query_timeout_seconds=0.01)

    assert result[0]["status"] in {"partial", "failed"}
    assert result[0]["timed_out"] is True


@pytest.mark.asyncio
async def test_degraded_queries_not_counted_for_axis_coverage(monkeypatch):
    """Degraded queries (meaningful < 2) must NOT mark validation axes as covered."""
    agent = _make_literature_agent()

    async def fake_search(query, max_results=20):
        return [{"pmid": "1", "title": "X in tissue", "abstract": "X gene study", "year": "2024", "authors": "A", "journal": "J"}]

    async def fake_chat(prompt, temperature=0.3):
        return json.dumps({"items": [{"pmid": "1", "bucket": "direct", "relevance": "medium", "why_it_matters": "Direct."}], "off_topic": False, "contradictory_ambiguous": False, "species_tissue_qualification": "", "summary": "Some."}), False

    monkeypatch.setattr(agent, "_pubmed_search", fake_search)
    monkeypatch.setattr(agent, "_chat_with_fallback", fake_chat)
    monkeypatch.setattr(agent, "_extract_session_cached_pubmed_articles", lambda state: [])

    # Candidate with ONLY genes (meaningful=1) → all queries degraded
    result = await agent.evaluate_direction_novelty_candidates([
        {"direction_id": "dir_degraded", "direction": "Degraded test", "key_genes": ["X"], "key_cell_types": [], "key_pathways": [], "suggested_methods": [], "organism": "Mus musculus", "tissue": "test tissue", "experimental_context": "model"}
    ])

    axes = result[0]["negative_validation_axes"]
    assert not any(axes.values()), f"Degraded queries should not cover axes, got: {axes}"
    assert result[0]["negative_validation_confidence"] in ("none", "low"), \
        f"Should not be high/medium, got: {result[0]['negative_validation_confidence']}"


@pytest.mark.asyncio
async def test_relevant_count_matches_all_evidence(monkeypatch):
    """relevant_count should count all LLM evidence items (irrelevant already filtered)."""
    agent = _make_literature_agent()

    async def fake_search(query, max_results=20):
        return [{"pmid": "1", "title": "Test", "abstract": "Test", "year": "2024", "authors": "A", "journal": "J"}]

    async def fake_chat(prompt, temperature=0.3):
        return json.dumps({"items": [{"pmid": "1", "bucket": "direct", "relevance": "high", "why_it_matters": "Direct."}], "off_topic": False, "contradictory_ambiguous": False, "summary": "One."}), False

    monkeypatch.setattr(agent, "_pubmed_search", fake_search)
    monkeypatch.setattr(agent, "_chat_with_fallback", fake_chat)
    monkeypatch.setattr(agent, "_extract_session_cached_pubmed_articles", lambda state: [])

    result = await agent.evaluate_direction_novelty_candidates([
        {"direction_id": "dir_rel", "direction": "Relevant test", "key_genes": ["Y"], "key_cell_types": ["Z"], "key_pathways": ["W"], "suggested_methods": ["Q"], "organism": "Mus musculus", "tissue": "test", "experimental_context": "model"}
    ])

    for q in result[0]["queries"]:
        if q["result_count"] > 0:
            assert q["relevant_count"] == 1, f"relevant_count should be 1, got {q['relevant_count']}"


@pytest.mark.asyncio
async def test_direction_timeout_preserves_evidence_collected_before_deadline(monkeypatch):
    agent = _make_literature_agent()
    calls = {"count": 0}

    async def fake_execute_query_batch(query, candidate, matching_cache, timeout, deadline=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return (
                [PubMedEvidenceItem(
                    pmid="12345678",
                    title="Copper metabolism in mouse liver",
                    year="2024",
                    url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                    relevance="high",
                    supports="direct",
                    source="new_search",
                    species_context="Mus musculus",
                    tissue_context="Liver",
                    why_it_matters="Collected before the direction deadline expired.",
                )],
                {"species_tissue_qualification": "matched_species_tissue_available"},
                False,
                [],
            )
        await asyncio.sleep(0.05)
        return [], {}, False, []

    monkeypatch.setattr(agent, "_execute_query_batch", fake_execute_query_batch)

    result = await agent.evaluate_direction_novelty_candidates(
        [{
            "direction_id": "dir_001",
            "direction": "Resolve hepatic copper redistribution",
            "key_genes": ["Slc31a1"],
            "key_cell_types": ["Hepatocytes"],
            "key_pathways": ["Copper homeostasis"],
            "suggested_methods": [],
            "organism": "Mus musculus",
            "tissue": "liver",
        }],
        query_timeout_seconds=1.0,
        direction_timeout_seconds=0.01,
    )

    ev = result[0]
    assert ev["direction_id"] == "dir_001"
    assert ev["status"] == "partial"
    assert ev["timed_out"] is True
    assert ev["timeout_seconds"] == 0.01
    assert "Direction-level timeout exceeded" in ev["limitations"]
    assert ev["direct_evidence_count"] == 1
    assert ev["direct_evidence"][0]["pmid"] == "12345678"


@pytest.mark.asyncio
async def test_direction_timeout_before_any_query_keeps_zero_counts(monkeypatch):
    agent = _make_literature_agent()

    async def fake_execute_query_batch(query, candidate, matching_cache, timeout, deadline=None):
        raise AssertionError("No query should run after an already-expired direction deadline")

    monkeypatch.setattr(agent, "_execute_query_batch", fake_execute_query_batch)
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)

    result = await agent._evaluate_single_candidate(
        {
            "direction_id": "dir_002",
            "direction": "B cell infiltration",
            "key_genes": ["Cd19"],
            "key_cell_types": ["B cells"],
            "key_pathways": ["Inflammation"],
            "suggested_methods": [],
            "organism": "Mus musculus",
            "tissue": "liver",
        },
        cached_articles=[],
        query_timeout_seconds=1.0,
        deadline=99.0,
        direction_timeout_seconds=0.5,
    )

    assert result["status"] == "partial"
    assert result["timed_out"] is True
    assert result["timeout_seconds"] == 0.5
    assert result["direct_evidence_count"] == 0
    assert result["analogous_evidence_count"] == 0
    assert "Direction-level timeout exceeded" in result["limitations"]


@pytest.mark.asyncio
async def test_direction_deadline_limits_in_flight_query_batch_runtime(monkeypatch):
    agent = _make_literature_agent()

    async def slow_search(query, max_results=20):
        await asyncio.sleep(0.2)
        return []

    monkeypatch.setattr(agent, "_pubmed_search", slow_search)
    monkeypatch.setattr(agent, "_extract_session_cached_pubmed_articles", lambda state: [])

    started = time.monotonic()
    result = await agent.evaluate_direction_novelty_candidates(
        [{
            "direction_id": "dir_budget",
            "direction": "Budget-limited search",
            "key_genes": ["Slc31a1"],
            "key_cell_types": ["Hepatocytes"],
            "key_pathways": ["Copper homeostasis"],
            "suggested_methods": [],
            "organism": "Mus musculus",
            "tissue": "liver",
        }],
        query_timeout_seconds=1.0,
        direction_timeout_seconds=0.02,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.12
    assert result[0]["status"] == "partial"
    assert result[0]["timed_out"] is True
    assert result[0]["timeout_seconds"] == 0.02
    assert "Direction-level timeout exceeded" in result[0]["limitations"]
