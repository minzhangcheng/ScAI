import asyncio
import json
from unittest.mock import MagicMock

import pytest

from backend.agents.literature_agent import LiteratureAgent


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
