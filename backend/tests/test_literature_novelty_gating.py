import json

import pytest

from backend.agents.literature_agent import LiteratureAgent


def _make_literature_agent():
    from unittest.mock import MagicMock
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    provider.get_llm_client.return_value = MagicMock()
    agent = LiteratureAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


@pytest.mark.asyncio
async def test_no_evidence_low_coverage_yields_unknown_not_high(monkeypatch):
    agent = _make_literature_agent()

    async def fake_search(query, max_results=20):
        return []

    async def fake_chat(prompt, temperature=0.3):
        return json.dumps({"items": [], "off_topic": False,
                           "contradictory_ambiguous": False, "summary": ""}), False

    monkeypatch.setattr(agent, "_pubmed_search", fake_search)
    monkeypatch.setattr(agent, "_chat_with_fallback", fake_chat)
    monkeypatch.setattr(agent, "_extract_session_cached_pubmed_articles", lambda state: [])

    # Only one meaningful axis (genes), no tissue/cells/pathways/methods -> degraded queries,
    # axes mostly uncovered -> neg_conf low/none.
    result = await agent.evaluate_direction_novelty_candidates([
        {"direction_id": "dir_x", "direction": "Sparse", "key_genes": ["BRCA1"],
         "key_cell_types": [], "key_pathways": [], "suggested_methods": [],
         "organism": "", "tissue": "", "experimental_context": ""}
    ])

    assert result[0]["negative_validation_confidence"] in {"none", "low"}
    assert result[0]["literature_novelty"] == "unknown"


@pytest.mark.asyncio
async def test_analogous_only_is_not_high(monkeypatch):
    agent = _make_literature_agent()

    async def fake_search(query, max_results=20):
        return [{"pmid": "5", "title": "related", "abstract": "related mechanism",
                 "year": "2024", "authors": "A", "journal": "J"}]

    async def fake_chat(prompt, temperature=0.3):
        return json.dumps({
            "items": [{"pmid": "5", "bucket": "analogous", "relevance": "medium",
                       "why_it_matters": "related"}],
            "off_topic": False, "contradictory_ambiguous": False, "summary": ""}), False

    monkeypatch.setattr(agent, "_pubmed_search", fake_search)
    monkeypatch.setattr(agent, "_chat_with_fallback", fake_chat)
    monkeypatch.setattr(agent, "_extract_session_cached_pubmed_articles", lambda state: [])

    result = await agent.evaluate_direction_novelty_candidates([
        {"direction_id": "dir_y", "direction": "Has analogous", "key_genes": ["Stat1"],
         "key_cell_types": ["Hepatocyte"], "key_pathways": ["interferon"],
         "suggested_methods": ["scRNA-seq"], "organism": "Mus musculus",
         "tissue": "liver", "experimental_context": "model"}
    ])

    assert result[0]["analogous_evidence_count"] >= 1
    assert result[0]["literature_novelty"] != "high"
