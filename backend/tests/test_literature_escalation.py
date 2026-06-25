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
async def test_deep_triggered_when_methods_present_but_no_methodological_precedent(monkeypatch):
    agent = _make_literature_agent()

    async def fake_search(query, max_results=20):
        # Two direct hits so the direct<2 condition is FALSE; nothing methodological.
        return [
            {"pmid": "1", "title": "Stat1 direct", "abstract": "Stat1 interferon hepatocyte",
             "year": "2024", "authors": "A", "journal": "J"},
            {"pmid": "2", "title": "Stat1 direct two", "abstract": "Stat1 interferon hepatocyte",
             "year": "2024", "authors": "A", "journal": "J"},
        ]

    async def fake_chat(prompt, temperature=0.3):
        return json.dumps({
            "items": [
                {"pmid": "1", "bucket": "direct", "relevance": "high", "why_it_matters": "d"},
                {"pmid": "2", "bucket": "direct", "relevance": "high", "why_it_matters": "d"},
            ],
            "off_topic": False, "contradictory_ambiguous": False, "summary": ""}), False

    monkeypatch.setattr(agent, "_pubmed_search", fake_search)
    monkeypatch.setattr(agent, "_chat_with_fallback", fake_chat)
    monkeypatch.setattr(agent, "_extract_session_cached_pubmed_articles", lambda state: [])

    result = await agent.evaluate_direction_novelty_candidates([
        {"direction_id": "dir_m", "direction": "Method dependent", "key_genes": ["Stat1"],
         "key_cell_types": ["Hepatocyte"], "key_pathways": ["interferon"],
         "suggested_methods": ["RNA velocity"], "organism": "Mus musculus",
         "tissue": "liver", "experimental_context": "model"}
    ])

    assert result[0]["search_tier"] == "deep"
