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
async def test_tissue_axis_not_covered_without_tissue_or_cell_terms(monkeypatch):
    agent = _make_literature_agent()

    async def fake_search(query, max_results=20):
        return []

    async def fake_chat(prompt, temperature=0.3):
        return json.dumps({"items": [], "off_topic": False,
                           "contradictory_ambiguous": False, "summary": ""}), False

    monkeypatch.setattr(agent, "_pubmed_search", fake_search)
    monkeypatch.setattr(agent, "_chat_with_fallback", fake_chat)
    monkeypatch.setattr(agent, "_extract_session_cached_pubmed_articles", lambda state: [])

    # Genes + pathways present, but NO cell types and NO tissue -> tissue/cell axis must be False.
    result = await agent.evaluate_direction_novelty_candidates([
        {"direction_id": "dir_z", "direction": "Gene+pathway only", "key_genes": ["A", "B"],
         "key_cell_types": [], "key_pathways": ["P"], "suggested_methods": [],
         "organism": "Mus musculus", "tissue": "", "experimental_context": "model"}
    ])

    axes = result[0]["negative_validation_axes"]
    assert axes["tissue_or_cell_type"] is False
    assert result[0]["negative_validation_confidence"] != "high"
