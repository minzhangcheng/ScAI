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


def test_cross_species_query_without_genes_uses_pathway_or_cell():
    agent = _make_literature_agent()
    terms = {"genes": [], "pathways": ["interferon response"], "cells": ["Hepatocyte"],
             "methods": [], "organism": ["Mus musculus"], "tissue": ["liver"], "context": [""]}
    q = agent._build_cross_species_or_tissue(terms)
    assert q is not None
    assert "interferon response" in q
    # Cross-species: exclude the source organism.
    assert 'NOT "Mus musculus"' in q
    # Target tissue must NOT be force-included (that would block true cross-tissue evidence).
    assert '"liver"[tiab]' not in q


def test_cross_species_query_returns_none_without_any_core_term():
    agent = _make_literature_agent()
    terms = {"genes": [], "pathways": [], "cells": [],
             "methods": ["scRNA-seq"], "organism": ["Mus musculus"], "tissue": ["liver"], "context": [""]}
    assert agent._build_cross_species_or_tissue(terms) is None
