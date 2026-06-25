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


def test_non_overlapping_article_does_not_match():
    agent = _make_literature_agent()
    article = {"title": "Photosynthesis in maize leaves", "abstract": "chloroplast carbon fixation"}
    candidate = {"key_genes": ["Stat1"], "key_pathways": ["interferon"], "key_cell_types": ["Hepatocyte"]}
    assert agent._article_matches_candidate(article, candidate) is False


def test_overlapping_article_matches():
    agent = _make_literature_agent()
    article = {"title": "Stat1 drives interferon response", "abstract": "..."}
    candidate = {"key_genes": ["Stat1"], "key_pathways": ["interferon"], "key_cell_types": []}
    assert agent._article_matches_candidate(article, candidate) is True
