import inspect
from backend.agents.literature_agent import LiteratureAgent


def test_integration_search_method_exists():
    assert hasattr(LiteratureAgent, "integration_literature_search")


def test_integration_search_uses_research_brief_not_context():
    src = inspect.getsource(LiteratureAgent.integration_literature_search)
    assert "research_brief=" in src
    assert "context=" not in src
