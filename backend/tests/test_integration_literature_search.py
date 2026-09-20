import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.agents.literature_agent import LiteratureAgent


def test_integration_search_method_exists():
    assert hasattr(LiteratureAgent, "integration_literature_search")


def test_integration_search_uses_research_brief_not_context():
    src = inspect.getsource(LiteratureAgent.integration_literature_search)
    assert "research_brief=" in src
    assert "research_context=" in src
    assert "pi_plan_fn=pi_plan_fn" in src
    assert "pi_review_fn=pi_review_fn" in src
    assert "integration_report[:6000]" in src


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    {"source_failure": {"mode": "pubmed_llm", "reason": "ValueError: invalid XML"}},
    {"source_processing_error": "RuntimeError: extraction failed"},
    {"search_meta": {"evaluation_failed": True, "stop_reason": "evaluation_failed"}},
])
async def test_integration_projection_keeps_findings_and_technical_provenance(failure):
    result = {
        "status": "partial", "findings": ["retained finding" * 2000],
        "source_provenance": "knowledge_only", **failure,
    }
    agent = SimpleNamespace(_do_research=AsyncMock(return_value=result))
    projected = await LiteratureAgent.integration_literature_search(
        agent, SimpleNamespace(name="project", initial_instruction="question"), "findings",
    )

    assert "retained finding" in projected
    assert "knowledge_only" in projected
    assert "partial" in projected
    for key in failure:
        assert key in projected[:2000]
