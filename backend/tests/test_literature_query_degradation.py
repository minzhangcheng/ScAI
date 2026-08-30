from unittest.mock import AsyncMock

import pytest

from backend.agents.literature_agent import LiteratureAgent
from backend.agents.literature import query_builders


@pytest.mark.asyncio
async def test_empty_query_never_falls_back_to_topic_prefix(monkeypatch):
    agent = LiteratureAgent.__new__(LiteratureAgent)
    agent._pubmed_search = AsyncMock(return_value=[])
    monkeypatch.setattr(query_builders.asyncio, "sleep", AsyncMock())
    query = '"ferroptosis"[Title/Abstract] AND "liver"[Title/Abstract]'
    topic = "Species: Mus musculus\n\nSearch PubMed for ferroptosis evidence"

    final_query, articles = await agent._validate_and_degrade_query(query, topic)

    assert articles == []
    assert final_query == "ferroptosis AND liver"
    submitted = [call.args[0] for call in agent._pubmed_search.await_args_list]
    assert submitted == [query, "ferroptosis AND liver"]
    assert all("Search PubMed" not in item for item in submitted)


@pytest.mark.asyncio
async def test_query_errors_are_not_reported_as_empty_results():
    agent = LiteratureAgent.__new__(LiteratureAgent)
    agent._pubmed_search = AsyncMock(side_effect=RuntimeError("PubMed unavailable"))

    with pytest.raises(RuntimeError, match="PubMed unavailable"):
        await agent._validate_and_degrade_query("ferroptosis", "topic")
