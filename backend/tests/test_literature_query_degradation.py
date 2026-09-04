from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.literature_agent import LiteratureAgent
from backend.agents.literature import query_builders
from backend.utils.llm_client import UpstreamExhaustedError


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


@pytest.mark.asyncio
async def test_query_prompt_marks_study_material_as_context_not_required_filters():
    agent = LiteratureAgent.__new__(LiteratureAgent)
    agent._build_research_context = MagicMock(return_value={
        "topic": "Metallothionein induction during hepatic copper deficiency",
        "species": "Mus musculus",
        "initial_instruction": "ZnT1/CTR1 double-knockout mouse liver study",
        "biochemistry_indicators": [],
        "treatment_groups": ["Con", "DKO"],
        "previous_findings": "Prior liver cell-type annotation",
        "current_round_findings": "Mt1 and Mt2 were elevated",
    })
    agent._chat_with_fallback = AsyncMock(return_value=(
        "metallothionein AND copper deficiency", None
    ))

    await agent._generate_pubmed_query(
        "Metallothionein induction during hepatic copper deficiency"
    )

    prompt = agent._chat_with_fallback.await_args.args[0]
    assert "Authoritative research question" in prompt
    assert "Context for terminology and disambiguation only" in prompt
    assert "do not automatically convert it into PubMed filters" in prompt
    assert (
        "Identify species, tissue/organ, and disease model from the background "
        "and include as filters"
    ) not in prompt


@pytest.mark.asyncio
async def test_true_zero_gets_one_concept_preserving_revision():
    agent = LiteratureAgent.__new__(LiteratureAgent)
    agent._pubmed_search = AsyncMock(side_effect=[
        [],
        [{"pmid": "1", "title": "Relevant article"}],
    ])
    agent._revise_zero_hit_query = AsyncMock(
        return_value="metallothionein AND copper deficiency"
    )
    attempts = []

    final_query, articles = await agent._validate_and_degrade_query(
        "metallothionein AND copper deficiency AND DKO AND ZnT1 AND CTR1",
        "Metallothionein induction during hepatic copper deficiency",
        enable_semantic_revision=True,
        attempt_log=attempts,
    )

    assert final_query == "metallothionein AND copper deficiency"
    assert [a["pmid"] for a in articles] == ["1"]
    assert agent._revise_zero_hit_query.await_count == 1
    assert agent._pubmed_search.await_count == 2
    assert [item["kind"] for item in attempts] == [
        "initial", "zero_hit_revision"
    ]


@pytest.mark.asyncio
async def test_semantic_revision_does_not_search_an_unchanged_query_twice():
    agent = LiteratureAgent.__new__(LiteratureAgent)
    agent._pubmed_search = AsyncMock(return_value=[])
    agent._revise_zero_hit_query = AsyncMock(return_value="  COPPER   DEFICIENCY  ")
    attempts = []

    final_query, articles = await agent._validate_and_degrade_query(
        "copper deficiency",
        "Copper deficiency",
        enable_semantic_revision=True,
        attempt_log=attempts,
    )

    assert final_query == "copper deficiency"
    assert articles == []
    assert agent._pubmed_search.await_count == 1
    assert [item["kind"] for item in attempts] == ["initial"]


@pytest.mark.asyncio
async def test_zero_hit_revision_upstream_exhaustion_propagates():
    agent = LiteratureAgent.__new__(LiteratureAgent)
    agent._pubmed_search = AsyncMock(return_value=[])
    agent._revise_zero_hit_query = AsyncMock(side_effect=UpstreamExhaustedError(
        "upstream exhausted",
        agent_name="literature",
        model="test-model",
        llm_server="test-server",
    ))

    with pytest.raises(UpstreamExhaustedError):
        await agent._validate_and_degrade_query(
            "exact query",
            "Exact question",
            enable_semantic_revision=True,
        )

    assert agent._pubmed_search.await_count == 1
    assert agent._revise_zero_hit_query.await_count == 1


@pytest.mark.asyncio
async def test_semantic_path_retries_technical_error_without_revising_topic(monkeypatch):
    agent = LiteratureAgent.__new__(LiteratureAgent)
    agent._pubmed_search = AsyncMock(side_effect=RuntimeError("PubMed unavailable"))
    agent._revise_zero_hit_query = AsyncMock(return_value="broader query")
    monkeypatch.setattr(query_builders.asyncio, "sleep", AsyncMock())
    attempts = []

    with pytest.raises(RuntimeError, match="PubMed unavailable"):
        await agent._validate_and_degrade_query(
            "exact query",
            "Exact question",
            enable_semantic_revision=True,
            attempt_log=attempts,
        )

    assert agent._pubmed_search.await_count == 2
    assert agent._revise_zero_hit_query.await_count == 0
    assert [item["kind"] for item in attempts] == ["initial", "technical_retry"]
