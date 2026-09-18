"""Reusable literature instructions precede request-specific evidence."""
from unittest.mock import AsyncMock

import pytest

from backend.agents.literature_agent import LiteratureAgent


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["query", "revision", "review", "findings"])
async def test_literature_prefix_survives_topic_and_evidence_changes(kind):
    agent = object.__new__(LiteratureAgent)
    agent._chat_with_fallback = AsyncMock(return_value=("Result", False))
    agent.chat_without_history = AsyncMock(return_value="- Finding")
    prompts = []
    for label in ("Alpha", "Beta"):
        if kind == "query":
            assert await agent._generate_pubmed_query(label) == "Result"
            prompt = agent._chat_with_fallback.await_args.args[0]
            boundary = "Authoritative research question:"
            assert "Every top-level AND concept" in prompt.split(boundary)[0]
        elif kind == "revision":
            assert await agent._revise_zero_hit_query(label, f"{label} query") == "Result"
            prompt = agent._chat_with_fallback.await_args.args[0]
            boundary = "Authoritative research question:"
            assert f"Executed query:\n{label} query" in prompt
            assert "Do not replace the target mechanism" in prompt.split(boundary)[0]
        elif kind == "review":
            assert await agent._synthesize_pubmed_results(
                label, [{"pmid": "123", "title": label, "abstract": f"{label} evidence"}],
                evaluation={"reasoning": f"{label} caveat", "evidence_strength": "limited"},
                max_articles=1,
            ) == "Result"
            prompt = agent._chat_with_fallback.await_args.args[0]
            boundary = "Topic:"
            assert "Cite specific PMIDs" in prompt.split(boundary)[0]
            assert f"{label} evidence" in prompt and f"{label} caveat" in prompt
            assert "[PMID: 123]" in prompt
        else:
            assert await agent._extract_findings(f"{label} report") == ["Finding"]
            prompt = agent.chat_without_history.await_args.args[0]
            boundary = f"{label} report"
            assert "Please list 3-5" in prompt.split(boundary)[0]
        assert label in prompt
        prompts.append(prompt.split(boundary, 1)[0])
    assert prompts[0] == prompts[1]
