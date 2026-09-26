import urllib.parse
from unittest.mock import AsyncMock

import httpx
import pytest

from backend.agents.literature_agent import LiteratureAgent
from backend.agents.literature import pubmed as pubmed_module


_PUBMED_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


class _RecordingThrottle:
    def __init__(self):
        self.rates = []

    async def acquire(self, rate_per_second):
        self.rates.append(rate_per_second)


class _RecordingAsyncClient:
    responses = []
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, *, params=None):
        type(self).calls.append(("get", url, dict(params or {})))
        return type(self).responses.pop(0)

    async def post(self, url, *, data=None):
        type(self).calls.append(("post", url, dict(data or {})))
        return type(self).responses.pop(0)


def _search_params(term: str) -> dict:
    return {
        "db": "pubmed",
        "term": term,
        "retmax": "20",
        "retmode": "json",
        "sort": "relevance",
        "tool": "ScAI",
        "email": "tester@example.org",
    }


def _prepared_get_uri_length(url: str, params: dict) -> int:
    return len(url) + 1 + len(urllib.parse.urlencode(params))


def _new_pubmed_agent() -> LiteratureAgent:
    agent = LiteratureAgent.__new__(LiteratureAgent)
    agent.pubmed_email = "tester@example.org"
    agent.pubmed_api_key = ""
    agent.proxy_url = ""
    return agent


def _install_pubmed_transport(monkeypatch, responses, *, sleep_mock=None):
    _RecordingAsyncClient.responses = list(responses)
    _RecordingAsyncClient.calls = []
    throttle = _RecordingThrottle()
    monkeypatch.setattr(pubmed_module, "_entrez_throttle", throttle)
    monkeypatch.setattr(pubmed_module.httpx, "AsyncClient", _RecordingAsyncClient)
    if sleep_mock is not None:
        monkeypatch.setattr(pubmed_module.asyncio, "sleep", sleep_mock)
    return throttle


@pytest.mark.asyncio
async def test_oversized_pubmed_query_uses_post_with_identical_params(monkeypatch):
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    term = " OR ".join(f"Mtm{n}" for n in range(3000))
    params = _search_params(term)
    assert (
        _prepared_get_uri_length(_PUBMED_URL, params)
        >= pubmed_module.NCBI_MAX_GET_URI_CHARS
    )
    expected = httpx.Response(
        200,
        json={"esearchresult": {"idlist": []}},
        request=httpx.Request("POST", _PUBMED_URL),
    )
    throttle = _install_pubmed_transport(monkeypatch, [expected])

    response = await _new_pubmed_agent()._entrez_get(_PUBMED_URL, params, {})

    assert response is expected
    assert _RecordingAsyncClient.calls == [("post", _PUBMED_URL, params)]
    assert throttle.rates == [3.0]


@pytest.mark.asyncio
async def test_get_414_retries_once_as_post_with_second_throttle_slot(monkeypatch):
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    params = _search_params("copper AND Complex-IV")
    too_long = httpx.Response(
        414,
        text="Request-URI Too Long",
        request=httpx.Request("GET", _PUBMED_URL),
    )
    expected = httpx.Response(
        200,
        json={"esearchresult": {"idlist": ["123"]}},
        request=httpx.Request("POST", _PUBMED_URL),
    )
    throttle = _install_pubmed_transport(monkeypatch, [too_long, expected])

    response = await _new_pubmed_agent()._entrez_get(_PUBMED_URL, params, {})

    assert response is expected
    assert _RecordingAsyncClient.calls == [
        ("get", _PUBMED_URL, params),
        ("post", _PUBMED_URL, params),
    ]
    # Fallback POST is a second request and must take its own throttle slot.
    assert throttle.rates == [3.0, 3.0]


@pytest.mark.asyncio
async def test_query_errors_are_not_reported_as_empty_results():
    agent = LiteratureAgent.__new__(LiteratureAgent)
    agent._pubmed_search = AsyncMock(side_effect=RuntimeError("PubMed unavailable"))

    with pytest.raises(RuntimeError, match="PubMed unavailable"):
        await agent._validate_and_degrade_query("ferroptosis", "topic")


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
