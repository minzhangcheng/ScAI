import urllib.parse
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from backend.agents.literature_agent import LiteratureAgent
from backend.agents.literature import pubmed as pubmed_module
from backend.agents.literature import query_builders
from backend.utils.llm_client import UpstreamExhaustedError


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


def test_prepared_get_uri_length_matches_httpx_url():
    params = _search_params("metallothionein AND copper deficiency")
    params["api_key"] = "test-api-key"
    assert _prepared_get_uri_length(_PUBMED_URL, params) == len(
        str(httpx.URL(_PUBMED_URL, params=params))
    )


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
async def test_below_threshold_pubmed_query_keeps_get_transport(monkeypatch):
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    params = _search_params("ferroptosis AND liver")
    assert (
        _prepared_get_uri_length(_PUBMED_URL, params)
        < pubmed_module.NCBI_MAX_GET_URI_CHARS
    )
    expected = httpx.Response(
        200,
        json={"esearchresult": {"idlist": []}},
        request=httpx.Request("GET", _PUBMED_URL),
    )
    throttle = _install_pubmed_transport(monkeypatch, [expected])

    response = await _new_pubmed_agent()._entrez_get(_PUBMED_URL, params, {})

    assert response is expected
    assert _RecordingAsyncClient.calls == [("get", _PUBMED_URL, params)]
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
async def test_414_transport_switch_stays_post_after_429(monkeypatch):
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    sleep_mock = AsyncMock()
    params = _search_params("retained below-threshold GET params")
    responses = [
        httpx.Response(
            414,
            text="Request-URI Too Long",
            request=httpx.Request("GET", _PUBMED_URL),
        ),
        *[
            httpx.Response(
                429,
                text="Too Many Requests",
                request=httpx.Request("POST", _PUBMED_URL),
            )
            for _ in range(3)
        ],
        httpx.Response(
            200,
            json={"esearchresult": {"idlist": ["456"]}},
            request=httpx.Request("POST", _PUBMED_URL),
        ),
    ]
    throttle = _install_pubmed_transport(
        monkeypatch, responses, sleep_mock=sleep_mock
    )

    response = await _new_pubmed_agent()._entrez_get(_PUBMED_URL, params, {})

    assert response.status_code == 200
    assert [call[0] for call in _RecordingAsyncClient.calls] == [
        "get", "post", "post", "post", "post",
    ]
    assert all(call[2] == params for call in _RecordingAsyncClient.calls)
    # One top-level slot for the GET, one fallback slot, and one slot for each
    # of the three remaining outer attempts. The 414 switch consumed none of
    # the configured 429 retry budget.
    assert throttle.rates == [3.0, 3.0, 3.0, 3.0, 3.0]
    assert sleep_mock.await_count == 3


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
