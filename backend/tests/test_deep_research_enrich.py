"""Tests for GoogleDeepResearchClient citation URL resolution + title backfill.

Uses httpx.MockTransport so no real network calls happen. The client instance
is built via __new__ to bypass __init__ (no api_key needed); we then attach an
AsyncClient wired to a mock handler.
"""
import httpx
import pytest

from backend.utils.llm_client import GoogleDeepResearchClient

REDIRECT_URL = (
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC123"
)
REAL_URL = "https://pubmed.ncbi.nlm.nih.gov/12345678/"


def _client_with_handler(handler):
    c = GoogleDeepResearchClient.__new__(GoogleDeepResearchClient)
    c.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


@pytest.mark.asyncio
async def test_resolve_redirect_follows_location_header():
    def handler(request):
        return httpx.Response(302, headers={"location": REAL_URL})

    c = _client_with_handler(handler)
    out = await c._resolve_redirect(REDIRECT_URL)
    assert out == REAL_URL


@pytest.mark.asyncio
async def test_resolve_redirect_skips_non_grounding_url_without_network():
    def handler(request):  # pragma: no cover - must NOT be called
        raise AssertionError("network call made for a non-grounding URL")

    c = _client_with_handler(handler)
    out = await c._resolve_redirect("https://example.com/already-real")
    assert out == "https://example.com/already-real"


@pytest.mark.asyncio
async def test_resolve_redirect_returns_original_when_no_location():
    def handler(request):
        return httpx.Response(200, text="not a redirect")

    c = _client_with_handler(handler)
    out = await c._resolve_redirect(REDIRECT_URL)
    assert out == REDIRECT_URL


@pytest.mark.asyncio
async def test_resolve_redirect_returns_original_on_http_error():
    def handler(request):
        raise httpx.ConnectError("boom")

    c = _client_with_handler(handler)
    out = await c._resolve_redirect(REDIRECT_URL)
    assert out == REDIRECT_URL


@pytest.mark.asyncio
async def test_resolve_redirect_joins_relative_location():
    def handler(request):
        return httpx.Response(302, headers={"location": "/article/123"})

    c = _client_with_handler(handler)
    out = await c._resolve_redirect(REDIRECT_URL)
    # Relative "/article/123" joined against the request URL host.
    assert out == "https://vertexaisearch.cloud.google.com/article/123"


@pytest.mark.asyncio
async def test_resolve_redirect_returns_original_when_url_expired():
    # Documents the expired-redirect scenario: an expired grounding URL returns
    # 200 (not a redirect) so we fall back to the original. NOTE: this exercises
    # the same is_redirect==False branch as
    # test_resolve_redirect_returns_original_when_no_location — kept as named
    # documentation of the expiry case, not as new branch coverage.
    def handler(request):
        return httpx.Response(200, text="<html>expired</html>")

    c = _client_with_handler(handler)
    out = await c._resolve_redirect(REDIRECT_URL)
    assert out == REDIRECT_URL


@pytest.mark.asyncio
async def test_fetch_title_extracts_and_unescapes():
    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<html><head><title>  Liver Fibrosis &amp; T Cells  </title></head></html>",
        )

    c = _client_with_handler(handler)
    out = await c._fetch_title(REAL_URL)
    assert out == "Liver Fibrosis & T Cells"


@pytest.mark.asyncio
async def test_fetch_title_handles_attributes_and_newlines():
    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text='<title lang="en">\nMulti\nLine Title\n</title>',
        )

    c = _client_with_handler(handler)
    out = await c._fetch_title(REAL_URL)
    assert out == "Multi Line Title"


@pytest.mark.asyncio
async def test_fetch_title_skips_non_html():
    def handler(request):
        return httpx.Response(
            200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4"
        )

    c = _client_with_handler(handler)
    out = await c._fetch_title(REAL_URL)
    assert out == ""


@pytest.mark.asyncio
async def test_fetch_title_returns_empty_on_error():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    c = _client_with_handler(handler)
    out = await c._fetch_title(REAL_URL)
    assert out == ""


@pytest.mark.asyncio
async def test_fetch_title_returns_empty_when_no_title_tag():
    def handler(request):
        return httpx.Response(
            200, headers={"content-type": "text/html"}, text="<html><body>no title</body></html>"
        )

    c = _client_with_handler(handler)
    out = await c._fetch_title(REAL_URL)
    assert out == ""


def _routing_handler(request):
    """302 for grounding redirects → a domain-specific real URL; 200 HTML
    with a per-path <title> otherwise."""
    url = str(request.url)
    if GoogleDeepResearchClient.GROUNDING_REDIRECT_MARKER in url:
        # Map the redirect id suffix to a stable real URL.
        rid = url.rsplit("/", 1)[-1]
        return httpx.Response(302, headers={"location": f"https://nih.gov/article/{rid}"})
    rid = url.rsplit("/", 1)[-1]
    return httpx.Response(
        200,
        headers={"content-type": "text/html"},
        text=f"<title>Title for {rid}</title>",
    )


@pytest.mark.asyncio
async def test_enrich_source_resolves_then_titles():
    c = _client_with_handler(_routing_handler)
    out = await c._enrich_source({"title": "", "uri": REDIRECT_URL + "/one"})
    assert out == {"title": "Title for one", "uri": "https://nih.gov/article/one"}


@pytest.mark.asyncio
async def test_enrich_source_keeps_existing_title_and_skips_fetch():
    # If a title is already present, do not overwrite it; still resolve the URL.
    c = _client_with_handler(_routing_handler)
    out = await c._enrich_source({"title": "Pre-set", "uri": REDIRECT_URL + "/two"})
    assert out == {"title": "Pre-set", "uri": "https://nih.gov/article/two"}


@pytest.mark.asyncio
async def test_enrich_sources_preserves_order():
    c = _client_with_handler(_routing_handler)
    sources = [
        {"title": "", "uri": REDIRECT_URL + "/a"},
        {"title": "", "uri": REDIRECT_URL + "/b"},
        {"title": "", "uri": REDIRECT_URL + "/c"},
    ]
    out = await c.enrich_sources(sources)
    assert [s["uri"] for s in out] == [
        "https://nih.gov/article/a",
        "https://nih.gov/article/b",
        "https://nih.gov/article/c",
    ]
    assert [s["title"] for s in out] == [
        "Title for a",
        "Title for b",
        "Title for c",
    ]


@pytest.mark.asyncio
async def test_enrich_sources_empty_list_returns_empty():
    c = GoogleDeepResearchClient.__new__(GoogleDeepResearchClient)
    out = await c.enrich_sources([])
    assert out == []


@pytest.mark.asyncio
async def test_enrich_source_falls_back_on_failure():
    def handler(request):
        raise httpx.ConnectError("down")

    c = _client_with_handler(handler)
    out = await c._enrich_source({"title": "", "uri": REDIRECT_URL + "/x"})
    # Redirect resolve failed → original URL kept; title fetch failed → "".
    assert out == {"title": "", "uri": REDIRECT_URL + "/x"}


@pytest.mark.asyncio
async def test_research_enriches_sources(monkeypatch):
    c = _client_with_handler(_routing_handler)

    async def fake_start(query):
        return "interaction-1"

    async def fake_poll(interaction_id):
        return {
            "report": "Body with [cite: 1] and [cite: 2].",
            "sources": [
                {"title": "", "uri": REDIRECT_URL + "/p1"},
                {"title": "", "uri": REDIRECT_URL + "/p2"},
            ],
            "thinking": [],
        }

    monkeypatch.setattr(c, "start_research", fake_start)
    monkeypatch.setattr(c, "poll_research", fake_poll)

    result = await c.research("any query")

    # Report text is passed through untouched (inline [cite: N] preserved).
    assert result["report"] == "Body with [cite: 1] and [cite: 2]."
    # Sources are resolved + titled, in original order.
    assert result["sources"] == [
        {"title": "Title for p1", "uri": "https://nih.gov/article/p1"},
        {"title": "Title for p2", "uri": "https://nih.gov/article/p2"},
    ]
