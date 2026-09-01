"""Focused coverage for LLM credential probe URL normalization."""

import pytest

from backend.utils import llm_probe


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("base_url", "expected_models_url"),
    [
        ("https://llm.example/v1", "https://llm.example/v1/models"),
        (
            "https://llm.example/v1/chat/completions",
            "https://llm.example/v1/models",
        ),
        (
            "https://llm.example/v1/chat/completions/",
            "https://llm.example/v1/models",
        ),
        ("https://llm.example/v1/responses", "https://llm.example/v1/models"),
        ("https://llm.example/v1/messages", "https://llm.example/v1/models"),
    ],
)
async def test_validate_llm_credentials_normalizes_full_endpoint_urls(
    monkeypatch,
    base_url,
    expected_models_url,
):
    requested_urls = []

    class FakeResponse:
        status_code = 200

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def get(self, url, headers):
            requested_urls.append(url)
            return FakeResponse()

    monkeypatch.setattr(
        llm_probe.httpx,
        "AsyncClient",
        lambda **kwargs: FakeClient(),
    )

    result = await llm_probe.validate_llm_credentials("test-key", base_url)

    assert result == {"valid": True, "uncertain": False}
    assert requested_urls == [expected_models_url]
