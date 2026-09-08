from backend.tests.split_facade import setattr_facade
import pytest
import pytest_asyncio

from backend.utils import llm_client as llm_mod
from backend.utils.llm_client import OpenAICompatibleClient


class ProbeTransientError(RuntimeError):
    pass


@pytest_asyncio.fixture
async def client():
    value = OpenAICompatibleClient(
        api_key="test",
        base_url="https://example.test/v1",
        agent_name="pi",
    )
    try:
        yield value
    finally:
        await value.close()


@pytest.mark.asyncio
async def test_chat_probe_fallback_retries_transient_without_upstream_wrapper(monkeypatch, client):
    calls = {"count": 0}
    sentinel = object()

    setattr_facade(monkeypatch, "backend.utils.llm_client", "_route_protocol_order", lambda base_url, model: ["chat", "responses"])
    setattr_facade(monkeypatch, "backend.utils.llm_client", "is_transient_probe_error",
                    lambda exc, *, vision_request=False: isinstance(exc, ProbeTransientError))
    setattr_facade(monkeypatch, "backend.utils.llm_client", "PROBE_RETRY_DELAYS", [0, 0])

    async def make_inline_call(*, non_retryable_predicate=None, max_retries_override=None):
        calls["count"] += 1
        assert max_retries_override == 0
        assert non_retryable_predicate is not None
        if calls["count"] == 1:
            raise ProbeTransientError("temporary chat probe outage")
        return sentinel

    result = await client._dispatch_non_streaming(
        model="model-a",
        interaction_type="chat",
        log_messages=[],
        make_anthropic_adapter_call=None,
        make_responses_adapter_call=lambda adapter, skip_temperature: None,
        make_inline_call=make_inline_call,
    )

    assert result is sentinel
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_decided_chat_protocol_uses_normal_retry_settings(monkeypatch, client):
    seen = {"predicate": False, "retries": None}
    sentinel = object()

    async def make_inline_call(*, non_retryable_predicate=None, max_retries_override=None):
        seen["predicate"] = non_retryable_predicate is not None
        seen["retries"] = max_retries_override
        return sentinel

    setattr_facade(monkeypatch, "backend.utils.llm_client", "_route_protocol_order", lambda base_url, model: ["chat"])

    result = await client._dispatch_non_streaming(
        model="model-a",
        interaction_type="chat",
        log_messages=[],
        make_anthropic_adapter_call=None,
        make_responses_adapter_call=lambda adapter, skip_temperature: None,
        make_inline_call=make_inline_call,
    )

    assert result is sentinel
    assert seen == {"predicate": True, "retries": None}
