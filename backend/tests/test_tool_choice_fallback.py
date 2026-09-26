"""Tests for tool_choice BadRequestError fallback in OpenAICompatibleClient.

When a provider rejects tool_choice="required" (or any non-"auto"/"none"
value), the client must detect that specific failure, cache the model
as tool-choice-unsupported, and retry the request once with
tool_choice="auto" so the autonomous loop can keep going instead of
crashing the round.
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from openai import BadRequestError

from backend.utils import llm_client
from backend.utils.llm_client import UpstreamExhaustedError


@pytest.fixture
def no_retry_sleep(monkeypatch):
    """Neutralize retry backoff for tests that deliberately exhaust retries.

    Same seam as `captured_sleeps` in test_rate_limit_backoff.py. The retry
    count, the classification and the raised exception are untouched; only
    the wall clock between attempts is.
    """

    async def _instant(delay):
        return None

    monkeypatch.setattr(llm_client.asyncio, "sleep", _instant)


@pytest_asyncio.fixture
async def close_stub_clients():
    clients = []
    yield clients.append
    for client in clients:
        client.client.close = AsyncMock()
        client.http_client = MagicMock()
        client.http_client.aclose = AsyncMock()
        await client.close()


def _make_bad_request(message: str) -> BadRequestError:
    """Construct a BadRequestError with the given message body."""
    response = MagicMock()
    response.status_code = 400
    response.headers = {}
    return BadRequestError(
        message=message,
        response=response,
        body={"error": {"message": message}},
    )


class TestChatWithToolsFallback:
    """Integration: mock the underlying OpenAI client to raise a
    tool_choice BadRequestError on the first call and succeed on the
    retry — assert the client caches the model and retries with
    tool_choice='auto'."""

    @pytest.mark.asyncio
    async def test_demotes_to_auto_on_bad_request(
        self, monkeypatch, close_stub_clients,
    ):
        from backend.utils.llm_client import OpenAICompatibleClient

        client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
        close_stub_clients(client)
        client.agent_name = "test"
        client.base_url = "https://fake.example.com/v1/chat/completions"
        client.session_id = "test-session"
        client.on_retry = None
        client._temp_unsupported_models = set()
        client._toolchoice_unsupported_models = set()

        # Fake llm_logger so log_request / log_response no-op.
        class _NullLogger:
            def log_request(self, **kw): return None
            def log_response(self, **kw): return None
            async def log_response_async(self, **kw): return None
        client.llm_logger = _NullLogger()

        # Successful chat completion shape: response.choices[0].message
        # with empty tool_calls / content "ok".
        good_message = MagicMock()
        good_message.tool_calls = None
        good_message.content = "ok"
        good_message.model_dump = MagicMock(return_value={"content": "ok", "tool_calls": None})
        good_choice = MagicMock(message=good_message)
        good_response = MagicMock(choices=[good_choice], model="m", usage=None)

        bad_err = _make_bad_request("Invalid value for 'tool_choice'")

        create_mock = AsyncMock(side_effect=[bad_err, good_response])
        client.client = MagicMock()
        client.client.chat = MagicMock()
        client.client.chat.completions = MagicMock()
        client.client.chat.completions.create = create_mock

        result = await client.chat_with_tools(
            model="provider/some-model",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}],
            tool_choice="required",
        )

        assert result.content == "ok"
        # Two calls: first failed with tool_choice='required', second succeeded with tool_choice='auto'.
        assert create_mock.call_count == 2
        first_kwargs = create_mock.call_args_list[0].kwargs
        second_kwargs = create_mock.call_args_list[1].kwargs
        assert first_kwargs.get("tool_choice") == "required"
        assert second_kwargs.get("tool_choice") == "auto"
        # Model cached for future calls.
        assert "some-model" in client._toolchoice_unsupported_models

    @pytest.mark.asyncio
    async def test_unrelated_bad_request_propagates(
        self, monkeypatch, close_stub_clients, no_retry_sleep,
    ):
        from backend.utils.llm_client import OpenAICompatibleClient

        client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
        close_stub_clients(client)
        client.agent_name = "test"
        client.base_url = "https://fake.example.com/v1/chat/completions"
        client.session_id = "test-session"
        client.on_retry = None
        client._temp_unsupported_models = set()
        client._toolchoice_unsupported_models = set()

        class _NullLogger:
            def log_request(self, **kw): return None
            def log_response(self, **kw): return None
            async def log_response_async(self, **kw): return None
        client.llm_logger = _NullLogger()

        # BadRequest that is NOT tool_choice and NOT an adaptation signal: under
        # the blacklist retry model it is treated as transient, retried, and on
        # exhaustion surfaces as UpstreamExhaustedError (→ workflow HITL gate)
        # rather than the raw BadRequestError.
        other_err = _make_bad_request("Unknown error from upstream")
        create_mock = AsyncMock(side_effect=other_err)
        client.client = MagicMock()
        client.client.chat = MagicMock()
        client.client.chat.completions = MagicMock()
        client.client.chat.completions.create = create_mock

        with pytest.raises(UpstreamExhaustedError):
            await client.chat_with_tools(
                model="provider/some-model",
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}],
                tool_choice="required",
            )
        assert "some-model" not in client._toolchoice_unsupported_models
