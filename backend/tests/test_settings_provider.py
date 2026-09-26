"""Unit tests for AgentSettingsProvider."""
import asyncio
import os
os.environ.setdefault("SCAI_DATA_DIR", "/tmp/scai_test_data")

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from backend.models.schemas import AnalysisState
from backend.config._config import SystemConfig, APIConfig, ModelConfig, AgentTemperatures
from backend.config.agent_prompts import AgentPrompts
from backend.utils.settings_provider import AgentSettingsProvider


@pytest.fixture
def mock_global_client():
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    client.api_key = "test-key"
    return client


@pytest.fixture
def state():
    return AnalysisState(
        id="test-session",
        name="Test",
        initial_instruction="test",
        user_id="user1",
    )


@pytest.fixture
def config():
    return SystemConfig(
        api=APIConfig(api_key="global-key", llm_base_url="https://openrouter.ai/api/v1"),
        models=ModelConfig(),
        prompts=AgentPrompts(),
        temperatures=AgentTemperatures(),
    )


@pytest.fixture
def provider(state, config, mock_global_client):
    return AgentSettingsProvider(state, config, mock_global_client)


class TestGetLlmClient:
    def test_propagates_before_retry_to_global_cached_lazy_and_external_clients(
        self, provider, state, mock_global_client
    ):
        on_retry = lambda *args: None

        async def before_retry():
            return None

        provider.set_retry_callback(on_retry, before_retry)
        assert mock_global_client.on_retry is on_retry
        assert mock_global_client.before_retry is before_retry

        state.agent_server_configs["pi"] = {
            "llm_base_url": "https://custom.api/v1",
            "api_key": "custom-key",
        }
        with patch("backend.utils.settings_provider.OpenAICompatibleClient") as MockClient:
            cached = MagicMock()
            MockClient.return_value = cached
            assert provider.get_llm_client("pi") is cached
            assert cached.on_retry is on_retry
            assert cached.before_retry is before_retry

        state.agent_server_configs["literature"] = {
            "llm_base_url": "https://lazy.api/v1",
            "api_key": "lazy-key",
        }
        with patch("backend.utils.settings_provider.OpenAICompatibleClient") as MockClient:
            lazy = MagicMock()
            MockClient.return_value = lazy
            assert provider.get_llm_client("literature") is lazy
            assert lazy.on_retry is on_retry
            assert lazy.before_retry is before_retry

        external = MagicMock()
        provider.apply_retry_callback(external)
        assert external.on_retry is on_retry
        assert external.before_retry is before_retry

        registered = MagicMock()
        provider.register_client(("deep_research", "registered"), registered)
        assert registered.on_retry is on_retry
        assert registered.before_retry is before_retry

class TestGetStateFreshness:
    def test_reads_from_sessions_dict_when_session_id_set(self, state, config, mock_global_client):
        """When session_id is set, Provider should read from sessions[id], not cached ref."""
        from backend.state import sessions
        sessions["test-freshness"] = state
        provider = AgentSettingsProvider(state, config, mock_global_client, session_id="test-freshness")
        assert provider.get_model("pi") == config.models.pi_model
        # Mutate the state IN the sessions dict (simulating session_config route)
        sessions["test-freshness"].custom_models["pi"] = "fresh-model"
        assert provider.get_model("pi") == "fresh-model"
        # Replace the state object (simulating LangGraph node transition)
        new_state = AnalysisState(id="test-freshness", name="Test", initial_instruction="test", user_id="u1")
        sessions["test-freshness"] = new_state
        new_state.custom_models["pi"] = "post-checkpoint-model"
        assert provider.get_model("pi") == "post-checkpoint-model"
        del sessions["test-freshness"]

class TestCleanup:
    @pytest.mark.asyncio
    async def test_clear_cache_defers_close_until_client_lease_released(
        self, provider,
    ):
        client = AsyncMock()
        provider._client_cache[("pi", "https://custom.test/v1", "key")] = client
        provider.lease_client(client)

        provider.clear_agent_client_cache("pi")
        await asyncio.sleep(0)

        client.close.assert_not_awaited()
        assert client in provider._retired_clients

        provider.release_client(client)
        await asyncio.sleep(0)

        client.close.assert_awaited_once()
        assert client not in provider._retired_clients
        await provider.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_retains_failed_client_for_retry(self, provider):
        client = AsyncMock()
        client.close.side_effect = [RuntimeError("close failed"), None]
        provider._client_cache[("pi", "https://custom.test/v1", "key")] = client

        with pytest.raises(RuntimeError, match="failed to close 1"):
            await provider.cleanup()

        assert client in provider._retired_clients
        assert provider._cleanup_complete is False
        await provider.cleanup()
        assert client.close.await_count == 2
        assert provider._cleanup_complete is True

# --- BaseAgent integration tests ---
