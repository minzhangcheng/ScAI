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


class TestGetModel:
    def test_returns_global_default_when_no_override(self, provider, config):
        assert provider.get_model("pi") == config.models.pi_model

    def test_returns_custom_override(self, provider, state):
        state.custom_models["pi"] = "anthropic/claude-sonnet-4.6"
        assert provider.get_model("pi") == "anthropic/claude-sonnet-4.6"

    def test_override_takes_precedence(self, provider, state, config):
        state.custom_models["bioinformatics"] = "custom-model"
        assert provider.get_model("bioinformatics") == "custom-model"

    def test_missing_agent_returns_empty(self, provider):
        assert provider.get_model("nonexistent") == ""

    def test_hot_reload_model_change(self, provider, state, config):
        assert provider.get_model("pi") == config.models.pi_model
        state.custom_models["pi"] = "new-model"
        assert provider.get_model("pi") == "new-model"


class TestGetTemperature:
    def test_returns_global_default(self, provider, config):
        assert provider.get_temperature("pi") == config.temperatures.pi

    def test_returns_custom_override(self, provider, state):
        state.custom_temperatures["pi"] = 0.9
        assert provider.get_temperature("pi") == 0.9

    def test_hot_reload_temperature_change(self, provider, state, config):
        assert provider.get_temperature("pi") == config.temperatures.pi
        state.custom_temperatures["pi"] = 0.8
        assert provider.get_temperature("pi") == 0.8


class TestGetSystemPrompt:
    def test_returns_global_default_when_no_override(self, provider, config):
        result = provider.get_system_prompt("pi")
        assert result == config.prompts.pi_prompt

    def test_returns_custom_override(self, provider, state):
        state.custom_prompts["pi"] = "Custom PI prompt"
        assert provider.get_system_prompt("pi") == "Custom PI prompt"

    def test_hot_reload_prompt_change(self, provider, state, config):
        assert provider.get_system_prompt("pi") == config.prompts.pi_prompt
        state.custom_prompts["pi"] = "New prompt"
        assert provider.get_system_prompt("pi") == "New prompt"


class TestGetLlmClient:
    def test_returns_global_client_when_no_override(self, provider, mock_global_client):
        client = provider.get_llm_client("pi")
        assert client is mock_global_client

    def test_creates_new_client_when_override_differs(self, provider, state):
        state.agent_server_configs["pi"] = {
            "llm_base_url": "https://custom.api/v1",
            "api_key": "custom-key",
        }
        with patch("backend.utils.settings_provider.OpenAICompatibleClient") as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            client = provider.get_llm_client("pi")
            assert client is mock_instance
            MockClient.assert_called_once()

    def test_caches_client_by_url_and_key(self, provider, state):
        state.agent_server_configs["pi"] = {
            "llm_base_url": "https://custom.api/v1",
            "api_key": "custom-key",
        }
        with patch("backend.utils.settings_provider.OpenAICompatibleClient") as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            client1 = provider.get_llm_client("pi")
            client2 = provider.get_llm_client("pi")
            assert client1 is client2
            assert MockClient.call_count == 1

    def test_returns_global_client_when_override_matches_global(self, provider, state, mock_global_client, config):
        state.agent_server_configs["pi"] = {
            "llm_base_url": config.api.llm_base_url,
            "api_key": config.api.api_key,
        }
        client = provider.get_llm_client("pi")
        assert client is mock_global_client

    def test_injects_retry_callback_on_new_client(self, provider, state):
        """Newly created clients should get on_retry set from provider's retry_callback."""
        provider.set_retry_callback(lambda **kw: None)
        state.agent_server_configs["pi"] = {
            "llm_base_url": "https://custom.api/v1",
            "api_key": "custom-key",
        }
        with patch("backend.utils.settings_provider.OpenAICompatibleClient") as MockClient:
            mock_instance = MagicMock()
            mock_instance.on_retry = None
            MockClient.return_value = mock_instance
            provider.get_llm_client("pi")
            assert mock_instance.on_retry is not None

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

    def test_notification_update_does_not_clear_existing_pause_gate(
        self, provider, mock_global_client
    ):
        existing_on_retry = lambda *args: None
        mock_global_client.on_retry = existing_on_retry

        async def before_retry():
            return None

        provider.set_before_retry_callback(before_retry)
        assert mock_global_client.on_retry is existing_on_retry
        provider.set_retry_callback(lambda *args: None)
        assert mock_global_client.before_retry is before_retry


class TestGetVisionSetting:
    def test_returns_true_by_default(self, provider):
        assert provider.get_vision_setting("pi") is True

    def test_returns_custom_override(self, provider, state):
        state.agent_vision_settings["pi"] = False
        assert provider.get_vision_setting("pi") is False

    def test_hot_reload_vision_change(self, provider, state):
        assert provider.get_vision_setting("pi") is True
        state.agent_vision_settings["pi"] = False
        assert provider.get_vision_setting("pi") is False


class TestGetLiteratureConfig:
    def test_returns_default_config(self, provider):
        config = provider.get_literature_config()
        assert config["mode"] == "pubmed_llm"
        assert config["deep_research_model"] == "o4-mini-deep-research"
        assert config["max_search_rounds"] == 20

    def test_returns_custom_config(self, provider, state):
        state.literature_config["mode"] = "pubmed_llm"
        state.literature_config["max_search_rounds"] = 10
        config = provider.get_literature_config()
        assert config["mode"] == "pubmed_llm"
        assert config["max_search_rounds"] == 10


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

    def test_falls_back_to_direct_ref_when_session_not_in_dict(self, state, config, mock_global_client):
        """When sessions[id] is missing, Provider falls back to the direct reference."""
        provider = AgentSettingsProvider(state, config, mock_global_client, session_id="nonexistent-id")
        assert provider.get_model("pi") == config.models.pi_model
        state.custom_models["pi"] = "fallback-model"
        assert provider.get_model("pi") == "fallback-model"

    def test_uses_direct_ref_when_no_session_id(self, state, config, mock_global_client):
        """For tests that don't set session_id, use the direct reference."""
        provider = AgentSettingsProvider(state, config, mock_global_client)
        state.custom_models["pi"] = "direct-ref-model"
        assert provider.get_model("pi") == "direct-ref-model"


class TestCleanup:
    @pytest.mark.asyncio
    async def test_closes_cached_clients(self, provider, state):
        state.agent_server_configs["pi"] = {
            "llm_base_url": "https://custom.api/v1",
            "api_key": "custom-key",
        }
        with patch("backend.utils.settings_provider.OpenAICompatibleClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance
            provider.get_llm_client("pi")
            await provider.cleanup()
            mock_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_error_on_empty_cache(self, provider):
        await provider.cleanup()

    @pytest.mark.asyncio
    async def test_clear_cache_closes_retired_client_on_running_loop(
        self, provider,
    ):
        client = AsyncMock()
        provider._client_cache[("pi", "https://custom.test/v1", "key")] = client

        provider.clear_agent_client_cache("pi")
        await asyncio.sleep(0)

        client.close.assert_awaited_once()
        assert client not in provider._retired_clients
        await provider.cleanup()

    @pytest.mark.asyncio
    async def test_clear_all_cache_retires_every_owned_client(self, provider):
        first = AsyncMock()
        second = AsyncMock()
        provider._client_cache[("pi", "https://first.test/v1", "key")] = first
        provider._client_cache[
            ("literature", "https://second.test/v1", "key")
        ] = second

        provider.clear_client_cache()
        await asyncio.sleep(0)

        assert not provider._client_cache
        first.close.assert_awaited_once()
        second.close.assert_awaited_once()
        assert not provider._retired_clients
        await provider.cleanup()

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
    async def test_scheduled_close_rechecks_late_lease_before_transport_close(
        self, provider,
    ):
        client = AsyncMock()
        provider._client_cache[("pi", "https://custom.test/v1", "key")] = client

        provider.clear_agent_client_cache("pi")
        provider.lease_client(client)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        client.close.assert_not_awaited()
        assert client in provider._retired_clients

        provider.release_client(client)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        client.close.assert_awaited_once()
        assert client not in provider._retired_clients
        await provider.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_retains_in_use_client_for_retry(self, provider):
        client = AsyncMock()
        provider._client_cache[("pi", "https://custom.test/v1", "key")] = client
        provider.lease_client(client)

        with pytest.raises(RuntimeError, match="still in use"):
            await provider.cleanup()

        client.close.assert_not_awaited()
        assert client in provider._retired_clients
        provider.release_client(client)
        await asyncio.sleep(0)
        await provider.cleanup()
        client.close.assert_awaited_once()

    def test_clear_cache_retains_client_until_cleanup_without_running_loop(
        self, provider,
    ):
        client = AsyncMock()
        provider._client_cache[("pi", "https://custom.test/v1", "key")] = client

        provider.clear_agent_client_cache("pi")

        assert client in provider._retired_clients
        asyncio.run(provider.cleanup())
        client.close.assert_awaited_once()
        assert not provider._retired_clients

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

    @pytest.mark.asyncio
    async def test_cleanup_rejects_late_client_creation(self, provider, state):
        blocker = asyncio.Event()
        entered = asyncio.Event()
        client = AsyncMock()

        async def blocked_close():
            entered.set()
            await blocker.wait()

        client.close.side_effect = blocked_close
        provider._client_cache[("pi", "https://custom.test/v1", "key")] = client
        cleanup_task = asyncio.create_task(provider.cleanup())
        await entered.wait()

        state.agent_server_configs["literature"] = {
            "llm_base_url": "https://late.test/v1",
            "api_key": "late-key",
        }
        with pytest.raises(RuntimeError, match="provider is closing"):
            provider.get_llm_client("literature")
        with pytest.raises(RuntimeError, match="provider is closing"):
            provider.register_client(("late", "client"), AsyncMock())

        blocker.set()
        await cleanup_task
        assert client.close.await_count == 1
        assert provider._cleanup_complete is True

    @pytest.mark.asyncio
    async def test_cleanup_does_not_double_close_late_registered_client(
        self, provider,
    ):
        initial_entered = asyncio.Event()
        initial_release = asyncio.Event()
        late_entered = asyncio.Event()
        late_release = asyncio.Event()

        initial = AsyncMock()

        async def close_initial():
            initial_entered.set()
            await initial_release.wait()

        initial.close.side_effect = close_initial
        provider._client_cache[("pi", "https://initial.test/v1", "key")] = initial
        provider.clear_agent_client_cache("pi")
        await initial_entered.wait()

        cleanup_task = asyncio.create_task(provider.cleanup())
        await asyncio.sleep(0)

        late = AsyncMock()

        async def close_late():
            late_entered.set()
            await late_release.wait()

        late.close.side_effect = close_late
        with pytest.raises(RuntimeError, match="provider is closing"):
            provider.register_client(("late", "client"), late)
        await late_entered.wait()

        initial_release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert late.close.await_count == 1

        late_release.set()
        await cleanup_task
        assert late.close.await_count == 1
        assert provider._cleanup_complete is True


class TestRegisterClient:
    def test_register_external_client(self, provider):
        mock_client = MagicMock()
        provider.register_client(("deep_research", "deep_research"), mock_client)
        assert provider.get_llm_client("pi") is not mock_client
        # Registered client is in cache and will be cleaned up
        assert ("registered", "deep_research", "deep_research") in provider._client_cache

    @pytest.mark.asyncio
    async def test_reregister_same_key_retires_previous_client(self, provider):
        previous = AsyncMock()
        replacement = AsyncMock()

        provider.register_client(("deep_research", "deep_research"), previous)
        provider.register_client(("deep_research", "deep_research"), replacement)
        await asyncio.sleep(0)

        previous.close.assert_awaited_once()
        assert previous not in provider._retired_clients
        assert provider._client_cache[
            ("registered", "deep_research", "deep_research")
        ] is replacement
        await provider.cleanup()
        replacement.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_register_after_cleanup_keeps_ownership_with_caller(self, provider):
        await provider.cleanup()
        late_client = AsyncMock()

        with pytest.raises(RuntimeError, match="already closed"):
            provider.register_client(("late", "client"), late_client)

        late_client.close.assert_not_awaited()
        assert late_client not in provider._retired_clients
        assert not provider._client_cache

    def test_cached_clients_returns_all(self, provider, state):
        state.agent_server_configs["pi"] = {
            "llm_base_url": "https://custom.api/v1",
            "api_key": "custom-key",
        }
        with patch("backend.utils.settings_provider.OpenAICompatibleClient") as MockClient:
            mock1 = MagicMock()
            MockClient.return_value = mock1
            provider.get_llm_client("pi")
        mock2 = MagicMock()
        provider.register_client(("deep_research", "deep_research"), mock2)
        all_clients = list(provider.cached_clients())
        assert mock1 in all_clients
        assert mock2 in all_clients


# --- BaseAgent integration tests ---

from backend.agents.base import BaseAgent, AgentConfig
from backend.models.schemas import AgentType


class _StubAgent(BaseAgent):
    """Concrete agent for testing."""
    async def process(self, state, instruction, **kwargs):
        return {"result": "ok"}


class TestBaseAgentHotReload:
    def test_model_property_reads_from_provider(self, provider, state, config):
        agent = _StubAgent(
            agent_type=AgentType.PI,
            settings_provider=provider,
        )
        assert agent.model == config.models.pi_model
        state.custom_models["pi"] = "new-model"
        assert agent.model == "new-model"

    def test_system_prompt_property_reads_from_provider(self, provider, state, config):
        agent = _StubAgent(
            agent_type=AgentType.PI,
            settings_provider=provider,
        )
        assert agent.system_prompt == config.prompts.pi_prompt
        state.custom_prompts["pi"] = "Updated prompt"
        assert agent.system_prompt == "Updated prompt"

    def test_config_temperature_reads_from_provider(self, provider, state, config):
        agent = _StubAgent(
            agent_type=AgentType.PI,
            settings_provider=provider,
        )
        assert agent.config.temperature == config.temperatures.pi
        state.custom_temperatures["pi"] = 0.9
        assert agent.config.temperature == 0.9

    def test_llm_client_property_reads_from_provider(self, provider, mock_global_client):
        agent = _StubAgent(
            agent_type=AgentType.PI,
            settings_provider=provider,
        )
        assert agent.llm_client is mock_global_client

    def test_has_vision_reads_from_provider(self, provider, state):
        agent = _StubAgent(
            agent_type=AgentType.PI,
            settings_provider=provider,
        )
        assert agent.has_vision is True
        state.agent_vision_settings["pi"] = False
        assert agent.has_vision is False

    def test_has_vision_runtime_override(self, provider, state):
        """Runtime override (agent detects vision failure) still works."""
        agent = _StubAgent(
            agent_type=AgentType.PI,
            settings_provider=provider,
        )
        assert agent.has_vision is True
        agent._vision_disabled_runtime = True
        assert agent.has_vision is False
        # Re-enabling via provider should NOT override runtime flag
        assert agent.has_vision is False
        # Explicit set back to True clears runtime flag
        agent.has_vision = True
        assert agent.has_vision is True
