"""Unit tests for AgentSettingsProvider."""
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
        assert config["mode"] == "deep_research"
        assert config["max_search_rounds"] == 5

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


class TestRegisterClient:
    def test_register_external_client(self, provider):
        mock_client = MagicMock()
        provider.register_client(("deep_research", "deep_research"), mock_client)
        assert provider.get_llm_client("pi") is not mock_client
        # Registered client is in cache and will be cleaned up
        assert ("deep_research", "deep_research") in provider._client_cache

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