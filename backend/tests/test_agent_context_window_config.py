"""Per-agent context-window configuration and round-trip contracts."""

from unittest.mock import AsyncMock

import pytest

from backend.config import AgentContextWindows, SystemConfig
from backend.models.api_requests import (
    AgentServerConfigRequest,
    AgentSettingsUpdateRequest,
)
from backend.models.config_schema import AgentOverride, ConfigV1
from backend.models.schemas import AnalysisState
from backend.routes import session_config
from backend.services.config_io import (
    apply_config_to_session,
    build_agent_server_configs,
    export_session_to_config,
)
from backend.utils.settings_provider import AgentSettingsProvider


AGENTS = (
    "pi", "pathology", "bioinformatics", "biochemistry",
    "literature", "painter", "supervisor", "technique",
)


class _Request:
    class State:
        user_id = "user-1"

    state = State()


def _config_dict() -> dict:
    return {
        "schema_version": "1.0",
        "analysis": {
            "name": "Context",
            "initial_instruction": "Analyze",
            "round_wait_seconds": 0,
        },
        "api_config": {
            "default_llm": {
                "llm_base_url": "https://example.test/v1",
                "api_key": "secret",
                "model": "model",
            },
            "literature": {"mode": "disabled"},
        },
        "data": {"groups": [], "files": [], "indicators": []},
    }


def _settings_body(**updates) -> AgentSettingsUpdateRequest:
    values = {
        "agent": "pi",
        "prompt": "prompt",
        "model": "model",
        "temperature": 0.2,
        "llm_base_url": "https://example.test/v1",
        "api_key": "secret",
        "vision_enabled": True,
    }
    values.update(updates)
    return AgentSettingsUpdateRequest(**values)


def test_context_window_defaults_and_schema_floor():
    windows = AgentContextWindows()
    assert [windows.get(agent) for agent in AGENTS] == [200_000] * len(AGENTS)
    assert SystemConfig().context_windows.get("pi") == 200_000
    assert AgentOverride(context_window=128_000).context_window == 128_000
    with pytest.raises(ValueError):
        AgentOverride(context_window=8_191)


def test_provider_resolves_override_default_and_invalid_legacy_value():
    state = AnalysisState(user_id="user-1")
    config = SystemConfig(context_windows={"pi": 131_072})
    provider = AgentSettingsProvider(state=state, config=config, global_client=None)

    assert provider.get_context_window("pi") == 131_072
    state.agent_server_configs["pi"] = {"context_window": 98_304}
    assert provider.get_context_window("pi") == 98_304
    state.agent_server_configs["pi"] = {"context_window": "broken"}
    assert provider.get_context_window("pi") == 131_072
    state.agent_server_configs["pi"] = {"context_window": "98304"}
    assert provider.get_context_window("pi") == 131_072


def test_effective_default_participates_in_config_signature():
    state = AnalysisState(user_id="user-1")
    provider = AgentSettingsProvider(
        state=state,
        config=SystemConfig(context_windows={"pi": 100_000}),
        global_client=None,
    )
    before = provider.get_config_signature("pi")

    provider._config = SystemConfig(context_windows={"pi": 120_000})

    assert provider.get_config_signature("pi") != before


def test_config_v1_roundtrip_preserves_only_explicit_context_override():
    raw = _config_dict()
    raw["agents"] = {"pi": {"context_window": 128_000}}
    config = ConfigV1.model_validate(raw)
    built = build_agent_server_configs(config)

    assert built["pi"]["context_window"] == 128_000
    assert "context_window" not in built["supervisor"]

    state = apply_config_to_session(config, user_id="user-1")
    exported = export_session_to_config(state)
    assert exported["agents"]["pi"]["context_window"] == 128_000
    assert "context_window" not in exported["agents"].get("supervisor", {})


@pytest.mark.asyncio
async def test_agent_server_route_uses_omitted_null_and_integer_semantics(monkeypatch):
    state = AnalysisState(id="context-route", user_id="user-1")
    state.agent_server_configs["pi"] = {
        "llm_base_url": "https://old.test/v1",
        "api_key": "old-key",
        "context_window": 128_000,
    }
    monkeypatch.setattr(
        session_config, "require_session_ownership", AsyncMock(return_value=state)
    )
    monkeypatch.setattr(session_config, "update_session_cache", AsyncMock())
    monkeypatch.setattr(session_config, "async_save_session", AsyncMock())

    await session_config.update_agent_server_config(
        state.id,
        _Request(),
        AgentServerConfigRequest(agent="pi", llm_base_url="", api_key=""),
    )
    assert state.agent_server_configs["pi"]["context_window"] == 128_000

    await session_config.update_agent_server_config(
        state.id,
        _Request(),
        AgentServerConfigRequest(
            agent="pi", llm_base_url="", api_key="", context_window=None
        ),
    )
    assert "context_window" not in state.agent_server_configs["pi"]

    await session_config.update_agent_server_config(
        state.id,
        _Request(),
        AgentServerConfigRequest(
            agent="pi", llm_base_url="", api_key="", context_window=96_000
        ),
    )
    assert state.agent_server_configs["pi"]["context_window"] == 96_000


@pytest.mark.asyncio
async def test_atomic_settings_route_preserves_or_resets_context_override(monkeypatch):
    state = AnalysisState(id="context-atomic", user_id="user-1")
    state.agent_server_configs["pi"] = {"context_window": 128_000}
    monkeypatch.setattr(
        session_config, "require_session_ownership", AsyncMock(return_value=state)
    )
    monkeypatch.setattr(
        session_config, "resolve_config_for_user", AsyncMock(return_value=SystemConfig())
    )
    monkeypatch.setattr(session_config, "update_session_cache", AsyncMock())
    monkeypatch.setattr(session_config, "async_save_session", AsyncMock())

    response = await session_config.update_agent_settings(
        state.id, _Request(), _settings_body()
    )
    assert state.agent_server_configs["pi"]["context_window"] == 128_000
    assert response["agent_server_configs"]["pi"]["context_window"] == 128_000
    assert response["agent_server_configs"]["pi"]["effective_context_window"] == 128_000

    response = await session_config.update_agent_settings(
        state.id, _Request(), _settings_body(context_window=None)
    )
    assert "context_window" not in state.agent_server_configs["pi"]
    assert response["agent_server_configs"]["pi"]["context_window"] is None
    assert response["agent_server_configs"]["pi"]["effective_context_window"] == 200_000
