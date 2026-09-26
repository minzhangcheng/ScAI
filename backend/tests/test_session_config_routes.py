from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.config import SystemConfig
from backend.models.api_requests import (
    AgentSettingsUpdateRequest,
    LiteratureConfigRequest,
)
from backend.models.schemas import AnalysisState
from backend.routes import session_config
from backend.state import sessions, workflows
from backend.utils.settings_provider import MASKED_SECRET_VALUE
from backend.utils.secret_redaction import mask_api_key


class _Request:
    class State:
        user_id = "user-1"
    state = State()


def _state(session_id="route-config-test"):
    return AnalysisState(id=session_id, name="Route Config Test", user_id="user-1", initial_instruction="test")


@pytest.mark.asyncio
async def test_literature_config_route_preserves_masked_key_updates_cache_and_invalidates(monkeypatch):
    session_id = "route-config-test"
    state = _state(session_id)
    state.literature_config = {"deep_research_api_key": "old-key", "google_api_key": "old-google"}
    sessions[session_id] = state
    provider = MagicMock()
    workflow = MagicMock()
    workflow._settings_provider = provider
    workflow.literature_agent = MagicMock()
    workflow.literature_agent.refresh_runtime_config_if_needed = MagicMock()
    workflows[session_id] = workflow

    monkeypatch.setattr(session_config, "require_session_ownership", AsyncMock(return_value=state))
    save_mock = AsyncMock()
    cache_mock = AsyncMock()
    monkeypatch.setattr(session_config, "async_save_session", save_mock)
    monkeypatch.setattr(session_config, "update_session_cache", cache_mock)

    body = LiteratureConfigRequest(
        mode="deep_research",
        deep_research_model="o4-mini-deep-research",
        deep_research_llm_base_url="https://sg.uiuiapi.com/v1",
        deep_research_api_key=MASKED_SECRET_VALUE,
        google_api_key="",
        pubmed_email="user@example.com",
        max_search_rounds=6,
    )

    try:
        response = await session_config.update_literature_config(session_id, _Request(), body)
    finally:
        workflows.pop(session_id, None)
        sessions.pop(session_id, None)

    assert response.success is True
    assert state.literature_config["deep_research_api_key"] == "old-key"
    assert state.literature_config["google_api_key"] == ""
    cache_mock.assert_awaited_once_with(session_id, state)
    save_mock.assert_awaited_once_with(session_id, state)
    provider.clear_agent_client_cache.assert_called_once_with("literature")
    workflow.literature_agent.refresh_runtime_config_if_needed.assert_called_once()


@pytest.mark.asyncio
async def test_agent_settings_route_updates_once_and_invalidates_before_next_call(monkeypatch):
    session_id = "route-agent-settings-test"
    state = _state(session_id)
    state.agent_server_configs["pi"] = {
        "llm_base_url": "https://old.example/v1",
        "api_key": "sk-123456789",
    }
    sessions[session_id] = state
    provider = MagicMock()
    workflow = MagicMock()
    workflow._settings_provider = provider
    workflow.pi_agent = MagicMock()
    workflow.pi_agent.refresh_runtime_config_if_needed = MagicMock()
    workflows[session_id] = workflow
    events = []
    workflow.pi_agent.refresh_runtime_config_if_needed.side_effect = lambda: events.append("invalidate")

    monkeypatch.setattr(session_config, "require_session_ownership", AsyncMock(return_value=state))
    monkeypatch.setattr(
        session_config, "resolve_config_for_user", AsyncMock(return_value=SystemConfig())
    )
    save_mock = AsyncMock(side_effect=lambda *_args, **_kwargs: events.append("save"))
    cache_mock = AsyncMock(side_effect=lambda *_args, **_kwargs: events.append("cache"))
    monkeypatch.setattr(session_config, "async_save_session", save_mock)
    monkeypatch.setattr(session_config, "update_session_cache", cache_mock)
    database = object()
    monkeypatch.setattr(session_config, "get_db", AsyncMock(return_value=database))
    monkeypatch.setattr(
        session_config,
        "load_agent_default_secrets",
        AsyncMock(return_value={"agent_api_keys": {}, "literature": {}}),
    )
    save_defaults_mock = AsyncMock()
    monkeypatch.setattr(
        session_config, "save_agent_default_secrets", save_defaults_mock
    )

    body = AgentSettingsUpdateRequest(
        agent="pi",
        prompt="Updated PI prompt",
        model="mimo-v2.5",
        temperature=0.3,
        llm_base_url="https://new.example/v1",
        api_key=mask_api_key("sk-123456789"),
        vision_enabled=False,
        save_as_user_default=True,
    )
    try:
        response = await session_config.update_agent_settings(session_id, _Request(), body)
    finally:
        workflows.pop(session_id, None)
        sessions.pop(session_id, None)

    assert state.custom_prompts["pi"] == "Updated PI prompt"
    assert state.custom_models["pi"] == "mimo-v2.5"
    assert state.custom_temperatures["pi"] == 0.3
    assert state.agent_server_configs["pi"] == {
        "llm_base_url": "https://new.example/v1",
        "api_key": "sk-123456789",
    }
    assert state.agent_vision_settings["pi"] is False
    assert response["models"]["pi"] == "mimo-v2.5"
    assert response["default_settings_saved"] is True
    assert response["agent_server_configs"]["pi"]["api_key"] == mask_api_key("sk-123456789")
    cache_mock.assert_awaited_once_with(session_id, state)
    save_mock.assert_awaited_once_with(session_id, state)
    provider.clear_agent_client_cache.assert_called_once_with("pi")
    workflow.pi_agent.refresh_runtime_config_if_needed.assert_called_once()
    assert events == ["cache", "invalidate", "save"]
    save_defaults_mock.assert_awaited_once_with(
        database,
        "user-1",
        {"agent_api_keys": {"pi": "sk-123456789"}, "literature": {}},
    )
