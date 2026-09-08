import pytest
from fastapi import HTTPException

from backend.models.api_requests import AgentSettingsSnapshotRequest
from backend.models.schemas import AnalysisState
from backend.routes.sessions import _new_session_agent_settings


def _state() -> AnalysisState:
    return AnalysisState(
        id="new-settings-test",
        name="New Settings Test",
        user_id="user-1",
        initial_instruction="test",
    )


def test_new_session_freezes_default_without_creating_manual_override():
    from backend.config import SystemConfig
    from backend.utils.settings_provider import AgentSettingsProvider
    state = _state()
    config = SystemConfig(context_windows={"pi": 160000})
    _new_session_agent_settings(state, None, {}, config)
    assert state.agent_server_configs["pi"]["default_context_window"] == 160000
    assert "context_window" not in state.agent_server_configs["pi"]
    restored = AnalysisState.model_validate_json(state.model_dump_json())
    assert AgentSettingsProvider(restored, SystemConfig(), None).get_context_window("pi") == 160000


def test_new_session_copies_nonsecret_defaults_and_injects_saved_secrets():
    state = _state()
    settings = AgentSettingsSnapshotRequest.model_validate({
        "prompts": {"pi": "Saved prompt"},
        "models": {"pi": "saved-model"},
        "temperatures": {"pi": 0.2},
        "agent_server_configs": {
            "pi": {"llm_base_url": "http://localhost:8000/v1", "context_window": 32768},
        },
        "vision_settings": {"pi": False},
        "literature_config": {
            "mode": "deep_research",
            "deep_research_model": "saved-research-model",
            "deep_research_llm_base_url": "https://research.example/v1",
            "pubmed_email": "saved@example.test",
            "max_search_rounds": 7,
        },
    })

    _new_session_agent_settings(state, settings, {
        "agent_api_keys": {"pi": "agent-secret"},
        "literature": {
            "deep_research_api_key": "research-secret",
            "google_api_key": "google-secret",
        },
    })

    assert state.custom_prompts == {"pi": "Saved prompt"}
    assert state.custom_models == {"pi": "saved-model"}
    assert state.custom_temperatures == {"pi": 0.2}
    assert state.agent_vision_settings == {"pi": False}
    assert state.agent_server_configs["pi"] == {
        "llm_base_url": "http://localhost:8000/v1",
        "api_key": "agent-secret",
        "context_window": 32768,
    }
    assert state.literature_config["deep_research_api_key"] == "research-secret"
    assert state.literature_config["google_api_key"] == "google-secret"


def test_new_session_rejects_malformed_agent_server_url():
    state = _state()
    settings = AgentSettingsSnapshotRequest.model_validate({
        "agent_server_configs": {"pi": {"llm_base_url": "http:/broken"}},
    })

    with pytest.raises(HTTPException, match="valid http or https URL"):
        _new_session_agent_settings(state, settings, {})


def test_new_session_applies_saved_secrets_without_browser_snapshot():
    state = _state()

    _new_session_agent_settings(state, None, {
        "agent_api_keys": {"pi": "agent-secret"},
        "literature": {
            "deep_research_api_key": "research-secret",
            "google_api_key": "google-secret",
        },
    })

    assert state.agent_server_configs["pi"]["api_key"] == "agent-secret"
    assert state.literature_config["deep_research_api_key"] == "research-secret"
    assert state.literature_config["google_api_key"] == "google-secret"


def test_partial_literature_snapshot_only_overrides_explicit_fields():
    state = _state()
    state.literature_config["mode"] = "disabled"
    settings = AgentSettingsSnapshotRequest.model_validate({
        "literature_config": {"pubmed_email": "saved@example.test"},
    })

    _new_session_agent_settings(state, settings, {})

    assert state.literature_config["mode"] == "disabled"
    assert state.literature_config["pubmed_email"] == "saved@example.test"
