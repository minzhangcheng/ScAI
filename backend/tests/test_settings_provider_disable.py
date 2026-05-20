import pytest
from backend.utils.settings_provider import AgentSettingsProvider
from backend.models.schemas import AnalysisState
from backend.config._config import SystemConfig


def _make_provider(custom_models=None, custom_prompts=None):
    state = AnalysisState(session_id="t", user_id="test")
    if custom_models is not None:
        state.custom_models = custom_models
    if custom_prompts is not None:
        state.custom_prompts = custom_prompts
    cfg = SystemConfig()
    return AgentSettingsProvider(state=state, config=cfg, global_client=None, session_id=None)


def test_get_model_missing_key_returns_default():
    p = _make_provider(custom_models={})
    assert p.get_model("pathology") == p._config.models.pathology_model


def test_get_model_explicit_empty_means_disabled():
    p = _make_provider(custom_models={"pathology": ""})
    assert p.get_model("pathology") == ""


def test_get_model_whitespace_only_is_disabled():
    p = _make_provider(custom_models={"pathology": "   \t\n  "})
    assert p.get_model("pathology") == ""


def test_get_model_explicit_value_wins_over_default():
    p = _make_provider(custom_models={"pathology": "my/model"})
    assert p.get_model("pathology") == "my/model"


def test_get_system_prompt_explicit_empty_returns_empty():
    p = _make_provider(custom_prompts={"pathology": ""})
    assert p.get_system_prompt("pathology") == ""


def test_get_system_prompt_missing_key_returns_default():
    p = _make_provider(custom_prompts={})
    assert p.get_system_prompt("pathology") == p._config.prompts.pathology_prompt


def test_get_system_prompt_whitespace_only_returns_empty():
    p = _make_provider(custom_prompts={"pathology": "   \t\n  "})
    assert p.get_system_prompt("pathology") == ""
