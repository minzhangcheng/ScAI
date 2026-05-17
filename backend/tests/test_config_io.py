"""Tests for config I/O roundtrip (apply_config_to_session / export_session_to_config)."""
import os

import pytest
from backend.models.config_schema import ConfigV1
from backend.services.config_io import apply_config_to_session, export_session_to_config


def _config_dict(**overrides) -> dict:
    """Return a minimal valid ConfigV1 dict."""
    base = {
        "schema_version": "1.0",
        "analysis": {"name": "Roundtrip Test", "initial_instruction": "Analyze cells", "round_wait_seconds": 60},
        "api_config": {
            "default_llm": {"llm_base_url": "https://api.test.com", "api_key": "sk-test", "model": "gpt-4o"},
            "literature": {"mode": "disabled"},
        },
        "data": {
            "groups": [
                {"key": "ctrl", "name": "Control", "sample_count": 3},
                {"key": "treat", "name": "Treatment", "sample_count": 3},
            ],
            "files": [
                {"local_path": "ctrl.h5ad", "section": "sequencing", "data_type": "h5ad", "group": "ctrl"},
                {"local_path": "treat.h5ad", "section": "sequencing", "data_type": "h5ad", "group": "treat"},
            ],
        },
        "agents": {
            "pi": {"model": "gpt-4o", "custom_prompt": "Focus on T cells", "vision": True},
        },
    }
    base.update(overrides)
    return base


class TestConfigRoundtrip:
    def test_basic_roundtrip(self):
        """apply then export preserves key fields."""
        config = ConfigV1.model_validate(_config_dict())
        state = apply_config_to_session(config, user_id="u1")

        assert state.user_id == "u1"
        assert state.name == "Roundtrip Test"
        assert len(state.treatment_groups) == 2
        assert len(state.data_files) == 2
        assert state.data_files[0].filename == "ctrl.h5ad"
        assert state.data_files[1].filename == "treat.h5ad"
        assert state.round_wait_seconds == 60

        exported = export_session_to_config(state)
        assert exported["analysis"]["name"] == "Roundtrip Test"
        assert exported["analysis"]["round_wait_seconds"] == 60

    def test_agent_custom_prompt_roundtrip(self):
        """custom_prompt field survives roundtrip (not 'prompt')."""
        config = ConfigV1.model_validate(_config_dict())
        state = apply_config_to_session(config, user_id="u1")
        assert "pi" in state.custom_prompts
        assert state.custom_prompts["pi"] == "Focus on T cells"

        exported = export_session_to_config(state)
        assert exported["agents"]["pi"]["custom_prompt"] == "Focus on T cells"
        assert "prompt" not in exported["agents"]["pi"]

    def test_agent_vision_roundtrip(self):
        """vision field survives roundtrip (not 'vision_enabled')."""
        config = ConfigV1.model_validate(_config_dict())
        state = apply_config_to_session(config, user_id="u1")
        assert "pi" in state.agent_vision_settings
        assert state.agent_vision_settings["pi"] is True

        exported = export_session_to_config(state)
        assert exported["agents"]["pi"]["vision"] is True
        assert "vision_enabled" not in exported["agents"]["pi"]

    def test_agent_model_roundtrip(self):
        config = ConfigV1.model_validate(_config_dict())
        state = apply_config_to_session(config, user_id="u1")
        assert "pi" in state.custom_models
        assert state.custom_models["pi"] == "gpt-4o"

        exported = export_session_to_config(state)
        assert exported["agents"]["pi"]["model"] == "gpt-4o"

    def test_all_agents_get_fallback_server_config(self):
        """Every agent gets server_config from default_llm if no override."""
        config = ConfigV1.model_validate(_config_dict())
        state = apply_config_to_session(config, user_id="u1")

        for name in ["pi", "pathology", "bioinformatics", "biochemistry",
                     "literature", "painter", "supervisor", "technique"]:
            assert name in state.agent_server_configs, f"{name} missing from agent_server_configs"
            cfg = state.agent_server_configs[name]
            assert cfg["base_url"] == "https://api.test.com"
            assert cfg["api_key"] == "sk-test"

    def test_literature_config_roundtrip(self):
        """Literature config roundtrips through nested dict."""
        d = _config_dict()
        d["api_config"]["literature"] = {
            "mode": "deep_research",
            "deep_research_model": "o4-mini",
            "max_search_rounds": 3,
        }
        config = ConfigV1.model_validate(d)
        state = apply_config_to_session(config, user_id="u1")
        assert state.literature_config["mode"] == "deep_research"
        assert state.literature_config["deep_research_model"] == "o4-mini"
        assert state.literature_config["max_search_rounds"] == 3

        exported = export_session_to_config(state)
        lit = exported["api_config"]["literature"]
        assert lit["mode"] == "deep_research"
        assert lit["deep_research_model"] == "o4-mini"

    def test_proxy_url_roundtrip(self):
        """proxy_url survives roundtrip via config_snapshot."""
        d = _config_dict()
        d["api_config"]["default_llm"]["proxy_url"] = "http://proxy:8080"
        config = ConfigV1.model_validate(d)
        state = apply_config_to_session(config, user_id="u1")

        # state has no proxy field; export reads from config_snapshot
        exported = export_session_to_config(state)
        assert exported["api_config"]["default_llm"]["proxy_url"] == "http://proxy:8080"

    def test_proxy_url_missing_is_none(self):
        """No proxy_url → export has None."""
        config = ConfigV1.model_validate(_config_dict())
        state = apply_config_to_session(config, user_id="u1")
        exported = export_session_to_config(state)
        assert exported["api_config"]["default_llm"]["proxy_url"] is None

    def test_config_snapshot_stored(self):
        """config_snapshot is set on state after apply."""
        config = ConfigV1.model_validate(_config_dict())
        state = apply_config_to_session(config, user_id="u1")
        assert state.config_snapshot is not None
        assert state.config_snapshot["schema_version"] == "1.0"

    def test_round_wait_minus1_preserved(self):
        """round_wait_seconds=-1 (manual skip) survives roundtrip."""
        d = _config_dict()
        d["analysis"]["round_wait_seconds"] = -1
        config = ConfigV1.model_validate(d)
        state = apply_config_to_session(config, user_id="u1")
        assert state.round_wait_seconds == -1

        exported = export_session_to_config(state)
        assert exported["analysis"]["round_wait_seconds"] == -1

    def test_export_validates_against_configv1(self):
        """Exported dict must pass ConfigV1 validation."""
        config = ConfigV1.model_validate(_config_dict())
        state = apply_config_to_session(config, user_id="u1")
        exported = export_session_to_config(state)
        # Should not raise
        ConfigV1.model_validate(exported)

    def test_placeholder_files_have_empty_path(self):
        """After apply, files are placeholders with empty file_path."""
        config = ConfigV1.model_validate(_config_dict())
        state = apply_config_to_session(config, user_id="u1")
        for df in state.data_files:
            assert df.file_path == ""
            assert df.filename != ""

    def test_max_search_rounds_default_aligned(self):
        """Import fallback should be 5 (matching schema default)."""
        d = _config_dict()
        d["api_config"]["literature"] = {"mode": "disabled"}
        config = ConfigV1.model_validate(d)
        state = apply_config_to_session(config, user_id="u1")
        # Fallback when max_search_rounds not set: should be 5
        assert state.literature_config["max_search_rounds"] == 5


def test_bioinfo_max_steps_default():
    """bioinfo_max_steps defaults to 200 and roundtrips correctly."""
    from backend.config._config import SystemConfig
    cfg = SystemConfig()
    assert cfg.bioinfo_max_steps == 200
    data = cfg.model_dump(mode="json")
    restored = SystemConfig.model_validate(data)
    assert restored.bioinfo_max_steps == 200


def test_bioinfo_max_steps_custom():
    """bioinfo_max_steps can be overridden."""
    from backend.config._config import SystemConfig
    cfg = SystemConfig(bioinfo_max_steps=500)
    assert cfg.bioinfo_max_steps == 500