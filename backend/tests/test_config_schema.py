"""Tests for ConfigV1 schema validation and field mapping."""
import os

import pytest
from pydantic import ValidationError
from backend.models.config_schema import (
    ConfigV1, AnalysisSection, DefaultLlm, LiteratureConfig,
    ApiConfigSection, DataSection, AgentOverride, GroupEntry, FileEntry,
)


def _valid_config(**overrides) -> dict:
    """Return a minimal valid ConfigV1 dict with optional overrides."""
    base = {
        "schema_version": "1.0",
        "analysis": {"name": "Test", "initial_instruction": "Analyze", "round_wait_seconds": 300},
        "api_config": {
            "default_llm": {"llm_base_url": "https://api.test.com", "api_key": "sk-test"},
            "literature": {"mode": "disabled"},
        },
    }
    base.update(overrides)
    return base


class TestConfigV1Validation:
    def test_minimal_valid(self):
        c = ConfigV1.model_validate(_valid_config())
        assert c.schema_version == "1.0"
        assert c.analysis.name == "Test"
        assert c.api_config.default_llm.llm_base_url == "https://api.test.com"

    def test_round_wait_minus1_allowed(self):
        """round_wait_seconds=-1 is valid (manual skip mode)."""
        cfg = _valid_config()
        cfg["analysis"]["round_wait_seconds"] = -1
        c = ConfigV1.model_validate(cfg)
        assert c.analysis.round_wait_seconds == -1

    def test_round_wait_below_minus1_rejected(self):
        cfg = _valid_config()
        cfg["analysis"]["round_wait_seconds"] = -2
        with pytest.raises(ValidationError):
            ConfigV1.model_validate(cfg)

    def test_round_wait_above_3600_rejected(self):
        cfg = _valid_config()
        cfg["analysis"]["round_wait_seconds"] = 3601
        with pytest.raises(ValidationError):
            ConfigV1.model_validate(cfg)

    def test_max_rounds_optional_and_bounded(self):
        cfg = _valid_config()
        assert ConfigV1.model_validate(cfg).analysis.max_rounds is None
        for value in (1, 50):
            cfg["analysis"]["max_rounds"] = value
            assert ConfigV1.model_validate(cfg).analysis.max_rounds == value
        for value in (0, 51):
            cfg["analysis"]["max_rounds"] = value
            with pytest.raises(ValidationError):
                ConfigV1.model_validate(cfg)

    def test_unknown_agent_name_rejected(self):
        cfg = _valid_config(agents={"unknown_agent": {"model": "gpt-4"}})
        with pytest.raises(ValidationError):
            ConfigV1.model_validate(cfg)

    def test_valid_agent_names(self):
        cfg = _valid_config(agents={
            "pi": {"model": "gpt-4"},
            "pathology": {"custom_prompt": "Be precise"},
            "biochemistry": {"vision": True},
        })
        c = ConfigV1.model_validate(cfg)
        assert c.agents["pi"].model == "gpt-4"
        assert c.agents["pathology"].custom_prompt == "Be precise"
        assert c.agents["biochemistry"].vision is True

    def test_file_group_ref_validated(self):
        cfg = _valid_config()
        cfg["data"] = {
            "groups": [{"key": "ctrl", "name": "Control", "sample_count": 3}],
            "files": [{"local_path": "data.csv", "section": "sequencing", "data_type": "h5ad", "group": "nonexistent"}],
        }
        with pytest.raises(ValidationError):
            ConfigV1.model_validate(cfg)

    def test_file_group_ref_valid(self):
        cfg = _valid_config()
        cfg["data"] = {
            "groups": [{"key": "ctrl", "name": "Control", "sample_count": 3}],
            "files": [{"local_path": "data.csv", "section": "sequencing", "data_type": "h5ad", "group": "ctrl"}],
        }
        c = ConfigV1.model_validate(cfg)
        assert c.data.files[0].group == "ctrl"

    def test_default_llm_model_optional(self):
        """model field has default empty string."""
        cfg = _valid_config()
        c = ConfigV1.model_validate(cfg)
        assert c.api_config.default_llm.model == ""

    def test_default_llm_with_model(self):
        cfg = _valid_config()
        cfg["api_config"]["default_llm"]["model"] = "gpt-4o"
        c = ConfigV1.model_validate(cfg)
        assert c.api_config.default_llm.model == "gpt-4o"

    def test_proxy_url_optional(self):
        cfg = _valid_config()
        cfg["api_config"]["default_llm"]["proxy_url"] = "http://proxy:8080"
        c = ConfigV1.model_validate(cfg)
        assert c.api_config.default_llm.proxy_url == "http://proxy:8080"

    def test_literature_max_search_rounds_default(self):
        cfg = _valid_config()
        c = ConfigV1.model_validate(cfg)
        # Product default raised 10 -> 20 in ver 0.34.7 across every config
        # layer (models/config_schema.py:35).
        assert c.api_config.literature.max_search_rounds == 20

    def test_literature_max_search_rounds_bounds(self):
        cfg = _valid_config()
        cfg["api_config"]["literature"]["max_search_rounds"] = 0
        with pytest.raises(ValidationError):
            ConfigV1.model_validate(cfg)

        cfg["api_config"]["literature"]["max_search_rounds"] = 51
        with pytest.raises(ValidationError):
            ConfigV1.model_validate(cfg)

        cfg["api_config"]["literature"]["max_search_rounds"] = 50
        c = ConfigV1.model_validate(cfg)
        assert c.api_config.literature.max_search_rounds == 50

    def test_agent_override_fields(self):
        """Verify field names: custom_prompt (not prompt), vision (not vision_enabled)."""
        ov = AgentOverride(model="m", custom_prompt="p", vision=True, temperature=0.5)
        assert ov.model == "m"
        assert ov.custom_prompt == "p"
        assert ov.vision is True
        assert ov.temperature == 0.5
        # Ensure old field names don't exist
        assert not hasattr(ov, "prompt")
        assert not hasattr(ov, "vision_enabled")

    def test_empty_analysis_name_rejected(self):
        cfg = _valid_config()
        cfg["analysis"]["name"] = ""
        with pytest.raises(ValidationError):
            ConfigV1.model_validate(cfg)

    def test_group_key_required(self):
        g = GroupEntry(key="ctrl", name="Control", sample_count=3)
        assert g.key == "ctrl"

    def test_group_key_empty_rejected(self):
        with pytest.raises(ValidationError):
            GroupEntry(key="", name="Control", sample_count=3)
