"""Tests for config I/O roundtrip (apply_config_to_session / export_session_to_config)."""
import os

import pytest
from backend.models.config_schema import ConfigV1
from backend.models.schemas import AnalysisState
from backend.services.config_io import apply_config_to_session, export_session_to_config
from backend.services.report.artifacts import ARTIFACT_REGISTRY_VERSION


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
        assert state.artifact_registry_version == ARTIFACT_REGISTRY_VERSION

        exported = export_session_to_config(state)
        assert exported["analysis"]["name"] == "Roundtrip Test"
        assert exported["analysis"]["round_wait_seconds"] == 60

    def test_max_rounds_roundtrip(self):
        data = _config_dict()
        data["analysis"]["max_rounds"] = 7
        config = ConfigV1.model_validate(data)
        state = apply_config_to_session(config, user_id="u1")

        assert state.max_rounds == 7
        assert export_session_to_config(state)["analysis"]["max_rounds"] == 7

    def test_max_rounds_absent_keeps_installation_default(self):
        config = ConfigV1.model_validate(_config_dict())
        state = apply_config_to_session(config, user_id="u1")
        installed_default = AnalysisState.model_fields["max_rounds"].default

        assert state.max_rounds == installed_default
        assert "max_rounds" not in export_session_to_config(state)["analysis"]

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

    def test_disabled_agent_stays_disabled_across_roundtrip(self):
        data = _config_dict()
        data["agents"]["pi"]["model"] = ""
        config = ConfigV1.model_validate(data)

        state = apply_config_to_session(config, user_id="u1")
        assert "pi" in state.custom_models
        assert state.custom_models["pi"] == ""
        assert state.agent_server_configs["pi"]["model"] == ""

        exported = export_session_to_config(state)
        assert exported["agents"]["pi"]["model"] == ""

        restored = apply_config_to_session(
            ConfigV1.model_validate(exported), user_id="u1"
        )
        assert restored.custom_models["pi"] == ""
        assert restored.agent_server_configs["pi"]["model"] == ""

    def test_absent_agent_keeps_installation_default(self):
        data = _config_dict()
        data["agents"] = {}
        config = ConfigV1.model_validate(data)

        state = apply_config_to_session(config, user_id="u1")
        assert "pi" not in state.custom_models
        assert "model" not in export_session_to_config(state)["agents"].get("pi", {})

    def test_all_agents_get_fallback_server_config(self):
        """Every agent gets server_config from default_llm if no override.

        The endpoint must be stored under the key the runtime reads. Writing
        ``base_url`` here while ``AgentSettingsProvider.get_llm_client`` and
        ``agent_llm.get_agent_llm_client`` both read ``llm_base_url`` dropped
        the endpoint from every imported configuration and left the paired
        credential to be sent to the destination's own global endpoint.
        """
        config = ConfigV1.model_validate(_config_dict())
        state = apply_config_to_session(config, user_id="u1")

        for name in ["pi", "pathology", "bioinformatics", "biochemistry",
                     "literature", "painter", "supervisor", "technique"]:
            assert name in state.agent_server_configs, f"{name} missing from agent_server_configs"
            cfg = state.agent_server_configs[name]
            assert cfg["llm_base_url"] == "https://api.test.com"
            assert cfg["api_key"] == "sk-test"

    def test_agent_server_config_key_matches_what_the_runtime_reads(self):
        """Pin the contract instead of a spelling: read it back the way agents do."""
        import inspect

        from backend.utils import agent_llm
        from backend.utils import settings_provider

        config = ConfigV1.model_validate(_config_dict())
        state = apply_config_to_session(config, user_id="u1")

        for module in (agent_llm, settings_provider):
            source = inspect.getsource(module)
            assert 'agent_cfg.get("llm_base_url"' in source, (
                f"{module.__name__} no longer reads llm_base_url; "
                "build_agent_server_configs must be updated to match"
            )

        assert all(
            "llm_base_url" in cfg for cfg in state.agent_server_configs.values()
        )

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
        """Import fallback should be 20 (matching schema default)."""
        d = _config_dict()
        d["api_config"]["literature"] = {"mode": "disabled"}
        config = ConfigV1.model_validate(d)
        state = apply_config_to_session(config, user_id="u1")
        # Fallback when max_search_rounds is not set should match the product default.
        assert state.literature_config["max_search_rounds"] == 20


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


class TestExportedLocalPath:
    def test_bound_files_export_real_distinct_relative_paths(self, tmp_path, monkeypatch):
        from backend.services import config_io
        from backend.models.schemas import AnalysisState, DataFile, TreatmentGroup

        session_root = tmp_path / "session"
        monkeypatch.setattr(
            config_io,
            "get_session_dir",
            lambda _session_id: session_root,
            raising=False,
        )
        group_a = "group-a"
        group_b = "group-b"
        state = AnalysisState(
            id="export-paths",
            user_id="u1",
            name="Export paths",
            treatment_groups=[
                TreatmentGroup(id=group_a, name="Control", sample_count=1, order=0),
                TreatmentGroup(id=group_b, name="Treatment", sample_count=1, order=1),
            ],
            data_files=[
                DataFile(
                    id="file-a",
                    filename="same.h5ad",
                    file_path=str(
                        session_root / "uploads/sequencing/group-a/_unsorted/same.h5ad"
                    ),
                    section="sequencing",
                    data_type="h5ad",
                    group_id=group_a,
                ),
                DataFile(
                    id="file-b",
                    filename="same.h5ad",
                    file_path=str(
                        session_root / "uploads/sequencing/group-b/2/same.h5ad"
                    ),
                    section="sequencing",
                    data_type="h5ad",
                    group_id=group_b,
                    sample_number=2,
                ),
            ],
        )

        exported = config_io.export_session_to_config(state)
        assert [entry["local_path"] for entry in exported["data"]["files"]] == [
            "uploads/sequencing/group-a/_unsorted/same.h5ad",
            "uploads/sequencing/group-b/2/same.h5ad",
        ]
