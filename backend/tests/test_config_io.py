"""Tests for config I/O roundtrip (apply_config_to_session / export_session_to_config)."""

from backend.models.config_schema import ConfigV1
from backend.models.schemas import AnalysisState
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
