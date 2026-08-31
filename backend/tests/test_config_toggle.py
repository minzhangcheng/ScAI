import importlib
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)


class TestReproducibilityToggleConfig:
    """Verify SystemConfig.enable_reproducibility defaults and explicit values."""

    def test_explicit_true(self):
        from backend.config._config import SystemConfig
        assert SystemConfig(enable_reproducibility=True).enable_reproducibility is True

    def test_explicit_false(self):
        from backend.config._config import SystemConfig
        assert SystemConfig(enable_reproducibility=False).enable_reproducibility is False

    def test_default_config_default_is_false(self):
        """Module-level default_config defaults enable_reproducibility to False."""
        from backend.config._config import default_config
        assert default_config.enable_reproducibility is False


class TestReproducibilityToggleEnvVar:
    """Verify SCAI_ENABLE_REPRODUCIBILITY controls the toggle at import time."""

    @pytest.mark.parametrize("value,expected", [
        ("true", True), ("True", True), ("TRUE", True),
        ("yes", True), ("1", True),
        ("false", False), ("0", False), ("no", False),
        ("", False),
    ])
    def test_env_var(self, monkeypatch, value, expected):
        import backend.config._config as _cfg
        monkeypatch.setenv("SCAI_ENABLE_REPRODUCIBILITY", value)
        importlib.reload(_cfg)
        assert _cfg.default_config.enable_reproducibility is expected

    def test_env_var_unset_defaults_false(self, monkeypatch):
        monkeypatch.delenv("SCAI_ENABLE_REPRODUCIBILITY", raising=False)
        import backend.config._config as _cfg
        importlib.reload(_cfg)
        assert _cfg.default_config.enable_reproducibility is False


class TestReproducibilityToggleEnvVarE2E:
    """End-to-end baseline: env var works through a full subprocess import."""

    def test_env_var_true_subprocess(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "from backend.config._config import default_config; "
             "print(default_config.enable_reproducibility)"],
            env={**os.environ, "SCAI_ENABLE_REPRODUCIBILITY": "true"},
            capture_output=True, text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.stdout.strip() == "True"


class TestMethodDecisionGuidanceToggle:
    """The experimental method-card pilot is process-scoped and default-off."""

    def test_explicit_values_and_provider(self):
        from backend.config._config import SystemConfig
        from backend.models.schemas import AnalysisState
        from backend.utils.settings_provider import AgentSettingsProvider

        enabled = SystemConfig(enable_method_decision_guidance=True)
        disabled = SystemConfig(enable_method_decision_guidance=False)
        assert enabled.enable_method_decision_guidance is True
        assert disabled.enable_method_decision_guidance is False
        provider = AgentSettingsProvider(
            AnalysisState(user_id="toggle-test"), enabled, global_client=None
        )
        assert provider.get_method_decision_guidance_enabled() is True

    def test_default_is_false(self):
        from backend.config._config import default_config

        assert default_config.enable_method_decision_guidance is False

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("true", True), ("yes", True), ("1", True), ("false", False), ("0", False), ("", False)],
    )
    def test_environment_value(self, monkeypatch, value, expected):
        import backend.config._config as _cfg

        monkeypatch.setenv("SCAI_ENABLE_METHOD_DECISION_GUIDANCE", value)
        importlib.reload(_cfg)
        assert _cfg.default_config.enable_method_decision_guidance is expected


class TestGeneratorReproducibilityToggle:
    """Verify the conditional branch in generator.py works correctly."""

    def test_maybe_generate_reproducibility_skips_when_disabled(self, monkeypatch):
        """When enable_reproducibility=False, no bundle is generated and path is None."""
        from backend.services.report import generator as gen_mod
        monkeypatch.setattr(gen_mod.default_config, "enable_reproducibility", False)

        state = MagicMock(id="test-session", data_files=[], round_results=[])
        gen = gen_mod.DetailedReportGenerator.__new__(gen_mod.DetailedReportGenerator)
        result = gen._maybe_generate_reproducibility_bundle(
            state=state,
            notebook_src=None,
        )
        assert result is None

    def test_maybe_generate_reproducibility_runs_when_enabled(self, monkeypatch, tmp_path):
        """When enable_reproducibility=True, the method attempts bundle generation."""
        from backend.services.report import generator as gen_mod
        monkeypatch.setattr(gen_mod.default_config, "enable_reproducibility", True)

        mock_instance = MagicMock()
        fake_path = str(tmp_path / "reproducibility")
        mock_instance.generate.return_value = fake_path
        mock_cls = MagicMock(return_value=mock_instance)

        monkeypatch.setattr(gen_mod, "get_session_workspace", lambda sid: tmp_path)
        monkeypatch.setattr(gen_mod, "get_session_uploads", lambda sid: tmp_path)

        import backend.services.reproducibility as repro_mod
        monkeypatch.setattr(repro_mod, "ReproducibilityBundle", mock_cls)

        state = MagicMock(id="test-session", data_files=[], round_results=[])
        gen = gen_mod.DetailedReportGenerator.__new__(gen_mod.DetailedReportGenerator)
        result = gen._maybe_generate_reproducibility_bundle(
            state=state,
            notebook_src=tmp_path / "notebook.ipynb",
        )
        assert result == fake_path
        mock_cls.assert_called_once()

    def test_bundle_uses_bound_file_paths(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        from backend.services.report import generator as gen_mod

        monkeypatch.setattr(gen_mod.default_config, "enable_reproducibility", True)
        first = tmp_path / "uploads/sequencing/group-a/_unsorted/same.h5ad"
        second = tmp_path / "uploads/sequencing/group-b/2/same.h5ad"
        first.parent.mkdir(parents=True)
        second.parent.mkdir(parents=True)
        first.write_bytes(b"a")
        second.write_bytes(b"b")

        bundle = MagicMock()
        bundle.generate.return_value = str(tmp_path / "bundle")
        bundle_cls = MagicMock(return_value=bundle)
        import backend.services.reproducibility as repro_mod

        monkeypatch.setattr(repro_mod, "ReproducibilityBundle", bundle_cls)
        monkeypatch.setattr(gen_mod, "get_session_workspace", lambda _sid: tmp_path)
        monkeypatch.setattr(
            gen_mod,
            "get_session_uploads",
            lambda _sid: tmp_path / "uploads",
        )

        state = SimpleNamespace(
            id="bound-paths",
            data_files=[
                SimpleNamespace(filename="same.h5ad", file_path=str(first)),
                SimpleNamespace(filename="same.h5ad", file_path=str(second)),
            ],
            round_results=[],
        )
        gen = gen_mod.DetailedReportGenerator.__new__(gen_mod.DetailedReportGenerator)

        result = gen._maybe_generate_reproducibility_bundle(
            state=state,
            notebook_src=tmp_path / "notebook.ipynb",
        )

        assert result == str(tmp_path / "bundle")
        assert bundle.generate.call_args.kwargs["upload_paths"] == [
            str(first),
            str(second),
        ]
