import importlib
import os


def test_nested_data_dir(monkeypatch, tmp_path):
    nested = tmp_path / "scai" / "prod"
    monkeypatch.setenv("SCAI_DATA_DIR", str(nested))
    import backend.config._config as cfg
    importlib.reload(cfg)
    assert nested.exists()
