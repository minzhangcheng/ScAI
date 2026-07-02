import pytest
from pathlib import Path


def test_restore_rejects_tampered_pkl(tmp_path, monkeypatch):
    monkeypatch.setenv("SCAI_CONFIG_SECRET", "test-secret-for-integrity-12345")
    from backend.utils import sandbox

    integrity_dir = tmp_path / ".integrity"
    integrity_dir.mkdir()
    pkl = tmp_path / "s_session.pkl"
    pkl.write_bytes(b"payload-A")
    sandbox._write_pkl_signature(pkl, integrity_dir)
    pkl.write_bytes(b"payload-EVIL")
    assert sandbox._verify_pkl_integrity(pkl, integrity_dir) is False


def test_restore_accepts_untampered(tmp_path):
    import os
    os.environ.setdefault("SCAI_CONFIG_SECRET", "test-secret-for-integrity-12345")
    from backend.utils import sandbox

    integrity_dir = tmp_path / ".integrity"
    integrity_dir.mkdir()
    pkl = tmp_path / "s_session.pkl"
    pkl.write_bytes(b"payload-A")
    sandbox._write_pkl_signature(pkl, integrity_dir)
    assert sandbox._verify_pkl_integrity(pkl, integrity_dir) is True


def test_integrity_dir_not_sandbox_writable(tmp_path, monkeypatch):
    import backend.config._config as cfg
    from backend.utils import sandbox

    fake_data_base = Path("/nonexistent-scai-integrity-test-path")
    monkeypatch.setattr(cfg, "DATA_BASE_DIR", fake_data_base)

    sb = sandbox.LocalJupyterSandbox(session_id="s", working_dir=str(tmp_path / "workspace"))

    integ = str(fake_data_base / "integrity" / "s" / "x")
    assert not any(
        sandbox._path_is_within(integ, root) for root in sb._allowed_write_roots()
    )
