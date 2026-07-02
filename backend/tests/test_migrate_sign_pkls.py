import pytest
from pathlib import Path


def test_migration_signs_unsigned_pkl(tmp_path, monkeypatch):
    monkeypatch.setenv("SCAI_CONFIG_SECRET", "test-secret-for-integrity-12345")
    from backend.utils import sandbox
    import backend.migrate_sign_session_pkls as mig

    sid = "sess1"
    states = tmp_path / "sessions" / sid / "states"
    states.mkdir(parents=True)
    pkl = states / f"{sid}_session.pkl"
    pkl.write_bytes(b"payload")
    monkeypatch.setattr(mig, "iter_session_pkls", lambda: [(sid, pkl)])
    mig.sign_all()
    assert sandbox._verify_pkl_integrity(pkl, sandbox._integrity_dir(sid)) is True
