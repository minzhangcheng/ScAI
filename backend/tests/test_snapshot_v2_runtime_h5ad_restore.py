import json
from pathlib import Path

import pytest

from backend.services.snapshot_v2.runtime_names import resolve_confined_runtime_h5ad_path
from backend.utils.runtime_apply import (
    RUNTIME_H5AD_MANIFEST_FILENAME,
    quarantine_live_runtime,
    runtime_h5ad_manifest_path,
    write_runtime_h5ad_manifest,
)


def test_write_runtime_h5ad_manifest_records_validated_relative_paths(tmp_path, monkeypatch):
    session_id = "sid-h5ad-manifest"
    states = tmp_path / "sessions" / session_id / "states"
    states.mkdir(parents=True)
    monkeypatch.setattr(
        "backend.utils.runtime_apply.get_session_states",
        lambda _session_id: states,
    )

    written = write_runtime_h5ad_manifest(
        session_id,
        {"adata": "a" * 64},
    )

    assert written == states / RUNTIME_H5AD_MANIFEST_FILENAME
    payload = json.loads(written.read_text())
    assert payload == {
        "version": 1,
        "entries": {
            "adata": {
                "path": "adata.h5ad",
                "sha256": "a" * 64,
            }
        },
    }
    assert runtime_h5ad_manifest_path(session_id) == written


def test_quarantine_live_runtime_clears_h5ad_manifest_and_sidecars(tmp_path, monkeypatch):
    session_id = "sid-h5ad-quarantine"
    states = tmp_path / "sessions" / session_id / "states"
    states.mkdir(parents=True)
    monkeypatch.setattr(
        "backend.utils.runtime_apply.get_session_states",
        lambda _session_id: states,
    )

    live_pkl = states / f"{session_id}_session.pkl"
    live_pkl.write_bytes(b"pkl")
    h5ad = states / "adata.h5ad"
    h5ad.write_bytes(b"H5AD")
    other_h5ad = states / "raw.h5ad"
    other_h5ad.write_bytes(b"RAW")
    write_runtime_h5ad_manifest(session_id, {"adata": "b" * 64})
    manifest = runtime_h5ad_manifest_path(session_id)
    assert manifest.exists()

    quarantine_live_runtime(session_id)

    assert not live_pkl.exists()
    assert live_pkl.with_suffix(live_pkl.suffix + ".quarantine").exists()
    assert not manifest.exists()
    assert manifest.with_suffix(manifest.suffix + ".quarantine").exists()
    assert not h5ad.exists()
    assert h5ad.with_suffix(h5ad.suffix + ".quarantine").exists()
    assert not other_h5ad.exists()
    assert other_h5ad.with_suffix(other_h5ad.suffix + ".quarantine").exists()


@pytest.mark.parametrize(
    "entry_path",
    [
        "../escape.h5ad",
        "subdir/adata.h5ad",
        "/tmp/adata.h5ad",
        "other.h5ad",
        "adata.h5ad/../x",
    ],
)
def test_resolve_confined_runtime_h5ad_path_rejects_unsafe_paths(tmp_path, entry_path):
    states = tmp_path / "states"
    states.mkdir()
    with pytest.raises(ValueError):
        resolve_confined_runtime_h5ad_path(states, "adata", entry_path)


def test_resolve_confined_runtime_h5ad_path_accepts_exact_basename(tmp_path):
    states = tmp_path / "states"
    states.mkdir()
    path = resolve_confined_runtime_h5ad_path(states, "adata", "adata.h5ad")
    assert path == (states / "adata.h5ad").resolve()
    assert path.parent == states.resolve()


def test_install_live_runtime_h5ad_set_empty_clears_stale_sidecars(tmp_path, monkeypatch):
    """Complete residual-only runtime must not leave prior live h5ad+manifest."""
    from backend.utils.runtime_apply import install_live_runtime_h5ad_set

    session_id = "sid-h5ad-empty-apply"
    states = tmp_path / "sessions" / session_id / "states"
    states.mkdir(parents=True)
    monkeypatch.setattr(
        "backend.utils.runtime_apply.get_session_states",
        lambda _session_id: states,
    )

    stale_h5ad = states / "adata.h5ad"
    stale_h5ad.write_bytes(b"STALE-H5AD")
    write_runtime_h5ad_manifest(session_id, {"adata": "c" * 64})
    manifest = runtime_h5ad_manifest_path(session_id)
    assert manifest.exists()
    assert stale_h5ad.exists()

    # Target set is empty (pkl-only / residual-only complete runtime).
    hashes = install_live_runtime_h5ad_set(session_id, {})

    assert hashes == {}
    payload = json.loads(manifest.read_text())
    assert payload == {"version": 1, "entries": {}}
    assert not stale_h5ad.exists()
    assert stale_h5ad.with_suffix(stale_h5ad.suffix + ".quarantine").exists()


def test_recovery_forward_complete_empty_h5ad_without_runtime_dir(
    tmp_path, monkeypatch,
):
    """Pkl-only staged targets still clear stale live h5ad+manifest on recovery."""
    from backend.services.snapshot_v2.recovery import _RecoveryRestorer

    session_id = "sid-h5ad-recovery-empty"
    session_dir = tmp_path / "session"
    workspace = session_dir / "workspace"
    states = session_dir / "states"
    staging_root = tmp_path / "staging"
    for d in (workspace, states, staging_root):
        d.mkdir(parents=True)

    class FakePaths:
        def __init__(self, root: Path, staging_dir: Path):
            self.root = root / "v2"
            self.objects = root / "objects"
            self.staging = staging_dir
            self.journals = root / "journals"
            self.integrity = root / "integrity"
            self.committed = root / "committed"

    monkeypatch.setattr(
        "backend.utils.runtime_apply.get_session_states",
        lambda _session_id: states,
    )

    stale_h5ad = states / "adata.h5ad"
    stale_h5ad.write_bytes(b"STALE-LIVE-H5AD")
    write_runtime_h5ad_manifest(session_id, {"adata": "f" * 64})
    manifest = runtime_h5ad_manifest_path(session_id)
    assert manifest.exists()

    tx_id = "restore-empty-h5ad"
    staged_root = staging_root / tx_id
    # Staging present, but no runtime/ dir (pkl-only complete target).
    staged_root.mkdir(parents=True)
    (staged_root / "analysis_state.json").write_text(
        json.dumps({"id": session_id, "current_round": 1})
    )

    journal = {
        "transaction_id": tx_id,
        "session_id": session_id,
        "applied_step_ledger": [
            {"step": "workspace", "completed": True},
            {"step": "state", "completed": True},
        ],
    }
    restorer = _RecoveryRestorer(
        FakePaths(tmp_path, staging_root), session_dir, workspace, states,
    )
    ok = restorer.forward_complete(journal)
    assert ok is True

    assert not stale_h5ad.exists()
    assert stale_h5ad.with_suffix(stale_h5ad.suffix + ".quarantine").exists()
    payload = json.loads(manifest.read_text())
    assert payload == {"version": 1, "entries": {}}
    assert any(
        step.get("step") == "runtime_sig" and step.get("completed")
        for step in journal["applied_step_ledger"]
    )


def test_install_live_runtime_h5ad_set_replaces_and_drops_unreferenced(tmp_path, monkeypatch):
    """Non-empty install rewrites manifest and quarantines names not in target."""
    import hashlib

    from backend.utils.runtime_apply import install_live_runtime_h5ad_set

    session_id = "sid-h5ad-partial-apply"
    states = tmp_path / "sessions" / session_id / "states"
    states.mkdir(parents=True)
    monkeypatch.setattr(
        "backend.utils.runtime_apply.get_session_states",
        lambda _session_id: states,
    )

    old_adata = states / "adata.h5ad"
    old_adata.write_bytes(b"OLD-ADATA")
    old_raw = states / "raw.h5ad"
    old_raw.write_bytes(b"OLD-RAW")
    write_runtime_h5ad_manifest(
        session_id,
        {"adata": "d" * 64, "raw": "e" * 64},
    )

    staged = tmp_path / "staged_adata.h5ad"
    staged.write_bytes(b"NEW-ADATA")
    hashes = install_live_runtime_h5ad_set(session_id, {"adata": str(staged)})

    expected = hashlib.sha256(b"NEW-ADATA").hexdigest()
    assert hashes == {"adata": expected}
    assert old_adata.read_bytes() == b"NEW-ADATA"
    assert not old_raw.exists()
    assert old_raw.with_suffix(old_raw.suffix + ".quarantine").exists()
    payload = json.loads(runtime_h5ad_manifest_path(session_id).read_text())
    assert payload["entries"] == {
        "adata": {"path": "adata.h5ad", "sha256": expected},
    }
