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


@pytest.mark.asyncio
async def test_empty_complete_runtime_removes_stale_residual(tmp_path, monkeypatch):
    from backend.services.snapshot_v2.restore import RestoreCoordinator, StagedRestore

    session_id = "sid-empty-complete"
    session_dir = tmp_path / "session"
    workspace = session_dir / "workspace"
    states = session_dir / "states"
    staged_workspace = tmp_path / "staged-workspace"
    for path in (workspace, states, staged_workspace):
        path.mkdir(parents=True)

    class Paths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"

    for path in (
        Paths.root, Paths.objects, Paths.staging, Paths.journals,
        Paths.integrity, Paths.committed,
    ):
        path.mkdir(parents=True)

    monkeypatch.setattr(
        "backend.utils.runtime_apply.get_session_states",
        lambda _session_id: states,
    )
    live_pkl = states / f"{session_id}_session.pkl"
    live_pkl.write_bytes(b"STALE-PKL")
    (states / "adata.h5ad").write_bytes(b"STALE-H5AD")
    write_runtime_h5ad_manifest(session_id, {"adata": "a" * 64})

    staged = StagedRestore(
        transaction_id="restore-empty-complete",
        session_id=session_id,
        snapshot_id="snap-empty-complete",
        recovery_mode="rollback",
        pre_restore_ref="snap-before",
        branch_id="main",
        continuation_kind="idle",
        manifest={
            "runtime": {
                "status": "complete",
                "residual_ref": None,
                "adata_refs": {},
            }
        },
        staged_workspace=str(staged_workspace),
    )

    result = await RestoreCoordinator().apply(
        staged,
        Paths,
        session_dir=str(session_dir),
        workspace_root=str(workspace),
        states_root=str(states),
        runtime_choice="keep",
    )

    assert result.runtime_restore_status == "complete"
    assert result.runtime_applied is True
    assert not live_pkl.exists()
    assert not (states / "adata.h5ad").exists()
    manifest = json.loads(runtime_h5ad_manifest_path(session_id).read_text())
    assert manifest == {"version": 1, "entries": {}}


def test_repeated_quarantine_preserves_prior_runtime(tmp_path, monkeypatch):
    session_id = "sid-repeat-quarantine"
    states = tmp_path / "states"
    states.mkdir()
    monkeypatch.setattr(
        "backend.utils.runtime_apply.get_session_states",
        lambda _session_id: states,
    )
    live_pkl = states / f"{session_id}_session.pkl"

    live_pkl.write_bytes(b"FIRST")
    quarantine_live_runtime(session_id)
    first_quarantine = live_pkl.with_suffix(live_pkl.suffix + ".quarantine")
    assert first_quarantine.read_bytes() == b"FIRST"

    live_pkl.write_bytes(b"SECOND")
    quarantine_live_runtime(session_id)

    quarantines = sorted(states.glob(f"{live_pkl.name}.quarantine*"))
    assert len(quarantines) == 2
    assert {path.read_bytes() for path in quarantines} == {b"FIRST", b"SECOND"}


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
    stale_pkl = states / f"{session_id}_session.pkl"
    stale_pkl.write_bytes(b"STALE-LIVE-PKL")
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
        "target_runtime_status": "complete",
        "target_runtime_has_residual": False,
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
    assert not stale_pkl.exists()
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
