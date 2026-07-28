"""Recovery rollback must leave the session consistent, not half-rolled-back.

apply() installs the target snapshot's h5ad sidecars, its sidecar manifest and its
LangGraph checkpoint. Rollback used to put back only the sidecar *files*, leaving
the manifest describing the target (so the kernel refuses to rehydrate on a hash
mismatch) and the checkpoint pointing at the target graph position (so the session
resumes at a node whose workspace and runtime were just rolled back).
"""
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from backend.services.snapshot_v2.recovery import _RecoveryRestorer


def _store_object(paths, payload: bytes) -> str:
    sha = hashlib.sha256(payload).hexdigest()
    (Path(paths.objects) / sha).write_bytes(payload)
    return sha


class _Paths:
    def __init__(self, root: Path):
        self.root = root / "v2"
        self.objects = root / "objects"
        self.staging = root / "staging"
        self.journals = root / "journals"
        self.integrity = root / "integrity"
        self.committed = root / "committed"
        for d in (self.root, self.objects, self.staging, self.journals,
                  self.integrity, self.committed):
            d.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A session laid out the way restore leaves it mid-apply."""
    from backend.services.snapshot_v2 import recovery as recovery_mod
    from backend.tests.snapshot_v2_fixtures import (
        make_snapshot_v2_manifest,
        write_manifest_to_store,
    )
    from backend.utils import runtime_apply

    session_id = "rollback-probe"
    paths = _Paths(tmp_path)
    session_dir = tmp_path / "session"
    states_root = session_dir / "states"
    workspace_root = session_dir / "workspace"
    for d in (session_dir, states_root, workspace_root):
        d.mkdir(parents=True, exist_ok=True)

    # write_runtime_h5ad_manifest resolves the states dir from the session id.
    monkeypatch.setattr(runtime_apply, "_states_dir", lambda _sid: states_root)

    return {
        "session_id": session_id,
        "paths": paths,
        "session_dir": session_dir,
        "states_root": states_root,
        "workspace_root": workspace_root,
        "make": make_snapshot_v2_manifest,
        "publish": write_manifest_to_store,
        "recovery_mod": recovery_mod,
    }


def _pre_restore_manifest(env, *, adata_refs=None, checkpoint_payload=None):
    from backend.services.snapshot_v2.models import ObjectRef

    paths = env["paths"]

    # The fixture's default refs are placeholder hashes; rollback verifies every
    # ref against the store, so back them with real objects.
    state_payload = json.dumps({"id": env["session_id"]}).encode()
    state_sha = _store_object(paths, state_payload)
    nb_payload = b'{"cells": [], "nbformat": 4, "nbformat_minor": 5}'
    nb_sha = _store_object(paths, nb_payload)

    overrides = {
        "snapshot_id": "pre-restore-1",
        "session_id": env["session_id"],
        "workspace": [],
        "analysis_state_ref": ObjectRef(
            sha256=state_sha, size=len(state_payload),
            media_type="application/json",
        ).model_dump(mode="json"),
        "notebook_ref": ObjectRef(
            sha256=nb_sha, size=len(nb_payload),
            media_type="application/x-ipynb+json",
        ).model_dump(mode="json"),
    }

    runtime = {
        "status": "state_only",
        "dependencies": {"required": {}},
        "residual_ref": None,
        "captured_residual_names": [],
        "adata_refs": {},
    }
    if adata_refs:
        refs = {}
        for name, payload in adata_refs.items():
            sha = _store_object(paths, payload)
            refs[name] = ObjectRef(
                sha256=sha, size=len(payload), media_type="application/x-hdf5",
            ).model_dump(mode="json")
        runtime["adata_refs"] = refs
    overrides["runtime"] = runtime

    if checkpoint_payload is not None:
        sha = _store_object(paths, checkpoint_payload)
        overrides["checkpoint_ref"] = ObjectRef(
            sha256=sha, size=len(checkpoint_payload),
            media_type="application/x-sqlite3",
        ).model_dump(mode="json")

    manifest = env["make"](**overrides)
    env["publish"](manifest, paths.committed, paths.integrity)
    return manifest


def _restorer(env):
    return _RecoveryRestorer(
        env["paths"], env["session_dir"], env["workspace_root"], env["states_root"],
    )


def _journal(env):
    return {
        "transaction_id": "tx-1",
        "session_id": env["session_id"],
        "pre_restore_ref": "pre-restore-1",
    }


def _make_sqlite(path: Path, value: str) -> bytes:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t(v TEXT)")
    conn.execute("INSERT INTO t VALUES(?)", (value,))
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    return path.read_bytes()


def test_rollback_rewrites_the_h5ad_sidecar_manifest(env):
    """Otherwise the kernel hash-checks the restored sidecars against the
    target's hashes and refuses to start."""
    from backend.utils.runtime_apply import (
        runtime_h5ad_manifest_path,
        write_runtime_h5ad_manifest,
    )

    pre_payload = b"PRE-RESTORE-ADATA"
    _pre_restore_manifest(env, adata_refs={"adata": pre_payload})

    # apply() left the manifest describing the TARGET snapshot.
    write_runtime_h5ad_manifest(env["session_id"], {"adata": "f" * 64})

    assert _restorer(env).rollback(_journal(env)) is True

    manifest = json.loads(
        runtime_h5ad_manifest_path(env["session_id"]).read_text(encoding="utf-8")
    )
    recorded = manifest["entries"]["adata"]["sha256"]
    on_disk = hashlib.sha256(
        (env["states_root"] / "adata.h5ad").read_bytes()
    ).hexdigest()

    assert recorded == on_disk, "manifest hash does not match the restored sidecar"
    assert recorded == hashlib.sha256(pre_payload).hexdigest()


def test_rollback_empties_the_manifest_when_pre_restore_had_no_h5ad(env):
    """A stale manifest is as fatal as a wrong one — it names files that are
    no longer the session's runtime."""
    from backend.utils.runtime_apply import (
        runtime_h5ad_manifest_path,
        write_runtime_h5ad_manifest,
    )

    _pre_restore_manifest(env)  # state_only: no adata refs
    write_runtime_h5ad_manifest(env["session_id"], {"adata": "f" * 64})

    assert _restorer(env).rollback(_journal(env)) is True

    manifest = json.loads(
        runtime_h5ad_manifest_path(env["session_id"]).read_text(encoding="utf-8")
    )
    assert manifest["entries"] == {}


def test_rollback_restores_the_langgraph_checkpoint(env, tmp_path):
    """Without this the session resumes at the TARGET graph node against a
    pre_restore workspace and runtime."""
    from backend.services.snapshot_v2.checkpoint_paths import checkpoint_sqlite_path

    pre_bytes = _make_sqlite(tmp_path / "pre.sqlite", "PRE_RESTORE")
    _pre_restore_manifest(env, checkpoint_payload=pre_bytes)

    # apply() installed the target checkpoint.
    live = checkpoint_sqlite_path(env["states_root"])
    _make_sqlite(live, "TARGET")

    assert _restorer(env).rollback(_journal(env)) is True

    conn = sqlite3.connect(live)
    try:
        assert conn.execute("SELECT v FROM t").fetchone()[0] == "PRE_RESTORE"
    finally:
        conn.close()


def test_rollback_keeps_the_wal_so_uncaptured_commits_survive(env, tmp_path):
    """Capture stores the MAIN FILE only, so commits still sitting in the WAL
    were never in the snapshot. apply() replaces only the main file, so that WAL
    still belongs to the file being put back — replaying it restores graph
    progress the snapshot could not capture. Deleting it silently drops that
    progress (measured: keep -> [PRE_A, PRE_IN_WAL], delete -> [PRE_A])."""
    from backend.services.snapshot_v2.checkpoint_paths import checkpoint_sqlite_path

    live = checkpoint_sqlite_path(env["states_root"])

    # A live WAL-mode DB: one row checkpointed into the main file, one left in
    # the -wal. Keep the connection open, exactly as the workflow does.
    wf = sqlite3.connect(live)
    wf.execute("PRAGMA journal_mode=WAL")
    wf.execute("CREATE TABLE t(v TEXT)")
    wf.execute("INSERT INTO t VALUES('IN_MAIN')")
    wf.commit()
    wf.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    wf.execute("INSERT INTO t VALUES('ONLY_IN_WAL')")
    wf.commit()
    try:
        captured_main = live.read_bytes()   # what put_file() would store
        assert live.with_name(live.name + "-wal").exists()

        _pre_restore_manifest(env, checkpoint_payload=captured_main)

        # apply() replaced the main file with the target's; -wal untouched.
        target = tmp_path / "target.sqlite"
        live.write_bytes(_make_sqlite(target, "TARGET"))

        assert _restorer(env).rollback(_journal(env)) is True
    finally:
        wf.close()

    conn = sqlite3.connect(live)
    try:
        rows = {r[0] for r in conn.execute("SELECT v FROM t")}
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()

    assert "IN_MAIN" in rows
    assert "ONLY_IN_WAL" in rows, (
        "the WAL was discarded, losing commits the snapshot never captured"
    )
    assert "TARGET" not in rows


def test_rollback_removes_the_checkpoint_when_pre_restore_had_none(env):
    """No pre_restore checkpoint means the session had none: anything live was
    seeded by apply() from the target, so the whole WAL triple must go."""
    from backend.services.snapshot_v2.checkpoint_paths import checkpoint_sqlite_path

    _pre_restore_manifest(env)  # checkpoint_ref is None

    live = checkpoint_sqlite_path(env["states_root"])
    live.write_bytes(b"target checkpoint installed by apply")
    wal = live.with_name(live.name + "-wal")
    shm = live.with_name(live.name + "-shm")
    wal.write_bytes(b"target wal")
    shm.write_bytes(b"target shm")

    assert _restorer(env).rollback(_journal(env)) is True

    assert not live.exists(), "target checkpoint survived rollback"
    assert not wal.exists(), "orphan -wal would be replayed into a missing DB"
    assert not shm.exists()


def test_rollback_quarantines_sidecars_the_target_added(env):
    """Rewriting the manifest is not enough: a sidecar the TARGET installed but
    pre_restore never had would stay live in the states dir, and the `*.h5ad`
    sweep still finds files the manifest does not name."""
    from backend.utils.runtime_apply import write_runtime_h5ad_manifest

    _pre_restore_manifest(env, adata_refs={"adata": b"PRE-RESTORE-ADATA"})

    # apply() installed two sidecars; pre_restore only ever had `adata`.
    states = env["states_root"]
    (states / "bdata.h5ad").write_bytes(b"TARGET-ONLY-ADATA")
    write_runtime_h5ad_manifest(
        env["session_id"], {"adata": "f" * 64, "bdata": "e" * 64},
    )

    assert _restorer(env).rollback(_journal(env)) is True

    assert (states / "adata.h5ad").exists()
    assert not (states / "bdata.h5ad").exists(), (
        "sidecar added by the restore survived rollback"
    )
