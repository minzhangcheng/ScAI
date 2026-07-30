"""Edge-2 recovery hardening: the seams where a failed restore is un-fenced.

Four independent invariants, all of them about *who owns the live session after
a restore went wrong*:

* a load that deliberately DEFERRED recovery must not clear an existing block —
  skipping the check is not evidence that nothing is wrong;
* recovery that already rewrote ``session.json`` must say so even when it then
  fails, or the caller keeps the stale cached state and writes it straight back;
* the checkpoint ``-wal``/``-shm`` pair holds graph commits the capture never
  saw, so ``apply()`` may not destroy it while ``rollback()`` still depends on
  it;
* the V2 listing reconcile projects V2 manifests and must not evict the legacy
  V1 records it knows nothing about.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from backend.models.schemas import AnalysisState, Snapshot
from backend.services.snapshot_v2.recovery import _RecoveryRestorer


# -- shared fixtures ----------------------------------------------------------


class _Paths:
    """Minimal SnapshotV2Paths stand-in (same attribute surface recovery uses)."""

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


def _store_object(paths, payload: bytes) -> str:
    sha = hashlib.sha256(payload).hexdigest()
    (Path(paths.objects) / sha).write_bytes(payload)
    return sha


def _make_sqlite(path: Path, value: str) -> bytes:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t(v TEXT)")
    conn.execute("INSERT INTO t VALUES(?)", (value,))
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    return path.read_bytes()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A session laid out the way restore leaves it mid-apply."""
    from backend.tests.snapshot_v2_fixtures import (
        make_snapshot_v2_manifest,
        write_manifest_to_store,
    )
    from backend.utils import runtime_apply

    session_id = "edge2-recovery-probe"
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
    }


def _pre_restore_manifest(env, *, checkpoint_payload=None, drop_notebook_object=False):
    """Publish a verified pre_restore manifest backed by real objects."""
    from backend.services.snapshot_v2.models import ObjectRef

    paths = env["paths"]

    state_payload = json.dumps({"id": env["session_id"], "marker": "PRE"}).encode()
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
        "runtime": {
            "status": "state_only",
            "dependencies": {"required": {}},
            "residual_ref": None,
            "captured_residual_names": [],
            "adata_refs": {},
        },
    }
    if checkpoint_payload is not None:
        sha = _store_object(paths, checkpoint_payload)
        overrides["checkpoint_ref"] = ObjectRef(
            sha256=sha, size=len(checkpoint_payload),
            media_type="application/x-sqlite3",
        ).model_dump(mode="json")

    manifest = env["make"](**overrides)
    env["publish"](manifest, paths.committed, paths.integrity)

    if drop_notebook_object:
        # The notebook copy is the first failable step AFTER rollback has already
        # streamed the pre_restore session.json over the live file.
        (Path(paths.objects) / nb_sha).unlink()

    return manifest, state_payload


def _restorer(env):
    return _RecoveryRestorer(
        env["paths"], env["session_dir"], env["workspace_root"], env["states_root"],
    )


def _journal(env, tx_id: str = "tx-1") -> dict:
    return {
        "transaction_id": tx_id,
        "session_id": env["session_id"],
        "pre_restore_ref": "pre-restore-1",
    }


# -- F01: a deferred recovery pass must not unblock a broken session ----------


@pytest.mark.asyncio
async def test_deferred_recovery_does_not_clear_a_block_on_the_disk_load_path(
    monkeypatch,
):
    """The cached branch already refuses this; the disk-load branch is the one a
    parked session actually reaches.

    Eviction drops the cache entry but leaves ``_recovery_blocked_sessions`` and
    the persisted ``recovery_blocked`` flag, so the next ``/resume`` — which
    takes the session lock *before* loading, making the deferral fire — served
    the load from disk. Clearing the block there hands ``assert_recovery_clear``
    a clean bill of health for a session whose restore_apply journal is still
    APPLYING.
    """
    import backend.config as config_mod
    from backend.services import session_store
    from backend.state import get_session_lock, session_access_times, sessions

    sid = f"edge2-disk-defer-{uuid.uuid4().hex[:8]}"
    ran: list = []
    persisted: list = []

    state = AnalysisState(id=sid, user_id="u1")
    state.recovery_blocked = True
    state.snapshot_transaction = {
        "transaction_id": "tx-dead",
        "status": "applying",
        "recovery_error": {"code": "RECOVERY_BLOCKED", "message": "half-applied"},
    }

    async def _never(_sid):
        ran.append(True)
        return False, None

    async def _no_cache(_sid):
        return None

    async def _persist(_sid, _state):
        persisted.append(_sid)

    monkeypatch.setattr(config_mod, "is_snapshot_v2_enabled", lambda: True)
    monkeypatch.setattr(session_store, "_recover_snapshot_v2_on_load", _never)
    monkeypatch.setattr(session_store, "_get_cached_session", _no_cache)
    monkeypatch.setattr(session_store, "load_session_from_disk", lambda _sid: state)
    monkeypatch.setattr(session_store, "_persist_recovery_marker", _persist)

    session_store._recovery_blocked_sessions.add(sid)
    try:
        async with get_session_lock(sid):
            await session_store.ensure_session_loaded(sid)
        still_marked = sid in session_store._recovery_blocked_sessions
    finally:
        session_store._recovery_blocked_sessions.discard(sid)
        sessions.pop(sid, None)
        session_access_times.pop(sid, None)

    assert not ran, "recovery ran while the route already held the session lock"
    assert still_marked, "a deferred load discarded the process-wide recovery marker"
    assert state.recovery_blocked is True, "a deferred load unblocked the session"
    assert "recovery_error" in (state.snapshot_transaction or {}), (
        "a deferred load stripped the recovery error the UI reports"
    )
    assert not persisted, "a deferred load persisted an unblocked state to disk"


# -- F02: report the session.json rewrite the moment it becomes durable -------


def test_rollback_reports_the_session_json_rewrite_before_a_later_step_fails(env):
    """``rollback()`` writes the live session.json several failable steps before
    it returns, so completion is the wrong moment to report the rewrite."""
    from backend.services.snapshot_v2.recovery import RecoveryFailure

    _, pre_state_payload = _pre_restore_manifest(env, drop_notebook_object=True)
    live_state = env["session_dir"] / "session.json"
    live_state.write_text(json.dumps({"marker": "TARGET"}), encoding="utf-8")

    restorer = _restorer(env)
    with pytest.raises(RecoveryFailure):
        restorer.rollback(_journal(env))

    assert live_state.read_bytes() == pre_state_payload, (
        "precondition: rollback should have restored session.json before failing"
    )
    assert restorer.session_json_rewritten is True, (
        "rollback rewrote the live session.json but did not report it, so the "
        "caller keeps the cached restore-target state"
    )


@pytest.mark.asyncio
async def test_recovery_reports_the_rewrite_when_rollback_fails_afterwards(env):
    """The 0.36.12 cache eviction is gated on ``session_state_rewritten``; a
    rollback that fails *after* the durable write must still arm it."""
    from backend.services.snapshot_v2.recovery import recover_snapshot_transactions
    from backend.services.snapshot_v2.restore_parts.journal import (
        _write_journal_hmac_signed,
    )

    _, pre_state_payload = _pre_restore_manifest(env, drop_notebook_object=True)
    live_state = env["session_dir"] / "session.json"
    live_state.write_text(json.dumps({"marker": "TARGET"}), encoding="utf-8")

    tx_id = "tx-rollback-partial"
    _write_journal_hmac_signed(env["paths"], tx_id, {
        "transaction_id": tx_id,
        "session_id": env["session_id"],
        "target_snapshot_id": "target-1",
        "pre_restore_ref": "pre-restore-1",
        "recovery_mode": "rollback",
        "status": "APPLYING",
        "applied_step_ledger": [],
    })

    result = await recover_snapshot_transactions(
        session_id=env["session_id"],
        paths=env["paths"],
        session_dir=str(env["session_dir"]),
        workspace_root=str(env["workspace_root"]),
        states_root=str(env["states_root"]),
    )

    assert result.mutations_blocked is True, "precondition: the rollback failed"
    assert live_state.read_bytes() == pre_state_payload
    assert result.session_state_rewritten is True, (
        "recovery rewrote session.json and then failed; without the flag the "
        "caller never evicts the cache and persists the target state back"
    )


# -- F05: the checkpoint WAL companions outlive apply() for rollback ----------


@pytest.mark.asyncio
async def test_rollback_after_apply_keeps_the_uncaptured_wal_commits(env, tmp_path):
    """Capture stores the checkpoint MAIN FILE only, so commits still sitting in
    the ``-wal`` were never in the pre_restore snapshot. ``rollback()`` puts that
    same main file back and depends on the companions to recover them — so
    ``apply()`` may remove them from the live triple (SQLite would otherwise
    replay them over the TARGET image) but must not destroy them.

    Drives the real seam: no workflow object exists, so
    ``_retire_superseded_workflow`` never closes the connection and the orphan
    WAL an unclean shutdown left behind is still there when apply runs.
    """
    from backend.services.snapshot_v2.checkpoint_paths import checkpoint_sqlite_path
    from backend.services.snapshot_v2.restore_parts.journal import StagedRestore
    from backend.services.snapshot_v2.restore_parts.coordinator import (
        RestoreCoordinator,
    )

    live = checkpoint_sqlite_path(env["states_root"])

    # A WAL-mode DB with one row folded into the main file and one still in the
    # -wal, then abandoned without a close: the triple is snapshotted while the
    # connection is open and written back after it closed, which is exactly the
    # orphan an unclean shutdown leaves behind.
    wf = sqlite3.connect(live)
    wf.execute("PRAGMA journal_mode=WAL")
    wf.execute("CREATE TABLE t(v TEXT)")
    wf.execute("INSERT INTO t VALUES('IN_MAIN')")
    wf.commit()
    wf.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    wf.execute("INSERT INTO t VALUES('ONLY_IN_WAL')")
    wf.commit()
    companions = ("-wal", "-shm")
    crashed = {"": live.read_bytes()}
    for companion in companions:
        crashed[companion] = live.with_name(live.name + companion).read_bytes()
    wf.close()
    for suffix, payload in crashed.items():
        live.with_name(live.name + suffix).write_bytes(payload)

    captured_main = crashed[""]  # exactly what capture's put_file() stores

    _pre_restore_manifest(env, checkpoint_payload=captured_main)

    tx_id = "tx-wal"
    staged_root = Path(env["paths"].staging) / tx_id
    (staged_root / "workspace").mkdir(parents=True, exist_ok=True)
    staged_ckpt = checkpoint_sqlite_path(staged_root)
    staged_ckpt.write_bytes(_make_sqlite(tmp_path / "target.sqlite", "TARGET"))

    staged = StagedRestore(
        transaction_id=tx_id,
        session_id=env["session_id"],
        snapshot_id="target-1",
        recovery_mode="rollback",
        pre_restore_ref="pre-restore-1",
        branch_id="main",
        continuation_kind="idle",
        manifest={},
        staged_workspace=str(staged_root / "workspace"),
        staged_checkpoint=str(staged_ckpt),
    )

    await RestoreCoordinator().apply(
        staged,
        env["paths"],
        session_dir=str(env["session_dir"]),
        workspace_root=str(env["workspace_root"]),
        states_root=str(env["states_root"]),
        runtime_choice="keep",
    )

    assert not live.with_name(live.name + "-wal").exists(), (
        "precondition: apply must not leave a companion SQLite would replay "
        "over the restored image"
    )

    assert _restorer(env).rollback(_journal(env, tx_id)) is True

    conn = sqlite3.connect(live)
    try:
        rows = {r[0] for r in conn.execute("SELECT v FROM t")}
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()

    assert "IN_MAIN" in rows
    assert "ONLY_IN_WAL" in rows, (
        "apply() destroyed the only copy of graph commits the capture never "
        "held, so the rollback silently resumed at an earlier node"
    )


# -- F28: the V2 listing reconcile must not evict legacy V1 records -----------


def _v1_snapshot(name: str = "legacy-1") -> Snapshot:
    return Snapshot(
        id=f"v1-{name}",
        name=name,
        phase="manual",
        state_file=f"{name}.json",
    )


def _v2_snapshot(name: str = "orphan") -> Snapshot:
    return Snapshot(
        id=f"v2-{name}",
        name=name,
        phase="manual",
        format_version=2,
    )


def _service(session_id, paths):
    from backend.services.snapshot_v2.service import SnapshotService

    return SnapshotService(session_id, paths)


def test_listing_reconcile_carries_legacy_v1_records_forward(env):
    """V2 manifests are authoritative for V2 projections only. A V1 record has
    no manifest by construction, so rebuilding from manifests alone deletes it —
    and with it the only handle the V1 inventory/delete routes have on files
    that GC never sweeps."""
    svc = _service(env["session_id"], env["paths"])

    rebuilt = svc.rebuild_snapshots_from_manifests([_v1_snapshot(), _v2_snapshot()])

    ids = [s.id for s in rebuilt]
    assert "v1-legacy-1" in ids, "the reconcile dropped a legacy V1 record"
    assert "v2-orphan" not in ids, (
        "a V2 projection whose committed manifest is gone must still disappear"
    )


def test_listing_reconcile_defaults_to_the_cached_state(env, monkeypatch):
    """The capture/delete call sites pass no baseline, so the default must come
    from the session cache — otherwise every capture re-drops the V1 records the
    load path just preserved."""
    from backend.state import sessions

    sid = env["session_id"]
    state = AnalysisState(id=sid, user_id="u1")
    state.snapshots = [_v1_snapshot(), _v2_snapshot()]
    monkeypatch.setitem(sessions, sid, state)

    rebuilt = _service(sid, env["paths"]).rebuild_snapshots_from_manifests()

    ids = [s.id for s in rebuilt]
    assert "v1-legacy-1" in ids
    assert "v2-orphan" not in ids


@pytest.mark.asyncio
async def test_session_load_keeps_legacy_v1_snapshot_records(monkeypatch, tmp_path):
    """End-to-end at the decisive seam: ``ensure_session_loaded`` assigns the
    reconcile result wholesale, so a V2-only projection is what the next save
    persists."""
    import backend.config as config_mod
    from backend.services import session_store
    from backend.state import session_access_times, sessions

    sid = f"edge2-legacy-{uuid.uuid4().hex[:8]}"
    state = AnalysisState(id=sid, user_id="u1")
    state.snapshots = [_v1_snapshot()]

    async def _clean(_sid):
        return False, None

    async def _no_cache(_sid):
        return None

    monkeypatch.setattr(config_mod, "is_snapshot_v2_enabled", lambda: True)
    monkeypatch.setattr(session_store, "_recover_snapshot_v2_on_load", _clean)
    monkeypatch.setattr(session_store, "_get_cached_session", _no_cache)
    monkeypatch.setattr(session_store, "load_session_from_disk", lambda _sid: state)
    monkeypatch.setattr(
        "backend.services.snapshot_v2.paths.get_session_dir",
        lambda _sid: tmp_path / "sessions" / _sid,
    )

    try:
        loaded = await session_store.ensure_session_loaded(sid)
        ids = [s.id for s in (loaded.snapshots or [])]
    finally:
        sessions.pop(sid, None)
        session_access_times.pop(sid, None)

    assert "v1-legacy-1" in ids, (
        "the load-path reconcile dropped the legacy V1 record, so the V1 "
        "inventory reports 0 and the V1 delete route answers 404"
    )
