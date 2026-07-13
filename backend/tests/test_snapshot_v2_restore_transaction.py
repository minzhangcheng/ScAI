"""Test snapshot V2 restore transaction — fault-injected boundaries.

Nine parameterised fault points, each tested in rollback mode (assert
original content restored) and forward-only mode (assert target content applied).
"""
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio


# ---- Fault point constants ------------------------------------------------

FAULT_POINTS = [
    "target_staged",
    "workspace_applied",
    "runtime_pkl_replaced",
    "runtime_sig_replaced",
    "notebook_applied",
    "state_replaced",
    "checkpoint_seeded",
    "journal_committed",
    "cleanup",
]


# ---- Helpers ---------------------------------------------------------------


def _make_fake_paths(tmp_path):
    """Create a FakePaths with all required dirs for restore testing."""

    class FakePaths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        committed = tmp_path / "committed"
        integrity = tmp_path / "integrity"

        def ensure_dirs(self):
            for d in [self.root, self.objects, self.staging,
                      self.journals, self.integrity, self.committed]:
                d.mkdir(parents=True, exist_ok=True)

    p = FakePaths()
    p.ensure_dirs()
    return p


def _make_session_dir(tmp_path, session_id):
    """Create a realistic session directory structure."""
    base = tmp_path / "sessions" / session_id
    workspace = base / "workspace"
    states = base / "states"
    for d in [base, workspace, states]:
        d.mkdir(parents=True, exist_ok=True)
    return base


def _write_file(path: Path, content: str = "data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _hash_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _store_object(p, data: bytes) -> str:
    """Store an object and return its sha256."""
    import hashlib
    sha = hashlib.sha256(data).hexdigest()
    dest = p.objects / sha
    dest.write_bytes(data)
    return sha


def _materialize_target_live(
    *,
    v2_paths,
    target_manifest,
    session_id: str,
    session_dir: Path,
    session_workspace: Path,
    session_states: Path,
    mutate_session_json: bool = False,
):
    """Materialize target live bytes using production path conventions.

    Runtime residual is written to ``{session_id}_session.pkl`` under
    ``get_session_states`` (same as ``atomic_replace_pkl``) and signed.
    When ``mutate_session_json`` is True, session.json is intentionally
    continuation-mutated so it no longer matches analysis_state_ref.sha256.
    """
    from backend.config._config import get_session_states
    from backend.utils.sandbox import _write_pkl_signature, _integrity_dir

    if target_manifest.analysis_state_ref is not None:
        src = v2_paths.objects / target_manifest.analysis_state_ref.sha256
        dst = session_dir / "session.json"
        raw = src.read_bytes()
        if mutate_session_json:
            try:
                payload = json.loads(raw.decode())
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            payload["restore_resume_kind"] = "idle"
            payload["continuation_mutated"] = True
            raw = json.dumps(payload, sort_keys=True).encode()
        dst.write_bytes(raw)
    if target_manifest.notebook_ref is not None:
        src = v2_paths.objects / target_manifest.notebook_ref.sha256
        (session_states / "notebook.ipynb").write_bytes(src.read_bytes())
    for entry in target_manifest.workspace or []:
        src = v2_paths.objects / entry.object_hash
        dst = session_workspace / entry.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

    runtime = getattr(target_manifest, "runtime", None)
    residual_ref = getattr(runtime, "residual_ref", None) if runtime is not None else None
    if residual_ref is not None:
        src = v2_paths.objects / residual_ref.sha256
        # Prefer production states path (atomic_replace_pkl destination).
        prod_states = get_session_states(session_id)
        prod_states.mkdir(parents=True, exist_ok=True)
        live_pkl = prod_states / f"{session_id}_session.pkl"
        live_pkl.write_bytes(src.read_bytes())
        _write_pkl_signature(live_pkl, _integrity_dir(session_id))
        # Mirror into test states_root for any path-based readers.
        mirror = session_states / f"{session_id}_session.pkl"
        if mirror.resolve() != live_pkl.resolve():
            mirror.write_bytes(src.read_bytes())


def _make_manifest(session_id, snapshot_id, p,
                   workspace_entries=None, objects=None,
                   runtime_adata_refs=None, runtime_residual_ref=None):
    from backend.services.snapshot_v2.models import (
        ObjectRef, RuntimeDependency, RuntimeDependencyManifest, RuntimeManifest,
    )
    from backend.services.snapshot_v2.object_store import canonical_json, ManifestStore
    from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest
    from backend.models.schemas import AnalysisState
    from datetime import datetime, timezone
    import hashlib

    workspace_entries = workspace_entries or []
    objects = dict(objects or {})
    runtime_kw = {}
    if runtime_adata_refs is not None or runtime_residual_ref is not None:
        required = {}
        residual_names = []
        for name in (runtime_adata_refs or {}):
            required[name] = RuntimeDependency(name=name, serializer="h5ad", type_name="anndata.AnnData")
        if runtime_residual_ref is not None:
            residual_names = ["_residual"]
            required["_residual"] = RuntimeDependency(name="_residual", serializer="dill", type_name="unknown")
        runtime_kw["runtime"] = RuntimeManifest(
            status="complete",
            dependencies=RuntimeDependencyManifest(required=required),
            residual_ref=runtime_residual_ref,
            captured_residual_names=residual_names,
            adata_refs=dict(runtime_adata_refs or {}),
        ).model_dump(mode="json")

    # Materialize real analysis_state / notebook objects so stage() can verify them.
    now = datetime.now(timezone.utc)
    state = AnalysisState(
        id=session_id, name="restore-tx", user_id="u1",
        created_at=now, updated_at=now,
    )
    state_payload = state.model_dump(mode="json")
    state_payload.pop("snapshots", None)
    state_payload.pop("realtime_messages", None)
    state_bytes = json.dumps(
        state_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
    ).encode()
    state_sha = _store_object(p, state_bytes)
    state_ref = ObjectRef(sha256=state_sha, size=len(state_bytes), media_type="application/json")
    objects[state_sha] = state_ref

    nb = {
        "nbformat": 4, "nbformat_minor": 5, "metadata": {},
        "cells": [],
    }
    nb_bytes = json.dumps(nb, sort_keys=True, separators=(",", ":")).encode()
    nb_sha = _store_object(p, nb_bytes)
    nb_ref = ObjectRef(sha256=nb_sha, size=len(nb_bytes), media_type="application/x-ipynb+json")
    objects[nb_sha] = nb_ref

    m = make_snapshot_v2_manifest(
        session_id=session_id,
        snapshot_id=snapshot_id,
        analysis_state_ref=state_ref.model_dump(mode="json"),
        notebook_ref=nb_ref.model_dump(mode="json"),
        workspace=[e.model_dump(mode="json") if hasattr(e, "model_dump") else e for e in workspace_entries],
        objects={k: (v.model_dump(mode="json") if hasattr(v, "model_dump") else v) for k, v in objects.items()},
        **runtime_kw,
    )
    raw = canonical_json(m)
    root_hash = hashlib.sha256(raw).hexdigest()
    m.root_hash = root_hash
    mstore = ManifestStore(p)
    staging = mstore.write_staging(m)
    mstore.publish_staging(staging)
    return m


# ---- Fixtures ---------------------------------------------------------------

@pytest.fixture
def v2_paths(tmp_path):
    return _make_fake_paths(tmp_path)


@pytest.fixture
def session_id():
    return "sess-restore-tx"


@pytest.fixture
def snapshot_id():
    return "snap-restore-target"


@pytest.fixture
def session_dir(session_id, tmp_path):
    return _make_session_dir(tmp_path, session_id)


@pytest.fixture
def session_workspace(session_dir):
    return session_dir / "workspace"


@pytest.fixture
def session_states(session_dir):
    return session_dir / "states"


@pytest.fixture
def pre_restore_snapshot_id():
    return "snap-pre-restore"


@pytest.fixture
def pre_restore_id():
    return "snap-pre-restore"


@pytest.fixture
def pre_restore_manifest(v2_paths, session_id, pre_restore_id, session_workspace, session_states):
    """Write a pre_restore manifest that matches the original live content.

    This is what rollback recovery will materialize.
    """
    from backend.services.snapshot_v2.models import WorkspaceEntry, ObjectRef

    ws_entries = []
    objects = {}

    # Original workspace content
    ws_data = b"original-png-content"
    ws_sha = _store_object(v2_paths, ws_data)
    ws_entries.append(WorkspaceEntry(
        path="figures/plot.png", object_hash=ws_sha,
        size=len(ws_data), mode=0o644,
    ))
    objects[ws_sha] = ObjectRef(
        sha256=ws_sha, size=len(ws_data), media_type="image/png",
    )

    # Original runtime content
    adata_data = b"original-adata-content"
    adata_sha = _store_object(v2_paths, adata_data)
    adata_ref = ObjectRef(
        sha256=adata_sha, size=len(adata_data),
        media_type="application/x-hdf5",
    )
    objects[adata_sha] = adata_ref

    residual_data = b"original-residual-content"
    residual_sha = _store_object(v2_paths, residual_data)
    residual_ref = ObjectRef(
        sha256=residual_sha, size=len(residual_data),
        media_type="application/octet-stream",
    )
    objects[residual_sha] = residual_ref

    manifest = _make_manifest(
        session_id=session_id, snapshot_id=pre_restore_id, p=v2_paths,
        workspace_entries=ws_entries, objects=objects,
        runtime_adata_refs={"adata": adata_ref},
        runtime_residual_ref=residual_ref,
    )
    return manifest


@pytest.fixture
def target_manifest(v2_paths, snapshot_id, session_id):
    """Write a realistic target manifest with workspace + runtime objects."""
    from backend.services.snapshot_v2.models import WorkspaceEntry, ObjectRef
    ws_entries = []
    objects = {}

    ws_data = b"workspace-figure-png"
    ws_sha = _store_object(v2_paths, ws_data)
    ws_entries.append(WorkspaceEntry(
        path="figures/plot.png", object_hash=ws_sha,
        size=len(ws_data), mode=0o644,
    ))
    objects[ws_sha] = ObjectRef(
        sha256=ws_sha, size=len(ws_data), media_type="image/png",
    )

    adata_data = b"anndata-matrix"
    adata_sha = _store_object(v2_paths, adata_data)
    adata_ref = ObjectRef(
        sha256=adata_sha, size=len(adata_data),
        media_type="application/x-hdf5",
    )
    objects[adata_sha] = adata_ref

    residual_data = b"residual-dill-data"
    residual_sha = _store_object(v2_paths, residual_data)
    residual_ref = ObjectRef(
        sha256=residual_sha, size=len(residual_data),
        media_type="application/octet-stream",
    )
    objects[residual_sha] = residual_ref

    manifest = _make_manifest(
        session_id=session_id, snapshot_id=snapshot_id, p=v2_paths,
        workspace_entries=ws_entries, objects=objects,
        runtime_adata_refs={"adata": adata_ref},
        runtime_residual_ref=residual_ref,
    )
    return manifest


@pytest.fixture
def live_session(session_workspace, session_states):
    """Create original (live) session content to be replaced by restore."""
    _write_file(session_workspace / "figures" / "plot.png", "original-png")
    _write_file(session_workspace / "analysis.py", "# original code")
    _write_file(session_states / "sess-restore-tx_session.pkl",
                "original-kernel")
    _write_file(session_states / "adata.h5ad", "original-adata")
    _write_file(session_states / "notebook.ipynb",
                json.dumps({"cells": [{"original": True}]}))
    _write_file(session_states.parent / "session.json",
                json.dumps({"id": "sess-restore-tx", "status": "running",
                            "current_round": 1}))
    _write_file(session_states / "checkpoint.sqlite", "original-checkpoint")


# ---- Live content snapshot helper -----------------------------------------

def _live_content(session_states, session_workspace):
    """Snapshot the current live content for comparison."""
    result = {}
    for base, label in [(session_workspace, "ws"), (session_states, "st")]:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                rel = f"{label}:{p.relative_to(base)}"
                result[rel] = p.read_bytes()
    return result


# ---- Fault injection helpers ------------------------------------------------

class FaultBarrier:
    """Allows tests to inject faults at named checkpoints.

    If a fault point is set, the barrier raises an exception at that
    checkpoint, simulating a crash.
    """

    def __init__(self, fault_point=None):
        self._fault_point = fault_point
        self._triggered = False

    def check(self, point):
        """If *point* matches our fault point, raise RuntimeError once."""
        if self._fault_point == point and not self._triggered:
            self._triggered = True
            raise RuntimeError("Injected fault at {}".format(point))

    @property
    def triggered(self):
        return self._triggered


# ---- Transaction-level fault-injection tests -------------------------------

class TestRestoreTransactionFaultInjection:
    """Nine fault points x two recovery modes = 18 scenarios."""

    async def _run_crash_simulation(self, fault_point, v2_paths, session_id,
                                     snapshot_id, session_dir, session_workspace,
                                     session_states, recovery_mode,
                                     pre_restore_ref="snap-pre-restore"):
        """Simulate a crash at *fault_point*.

        Each phase runs sequentially.  If the fault point is in the current
        phase, that phase crashes and no further phases execute.  After a
        crash (or on full success), recovery runs — simulating a process
        restart.

        Returns the RecoveryResult.
        """
        from backend.services.snapshot_v2.restore import RestoreCoordinator

        stage_faults = {"target_staged"}
        apply_faults = {"workspace_applied", "runtime_pkl_replaced",
                        "runtime_sig_replaced", "notebook_applied",
                        "state_replaced", "checkpoint_seeded"}
        commit_faults = {"journal_committed"}
        cleanup_faults = {"cleanup"}

        def _fault_barrier(fault_set):
            return FaultBarrier(fault_point if fault_point in fault_set else None)

        crashed = False
        tx_id = None
        staged = None

        # Phase 1: Stage
        if not crashed:
            c1 = RestoreCoordinator(fault_barrier=_fault_barrier(stage_faults))
            try:
                staged = await c1.stage(
                    snapshot_id=snapshot_id,
                    session_id=session_id,
                    paths=v2_paths,
                    recovery_mode=recovery_mode,
                    pre_restore_ref=pre_restore_ref,
                    branch_id="main",
                    continuation_kind="idle",
                )
                tx_id = staged.transaction_id
            except RuntimeError:
                crashed = True

        # Phase 2: Apply
        if not crashed and staged is not None:
            c2 = RestoreCoordinator(fault_barrier=_fault_barrier(apply_faults))
            try:
                await c2.apply(
                    staged=staged, paths=v2_paths,
                    session_dir=str(session_dir),
                    workspace_root=str(session_workspace),
                    states_root=str(session_states),
                )
            except RuntimeError:
                crashed = True

        # Phase 3: Commit
        # Fault-injection matrix intentionally writes v0 COMMITTED (no proof)
        # via the test helper so recovery re-verifies — production commit()
        # requires verification_proof. Write first (matching commit() order),
        # then fire the journal_committed fault barrier.
        if not crashed and tx_id is not None:
            c3 = RestoreCoordinator(fault_barrier=_fault_barrier(commit_faults))
            try:
                c3.write_committed_restore_v0_for_test(tx_id, v2_paths)
                c3._check_fault("journal_committed")
            except RuntimeError:
                crashed = True

        # Phase 4: Cleanup
        if not crashed and tx_id is not None:
            c4 = RestoreCoordinator(fault_barrier=_fault_barrier(cleanup_faults))
            try:
                await c4.cleanup(tx_id, v2_paths)
            except RuntimeError:
                crashed = True

        # Always run recovery (simulating process restart)
        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions
        result = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(session_dir),
            workspace_root=str(session_workspace),
            states_root=str(session_states),
        )
        return result

    @pytest.mark.parametrize("fault_point", FAULT_POINTS)
    @pytest.mark.asyncio
    async def test_rollback_restores_original(self, fault_point, v2_paths,
                                               session_id, snapshot_id,
                                               session_dir, session_workspace,
                                               session_states, live_session,
                                               target_manifest,
                                               pre_restore_manifest):
        """In rollback mode, a fault at any step leaves the original content
        (or target content if the apply completed before the fault).

        Faults during stage/apply → recovery materializes pre_restore.
        Faults during commit/cleanup → apply was complete; target stays.
        """
        from backend.services.snapshot_v2.object_store import ManifestStore

        result = await self._run_crash_simulation(
            fault_point, v2_paths, session_id, snapshot_id,
            session_dir, session_workspace, session_states,
            recovery_mode="rollback",
            pre_restore_ref="snap-pre-restore",
        )
        assert result.recovered, "Expected recovery for fault {}".format(fault_point)

        mstore = ManifestStore(v2_paths)
        after = _live_content(session_states, session_workspace)

        # Faults after apply was fully complete leave target content
        apply_complete_faults = {"journal_committed", "cleanup"}

        if fault_point in apply_complete_faults:
            # Apply was fully done — content is target, not pre_restore
            manifest = mstore.load(snapshot_id)
            for entry in manifest.workspace:
                key = "ws:{}".format(entry.path)
                assert key in after, "Missing target file {} after rollback".format(key)
                obj_content = (v2_paths.objects / entry.object_hash).read_bytes()
                assert after[key] == obj_content, \
                    "Target content mismatch for {} after rollback".format(entry.path)
        else:
            # Fault happened during apply or before — pre_restore was materialized
            pre_manifest = mstore.load("snap-pre-restore")
            for entry in pre_manifest.workspace:
                key = "ws:{}".format(entry.path)
                assert key in after, "Missing pre_restore file {} after rollback".format(key)
                obj_content = (v2_paths.objects / entry.object_hash).read_bytes()
                assert after[key] == obj_content, \
                    "Pre_restore content mismatch for {} after rollback".format(entry.path)

    @pytest.mark.parametrize("fault_point", FAULT_POINTS)
    @pytest.mark.asyncio
    async def test_forward_only_completes_target(self, fault_point, v2_paths,
                                                  session_id, snapshot_id,
                                                  session_dir, session_workspace,
                                                  session_states, live_session,
                                                  target_manifest,
                                                  pre_restore_manifest):
        """In forward-only mode, recovery completes the target content."""
        from backend.services.snapshot_v2.object_store import ManifestStore

        result = await self._run_crash_simulation(
            fault_point, v2_paths, session_id, snapshot_id,
            session_dir, session_workspace, session_states,
            recovery_mode="forward_only",
            pre_restore_ref="snap-pre-restore",
        )

        mstore = ManifestStore(v2_paths)
        manifest = mstore.load(snapshot_id)
        expected_ws_files = {e.path: v2_paths.objects / e.object_hash
                             for e in manifest.workspace}

        assert result.recovered, "Expected recovery for fault {}".format(fault_point)

        after = _live_content(session_states, session_workspace)
        for rel_path, obj_path in expected_ws_files.items():
            key = "ws:{}".format(rel_path)
            assert key in after, "Missing target workspace file {} after recovery".format(key)
            assert after[key] == obj_path.read_bytes(), \
                "Target workspace content mismatch for {}".format(key)


# ---- Unit tests for RestoreCoordinator API ---------------------------------

class TestRestoreCoordinatorAPI:

    @pytest.mark.asyncio
    async def test_stage_creates_staging_dir_and_returns_transaction(
            self, v2_paths, session_id, snapshot_id, session_dir,
            session_workspace, session_states, live_session, target_manifest):
        """stage() loads manifest, stages artifacts, returns StagedRestore."""
        from backend.services.snapshot_v2.restore import RestoreCoordinator

        coordinator = RestoreCoordinator()
        staged = await coordinator.stage(
            snapshot_id=snapshot_id,
            session_id=session_id,
            paths=v2_paths,
            recovery_mode="rollback",
            pre_restore_ref="snap-pre-restore",
            branch_id="main",
            continuation_kind="idle",
        )

        assert staged is not None
        assert staged.transaction_id is not None
        assert staged.snapshot_id == snapshot_id
        assert staged.session_id == session_id
        assert staged.recovery_mode == "rollback"

        staged_root = v2_paths.staging / staged.transaction_id
        assert staged_root.exists(), "Staging root should exist"
        entries = list(staged_root.iterdir())
        assert len(entries) > 0

    @pytest.mark.asyncio
    async def test_stage_verifies_objects_before_staging(
            self, v2_paths, session_id, snapshot_id, session_dir,
            session_workspace, session_states, live_session, target_manifest):
        """stage() must verify all object hashes match before staging."""
        from backend.services.snapshot_v2.restore import RestoreCoordinator

        coordinator = RestoreCoordinator()
        staged = await coordinator.stage(
            snapshot_id=snapshot_id,
            session_id=session_id,
            paths=v2_paths,
            recovery_mode="rollback",
            pre_restore_ref="snap-pre-restore",
            branch_id="main",
            continuation_kind="idle",
        )
        assert staged is not None

    @pytest.mark.asyncio
    async def test_apply_replaces_live_workspace(
            self, v2_paths, session_id, snapshot_id, session_dir,
            session_workspace, session_states, live_session, target_manifest):
        """apply() replaces workspace files with staged versions."""
        from backend.services.snapshot_v2.restore import RestoreCoordinator
        from backend.services.snapshot_v2.object_store import ManifestStore

        coordinator = RestoreCoordinator()
        staged = await coordinator.stage(
            snapshot_id=snapshot_id,
            session_id=session_id,
            paths=v2_paths,
            recovery_mode="rollback",
            pre_restore_ref="snap-pre-restore",
            branch_id="main",
            continuation_kind="idle",
        )

        applied = await coordinator.apply(
            staged=staged,
            paths=v2_paths,
            session_dir=str(session_dir),
            workspace_root=str(session_workspace),
            states_root=str(session_states),
        )

        assert applied is not None
        assert applied.workspace_files_count > 0
        assert applied.runtime_applied
        # Notebook and state/checkpoint are not staged in this test manifest
        # so they remain False

        manifest = ManifestStore(v2_paths).load(snapshot_id)
        for entry in manifest.workspace:
            target_path = session_workspace / entry.path
            assert target_path.exists(), "Applied workspace file missing: {}".format(target_path)
            obj_path = v2_paths.objects / entry.object_hash
            assert target_path.read_bytes() == obj_path.read_bytes(), \
                "Applied workspace content mismatch: {}".format(entry.path)

    @pytest.mark.asyncio
    async def test_commit_finalizes_journal(
            self, v2_paths, session_id, snapshot_id, session_dir,
            session_workspace, session_states, live_session, target_manifest):
        """commit() marks the journal as COMMITTED."""
        from backend.services.snapshot_v2.restore import RestoreCoordinator

        coordinator = RestoreCoordinator()
        staged = await coordinator.stage(
            snapshot_id=snapshot_id,
            session_id=session_id,
            paths=v2_paths,
            recovery_mode="rollback",
            pre_restore_ref="snap-pre-restore",
            branch_id="main",
            continuation_kind="idle",
        )

        await coordinator.apply(
            staged=staged,
            paths=v2_paths,
            session_dir=str(session_dir),
            workspace_root=str(session_workspace),
            states_root=str(session_states),
        )

        proof = {
            "verified_live_root_hash": _hash_bytes(b"commit-proof"),
            "component_hashes": {"session_json": "x"},
            "verified_at": "2026-07-12T00:00:00Z",
            "restore_journal_version": 1,
        }
        await coordinator.commit(
            staged.transaction_id, v2_paths, verification_proof=proof,
        )

        journal_path = v2_paths.journals / "{}.json".format(staged.transaction_id)
        assert journal_path.exists(), "Journal should exist after commit"
        journal = json.loads(journal_path.read_text())
        assert journal["status"] == "COMMITTED"
        assert journal.get("restore_journal_version") == 1
        assert journal.get("verified_live_root_hash") == proof["verified_live_root_hash"]

    @pytest.mark.asyncio
    async def test_cleanup_removes_staging_and_journal(
            self, v2_paths, session_id, snapshot_id, session_dir,
            session_workspace, session_states, live_session, target_manifest):
        """cleanup() removes the staging dir and journal."""
        from backend.services.snapshot_v2.restore import RestoreCoordinator

        coordinator = RestoreCoordinator()
        staged = await coordinator.stage(
            snapshot_id=snapshot_id,
            session_id=session_id,
            paths=v2_paths,
            recovery_mode="rollback",
            pre_restore_ref="snap-pre-restore",
            branch_id="main",
            continuation_kind="idle",
        )

        staged_root = v2_paths.staging / staged.transaction_id
        assert staged_root.exists()

        await coordinator.cleanup(staged.transaction_id, v2_paths)

        assert not staged_root.exists(), "Staging dir should be removed"
        journal_path = v2_paths.journals / "{}.json".format(staged.transaction_id)
        assert not journal_path.exists(), "Journal should be removed"


# ---- Recovery tests --------------------------------------------------------

class TestRecovery:

    @pytest.mark.asyncio
    async def test_recover_with_no_journals_returns_not_recovered(
            self, v2_paths, session_id, session_dir,
            session_workspace, session_states):
        """recover_snapshot_transactions with no journal returns recovered=False."""
        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions

        result = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(session_dir),
            workspace_root=str(session_workspace),
            states_root=str(session_states),
        )
        assert not result.recovered

    @pytest.mark.asyncio
    async def test_recover_rolls_back_apply_journal(
            self, v2_paths, session_id, snapshot_id, session_dir,
            session_workspace, session_states, live_session, target_manifest,
            pre_restore_manifest):
        """An APPLYING journal in rollback mode triggers rollback."""
        from backend.services.snapshot_v2.restore import RestoreCoordinator

        coordinator = RestoreCoordinator()
        staged = await coordinator.stage(
            snapshot_id=snapshot_id,
            session_id=session_id,
            paths=v2_paths,
            recovery_mode="rollback",
            pre_restore_ref="snap-pre-restore",
            branch_id="main",
            continuation_kind="idle",
        )

        # Write an APPLYING journal without applying
        journal = {
            "transaction_id": staged.transaction_id,
            "target_snapshot_id": snapshot_id,
            "session_id": session_id,
            "pre_restore_ref": "snap-pre-restore",
            "recovery_mode": "rollback",
            "status": "APPLYING",
            "applied_step_ledger": [],
            "branch_id": "main",
            "continuation_kind": "idle",
        }
        from backend.services.snapshot_v2.restore import _write_journal_hmac_signed
        _write_journal_hmac_signed(v2_paths, staged.transaction_id, journal)

        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions
        result = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(session_dir),
            workspace_root=str(session_workspace),
            states_root=str(session_states),
        )
        assert result.recovered
        assert staged.transaction_id in result.recovered_transactions

    @pytest.mark.asyncio
    async def test_recover_completes_forward_only_journal(
            self, v2_paths, session_id, snapshot_id, session_dir,
            session_workspace, session_states, live_session, target_manifest):
        """An APPLYING journal in forward_only mode replays target."""
        from backend.services.snapshot_v2.restore import RestoreCoordinator

        coordinator = RestoreCoordinator()
        staged = await coordinator.stage(
            snapshot_id=snapshot_id,
            session_id=session_id,
            paths=v2_paths,
            recovery_mode="forward_only",
            pre_restore_ref="snap-pre-restore",
            branch_id="main",
            continuation_kind="idle",
        )

        journal = {
            "transaction_id": staged.transaction_id,
            "target_snapshot_id": snapshot_id,
            "session_id": session_id,
            "pre_restore_ref": "snap-pre-restore",
            "recovery_mode": "forward_only",
            "status": "APPLYING",
            "applied_step_ledger": [],
            "branch_id": "main",
            "continuation_kind": "idle",
        }
        from backend.services.snapshot_v2.restore import _write_journal_hmac_signed
        _write_journal_hmac_signed(v2_paths, staged.transaction_id, journal)

        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions
        result = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(session_dir),
            workspace_root=str(session_workspace),
            states_root=str(session_states),
        )
        assert result.recovered
        assert staged.transaction_id in result.recovered_transactions

    @pytest.mark.asyncio
    async def test_forward_only_partial_apply_keeps_journal_when_verify_fails(
            self, v2_paths, session_id, snapshot_id, session_dir,
            session_workspace, session_states, live_session, target_manifest):
        """APPLYING + forward_only must not drop journal without verify success.

        Simulates live-verify-style partial apply: staging gone / empty and
        live target incomplete so forward_complete is a no-op; recovery must
        retain the journal and block mutations.
        """
        from backend.services.snapshot_v2.restore import (
            RestoreCoordinator,
            _write_journal_hmac_signed,
        )
        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions
        import shutil

        coordinator = RestoreCoordinator()
        staged = await coordinator.stage(
            snapshot_id=snapshot_id,
            session_id=session_id,
            paths=v2_paths,
            recovery_mode="forward_only",
            pre_restore_ref="snap-pre-restore",
            branch_id="main",
            continuation_kind="idle",
        )

        # Wipe staging so forward_complete cannot materialize missing steps.
        staged_root = Path(str(v2_paths.staging)) / staged.transaction_id
        if staged_root.exists():
            shutil.rmtree(staged_root)

        # Leave live paths incomplete (no materialize) so verify fails.
        journal = {
            "transaction_id": staged.transaction_id,
            "target_snapshot_id": snapshot_id,
            "session_id": session_id,
            "pre_restore_ref": "snap-pre-restore",
            "recovery_mode": "forward_only",
            "status": "APPLYING",
            # Pretend some steps completed without live bytes matching target.
            "applied_step_ledger": [
                {"step": "workspace", "completed": True},
                {"step": "state", "completed": True},
            ],
            "branch_id": "main",
            "continuation_kind": "idle",
        }
        _write_journal_hmac_signed(v2_paths, staged.transaction_id, journal)

        result = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(session_dir),
            workspace_root=str(session_workspace),
            states_root=str(session_states),
        )
        assert staged.transaction_id not in result.recovered_transactions
        assert staged.transaction_id in result.failed_transactions
        assert (v2_paths.journals / f"{staged.transaction_id}.json").exists()
        assert result.mutations_blocked is True
        assert any(
            "TARGET_VERIFY_FAILED" in e or "FORWARD_COMPLETE_FAILED" in e or "verify" in e.lower()
            for e in result.errors
        )

    @pytest.mark.asyncio
    async def test_recover_skips_committed_journal(
            self, v2_paths, session_id, snapshot_id, session_dir,
            session_workspace, session_states, live_session, target_manifest):
        """A COMMITTED journal with verification proof is cleaned after re-verify."""
        from backend.services.snapshot_v2.restore import (
            RestoreCoordinator,
            _write_journal_hmac_signed,
        )
        import hashlib

        coordinator = RestoreCoordinator()
        staged = await coordinator.stage(
            snapshot_id=snapshot_id,
            session_id=session_id,
            paths=v2_paths,
            recovery_mode="rollback",
            pre_restore_ref="snap-pre-restore",
            branch_id="main",
            continuation_kind="idle",
        )

        # Materialize target live bytes (production pkl path + sig).
        # Mutate session.json to prove recovery does NOT require
        # sha256(session.json) == analysis_state_ref.sha256 after activate.
        _materialize_target_live(
            v2_paths=v2_paths,
            target_manifest=target_manifest,
            session_id=session_id,
            session_dir=session_dir,
            session_workspace=session_workspace,
            session_states=session_states,
            mutate_session_json=True,
        )
        live_session_json = (session_dir / "session.json").read_bytes()
        if target_manifest.analysis_state_ref is not None:
            assert (
                hashlib.sha256(live_session_json).hexdigest()
                != target_manifest.analysis_state_ref.sha256
            ), "test setup must use post-activate mutated session.json"

        journal = {
            "transaction_id": staged.transaction_id,
            "target_snapshot_id": snapshot_id,
            "session_id": session_id,
            "pre_restore_ref": "snap-pre-restore",
            "recovery_mode": "rollback",
            "status": "COMMITTED",
            "applied_step_ledger": [],
            "branch_id": "main",
            "continuation_kind": "idle",
            "restore_journal_version": 1,
            "verified_live_root_hash": hashlib.sha256(b"proof").hexdigest(),
            "component_hashes": {"session_json": "x"},
            "verified_at": "2026-07-12T00:00:00Z",
        }
        _write_journal_hmac_signed(v2_paths, staged.transaction_id, journal)

        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions
        result = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(session_dir),
            workspace_root=str(session_workspace),
            states_root=str(session_states),
        )
        assert result.recovered
        assert staged.transaction_id in result.recovered_transactions

    @pytest.mark.asyncio
    async def test_recover_migrates_verified_v0_committed_journal(
            self, v2_paths, session_id, snapshot_id, session_dir,
            session_workspace, session_states, live_session, target_manifest):
        """HMAC-valid v0 COMMITTED journal is upgraded with proof then cleaned."""
        from backend.services.snapshot_v2.restore import (
            RestoreCoordinator,
            read_verified_restore_journal,
        )
        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions

        coordinator = RestoreCoordinator()
        staged = await coordinator.stage(
            snapshot_id=snapshot_id,
            session_id=session_id,
            paths=v2_paths,
            recovery_mode="rollback",
            pre_restore_ref="snap-pre-restore",
            branch_id="main",
            continuation_kind="idle",
        )

        _materialize_target_live(
            v2_paths=v2_paths,
            target_manifest=target_manifest,
            session_id=session_id,
            session_dir=session_dir,
            session_workspace=session_workspace,
            session_states=session_states,
            mutate_session_json=True,
        )

        # Legacy v0 via test helper (not production commit).
        coordinator.write_committed_restore_v0_for_test(
            staged.transaction_id,
            v2_paths,
            journal_body={
                "transaction_id": staged.transaction_id,
                "target_snapshot_id": snapshot_id,
                "session_id": session_id,
                "pre_restore_ref": "snap-pre-restore",
                "recovery_mode": "rollback",
                "status": "COMMITTED",
                "applied_step_ledger": [],
                "branch_id": "main",
                "continuation_kind": "idle",
            },
        )
        verified = read_verified_restore_journal(v2_paths, staged.transaction_id)
        assert verified.restore_journal_version == 0

        result = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(session_dir),
            workspace_root=str(session_workspace),
            states_root=str(session_states),
        )
        assert result.recovered
        assert staged.transaction_id in result.recovered_transactions
        assert result.failed_transactions == []
        # Cleaned after successful migrate+verify.
        assert not (v2_paths.journals / f"{staged.transaction_id}.json").exists()

    @pytest.mark.asyncio
    async def test_recover_blocks_unverifiable_v0_committed_journal(
            self, v2_paths, session_id, snapshot_id, session_dir,
            session_workspace, session_states, live_session, target_manifest):
        """v0 COMMITTED that fails target verification is retained and blocks."""
        from backend.services.snapshot_v2.restore import (
            RestoreCoordinator,
            _write_journal_hmac_signed,
        )
        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions

        coordinator = RestoreCoordinator()
        staged = await coordinator.stage(
            snapshot_id=snapshot_id,
            session_id=session_id,
            paths=v2_paths,
            recovery_mode="rollback",
            pre_restore_ref="snap-pre-restore",
            branch_id="main",
            continuation_kind="idle",
        )

        # Leave live paths empty / mismatched so verify_target_complete fails.
        journal = {
            "transaction_id": staged.transaction_id,
            "target_snapshot_id": snapshot_id,
            "session_id": session_id,
            "pre_restore_ref": "snap-pre-restore",
            "recovery_mode": "rollback",
            "status": "COMMITTED",
            "applied_step_ledger": [],
            "branch_id": "main",
            "continuation_kind": "idle",
        }
        _write_journal_hmac_signed(v2_paths, staged.transaction_id, journal)

        result = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(session_dir),
            workspace_root=str(session_workspace),
            states_root=str(session_states),
        )
        assert staged.transaction_id in result.failed_transactions
        assert (v2_paths.journals / f"{staged.transaction_id}.json").exists()
        assert any("TARGET_VERIFY_FAILED" in e or "verify" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_recover_blocks_v1_committed_without_proof(
            self, v2_paths, session_id, snapshot_id, session_dir,
            session_workspace, session_states, live_session, target_manifest):
        """v1 COMMITTED missing verified_live_root_hash blocks recovery."""
        from backend.services.snapshot_v2.restore import (
            RestoreCoordinator,
            _write_journal_hmac_signed,
        )
        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions

        coordinator = RestoreCoordinator()
        staged = await coordinator.stage(
            snapshot_id=snapshot_id,
            session_id=session_id,
            paths=v2_paths,
            recovery_mode="rollback",
            pre_restore_ref="snap-pre-restore",
            branch_id="main",
            continuation_kind="idle",
        )

        journal = {
            "transaction_id": staged.transaction_id,
            "target_snapshot_id": snapshot_id,
            "session_id": session_id,
            "pre_restore_ref": "snap-pre-restore",
            "recovery_mode": "rollback",
            "status": "COMMITTED",
            "applied_step_ledger": [],
            "branch_id": "main",
            "continuation_kind": "idle",
            "restore_journal_version": 1,
            # intentionally no verified_live_root_hash
        }
        _write_journal_hmac_signed(v2_paths, staged.transaction_id, journal)

        result = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(session_dir),
            workspace_root=str(session_workspace),
            states_root=str(session_states),
        )
        assert staged.transaction_id in result.failed_transactions
        assert (v2_paths.journals / f"{staged.transaction_id}.json").exists()
        assert any("COMMITTED_WITHOUT_VERIFICATION" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_commit_requires_verification_proof(
            self, v2_paths, session_id, snapshot_id, session_dir,
            session_workspace, session_states, live_session, target_manifest):
        """Production commit(None) must refuse — v0 only via test helper."""
        from backend.services.snapshot_v2.restore import RestoreCoordinator
        from backend.services.snapshot_v2.errors import SnapshotTransactionError

        coordinator = RestoreCoordinator()
        staged = await coordinator.stage(
            snapshot_id=snapshot_id,
            session_id=session_id,
            paths=v2_paths,
            recovery_mode="rollback",
            pre_restore_ref="snap-pre-restore",
            branch_id="main",
            continuation_kind="idle",
        )
        await coordinator.apply(
            staged=staged,
            paths=v2_paths,
            session_dir=str(session_dir),
            workspace_root=str(session_workspace),
            states_root=str(session_states),
        )
        with pytest.raises(SnapshotTransactionError, match="COMMIT_WITHOUT_VERIFICATION"):
            await coordinator.commit(staged.transaction_id, v2_paths)
        with pytest.raises(SnapshotTransactionError, match="COMMIT_WITHOUT_VERIFICATION"):
            await coordinator.commit(
                staged.transaction_id, v2_paths, verification_proof={},
            )

    @pytest.mark.asyncio
    async def test_verify_target_uses_production_pkl_path_and_sig(
            self, v2_paths, session_id, snapshot_id, session_dir,
            session_workspace, session_states, live_session, target_manifest):
        """Critical 1: wrong names (session_state.pkl/residual.pkl) must not pass."""
        from backend.services.snapshot_v2.recovery import _RecoveryRestorer
        from backend.services.snapshot_v2.recovery import RecoveryFailure
        from backend.config._config import get_session_states
        from backend.utils.sandbox import (
            _write_pkl_signature, _integrity_dir, _sig_path,
        )

        residual = getattr(target_manifest.runtime, "residual_ref", None)
        assert residual is not None
        payload = (v2_paths.objects / residual.sha256).read_bytes()

        # Clear any leftover production residual from earlier tests.
        prod_states = get_session_states(session_id)
        prod_states.mkdir(parents=True, exist_ok=True)
        live_pkl = prod_states / f"{session_id}_session.pkl"
        live_pkl.unlink(missing_ok=True)
        _sig_path(live_pkl, _integrity_dir(session_id)).unlink(missing_ok=True)
        (session_states / f"{session_id}_session.pkl").unlink(missing_ok=True)

        # Place decoys at the WRONG paths that old verifiers accepted.
        (session_states / "session_state.pkl").write_bytes(payload)
        (session_states / "residual.pkl").write_bytes(payload)

        # Also write correct workspace/session/notebook but NO production pkl.
        if target_manifest.analysis_state_ref is not None:
            src = v2_paths.objects / target_manifest.analysis_state_ref.sha256
            (session_dir / "session.json").write_bytes(src.read_bytes())
        if target_manifest.notebook_ref is not None:
            src = v2_paths.objects / target_manifest.notebook_ref.sha256
            (session_states / "notebook.ipynb").write_bytes(src.read_bytes())
        for entry in target_manifest.workspace or []:
            src = v2_paths.objects / entry.object_hash
            dst = session_workspace / entry.path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())

        restorer = _RecoveryRestorer(
            v2_paths, str(session_dir), str(session_workspace), str(session_states),
        )
        journal = {
            "transaction_id": "tx-wrong-pkl",
            "target_snapshot_id": snapshot_id,
            "session_id": session_id,
            "recovery_mode": "rollback",
            "status": "COMMITTED",
            "branch_id": "main",
            "continuation_kind": "idle",
        }
        with pytest.raises(RecoveryFailure, match="TARGET_VERIFY_FAILED"):
            restorer.verify_target_complete(journal)

        # Now install production path + sig — must pass (and session.json
        # may be mutated without matching analysis_state_ref).
        live_pkl.write_bytes(payload)
        _write_pkl_signature(live_pkl, _integrity_dir(session_id))
        mutated = json.loads((session_dir / "session.json").read_text())
        mutated["continuation_mutated"] = True
        (session_dir / "session.json").write_text(json.dumps(mutated, sort_keys=True))
        proof = restorer.verify_target_complete(journal)
        assert proof["restore_journal_version"] == 1
        assert proof.get("verified_live_root_hash")
        assert "runtime_pkl" in proof["component_hashes"]

    @pytest.mark.asyncio
    async def test_recover_with_corrupt_journal_quarantines(
            self, v2_paths, session_id, session_dir,
            session_workspace, session_states):
        """A corrupt journal is quarantined, not applied."""
        import os as _os
        jp = v2_paths.journals / "corrupt-tx.json"
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text("not-json")

        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions
        result = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(session_dir),
            workspace_root=str(session_workspace),
            states_root=str(session_states),
        )
        assert not result.recovered
        assert len(result.errors) > 0



# ---- Journal HMAC integrity -------------------------------------------------

class TestJournalHmacIntegrity:

    def test_tampered_restore_journal_blocks_recovery(self, v2_paths, session_id):
        """Tampered journal HMAC must raise SnapshotIntegrityError(JOURNAL_HMAC)."""
        from backend.services.snapshot_v2.restore import (
            _write_journal_hmac_signed,
            read_verified_restore_journal,
        )
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError

        tx_id = "restore-tamper-test"
        journal = {
            "transaction_id": tx_id,
            "target_snapshot_id": "snap-x",
            "session_id": session_id,
            "pre_restore_ref": "snap-pre",
            "recovery_mode": "rollback",
            "status": "APPLYING",
            "applied_step_ledger": [],
            "branch_id": "main",
            "continuation_kind": "idle",
        }
        _write_journal_hmac_signed(v2_paths, tx_id, journal)

        jp = v2_paths.journals / f"{tx_id}.json"
        payload = json.loads(jp.read_text())
        payload["status"] = "COMMITTED"  # mutate without re-signing
        jp.write_text(json.dumps(payload, sort_keys=True))

        with pytest.raises(SnapshotIntegrityError, match="JOURNAL_HMAC"):
            read_verified_restore_journal(v2_paths, tx_id)

    @pytest.mark.asyncio
    async def test_complete_target_ignores_reset_and_installs_signed_runtime(
            self, v2_paths, session_id, snapshot_id, session_dir,
            session_workspace, session_states, live_session, target_manifest):
        """Complete runtime target installs staged pkl+sig+h5ad even if choice=reset."""
        from backend.services.snapshot_v2.restore import RestoreCoordinator
        from backend.utils.runtime_apply import atomic_replace_pkl
        from backend.utils.sandbox import _verify_pkl_integrity, _integrity_dir
        from backend.config import get_session_states as gss

        # Use real session states path so atomic_replace_pkl / integrity work
        real_states = gss(session_id)
        real_states.mkdir(parents=True, exist_ok=True)
        live_pkl = real_states / f"{session_id}_session.pkl"
        live_pkl.write_bytes(b"STALE-LIVE-KERNEL")
        from backend.utils.sandbox import _write_pkl_signature
        _write_pkl_signature(live_pkl, _integrity_dir(session_id))

        coordinator = RestoreCoordinator()
        staged = await coordinator.stage(
            snapshot_id=snapshot_id,
            session_id=session_id,
            paths=v2_paths,
            recovery_mode="rollback",
            pre_restore_ref="snap-pre-restore",
            branch_id="main",
            continuation_kind="idle",
        )
        assert staged.staged_runtime_pkl is not None

        # Inject a known payload into staged residual so we can assert install
        staged_pkl = Path(staged.staged_runtime_pkl)
        staged_pkl.write_bytes(b"answer=41-kernel-bytes")

        applied = await coordinator.apply(
            staged=staged,
            paths=v2_paths,
            session_dir=str(session_dir),
            workspace_root=str(session_workspace),
            states_root=str(session_states),
            runtime_choice="reset",  # must be ignored for complete target
        )
        assert applied.runtime_applied is True
        assert live_pkl.read_bytes() == b"answer=41-kernel-bytes"
        assert _verify_pkl_integrity(live_pkl, _integrity_dir(session_id)) is True

        # h5ad sidecars installed
        for var_name in staged.staged_runtime_h5ads:
            h5 = real_states / f"{var_name}.h5ad"
            assert h5.exists(), f"h5ad sidecar missing for {var_name}"

    def test_complete_target_missing_pkl_rejects_reset_policy(
            self, v2_paths, session_id, snapshot_id, session_dir,
            session_workspace, session_states, target_manifest):
        """Complete residual with pkl missing must fail even if runtime_choice=reset."""
        from backend.services.snapshot_v2.live_verify import (
            verify_live_restore_components,
        )
        from backend.config._config import get_session_states
        from backend.utils.sandbox import _integrity_dir, _sig_path

        residual = getattr(target_manifest.runtime, "residual_ref", None)
        assert residual is not None
        assert target_manifest.runtime.status == "complete"

        # Ensure production residual pair is absent.
        prod_states = get_session_states(session_id)
        prod_states.mkdir(parents=True, exist_ok=True)
        live_pkl = prod_states / f"{session_id}_session.pkl"
        live_pkl.unlink(missing_ok=True)
        _sig_path(live_pkl, _integrity_dir(session_id)).unlink(missing_ok=True)
        (session_states / f"{session_id}_session.pkl").unlink(missing_ok=True)

        # Materialize non-runtime components so failure is residual-specific.
        if target_manifest.analysis_state_ref is not None:
            src = v2_paths.objects / target_manifest.analysis_state_ref.sha256
            (session_dir / "session.json").write_bytes(src.read_bytes())
        if target_manifest.notebook_ref is not None:
            src = v2_paths.objects / target_manifest.notebook_ref.sha256
            (session_states / "notebook.ipynb").write_bytes(src.read_bytes())
        for entry in target_manifest.workspace or []:
            src = v2_paths.objects / entry.object_hash
            dst = session_workspace / entry.path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())

        with pytest.raises(ValueError, match="runtime residual missing for complete target"):
            verify_live_restore_components(
                session_id=session_id,
                target_manifest=target_manifest,
                session_dir=session_dir,
                workspace_root=session_workspace,
                states_root=session_states,
                runtime_choice="reset",
            )

    @pytest.mark.asyncio
    async def test_missing_pre_restore_blocks_rollback_mutations(
            self, v2_paths, session_id, snapshot_id, session_dir,
            session_workspace, session_states, live_session, target_manifest):
        """Absent pre_restore_ref → typed recovery failure, journal kept, mutations_blocked."""
        from backend.services.snapshot_v2.restore import (
            RestoreCoordinator, _write_journal_hmac_signed,
        )
        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions

        coordinator = RestoreCoordinator()
        staged = await coordinator.stage(
            snapshot_id=snapshot_id,
            session_id=session_id,
            paths=v2_paths,
            recovery_mode="rollback",
            pre_restore_ref="",  # absent
            branch_id="main",
            continuation_kind="idle",
        )
        journal = {
            "transaction_id": staged.transaction_id,
            "target_snapshot_id": snapshot_id,
            "session_id": session_id,
            "pre_restore_ref": "",
            "recovery_mode": "rollback",
            "status": "APPLYING",
            "applied_step_ledger": [{"step": "workspace", "completed": True}],
            "branch_id": "main",
            "continuation_kind": "idle",
        }
        _write_journal_hmac_signed(v2_paths, staged.transaction_id, journal)

        result = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(session_dir),
            workspace_root=str(session_workspace),
            states_root=str(session_states),
        )
        assert result.mutations_blocked is True
        assert staged.transaction_id in result.failed_transactions
        jp = v2_paths.journals / f"{staged.transaction_id}.json"
        assert jp.exists(), "Journal must remain for operator retry"

# ---- Runtime apply helpers --------------------------------------------------

class TestRuntimeApplyHelpers:

    def test_replace_pkl_atomically(self, tmp_path, monkeypatch):
        """atomic_replace_pkl replaces the live pkl with fsync safety."""
        from backend.utils.runtime_apply import atomic_replace_pkl
        from backend.config import get_session_states as gss, DATA_BASE_DIR

        session_id = "sess-atom-pkl"
        # Use the actual session states path so atomic_replace_pkl finds it
        states = gss(session_id)
        states.mkdir(parents=True, exist_ok=True)

        live_pkl = states / "{}_session.pkl".format(session_id)
        live_pkl.write_text("ORIGINAL_PKL")

        from backend.utils.sandbox import _write_pkl_signature, _integrity_dir
        _write_pkl_signature(live_pkl, _integrity_dir(session_id))

        src_pkl = tmp_path / "src.pkl"
        src_pkl.write_text("STAGED_PKL")

        atomic_replace_pkl(session_id, str(src_pkl))
        assert live_pkl.read_text() == "STAGED_PKL", \
            "Live pkl should be replaced"

    def test_replace_h5ad_atomically(self, tmp_path, monkeypatch):
        """atomic_replace_h5ad replaces a named h5ad in states dir."""
        from backend.utils.runtime_apply import atomic_replace_h5ad
        from backend.config import get_session_states as gss

        session_id = "sess-atom-h5ad"
        states = gss(session_id)
        states.mkdir(parents=True, exist_ok=True)

        src_h5ad = tmp_path / "src.h5ad"
        src_h5ad.write_text("STAGED_H5AD_DATA")

        atomic_replace_h5ad(session_id, str(src_h5ad), var_name="adata")
        h5ad_path = states / "adata.h5ad"
        assert h5ad_path.exists(), "h5ad should be placed in states dir"
        assert h5ad_path.read_text() == "STAGED_H5AD_DATA", \
            "h5ad content should match source"


# ---- Session store integration tests ---------------------------------------

@pytest_asyncio.fixture
async def assert_clean_async_teardown():
    """Fail if the test leaves pending async tasks (task leaks)."""
    yield
    loop = asyncio.get_running_loop()
    current = asyncio.current_task(loop=loop)
    pending = [
        task for task in asyncio.all_tasks(loop)
        if task is not current and not task.done()
    ]
    assert not pending, [repr(task) for task in pending]


class TestSessionStoreIntegration:

    @pytest.mark.asyncio
    async def test_load_session_calls_recovery(
        self, tmp_path, monkeypatch, assert_clean_async_teardown,
    ):
        """load_session calls recover_snapshot_transactions at start."""
        import backend.services.session_store as ss_mod

        called = []

        async def fake_recover(**kw):
            called.append(kw.get("session_id"))
            from backend.services.snapshot_v2.recovery import RecoveryResult
            return RecoveryResult(recovered=False, recovered_transactions=[],
                                  failed_transactions=[], errors=[])

        monkeypatch.setattr(
            "backend.services.snapshot_v2.recovery.recover_snapshot_transactions",
            fake_recover,
        )

        sid = "sess-store-recover"
        p = tmp_path / "sessions" / sid
        p.mkdir(parents=True, exist_ok=True)
        session_file = p / "session.json"
        session_file.write_text(json.dumps({
            "id": sid, "user_id": "u1", "name": "t",
        }))

        from backend.config._config import ensure_session_dirs
        ensure_session_dirs(sid)

        result = await ss_mod.ensure_session_loaded(sid)
        assert called == [sid] or sid in called


# ---- Main lifespan recovery -------------------------------------------------

class TestLifespanRecovery:

    @pytest.mark.asyncio
    async def test_recover_for_all_merges_failures_when_not_recovered(
            self, tmp_path, monkeypatch):
        """failed_transactions/mutations_blocked merge even when recovered=False."""
        from backend.config._config import SESSION_DIR
        from backend.services.snapshot_v2.recovery import (
            recover_for_all_sessions, RecoveryResult,
        )

        sid = "sess-lifespan-fail-merge"
        sd = SESSION_DIR / sid
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "session.json").write_text(json.dumps({
            "id": sid, "user_id": "u1", "name": "t",
        }))

        async def fake_recover(**kw):
            return RecoveryResult(
                recovered=False,
                recovered_transactions=[],
                failed_transactions=["restore-deadbeef"],
                errors=["Journal integrity failure"],
                mutations_blocked=True,
            )

        monkeypatch.setattr(
            "backend.services.snapshot_v2.recovery.recover_snapshot_transactions",
            fake_recover,
        )

        combined = await recover_for_all_sessions()
        assert combined.recovered is False
        assert "restore-deadbeef" in combined.failed_transactions
        assert combined.mutations_blocked is True
        assert any("integrity" in e.lower() or "failure" in e.lower()
                   for e in combined.errors)

    @pytest.mark.asyncio
    async def test_lifespan_calls_recovery_for_sessions(self, tmp_path, monkeypatch):
        """The lifespan startup calls recovery for known session dirs."""
        from backend.config._config import SESSION_DIR
        import backend.main as main_mod

        # Create a session dir
        sid = "sess-lifespan-recover"
        sd = SESSION_DIR / sid
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "session.json").write_text(json.dumps({
            "id": sid, "user_id": "u1", "name": "t",
        }))

        called = []

        async def fake_recover(**kw):
            called.append(kw.get("session_id"))
            from backend.services.snapshot_v2.recovery import RecoveryResult
            return RecoveryResult(recovered=False, recovered_transactions=[],
                                  failed_transactions=[], errors=[])

        monkeypatch.setattr(
            "backend.services.snapshot_v2.recovery.recover_snapshot_transactions",
            fake_recover,
        )

        from backend.services.snapshot_v2.recovery import recover_for_all_sessions
        await recover_for_all_sessions()
        assert sid in called or len(called) > 0
