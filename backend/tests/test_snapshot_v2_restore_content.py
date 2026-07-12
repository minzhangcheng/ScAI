"""Tests for Snapshot V2 restore content staging and workspace scope.

Batch 2 Task 2: stage must materialize every declared manifest artifact
(analysis_state, notebook, checkpoint, runtime) and workspace apply/remove
must stay inside WorkspaceScope.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_paths(tmp_path):
    class FakePaths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        committed = tmp_path / "committed"
        integrity = tmp_path / "integrity"

        def ensure_dirs(self):
            for d in [
                self.root,
                self.objects,
                self.staging,
                self.journals,
                self.integrity,
                self.committed,
            ]:
                d.mkdir(parents=True, exist_ok=True)

    p = FakePaths()
    p.ensure_dirs()
    return p


def _store_object(paths, data: bytes) -> str:
    sha = hashlib.sha256(data).hexdigest()
    dest = paths.objects / sha
    dest.write_bytes(data)
    return sha


def _valid_analysis_state_bytes(session_id: str = "sess-stage") -> bytes:
    from backend.models.schemas import AnalysisState

    now = datetime.now(timezone.utc)
    state = AnalysisState(
        id=session_id,
        name="stage-test",
        user_id="u1",
        created_at=now,
        updated_at=now,
    )
    payload = state.model_dump(mode="json")
    payload.pop("snapshots", None)
    payload.pop("realtime_messages", None)
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
    ).encode("utf-8")


def _valid_notebook_bytes() -> bytes:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [{"cell_type": "code", "source": "print(1)", "metadata": {}, "outputs": []}],
    }
    return json.dumps(nb, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _make_complete_manifest(paths, session_id: str, snapshot_id: str):
    """Publish a complete Batch-1-style manifest with real objects on disk."""
    from backend.services.snapshot_v2.models import (
        ObjectRef,
        RuntimeDependency,
        RuntimeDependencyManifest,
        RuntimeManifest,
        WorkspaceEntry,
    )
    from backend.services.snapshot_v2.object_store import ManifestStore, canonical_json
    from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest

    objects: dict[str, ObjectRef] = {}

    state_data = _valid_analysis_state_bytes(session_id)
    state_sha = _store_object(paths, state_data)
    state_ref = ObjectRef(sha256=state_sha, size=len(state_data), media_type="application/json")
    objects[state_sha] = state_ref

    nb_data = _valid_notebook_bytes()
    nb_sha = _store_object(paths, nb_data)
    nb_ref = ObjectRef(
        sha256=nb_sha, size=len(nb_data), media_type="application/x-ipynb+json",
    )
    objects[nb_sha] = nb_ref

    ckpt_data = b"sqlite-checkpoint-bytes"
    ckpt_sha = _store_object(paths, ckpt_data)
    ckpt_ref = ObjectRef(
        sha256=ckpt_sha, size=len(ckpt_data), media_type="application/x-sqlite3",
    )
    objects[ckpt_sha] = ckpt_ref

    residual_data = b"residual-dill-payload"
    residual_sha = _store_object(paths, residual_data)
    residual_ref = ObjectRef(
        sha256=residual_sha, size=len(residual_data), media_type="application/octet-stream",
    )
    objects[residual_sha] = residual_ref

    adata_data = b"anndata-h5ad-payload"
    adata_sha = _store_object(paths, adata_data)
    adata_ref = ObjectRef(
        sha256=adata_sha, size=len(adata_data), media_type="application/x-hdf5",
    )
    objects[adata_sha] = adata_ref

    ws_data = b"workspace-figure-png"
    ws_sha = _store_object(paths, ws_data)
    ws_entry = WorkspaceEntry(
        path="figures/plot.png",
        object_hash=ws_sha,
        size=len(ws_data),
        mode=0o644,
    )
    objects[ws_sha] = ObjectRef(
        sha256=ws_sha, size=len(ws_data), media_type="image/png",
    )

    runtime = RuntimeManifest(
        status="complete",
        dependencies=RuntimeDependencyManifest(
            required={
                "adata": RuntimeDependency(
                    name="adata", serializer="h5ad", type_name="anndata.AnnData",
                ),
                "results": RuntimeDependency(
                    name="results", serializer="dill", type_name="builtins.dict",
                ),
            },
        ),
        residual_ref=residual_ref,
        captured_residual_names=["results"],
        adata_refs={"adata": adata_ref},
    )

    m = make_snapshot_v2_manifest(
        session_id=session_id,
        snapshot_id=snapshot_id,
        analysis_state_ref=state_ref.model_dump(mode="json"),
        notebook_ref=nb_ref.model_dump(mode="json"),
        checkpoint_ref=ckpt_ref.model_dump(mode="json"),
        runtime=runtime.model_dump(mode="json"),
        workspace=[ws_entry.model_dump(mode="json")],
        objects={k: v.model_dump(mode="json") for k, v in objects.items()},
    )
    raw = canonical_json(m)
    m.root_hash = hashlib.sha256(raw).hexdigest()
    mstore = ManifestStore(paths)
    staging = mstore.write_staging(m)
    mstore.publish_staging(staging)
    return m, {
        "state": state_data,
        "notebook": nb_data,
        "checkpoint": ckpt_data,
        "residual": residual_data,
        "adata": adata_data,
        "workspace": ws_data,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def v2_paths(tmp_path):
    return _make_fake_paths(tmp_path)


@pytest.fixture
def session_id():
    return "sess-stage-content"


@pytest.fixture
def snapshot_id():
    return "snap-stage-target"


@pytest.fixture
def complete_snapshot(v2_paths, session_id, snapshot_id):
    manifest, blobs = _make_complete_manifest(v2_paths, session_id, snapshot_id)
    return {"manifest": manifest, "blobs": blobs, "snapshot_id": snapshot_id}


# ---------------------------------------------------------------------------
# Stage materialization
# ---------------------------------------------------------------------------


class TestStageMaterializesContent:
    @pytest.mark.asyncio
    async def test_stage_materializes_all_manifest_content(
        self, v2_paths, session_id, complete_snapshot,
    ):
        from backend.services.snapshot_v2.restore import RestoreCoordinator

        coordinator = RestoreCoordinator()
        blobs = complete_snapshot["blobs"]
        staged = await coordinator.stage(
            complete_snapshot["snapshot_id"], session_id, v2_paths,
        )

        assert staged.staged_analysis_state is not None
        assert Path(staged.staged_analysis_state).read_bytes() == blobs["state"]

        assert staged.staged_notebook is not None
        assert Path(staged.staged_notebook).read_bytes() == blobs["notebook"]

        assert staged.staged_checkpoint is not None
        assert Path(staged.staged_checkpoint).exists()
        assert Path(staged.staged_checkpoint).read_bytes() == blobs["checkpoint"]

        assert staged.staged_runtime_pkl is not None
        assert Path(staged.staged_runtime_pkl).exists()
        assert Path(staged.staged_runtime_pkl).read_bytes() == blobs["residual"]

        assert "adata" in staged.staged_runtime_h5ads
        assert Path(staged.staged_runtime_h5ads["adata"]).read_bytes() == blobs["adata"]

        ws_file = Path(staged.staged_workspace) / "figures" / "plot.png"
        assert ws_file.read_bytes() == blobs["workspace"]

    @pytest.mark.asyncio
    async def test_stage_rejects_missing_declared_object(
        self, v2_paths, session_id, complete_snapshot,
    ):
        """Declared analysis_state_ref with missing object must fail staging."""
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError
        from backend.services.snapshot_v2.restore import RestoreCoordinator

        manifest = complete_snapshot["manifest"]
        # Delete the analysis state object bytes
        obj = v2_paths.objects / manifest.analysis_state_ref.sha256
        obj.unlink()

        coordinator = RestoreCoordinator()
        with pytest.raises(SnapshotIntegrityError):
            await coordinator.stage(
                complete_snapshot["snapshot_id"], session_id, v2_paths,
            )

    @pytest.mark.asyncio
    async def test_stage_rejects_invalid_analysis_state(
        self, v2_paths, session_id, snapshot_id,
    ):
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError
        from backend.services.snapshot_v2.models import ObjectRef, WorkspaceEntry
        from backend.services.snapshot_v2.object_store import ManifestStore, canonical_json
        from backend.services.snapshot_v2.restore import RestoreCoordinator
        from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest

        bad_state = b'{"not": "an analysis state"}'
        sha = _store_object(v2_paths, bad_state)
        state_ref = ObjectRef(sha256=sha, size=len(bad_state), media_type="application/json")

        nb = _valid_notebook_bytes()
        nb_sha = _store_object(v2_paths, nb)
        nb_ref = ObjectRef(
            sha256=nb_sha, size=len(nb), media_type="application/x-ipynb+json",
        )

        objects = {
            sha: state_ref.model_dump(mode="json"),
            nb_sha: nb_ref.model_dump(mode="json"),
        }
        m = make_snapshot_v2_manifest(
            session_id=session_id,
            snapshot_id=snapshot_id,
            analysis_state_ref=state_ref.model_dump(mode="json"),
            notebook_ref=nb_ref.model_dump(mode="json"),
            workspace=[],
            objects=objects,
        )
        raw = canonical_json(m)
        m.root_hash = hashlib.sha256(raw).hexdigest()
        mstore = ManifestStore(v2_paths)
        mstore.publish_staging(mstore.write_staging(m))

        coordinator = RestoreCoordinator()
        with pytest.raises((SnapshotIntegrityError, Exception)):
            await coordinator.stage(snapshot_id, session_id, v2_paths)

    @pytest.mark.asyncio
    async def test_stage_rejects_invalid_notebook_json(
        self, v2_paths, session_id, snapshot_id,
    ):
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError
        from backend.services.snapshot_v2.models import ObjectRef
        from backend.services.snapshot_v2.object_store import ManifestStore, canonical_json
        from backend.services.snapshot_v2.restore import RestoreCoordinator
        from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest

        state = _valid_analysis_state_bytes(session_id)
        state_sha = _store_object(v2_paths, state)
        state_ref = ObjectRef(
            sha256=state_sha, size=len(state), media_type="application/json",
        )

        bad_nb = b"not-json-notebook"
        nb_sha = _store_object(v2_paths, bad_nb)
        nb_ref = ObjectRef(
            sha256=nb_sha, size=len(bad_nb), media_type="application/x-ipynb+json",
        )

        objects = {
            state_sha: state_ref.model_dump(mode="json"),
            nb_sha: nb_ref.model_dump(mode="json"),
        }
        m = make_snapshot_v2_manifest(
            session_id=session_id,
            snapshot_id=snapshot_id,
            analysis_state_ref=state_ref.model_dump(mode="json"),
            notebook_ref=nb_ref.model_dump(mode="json"),
            workspace=[],
            objects=objects,
        )
        raw = canonical_json(m)
        m.root_hash = hashlib.sha256(raw).hexdigest()
        mstore = ManifestStore(v2_paths)
        mstore.publish_staging(mstore.write_staging(m))

        coordinator = RestoreCoordinator()
        with pytest.raises((SnapshotIntegrityError, Exception)):
            await coordinator.stage(snapshot_id, session_id, v2_paths)


# ---------------------------------------------------------------------------
# WorkspaceScope shared helper
# ---------------------------------------------------------------------------


class TestWorkspaceScope:
    def test_is_in_scope_root_and_recursive_dirs(self):
        from backend.services.snapshot_v2.workspace import WorkspaceScope

        scope = WorkspaceScope()
        assert scope.is_in_scope("summary.txt")
        assert scope.is_in_scope("figures/plot.png")
        assert scope.is_in_scope("tables/a/b.csv")
        assert scope.is_in_scope("reproducibility/x.tar.gz")
        assert not scope.is_in_scope("scratch/private.txt")
        assert not scope.is_in_scope("uploads/data.h5ad")
        assert not scope.is_in_scope("logs/run.log")

    def test_list_live_scoped_paths(self, tmp_path):
        from backend.services.snapshot_v2.workspace import WorkspaceScope

        ws = tmp_path / "workspace"
        (ws / "figures").mkdir(parents=True)
        (ws / "scratch").mkdir()
        (ws / "figures" / "a.png").write_bytes(b"a")
        (ws / "root.txt").write_bytes(b"r")
        (ws / "scratch" / "x.txt").write_bytes(b"x")

        scope = WorkspaceScope()
        live = scope.list_live_scoped_paths(ws)
        assert "figures/a.png" in live
        assert "root.txt" in live
        assert "scratch/x.txt" not in live

    def test_remove_added_files_only_in_scope(self, tmp_path):
        from backend.services.snapshot_v2.workspace import WorkspaceScope

        ws = tmp_path / "workspace"
        (ws / "figures").mkdir(parents=True)
        (ws / "scratch").mkdir()
        (ws / "uploads").mkdir()

        # Live files: one declared in staged set, one added later (in scope),
        # one out of scope.
        (ws / "figures" / "keep.png").write_bytes(b"keep")
        (ws / "figures" / "added.png").write_bytes(b"added")
        (ws / "root_added.txt").write_bytes(b"added-root")
        (ws / "scratch" / "private.txt").write_bytes(b"private")
        (ws / "uploads" / "data.h5ad").write_bytes(b"upload")

        staged_paths = {"figures/keep.png"}
        scope = WorkspaceScope()
        scope.remove_added_files(ws, staged_paths)

        assert (ws / "figures" / "keep.png").exists()
        assert not (ws / "figures" / "added.png").exists()
        assert not (ws / "root_added.txt").exists()
        # Out of scope must be preserved
        assert (ws / "scratch" / "private.txt").read_bytes() == b"private"
        assert (ws / "uploads" / "data.h5ad").read_bytes() == b"upload"


# ---------------------------------------------------------------------------
# Undeclared object rejection
# ---------------------------------------------------------------------------


class TestUndeclaredObjectRejection:
    @pytest.mark.asyncio
    async def test_stage_rejects_undeclared_workspace_object(
        self, v2_paths, session_id, snapshot_id,
    ):
        """Workspace entry object_hash not in manifest.objects → UNDECLARED_OBJECT."""
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError
        from backend.services.snapshot_v2.models import ObjectRef, WorkspaceEntry
        from backend.services.snapshot_v2.object_store import ManifestStore, canonical_json
        from backend.services.snapshot_v2.restore import RestoreCoordinator
        from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest

        state = _valid_analysis_state_bytes(session_id)
        state_sha = _store_object(v2_paths, state)
        state_ref = ObjectRef(
            sha256=state_sha, size=len(state), media_type="application/json",
        )
        nb = _valid_notebook_bytes()
        nb_sha = _store_object(v2_paths, nb)
        nb_ref = ObjectRef(
            sha256=nb_sha, size=len(nb), media_type="application/x-ipynb+json",
        )

        ws_data = b"orphan-ws"
        ws_sha = _store_object(v2_paths, ws_data)
        entry = WorkspaceEntry(
            path="figures/orphan.png",
            object_hash=ws_sha,
            size=len(ws_data),
            mode=0o644,
        )

        objects = {
            state_sha: state_ref.model_dump(mode="json"),
            nb_sha: nb_ref.model_dump(mode="json"),
        }
        m = make_snapshot_v2_manifest(
            session_id=session_id,
            snapshot_id=snapshot_id,
            analysis_state_ref=state_ref.model_dump(mode="json"),
            notebook_ref=nb_ref.model_dump(mode="json"),
            workspace=[entry.model_dump(mode="json")],
            objects=objects,
        )
        raw = canonical_json(m)
        m.root_hash = hashlib.sha256(raw).hexdigest()
        mstore = ManifestStore(v2_paths)
        mstore.publish_staging(mstore.write_staging(m))

        coordinator = RestoreCoordinator()
        with pytest.raises(SnapshotIntegrityError) as ei:
            await coordinator.stage(snapshot_id, session_id, v2_paths)
        assert ei.value.code == "UNDECLARED_OBJECT"

    @pytest.mark.asyncio
    async def test_stage_rejects_undeclared_analysis_state_ref(
        self, v2_paths, session_id, snapshot_id,
    ):
        """analysis_state_ref sha not in manifest.objects → UNDECLARED_OBJECT."""
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError
        from backend.services.snapshot_v2.models import ObjectRef
        from backend.services.snapshot_v2.object_store import ManifestStore, canonical_json
        from backend.services.snapshot_v2.restore import RestoreCoordinator
        from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest

        state = _valid_analysis_state_bytes(session_id)
        state_sha = _store_object(v2_paths, state)
        state_ref = ObjectRef(
            sha256=state_sha, size=len(state), media_type="application/json",
        )
        nb = _valid_notebook_bytes()
        nb_sha = _store_object(v2_paths, nb)
        nb_ref = ObjectRef(
            sha256=nb_sha, size=len(nb), media_type="application/x-ipynb+json",
        )

        objects = {
            nb_sha: nb_ref.model_dump(mode="json"),
        }
        m = make_snapshot_v2_manifest(
            session_id=session_id,
            snapshot_id=snapshot_id,
            analysis_state_ref=state_ref.model_dump(mode="json"),
            notebook_ref=nb_ref.model_dump(mode="json"),
            workspace=[],
            objects=objects,
        )
        raw = canonical_json(m)
        m.root_hash = hashlib.sha256(raw).hexdigest()
        mstore = ManifestStore(v2_paths)
        mstore.publish_staging(mstore.write_staging(m))

        coordinator = RestoreCoordinator()
        with pytest.raises(SnapshotIntegrityError) as ei:
            await coordinator.stage(snapshot_id, session_id, v2_paths)
        assert ei.value.code == "UNDECLARED_OBJECT"

    @pytest.mark.asyncio
    async def test_stage_rejects_undeclared_continuation_payload_ref(
        self, v2_paths, session_id, snapshot_id,
    ):
        """semantic_gate payload_ref not in objects → UNDECLARED_OBJECT."""
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError
        from backend.services.snapshot_v2.models import ObjectRef
        from backend.services.snapshot_v2.object_store import ManifestStore, canonical_json
        from backend.services.snapshot_v2.restore import RestoreCoordinator
        from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest

        state = _valid_analysis_state_bytes(session_id)
        state_sha = _store_object(v2_paths, state)
        state_ref = ObjectRef(
            sha256=state_sha, size=len(state), media_type="application/json",
        )
        nb = _valid_notebook_bytes()
        nb_sha = _store_object(v2_paths, nb)
        nb_ref = ObjectRef(
            sha256=nb_sha, size=len(nb), media_type="application/x-ipynb+json",
        )

        payload = b'{"gate":"annotation"}'
        payload_sha = _store_object(v2_paths, payload)
        payload_ref = ObjectRef(
            sha256=payload_sha, size=len(payload), media_type="application/json",
        )

        objects = {
            state_sha: state_ref.model_dump(mode="json"),
            nb_sha: nb_ref.model_dump(mode="json"),
        }
        m = make_snapshot_v2_manifest(
            session_id=session_id,
            snapshot_id=snapshot_id,
            analysis_state_ref=state_ref.model_dump(mode="json"),
            notebook_ref=nb_ref.model_dump(mode="json"),
            workspace=[],
            objects=objects,
            continuation={
                "kind": "semantic_gate",
                "graph_version": "2.0",
                "gate": "annotation_confirmation",
                "wait_reason": None,
                "payload_ref": payload_ref.model_dump(mode="json"),
            },
        )
        raw = canonical_json(m)
        m.root_hash = hashlib.sha256(raw).hexdigest()
        mstore = ManifestStore(v2_paths)
        mstore.publish_staging(mstore.write_staging(m))

        coordinator = RestoreCoordinator()
        with pytest.raises(SnapshotIntegrityError) as ei:
            await coordinator.stage(snapshot_id, session_id, v2_paths)
        assert ei.value.code == "UNDECLARED_OBJECT"

    @pytest.mark.asyncio
    async def test_stage_rejects_undeclared_workflow_state_ref(
        self, v2_paths, session_id, snapshot_id,
    ):
        """graph_node workflow_state_ref not in objects → UNDECLARED_OBJECT."""
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError
        from backend.services.snapshot_v2.models import ObjectRef
        from backend.services.snapshot_v2.object_store import ManifestStore, canonical_json
        from backend.services.snapshot_v2.restore import RestoreCoordinator
        from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest

        state = _valid_analysis_state_bytes(session_id)
        state_sha = _store_object(v2_paths, state)
        state_ref = ObjectRef(
            sha256=state_sha, size=len(state), media_type="application/json",
        )
        nb = _valid_notebook_bytes()
        nb_sha = _store_object(v2_paths, nb)
        nb_ref = ObjectRef(
            sha256=nb_sha, size=len(nb), media_type="application/x-ipynb+json",
        )

        ws_state = b'{"current_round": 1}'
        ws_sha = _store_object(v2_paths, ws_state)
        ws_ref = ObjectRef(
            sha256=ws_sha, size=len(ws_state), media_type="application/json",
        )

        objects = {
            state_sha: state_ref.model_dump(mode="json"),
            nb_sha: nb_ref.model_dump(mode="json"),
        }
        m = make_snapshot_v2_manifest(
            session_id=session_id,
            snapshot_id=snapshot_id,
            analysis_state_ref=state_ref.model_dump(mode="json"),
            notebook_ref=nb_ref.model_dump(mode="json"),
            workspace=[],
            objects=objects,
            continuation={
                "kind": "graph_node",
                "graph_version": "2.0",
                "next_node": "bioinformatics_analysis",
                "workflow_state_ref": ws_ref.model_dump(mode="json"),
                "planning_stage": "full",
            },
        )
        raw = canonical_json(m)
        m.root_hash = hashlib.sha256(raw).hexdigest()
        mstore = ManifestStore(v2_paths)
        mstore.publish_staging(mstore.write_staging(m))

        coordinator = RestoreCoordinator()
        with pytest.raises(SnapshotIntegrityError) as ei:
            await coordinator.stage(snapshot_id, session_id, v2_paths)
        assert ei.value.code == "UNDECLARED_OBJECT"


# ---------------------------------------------------------------------------
# Recovery WorkspaceScope wiring
# ---------------------------------------------------------------------------


class TestRecoveryWorkspaceScope:
    @pytest.mark.asyncio
    async def test_rollback_uses_scope_preserves_out_of_scope(
        self, v2_paths, session_id, tmp_path,
    ):
        """Rollback restores pre_restore scoped files and preserves uploads/scratch."""
        from backend.services.snapshot_v2.models import ObjectRef, WorkspaceEntry
        from backend.services.snapshot_v2.object_store import ManifestStore, canonical_json
        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions
        from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest

        pre_ws = b"pre-figure"
        pre_sha = _store_object(v2_paths, pre_ws)
        pre_entry = WorkspaceEntry(
            path="figures/pre.png",
            object_hash=pre_sha,
            size=len(pre_ws),
            mode=0o644,
        )

        state = _valid_analysis_state_bytes(session_id)
        state_sha = _store_object(v2_paths, state)
        state_ref = ObjectRef(
            sha256=state_sha, size=len(state), media_type="application/json",
        )
        nb = _valid_notebook_bytes()
        nb_sha = _store_object(v2_paths, nb)
        nb_ref = ObjectRef(
            sha256=nb_sha, size=len(nb), media_type="application/x-ipynb+json",
        )

        objects = {
            state_sha: state_ref.model_dump(mode="json"),
            nb_sha: nb_ref.model_dump(mode="json"),
            pre_sha: ObjectRef(
                sha256=pre_sha, size=len(pre_ws), media_type="image/png",
            ).model_dump(mode="json"),
        }
        pre_id = "snap-pre-restore"
        m = make_snapshot_v2_manifest(
            session_id=session_id,
            snapshot_id=pre_id,
            analysis_state_ref=state_ref.model_dump(mode="json"),
            notebook_ref=nb_ref.model_dump(mode="json"),
            workspace=[pre_entry.model_dump(mode="json")],
            objects=objects,
        )
        raw = canonical_json(m)
        m.root_hash = hashlib.sha256(raw).hexdigest()
        mstore = ManifestStore(v2_paths)
        mstore.publish_staging(mstore.write_staging(m))

        session_dir = tmp_path / "session"
        workspace = session_dir / "workspace"
        states = session_dir / "states"
        workspace.mkdir(parents=True)
        states.mkdir(parents=True)
        (workspace / "figures").mkdir()
        (workspace / "scratch").mkdir()
        (workspace / "uploads").mkdir()
        (workspace / "figures" / "later.png").write_bytes(b"later")
        (workspace / "scratch" / "private.txt").write_bytes(b"private")
        (workspace / "uploads" / "data.h5ad").write_bytes(b"upload")

        tx_id = "restore-scope-test01"
        journal = {
            "transaction_id": tx_id,
            "target_snapshot_id": "snap-target-x",
            "session_id": session_id,
            "pre_restore_ref": pre_id,
            "recovery_mode": "rollback",
            "status": "APPLYING",
            "applied_step_ledger": [{"step": "workspace", "completed": True}],
            "continuation_kind": "idle",
            "branch_id": "main",
        }
        from backend.services.snapshot_v2.restore import _write_journal_hmac_signed

        _write_journal_hmac_signed(v2_paths, tx_id, journal)

        result = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(session_dir),
            workspace_root=str(workspace),
            states_root=str(states),
        )
        assert result.recovered
        assert (workspace / "figures" / "pre.png").read_bytes() == pre_ws
        assert not (workspace / "figures" / "later.png").exists()
        assert (workspace / "scratch" / "private.txt").read_bytes() == b"private"
        assert (workspace / "uploads" / "data.h5ad").read_bytes() == b"upload"

    @pytest.mark.asyncio
    async def test_forward_complete_uses_scope(
        self, v2_paths, session_id, tmp_path,
    ):
        """forward_complete copies only scoped staged files and removes extras.

        forward_only recovery now re-verifies the target via
        ``verify_target_complete``, so the committed target manifest and the
        fully staged target (workspace + notebook + state) must be present. The
        assertions still prove scope: an out-of-scope staged file is not applied
        and out-of-scope live files are preserved.
        """
        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions
        from backend.services.snapshot_v2.restore import _write_journal_hmac_signed
        from backend.services.snapshot_v2.models import ObjectRef, WorkspaceEntry
        from backend.services.snapshot_v2.object_store import ManifestStore, canonical_json
        from backend.tests.snapshot_v2_fixtures import (
            make_snapshot_v2_manifest,
            complete_runtime,
        )

        session_dir = tmp_path / "session"
        workspace = session_dir / "workspace"
        states = session_dir / "states"
        workspace.mkdir(parents=True)
        states.mkdir(parents=True)
        (workspace / "figures").mkdir()
        (workspace / "scratch").mkdir()
        (workspace / "figures" / "old.png").write_bytes(b"old")
        (workspace / "scratch" / "keep.txt").write_bytes(b"keep")

        tx_id = "restore-fwd-scope01"
        target_id = "snap-fwd"

        # Stage the full target apply: scoped workspace + an out-of-scope file
        # (which must NOT be applied) + notebook + analysis state.
        staged = v2_paths.staging / tx_id
        (staged / "workspace" / "figures").mkdir(parents=True)
        new_png = b"new"
        (staged / "workspace" / "figures" / "new.png").write_bytes(new_png)
        (staged / "workspace" / "scratch").mkdir()
        (staged / "workspace" / "scratch" / "evil.txt").write_bytes(b"evil")
        nb_data = _valid_notebook_bytes()
        (staged / "notebook.ipynb").write_bytes(nb_data)
        state_data = _valid_analysis_state_bytes(session_id)
        (staged / "analysis_state.json").write_bytes(state_data)
        round_number = int(json.loads(state_data).get("current_round", 0))

        # Commit a snap-fwd manifest matching the materialized live target.
        state_sha = _store_object(v2_paths, state_data)
        nb_sha = _store_object(v2_paths, nb_data)
        ws_sha = _store_object(v2_paths, new_png)
        state_ref = ObjectRef(
            sha256=state_sha, size=len(state_data), media_type="application/json",
        )
        nb_ref = ObjectRef(
            sha256=nb_sha, size=len(nb_data), media_type="application/x-ipynb+json",
        )
        ws_ref = ObjectRef(sha256=ws_sha, size=len(new_png), media_type="image/png")
        ws_entry = WorkspaceEntry(
            path="figures/new.png", object_hash=ws_sha, size=len(new_png), mode=0o644,
        )
        m = make_snapshot_v2_manifest(
            session_id=session_id,
            snapshot_id=target_id,
            round_number=round_number,
            analysis_state_ref=state_ref.model_dump(mode="json"),
            notebook_ref=nb_ref.model_dump(mode="json"),
            checkpoint_ref=None,
            runtime=complete_runtime(required={}, captured={}),
            workspace=[ws_entry.model_dump(mode="json")],
            objects={
                state_sha: state_ref.model_dump(mode="json"),
                nb_sha: nb_ref.model_dump(mode="json"),
                ws_sha: ws_ref.model_dump(mode="json"),
            },
        )
        m.root_hash = hashlib.sha256(canonical_json(m)).hexdigest()
        mstore = ManifestStore(v2_paths)
        mstore.publish_staging(mstore.write_staging(m))

        journal = {
            "transaction_id": tx_id,
            "target_snapshot_id": target_id,
            "session_id": session_id,
            "pre_restore_ref": "",
            "recovery_mode": "forward_only",
            "status": "APPLYING",
            "applied_step_ledger": [],
            "continuation_kind": "idle",
            "branch_id": "main",
        }
        _write_journal_hmac_signed(v2_paths, tx_id, journal)

        result = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(session_dir),
            workspace_root=str(workspace),
            states_root=str(states),
        )
        assert result.recovered, result.errors
        # Scoped workspace applied and stale in-scope file removed.
        assert (workspace / "figures" / "new.png").read_bytes() == b"new"
        assert not (workspace / "figures" / "old.png").exists()
        # Out-of-scope live file preserved; out-of-scope staged file not applied.
        assert (workspace / "scratch" / "keep.txt").read_bytes() == b"keep"
        assert not (workspace / "scratch" / "evil.txt").exists()
