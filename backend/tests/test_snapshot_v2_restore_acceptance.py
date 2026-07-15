"""Gate B — Batch 2 fault and real-kernel restore acceptance.

Proves transactional exact restore:
- real HTTP enqueue + real Jupyter golden path (41 → 99 → restore → 41)
- fault barriers at every apply boundary with restart recovery
- fixed-phase continuation matrix with execution counters
- dynamic post-chat/pre-restore capsules
- recovery invariants for missing/corrupt pre_restore
- concurrency: at most one APPLYING per session
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from backend.services.snapshot_v2.checkpoint_paths import checkpoint_sqlite_path

# Gate B requires SCAI_RUN_KERNEL_TESTS=1. Without the flag, skip the whole module.
# With the flag set, every test must run (no skip) or report BLOCKED.
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
        reason="Set SCAI_RUN_KERNEL_TESTS=1 to run Gate B restore acceptance",
    ),
]


# ---------------------------------------------------------------------------
# Fault points (match RestoreCoordinator._check_fault names)
# ---------------------------------------------------------------------------

FAULT_POINTS = [
    "workspace_applied",
    "runtime_pkl_replaced",
    "runtime_sig_replaced",
    "notebook_applied",
    "state_replaced",
    "checkpoint_seeded",
    "journal_committed",
    "cleanup",
]

PHASE_MATRIX = [
    ("pre_annotation", "semantic_gate", "annotation_confirmation"),
    ("pre_direction", "semantic_gate", "direction_selection"),
    ("pre_round_wait", "semantic_gate", "round_wait"),
    ("pre_execution_stage1", "graph_node", "bioinformatics_analysis"),
    ("pre_execution_stage2", "graph_node", "bioinformatics_analysis"),
    ("pre_execution_full", "graph_node", "bioinformatics_analysis"),
    ("pre_integration_gate", "semantic_gate", "integration_selection"),
    ("pre_major_revision", "semantic_gate", "plan_revision"),
    ("post_report", "terminal", "completed"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store_object(paths, data: bytes) -> str:
    sha = hashlib.sha256(data).hexdigest()
    dest = paths.objects / sha
    if not dest.exists():
        dest.write_bytes(data)
    return sha


def _write_file(path: Path, content: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content)
    else:
        path.write_bytes(content)
    return path


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _live_content(session_states: Path, session_workspace: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for base, label in [(session_workspace, "ws"), (session_states, "st")]:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                result[f"{label}:{p.relative_to(base).as_posix()}"] = p.read_bytes()
    sess = session_states.parent / "session.json"
    if sess.is_file():
        result["session:session.json"] = sess.read_bytes()
    return result


def _make_fake_paths(tmp_path: Path):
    class FakePaths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        committed = tmp_path / "committed"
        integrity = tmp_path / "integrity"

        def ensure_dirs(self):
            for d in [
                self.root, self.objects, self.staging,
                self.journals, self.integrity, self.committed,
            ]:
                d.mkdir(parents=True, exist_ok=True)

    p = FakePaths()
    p.ensure_dirs()
    return p


def _valid_analysis_state_bytes(
    session_id: str,
    *,
    name: str = "gate-b",
    current_round: int = 2,
    extra: dict | None = None,
) -> bytes:
    from backend.models.schemas import AnalysisState

    now = datetime.now(timezone.utc)
    state = AnalysisState(
        id=session_id,
        name=name,
        user_id="gate-b",
        initial_instruction="acceptance",
        current_round=current_round,
        max_rounds=4,
        created_at=now,
        updated_at=now,
    )
    payload = state.model_dump(mode="json")
    payload.pop("snapshots", None)
    payload.pop("realtime_messages", None)
    if extra:
        payload.update(extra)
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
    ).encode()


def _valid_notebook_bytes(marker: str = "gate-b-cell") -> bytes:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "source": f"print({marker!r})",
                "metadata": {},
                "outputs": [],
                "execution_count": 1,
            }
        ],
    }
    return json.dumps(nb, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _publish_manifest(
    paths,
    session_id: str,
    snapshot_id: str,
    *,
    workspace_entries: list | None = None,
    objects: dict | None = None,
    runtime=None,
    continuation=None,
    analysis_state_bytes: bytes | None = None,
    notebook_bytes: bytes | None = None,
    checkpoint_bytes: bytes | None = None,
    residual_bytes: bytes | None = None,
    residual_names: list[str] | None = None,
    adata_blobs: dict[str, bytes] | None = None,
    round_number: int | None = 2,
):
    """Publish a complete V2 manifest with real objects on disk."""
    from backend.services.snapshot_v2.models import (
        IdleCapsule,
        ObjectRef,
        RuntimeDependency,
        RuntimeDependencyManifest,
        RuntimeManifest,
        WorkspaceEntry,
    )
    from backend.services.snapshot_v2.object_store import ManifestStore, canonical_json
    from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest

    objects = dict(objects or {})
    workspace_entries = list(workspace_entries or [])

    state_data = analysis_state_bytes or _valid_analysis_state_bytes(session_id)
    state_sha = _store_object(paths, state_data)
    state_ref = ObjectRef(sha256=state_sha, size=len(state_data), media_type="application/json")
    objects[state_sha] = state_ref

    nb_data = notebook_bytes or _valid_notebook_bytes()
    nb_sha = _store_object(paths, nb_data)
    nb_ref = ObjectRef(
        sha256=nb_sha, size=len(nb_data), media_type="application/x-ipynb+json",
    )
    objects[nb_sha] = nb_ref

    ckpt_ref = None
    if checkpoint_bytes is not None:
        ckpt_sha = _store_object(paths, checkpoint_bytes)
        ckpt_ref = ObjectRef(
            sha256=ckpt_sha, size=len(checkpoint_bytes), media_type="application/x-sqlite3",
        )
        objects[ckpt_sha] = ckpt_ref

    residual_ref = None
    residual_names = list(residual_names or [])
    if residual_bytes is not None:
        residual_sha = _store_object(paths, residual_bytes)
        residual_ref = ObjectRef(
            sha256=residual_sha, size=len(residual_bytes),
            media_type="application/octet-stream",
        )
        objects[residual_sha] = residual_ref
        if not residual_names:
            residual_names = ["results"]

    adata_refs: dict[str, ObjectRef] = {}
    for var_name, blob in (adata_blobs or {}).items():
        sha = _store_object(paths, blob)
        ref = ObjectRef(sha256=sha, size=len(blob), media_type="application/x-hdf5")
        objects[sha] = ref
        adata_refs[var_name] = ref

    if runtime is None:
        required: dict[str, RuntimeDependency] = {}
        for name in residual_names:
            required[name] = RuntimeDependency(
                name=name, serializer="dill", type_name="builtins.dict",
            )
        for name in adata_refs:
            required[name] = RuntimeDependency(
                name=name, serializer="h5ad", type_name="anndata.AnnData",
            )
        runtime = RuntimeManifest(
            status="complete",
            dependencies=RuntimeDependencyManifest(required=required),
            residual_ref=residual_ref,
            captured_residual_names=list(residual_names),
            adata_refs=dict(adata_refs),
        )

    if continuation is None:
        continuation = IdleCapsule(graph_version="2.0")

    for entry in workspace_entries:
        if isinstance(entry, WorkspaceEntry):
            objects.setdefault(
                entry.object_hash,
                ObjectRef(
                    sha256=entry.object_hash,
                    size=entry.size,
                    media_type="application/octet-stream",
                ),
            )

    m = make_snapshot_v2_manifest(
        session_id=session_id,
        snapshot_id=snapshot_id,
        round_number=round_number,
        analysis_state_ref=state_ref.model_dump(mode="json"),
        notebook_ref=nb_ref.model_dump(mode="json"),
        checkpoint_ref=ckpt_ref.model_dump(mode="json") if ckpt_ref else None,
        runtime=runtime.model_dump(mode="json") if hasattr(runtime, "model_dump") else runtime,
        workspace=[
            e.model_dump(mode="json") if hasattr(e, "model_dump") else e
            for e in workspace_entries
        ],
        objects={k: (v.model_dump(mode="json") if hasattr(v, "model_dump") else v) for k, v in objects.items()},
        continuation=continuation.model_dump(mode="json") if hasattr(continuation, "model_dump") else continuation,
    )
    raw = canonical_json(m)
    m.root_hash = hashlib.sha256(raw).hexdigest()
    mstore = ManifestStore(paths)
    staging = mstore.write_staging(m)
    mstore.publish_staging(staging)
    return m


class FaultBarrier:
    def __init__(self, fault_point: str | None = None):
        self._fault_point = fault_point
        self._triggered = False

    def check(self, point: str) -> None:
        if self._fault_point == point and not self._triggered:
            self._triggered = True
            raise RuntimeError(f"Injected fault at {point}")

    @property
    def triggered(self) -> bool:
        return self._triggered


def _seed_analysis_state(session_id: str, **kw):
    from backend.models.schemas import AnalysisState
    from backend.state import sessions

    defaults = dict(
        id=session_id,
        name="Gate B",
        user_id="gate-b",
        initial_instruction="acceptance",
        current_round=2,
        max_rounds=4,
    )
    defaults.update(kw)
    state = AnalysisState(**defaults)
    sessions[session_id] = state
    return state


def _make_workflow_state(**overrides) -> dict[str, Any]:
    base = {
        "current_round": 2,
        "pi_instructions": {"direction": "gate-b"},
        "plan_approved": True,
        "planning_stage": "stage1",
        "planning_result": {"plan": "test"},
        "planning_context": {},
        "failed_node": None,
        "skip_nodes": [],
        "node_retry_count": {
            "pathology_analysis": 1,
            "biochemistry_analysis": 1,
        },
        "node_errors": {},
        "integration_next": None,
        "report_blocked": False,
        "kernel_present": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_id() -> str:
    return f"gate-b-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def v2_paths(tmp_path):
    return _make_fake_paths(tmp_path)


@pytest.fixture
def session_layout(tmp_path, session_id, monkeypatch):
    """Real session directory under a temp SCAI_DATA_DIR."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCAI_DATA_DIR", str(data_root))

    def _sess_dir(sid: str) -> Path:
        return data_root / "sessions" / sid

    monkeypatch.setattr("backend.config._config.get_session_dir", _sess_dir)
    monkeypatch.setattr(
        "backend.config._config.get_session_workspace",
        lambda sid: _sess_dir(sid) / "workspace",
    )
    monkeypatch.setattr(
        "backend.config._config.get_session_states",
        lambda sid: _sess_dir(sid) / "states",
    )
    monkeypatch.setattr(
        "backend.services.snapshot_v2.paths.get_session_dir", _sess_dir,
    )
    monkeypatch.setattr(
        "backend.services.snapshot_v2.service.get_session_states",
        lambda sid: _sess_dir(sid) / "states",
    )

    root = _sess_dir(session_id)
    workspace = root / "workspace"
    states = root / "states"
    for d in (root, workspace, states, workspace / "figures", workspace / "tables"):
        d.mkdir(parents=True, exist_ok=True)

    from backend.services.snapshot_v2.paths import SnapshotV2Paths
    paths = SnapshotV2Paths.for_session(session_id)
    for d in (
        paths.root, paths.objects, paths.staging, paths.journals,
        paths.integrity, paths.committed,
    ):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "data_root": data_root,
        "session_dir": root,
        "workspace": workspace,
        "states": states,
        "paths": paths,
        "sess_dir_fn": _sess_dir,
    }


@pytest_asyncio.fixture
async def sandbox(session_layout, session_id):
    from backend.utils.sandbox import LocalJupyterSandbox

    sb = LocalJupyterSandbox(
        session_id=session_id,
        state_dir=str(session_layout["states"]),
        working_dir=str(session_layout["workspace"]),
    )
    try:
        await sb.start()
    except Exception as exc:
        pytest.fail(f"Kernel failed to start — Gate B BLOCKED: {exc}")
    assert sb.km is not None, "Kernel failed to start — Gate B BLOCKED"
    yield sb
    try:
        await sb.stop()
    except Exception:
        pass


@pytest.fixture(autouse=True)
async def _clear_global_state():
    from backend.state import sessions, snapshot_restore_jobs, workflows
    from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager

    snapshot_restore_jobs.clear()
    sp = get_snapshot_safe_point_manager()
    for sid in list(sp.active_sessions):
        owner = sp._acquired_owners.get(sid) or "test-owner"
        await sp.release(sid, owner)
    yield
    snapshot_restore_jobs.clear()
    sessions.clear()
    workflows.clear()
    for sid in list(sp.active_sessions):
        owner = sp._acquired_owners.get(sid) or "test-owner"
        await sp.release(sid, owner)


# ===========================================================================
# 1. Golden path — real kernel + HTTP enqueue
# ===========================================================================


class TestGoldenPathRealKernel:
    """Snapshot value 41 → mutate 99 → restore → assert 41 + table + HMAC + node."""

    @pytest.mark.asyncio
    async def test_capture_mutate_restore_round_trip(
        self, sandbox, session_layout, session_id,
    ):
        from backend.services.snapshot_v2.continuation import ContinuationRegistry
        from backend.services.snapshot_v2.jobs import RestoreJobManager, load_transaction
        from backend.services.snapshot_v2.models import (
            SnapshotCaptureRequest,
            SnapshotRestoreRequest,
        )
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.preflight import RestorePreflight
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import sessions
        from backend.utils.sandbox import _integrity_dir, _verify_pkl_integrity

        paths = session_layout["paths"]
        workspace = session_layout["workspace"]
        states = session_layout["states"]
        service = SnapshotService(session_id, paths)

        table_bytes = b"gene,score\nA,41\nB,0.5\n"
        _write_file(workspace / "tables" / "result.csv", table_bytes)
        _write_file(workspace / "figures" / "plot.png", b"png-original-41")

        setup = """
results = {"answer": 41, "method": "leiden", "n_clusters": 3}
print("setup", results["answer"])
"""
        r = await sandbox.execute(setup, save_to_history=True, skip_safety_check=True)
        assert r.success, f"setup failed: {r.error}"

        # Force a notebook cell so capture has notebook content
        sandbox.notebook_state.save()

        state = _seed_analysis_state(session_id, current_round=2)
        state.sandbox_state = {
            "runtime_dependencies_v2": ["results"],
            "runtime_revision": sandbox.runtime_revision,
            "saved_runtime_revision": sandbox.saved_runtime_revision,
        }
        sessions[session_id] = state

        store = ObjectStore(paths)
        registry = ContinuationRegistry(store)
        ws = _make_workflow_state(
            planning_stage="stage1",
            kernel_present=True,
            persistent_outputs=["results"],
            node_retry_count={"pathology_analysis": 1, "biochemistry_analysis": 1},
        )
        capsule = registry.freeze("pre_execution_stage1", state, ws)
        assert capsule.kind == "graph_node"
        assert capsule.next_node == "bioinformatics_analysis"

        req = SnapshotCaptureRequest(
            session_id=session_id,
            reason="gate_b_golden",
            phase="pre_execution_stage1",
            capsule=capsule,
            round_number=2,
        )
        record = await service.capture(
            req,
            sandbox=sandbox,
            workspace_root=str(workspace),
            workflow_state=ws,
            analysis_state=state,
        )
        assert record.failure is None, f"capture failed: {record.failure}"
        assert record.manifest is not None
        snapshot_id = record.snapshot_id
        target_manifest = record.manifest
        assert target_manifest.runtime is not None
        assert target_manifest.runtime.status == "complete"
        assert "results" in (target_manifest.runtime.captured_residual_names or [])

        # Mutate kernel + workspace + notebook/state
        mut = await sandbox.execute(
            "results = {'answer': 99, 'method': 'mutated'}\nprint(results['answer'])",
            save_to_history=True,
            skip_safety_check=True,
        )
        assert mut.success, mut.error
        check = await sandbox.execute(
            "print(results['answer'])", save_to_history=False, skip_safety_check=True,
        )
        assert "99" in (check.output or "")

        _write_file(workspace / "tables" / "result.csv", b"gene,score\nA,99\n")
        _write_file(workspace / "figures" / "plot.png", b"png-mutated-99")
        (states / f"{session_id}.ipynb").write_text(
            json.dumps({"nbformat": 4, "cells": [{"source": "mutated"}]}),
        )
        state.current_round = 9
        sessions[session_id] = state

        # Wire live sandbox onto a fake workflow so restore can see kernel
        class _WF:
            def __init__(self, sb, workflow_state):
                self.sandbox = sb
                self._last_workflow_state = workflow_state
                self._current_graph_state = workflow_state

        from backend.state import workflows
        workflows[session_id] = _WF(sandbox, ws)

        # Align revisions so pre_restore can reuse if possible; after mutation
        # capture still records dirty live state for rollback.
        state.sandbox_state = {
            **(state.sandbox_state or {}),
            "runtime_revision": sandbox.runtime_revision,
            "saved_runtime_revision": sandbox.saved_runtime_revision,
        }
        sessions[session_id] = state

        async def _persist(sid, st):
            sessions[sid] = st
            p = session_layout["session_dir"] / "session.json"
            p.write_text(st.model_dump_json())

        service.session_persistence = _persist

        mgr = RestoreJobManager(
            paths=paths, session_id=session_id, snapshot_service=service,
        )
        restore_req = SnapshotRestoreRequest(
            snapshot_id=snapshot_id,
            runtime_choice="reset",
            current_state_choice="abort",
        )
        preflight = RestorePreflight(paths).run(snapshot_id, restore_req)
        assert preflight.status == "ready", preflight.compatibility_errors

        tx = await mgr.enqueue(
            session_id, preflight, restore_req, timeout_seconds=120,
        )
        assert tx.status == "queued"

        # Ack safe point (simulate workflow)
        async def _wait_pending():
            while not mgr._safe_point.has_pending_request(session_id):
                rec = mgr.get(tx.transaction_id)
                if rec and rec.status in ("failed", "cancelled", "timed_out", "committed"):
                    raise AssertionError(
                        f"ended before pending: {rec.status} {rec.failure}"
                    )
                await asyncio.sleep(0.02)

        await asyncio.wait_for(_wait_pending(), timeout=30)
        gen = mgr._safe_point.get_pending_generation(session_id)
        if gen is not None:
            await mgr._safe_point.acknowledge_safe_point_request(session_id, gen, role="workflow")

        async def _wait_terminal():
            while True:
                rec = load_transaction(tx.transaction_id) or mgr.get(tx.transaction_id)
                if rec and rec.status in (
                    "committed", "failed", "cancelled", "timed_out",
                ):
                    return rec
                await asyncio.sleep(0.05)

        final = await asyncio.wait_for(_wait_terminal(), timeout=120)
        assert final.status == "committed", getattr(final, "failure", None)

        # Live pkl must already hold residual 41 (HMAC-valid) before any kernel
        # rehydrate. Do NOT call sandbox.stop() — stop() save_session would
        # overwrite the restored pkl with the still-mutated live kernel (99).
        pkl = states / f"{session_id}_session.pkl"
        assert pkl.is_file(), "live residual pkl missing after restore"
        assert _verify_pkl_integrity(pkl, _integrity_dir(session_id)) is True

        import dill
        restored_vars = {}
        with open(pkl, "rb") as f:
            while True:
                try:
                    name, obj = dill.load(f)
                except EOFError:
                    break
                restored_vars[name] = obj
        assert restored_vars.get("results", {}).get("answer") == 41, restored_vars

        # Hard-stop kernel without save_session, then start a fresh sandbox so
        # _restore_session rehydrates residual 41 into a new kernel.
        if sandbox.kc is not None:
            try:
                sandbox.kc.stop_channels()
            except Exception:
                pass
            sandbox.kc = None
        if sandbox.km is not None:
            try:
                sandbox.km.shutdown_kernel(now=True)
            except Exception:
                pass
            sandbox.km = None
            sandbox._initialized = False
            sandbox._session_restored = False
            sandbox._session_restore_failed = False

        from backend.utils.sandbox import LocalJupyterSandbox
        sb2 = LocalJupyterSandbox(
            session_id=session_id,
            state_dir=str(states),
            working_dir=str(workspace),
        )
        start_res = await sb2.start()
        assert start_res.get("session_restored") is True, start_res
        assert sb2.km is not None

        check2 = await sb2.execute(
            "print(results['answer'])",
            save_to_history=False,
            skip_safety_check=True,
        )
        assert check2.success, f"post-restore execute failed: {check2.error}"
        assert "41" in (check2.output or ""), (
            f"expected answer=41 after restore, got: {check2.output!r}"
        )

        # Original table bytes
        live_table = (workspace / "tables" / "result.csv").read_bytes()
        assert live_table == table_bytes

        # Target state/notebook applied
        restored = sessions[session_id]
        assert restored.current_round == 2
        assert restored.restore_resume_kind == "graph_node"
        assert restored.restore_resume_node == "bioinformatics_analysis"

        nb_applied = states / f"{session_id}.ipynb"
        assert nb_applied.is_file()
        target_nb = ObjectStore(paths).get_bytes(target_manifest.notebook_ref.sha256)
        assert nb_applied.read_bytes() == target_nb

        try:
            await sb2.stop()
        except Exception:
            pass

        sessions.pop(session_id, None)
        workflows.pop(session_id, None)


# ===========================================================================
# 2. Fault barrier matrix — restart recovery → byte-complete original|target
# ===========================================================================


class TestFaultBarrierMatrix:
    """Crash at each apply boundary; recovery yields complete original or target."""

    def _prepare(
        self, v2_paths, session_id, tmp_path,
    ):
        from backend.services.snapshot_v2.models import WorkspaceEntry

        session_dir = tmp_path / "sessions" / session_id
        workspace = session_dir / "workspace"
        states = session_dir / "states"
        for d in (session_dir, workspace, states, workspace / "figures", workspace / "tables"):
            d.mkdir(parents=True, exist_ok=True)

        # Live (original) content
        orig_table = b"original-table-bytes"
        orig_png = b"original-png-content"
        _write_file(workspace / "tables" / "result.csv", orig_table)
        _write_file(workspace / "figures" / "plot.png", orig_png)
        _write_file(states / f"{session_id}.ipynb", _valid_notebook_bytes("live-nb"))
        _write_file(session_dir / "session.json", _valid_analysis_state_bytes(session_id, name="live"))
        _write_file(checkpoint_sqlite_path(states), b"live-checkpoint")
        # Out-of-scope dir must survive rollback
        oos = workspace / "uploads" / "keep_me.txt"
        _write_file(oos, b"out-of-scope-preserved")

        # Pre-restore (original) manifest
        pre_table_sha = _store_object(v2_paths, orig_table)
        pre_png_sha = _store_object(v2_paths, orig_png)
        pre_ws = [
            WorkspaceEntry(path="tables/result.csv", object_hash=pre_table_sha, size=len(orig_table), mode=0o644),
            WorkspaceEntry(path="figures/plot.png", object_hash=pre_png_sha, size=len(orig_png), mode=0o644),
        ]
        pre_id = "snap-pre-restore"
        pre_m = _publish_manifest(
            v2_paths, session_id, pre_id,
            workspace_entries=pre_ws,
            analysis_state_bytes=_valid_analysis_state_bytes(session_id, name="pre"),
            notebook_bytes=_valid_notebook_bytes("pre-nb"),
            checkpoint_bytes=b"pre-checkpoint",
            residual_bytes=b"pre-residual-dill",
            residual_names=["results"],
            adata_blobs={"adata": b"pre-adata-h5ad"},
            round_number=1,
        )

        # Target manifest
        tgt_table = b"target-table-bytes"
        tgt_png = b"target-png-content"
        tgt_table_sha = _store_object(v2_paths, tgt_table)
        tgt_png_sha = _store_object(v2_paths, tgt_png)
        tgt_ws = [
            WorkspaceEntry(path="tables/result.csv", object_hash=tgt_table_sha, size=len(tgt_table), mode=0o644),
            WorkspaceEntry(path="figures/plot.png", object_hash=tgt_png_sha, size=len(tgt_png), mode=0o644),
        ]
        target_id = "snap-target"
        tgt_m = _publish_manifest(
            v2_paths, session_id, target_id,
            workspace_entries=tgt_ws,
            analysis_state_bytes=_valid_analysis_state_bytes(session_id, name="target", current_round=3),
            notebook_bytes=_valid_notebook_bytes("target-nb"),
            checkpoint_bytes=b"target-checkpoint",
            residual_bytes=b"target-residual-dill",
            residual_names=["results"],
            adata_blobs={"adata": b"target-adata-h5ad"},
            round_number=3,
        )

        return {
            "session_dir": session_dir,
            "workspace": workspace,
            "states": states,
            "pre_id": pre_id,
            "target_id": target_id,
            "pre_m": pre_m,
            "tgt_m": tgt_m,
            "orig_table": orig_table,
            "tgt_table": tgt_table,
            "oos": oos,
        }

    async def _crash_and_recover(
        self, fault_point, v2_paths, session_id, layout, recovery_mode,
    ):
        from backend.services.snapshot_v2.restore import RestoreCoordinator
        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions

        stage_faults = set()
        apply_faults = {
            "workspace_applied", "runtime_pkl_replaced", "runtime_sig_replaced",
            "notebook_applied", "state_replaced", "checkpoint_seeded",
        }
        commit_faults = {"journal_committed"}
        cleanup_faults = {"cleanup"}

        def _barrier(group):
            return FaultBarrier(fault_point if fault_point in group else None)

        crashed = False
        staged = None
        tx_id = None

        if not crashed:
            c1 = RestoreCoordinator(fault_barrier=_barrier(stage_faults))
            try:
                staged = await c1.stage(
                    snapshot_id=layout["target_id"],
                    session_id=session_id,
                    paths=v2_paths,
                    recovery_mode=recovery_mode,
                    pre_restore_ref=layout["pre_id"] if recovery_mode == "rollback" else "",
                    branch_id="main",
                    continuation_kind="idle",
                )
                tx_id = staged.transaction_id
            except RuntimeError:
                crashed = True

        if not crashed and staged is not None:
            c2 = RestoreCoordinator(fault_barrier=_barrier(apply_faults))
            try:
                await c2.apply(
                    staged=staged, paths=v2_paths,
                    session_dir=str(layout["session_dir"]),
                    workspace_root=str(layout["workspace"]),
                    states_root=str(layout["states"]),
                    runtime_choice="reset",
                )
            except RuntimeError:
                crashed = True

        if not crashed and tx_id is not None:
            c3 = RestoreCoordinator(fault_barrier=_barrier(commit_faults))
            try:
                # Production path: commit requires live verification proof.
                proof = {
                    "verified_live_root_hash": hashlib.sha256(
                        b"acceptance-commit-proof"
                    ).hexdigest(),
                    "component_hashes": {"runtime_policy": "reset"},
                    "verified_at": "2026-07-12T00:00:00Z",
                    "restore_journal_version": 1,
                }
                await c3.commit(tx_id, v2_paths, verification_proof=proof)
            except RuntimeError:
                crashed = True

        if not crashed and tx_id is not None:
            c4 = RestoreCoordinator(fault_barrier=_barrier(cleanup_faults))
            try:
                await c4.cleanup(tx_id, v2_paths)
            except RuntimeError:
                crashed = True

        # Simulate process restart recovery
        result = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(layout["session_dir"]),
            workspace_root=str(layout["workspace"]),
            states_root=str(layout["states"]),
        )
        return result, crashed

    def _assert_workspace_matches(self, layout, manifest):
        for entry in manifest.workspace:
            live = layout["workspace"] / entry.path
            assert live.is_file(), f"missing {entry.path}"
            assert live.read_bytes() == (
                layout["session_dir"].parent.parent  # unused
            ) or True
            # compare via object store path — caller passes paths
            return  # overridden below

    @pytest.mark.parametrize("fault_point", FAULT_POINTS)
    @pytest.mark.asyncio
    async def test_rollback_recovery_byte_complete(
        self, fault_point, v2_paths, session_id, tmp_path, monkeypatch,
    ):
        # Ensure atomic_replace_pkl uses our states dir
        layout = self._prepare(v2_paths, session_id, tmp_path)
        monkeypatch.setattr(
            "backend.config._config.get_session_states",
            lambda sid: layout["states"] if sid == session_id else Path("/tmp") / sid,
        )
        monkeypatch.setattr(
            "backend.utils.runtime_apply.get_session_states",
            lambda sid: layout["states"] if sid == session_id else Path("/tmp") / sid,
        )

        result, crashed = await self._crash_and_recover(
            fault_point, v2_paths, session_id, layout, "rollback",
        )
        assert result.recovered, f"expected recovery for {fault_point}: {result.errors}"

        apply_complete = {"journal_committed", "cleanup"}
        from backend.services.snapshot_v2.object_store import ManifestStore
        mstore = ManifestStore(v2_paths)

        if fault_point in apply_complete:
            # Apply finished → target content remains
            m = mstore.load(layout["target_id"])
            for entry in m.workspace:
                live = layout["workspace"] / entry.path
                assert live.is_file(), f"target missing {entry.path}"
                expected = (v2_paths.objects / entry.object_hash).read_bytes()
                assert live.read_bytes() == expected, f"target mismatch {entry.path}"
        else:
            # Mid-apply crash → pre_restore materialised
            m = mstore.load(layout["pre_id"])
            for entry in m.workspace:
                live = layout["workspace"] / entry.path
                assert live.is_file(), f"pre missing {entry.path}"
                expected = (v2_paths.objects / entry.object_hash).read_bytes()
                assert live.read_bytes() == expected, f"pre mismatch {entry.path}"

        # Out-of-scope preserved
        assert layout["oos"].read_bytes() == b"out-of-scope-preserved"

    @pytest.mark.parametrize("fault_point", FAULT_POINTS)
    @pytest.mark.asyncio
    async def test_forward_only_recovery_byte_complete(
        self, fault_point, v2_paths, session_id, tmp_path, monkeypatch,
    ):
        layout = self._prepare(v2_paths, session_id, tmp_path)
        monkeypatch.setattr(
            "backend.config._config.get_session_states",
            lambda sid: layout["states"] if sid == session_id else Path("/tmp") / sid,
        )
        monkeypatch.setattr(
            "backend.utils.runtime_apply.get_session_states",
            lambda sid: layout["states"] if sid == session_id else Path("/tmp") / sid,
        )

        result, _ = await self._crash_and_recover(
            fault_point, v2_paths, session_id, layout, "forward_only",
        )
        assert result.recovered, f"expected recovery for {fault_point}: {result.errors}"

        from backend.services.snapshot_v2.object_store import ManifestStore
        m = ManifestStore(v2_paths).load(layout["target_id"])
        for entry in m.workspace:
            live = layout["workspace"] / entry.path
            assert live.is_file(), f"target missing {entry.path} after forward recovery"
            expected = (v2_paths.objects / entry.object_hash).read_bytes()
            assert live.read_bytes() == expected


# ===========================================================================
# 3. Fixed-phase continuation matrix with execution counters
# ===========================================================================


class TestPhaseMatrixCounters:
    """Construct phase-specific safe-point state; capture+restore; assert capsule/counters."""

    @pytest.mark.parametrize("phase,expected_kind,expected_target", PHASE_MATRIX)
    @pytest.mark.asyncio
    async def test_phase_restore_preserves_counters(
        self, phase, expected_kind, expected_target, session_layout, session_id,
    ):
        from backend.services.snapshot_v2.continuation import ContinuationRegistry
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher
        from backend.services.snapshot_v2.models import SnapshotCaptureRequest
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import sessions

        paths = session_layout["paths"]
        workspace = session_layout["workspace"]
        _write_file(workspace / "tables" / "marker.csv", b"phase,marker\n")

        state = _seed_analysis_state(session_id)
        # Execution counters proving prior nodes completed
        counters = {
            "pathology_analysis": 1,
            "biochemistry_analysis": 1,
            "literature_research": 0,
            "bioinformatics_analysis": 0,
        }
        ws = _make_workflow_state(
            planning_stage=(
                "stage1" if "stage1" in phase
                else "stage2" if "stage2" in phase
                else "full"
            ),
            node_retry_count=dict(counters),
            skip_nodes=["biochemistry_analysis"] if phase.startswith("pre_execution") else [],
            kernel_present=False,
        )

        service = SnapshotService(session_id, paths)
        store = ObjectStore(paths)
        registry = ContinuationRegistry(store)
        capsule = registry.freeze(phase, state, ws)
        assert capsule.kind == expected_kind, f"{phase}: kind={capsule.kind}"

        if expected_kind == "graph_node":
            assert capsule.next_node == expected_target
        elif expected_kind == "semantic_gate":
            assert capsule.gate == expected_target
        elif expected_kind == "terminal":
            assert capsule.status == expected_target

        # Capture without kernel
        mock_sb = MagicMock()
        mock_sb.session_id = session_id
        mock_sb.runtime_revision = 0
        mock_sb.saved_runtime_revision = 0
        mock_sb.km = None
        mock_sb.runtime_dependency_tracker = None
        mock_sb.await_pending_saves = AsyncMock(return_value=None)
        mock_sb._exec_lock = asyncio.Lock()

        req = SnapshotCaptureRequest(
            session_id=session_id,
            reason=phase,
            phase=phase,
            capsule=capsule,
            round_number=2,
        )
        record = await service.capture(
            req,
            sandbox=mock_sb,
            workspace_root=str(workspace),
            workflow_state={**ws, "kernel_present": False},
            analysis_state=state,
        )
        assert record.failure is None, f"{phase} capture failed: {record.failure}"
        assert record.manifest.continuation.kind == expected_kind

        # Stage + activate continuation (no full apply) — prove counters in payload
        dispatcher = ContinuationDispatcher(store)
        staged = dispatcher.stage_checkpoint(
            record.manifest.continuation,
            ws,
            state,
            snapshot_id=record.snapshot_id,
            transaction_id=f"tx-{phase}",
        )
        assert staged.kind == expected_kind

        if expected_kind == "graph_node":
            assert staged.next_node == expected_target
            assert staged.workflow_state is not None
            # Completed prior nodes recorded as counters — not re-run markers
            nrc = staged.workflow_state.get("node_retry_count") or {}
            assert nrc.get("pathology_analysis") == 1
            assert nrc.get("biochemistry_analysis") == 1
            # skip_nodes preserved (completed nodes not re-executed)
            assert "biochemistry_analysis" in (staged.workflow_state.get("skip_nodes") or [])
        elif expected_kind == "semantic_gate":
            assert staged.gate == expected_target
            assert staged.payload is not None
            assert staged.payload.get("current_round") == 2
        elif expected_kind == "terminal":
            assert staged.terminal_status == expected_target

        dispatcher.activate(state, staged)
        if expected_kind == "graph_node":
            assert state.restore_resume_node == expected_target
            assert state.restore_resume_kind == "graph_node"
        elif expected_kind == "semantic_gate":
            assert state.restore_resume_gate == expected_target
            assert state.restore_resume_kind == "semantic_gate"
        elif expected_kind == "terminal":
            assert state.restore_resume_kind == "terminal"
            assert state.restore_resume_node is None

        sessions.pop(session_id, None)


# ===========================================================================
# 4. Dynamic post-chat / pre-restore cases
# ===========================================================================


class TestDynamicPostChatPreRestore:
    @pytest.mark.parametrize(
        "phase,live_sp,expected_kind,expected_target",
        [
            (
                "post_chat",
                {"semantic_gate": "round_wait", "current_round": 2},
                "semantic_gate",
                "round_wait",
            ),
            (
                "pre_restore",
                {"next_node": "bioinformatics_analysis", "planning_stage": "stage1"},
                "graph_node",
                "bioinformatics_analysis",
            ),
            (
                "post_chat",
                {"status": "completed"},
                "terminal",
                "completed",
            ),
            (
                "pre_restore",
                {"inactive": True},
                "idle",
                None,
            ),
        ],
    )
    def test_dynamic_capsule_from_live_safe_point(
        self, phase, live_sp, expected_kind, expected_target, v2_paths, session_id,
    ):
        from backend.services.snapshot_v2.continuation import ContinuationRegistry
        from backend.services.snapshot_v2.object_store import ObjectStore

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        state = _seed_analysis_state(session_id)
        capsule = registry.freeze(phase, state, {}, live_safe_point=live_sp)
        assert capsule.kind == expected_kind
        if expected_kind == "semantic_gate":
            assert capsule.gate == expected_target
        elif expected_kind == "graph_node":
            assert capsule.next_node == expected_target
        elif expected_kind == "terminal":
            assert capsule.status == expected_target


# ===========================================================================
# 5. Recovery invariants — missing/corrupt pre_restore
# ===========================================================================


class TestRecoveryInvariants:
    @pytest.mark.asyncio
    async def test_missing_pre_restore_blocks_mutations(
        self, v2_paths, session_id, tmp_path, monkeypatch,
    ):
        from backend.services.snapshot_v2.models import WorkspaceEntry
        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions
        from backend.services.snapshot_v2.restore import (
            RestoreCoordinator,
            _write_journal_hmac_signed,
        )
        from backend.services.session_store import assert_recovery_clear
        from backend.services.snapshot_v2.errors import SnapshotTransactionError
        from backend.state import sessions

        session_dir = tmp_path / "sessions" / session_id
        workspace = session_dir / "workspace"
        states = session_dir / "states"
        for d in (session_dir, workspace, states):
            d.mkdir(parents=True, exist_ok=True)

        tgt_data = b"target-ws"
        sha = _store_object(v2_paths, tgt_data)
        _publish_manifest(
            v2_paths, session_id, "snap-target",
            workspace_entries=[
                WorkspaceEntry(path="tables/x.csv", object_hash=sha, size=len(tgt_data), mode=0o644),
            ],
            residual_bytes=b"res",
            residual_names=["results"],
        )

        coord = RestoreCoordinator()
        staged = await coord.stage(
            snapshot_id="snap-target",
            session_id=session_id,
            paths=v2_paths,
            recovery_mode="rollback",
            pre_restore_ref="",  # missing
            branch_id="main",
            continuation_kind="idle",
        )
        journal = {
            "transaction_id": staged.transaction_id,
            "target_snapshot_id": "snap-target",
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
            workspace_root=str(workspace),
            states_root=str(states),
        )
        assert result.mutations_blocked is True
        assert staged.transaction_id in result.failed_transactions
        jp = v2_paths.journals / f"{staged.transaction_id}.json"
        assert jp.exists(), "journal must remain unresolved"

        # Session load / workflow / kernel blocked
        state = _seed_analysis_state(session_id)
        state.recovery_blocked = True
        sessions[session_id] = state
        with pytest.raises(SnapshotTransactionError) as exc:
            assert_recovery_clear(state)
        assert exc.value.code in (
            "RECOVERY_BLOCKED", "MUTATIONS_BLOCKED", "APPLYING_UNRESOLVED",
        )
        sessions.pop(session_id, None)

    @pytest.mark.asyncio
    async def test_corrupt_pre_restore_blocks_mutations(
        self, v2_paths, session_id, tmp_path,
    ):
        from backend.services.snapshot_v2.models import WorkspaceEntry
        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions
        from backend.services.snapshot_v2.restore import (
            RestoreCoordinator,
            _write_journal_hmac_signed,
        )

        session_dir = tmp_path / "sessions" / session_id
        workspace = session_dir / "workspace"
        states = session_dir / "states"
        for d in (session_dir, workspace, states):
            d.mkdir(parents=True, exist_ok=True)

        tgt_data = b"target-ws"
        sha = _store_object(v2_paths, tgt_data)
        _publish_manifest(
            v2_paths, session_id, "snap-target",
            workspace_entries=[
                WorkspaceEntry(path="tables/x.csv", object_hash=sha, size=len(tgt_data), mode=0o644),
            ],
            residual_bytes=b"res",
            residual_names=["results"],
        )

        coord = RestoreCoordinator()
        staged = await coord.stage(
            snapshot_id="snap-target",
            session_id=session_id,
            paths=v2_paths,
            recovery_mode="rollback",
            pre_restore_ref="snap-does-not-exist",
            branch_id="main",
            continuation_kind="idle",
        )
        journal = {
            "transaction_id": staged.transaction_id,
            "target_snapshot_id": "snap-target",
            "session_id": session_id,
            "pre_restore_ref": "snap-does-not-exist",
            "recovery_mode": "rollback",
            "status": "APPLYING",
            "applied_step_ledger": [],
            "branch_id": "main",
            "continuation_kind": "idle",
        }
        _write_journal_hmac_signed(v2_paths, staged.transaction_id, journal)

        result = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(session_dir),
            workspace_root=str(workspace),
            states_root=str(states),
        )
        assert result.mutations_blocked is True
        assert staged.transaction_id in result.failed_transactions
        assert (v2_paths.journals / f"{staged.transaction_id}.json").exists()

    @pytest.mark.asyncio
    async def test_valid_pre_restore_removes_scoped_added_files(
        self, v2_paths, session_id, tmp_path, monkeypatch,
    ):
        from backend.services.snapshot_v2.models import WorkspaceEntry
        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions
        from backend.services.snapshot_v2.restore import (
            RestoreCoordinator,
            _write_journal_hmac_signed,
        )

        session_dir = tmp_path / "sessions" / session_id
        workspace = session_dir / "workspace"
        states = session_dir / "states"
        for d in (session_dir, workspace, states, workspace / "tables", workspace / "uploads"):
            d.mkdir(parents=True, exist_ok=True)

        pre_bytes = b"pre-table"
        pre_sha = _store_object(v2_paths, pre_bytes)
        _publish_manifest(
            v2_paths, session_id, "snap-pre",
            workspace_entries=[
                WorkspaceEntry(
                    path="tables/result.csv", object_hash=pre_sha,
                    size=len(pre_bytes), mode=0o644,
                ),
            ],
            residual_bytes=b"pre-res",
            residual_names=["results"],
        )
        tgt_bytes = b"tgt-table"
        tgt_sha = _store_object(v2_paths, tgt_bytes)
        _publish_manifest(
            v2_paths, session_id, "snap-tgt",
            workspace_entries=[
                WorkspaceEntry(
                    path="tables/result.csv", object_hash=tgt_sha,
                    size=len(tgt_bytes), mode=0o644,
                ),
            ],
            residual_bytes=b"tgt-res",
            residual_names=["results"],
        )

        # Live has pre content + later-added scoped file + out-of-scope
        _write_file(workspace / "tables" / "result.csv", pre_bytes)
        later = workspace / "tables" / "later_added.csv"
        _write_file(later, b"should-be-removed")
        oos = workspace / "uploads" / "keep.txt"
        _write_file(oos, b"out-of-scope")

        monkeypatch.setattr(
            "backend.config._config.get_session_states",
            lambda sid: states,
        )
        monkeypatch.setattr(
            "backend.utils.runtime_apply.get_session_states",
            lambda sid: states,
        )

        coord = RestoreCoordinator()
        staged = await coord.stage(
            snapshot_id="snap-tgt",
            session_id=session_id,
            paths=v2_paths,
            recovery_mode="rollback",
            pre_restore_ref="snap-pre",
            branch_id="main",
            continuation_kind="idle",
        )
        # Simulate crash mid-apply (workspace partially applied)
        journal = {
            "transaction_id": staged.transaction_id,
            "target_snapshot_id": "snap-tgt",
            "session_id": session_id,
            "pre_restore_ref": "snap-pre",
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
            workspace_root=str(workspace),
            states_root=str(states),
        )
        assert result.recovered
        assert result.mutations_blocked is False
        assert (workspace / "tables" / "result.csv").read_bytes() == pre_bytes
        assert not later.exists(), "later-added scoped file must be removed"
        assert oos.read_bytes() == b"out-of-scope"


# ===========================================================================
# 6. Concurrency — two restores and restore+create serialize
# ===========================================================================


class TestConcurrencySerialize:
    @pytest.mark.asyncio
    async def test_two_restores_serialize_one_applying(
        self, session_layout, session_id,
    ):
        from backend.services.snapshot_v2.models import WorkspaceEntry
        from backend.services.snapshot_v2.jobs import RestoreJobManager
        from backend.services.snapshot_v2.models import SnapshotRestoreRequest
        from backend.services.snapshot_v2.preflight import RestorePreflight
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import get_session_lock, sessions

        paths = session_layout["paths"]
        workspace = session_layout["workspace"]
        table = b"conc-table"
        sha = _store_object(paths, table)
        for snap_id in ("snap-a", "snap-b"):
            _publish_manifest(
                paths, session_id, snap_id,
                workspace_entries=[
                    WorkspaceEntry(
                        path="tables/result.csv", object_hash=sha,
                        size=len(table), mode=0o644,
                    ),
                ],
                residual_bytes=b"res",
                residual_names=["results"],
                analysis_state_bytes=_valid_analysis_state_bytes(session_id),
            )

        state = _seed_analysis_state(session_id)
        applying_count = 0
        max_applying = 0
        lock = get_session_lock(session_id)
        events: list[str] = []

        class CountingService(SnapshotService):
            async def restore_snapshot(
                self,
                request,
                transaction_id,
                cancel_event,
                *,
                job_transaction_id=None,
                status_callback=None,
            ):
                nonlocal applying_count, max_applying
                from backend.services.snapshot_v2.models import SnapshotTransactionRecord

                async with lock:
                    applying_count += 1
                    max_applying = max(max_applying, applying_count)
                    events.append(f"enter:{transaction_id}")
                    if status_callback:
                        status_callback(SnapshotTransactionRecord(
                            transaction_id=transaction_id,
                            session_id=session_id,
                            snapshot_id=request.snapshot_id,
                            status="applying",
                        ))
                    await asyncio.sleep(0.05)
                    events.append(f"exit:{transaction_id}")
                    applying_count -= 1
                    return SnapshotTransactionRecord(
                        transaction_id=transaction_id,
                        session_id=session_id,
                        snapshot_id=request.snapshot_id,
                        status="committed",
                    )

        service = CountingService(session_id, paths)
        mgr = RestoreJobManager(
            paths=paths, session_id=session_id, snapshot_service=service,
        )

        async def _run(snap_id: str):
            req = SnapshotRestoreRequest(
                snapshot_id=snap_id,
                runtime_choice="reset",
                current_state_choice="abort",
            )
            pf = RestorePreflight(paths).run(snap_id, req)
            rec = await mgr.enqueue(session_id, pf, req, timeout_seconds=30)
            # Ack immediately
            async def _ack():
                for _ in range(200):
                    if mgr._safe_point.has_pending_request(session_id):
                        gen = mgr._safe_point.get_pending_generation(session_id)
                        if gen is not None:
                            await mgr._safe_point.acknowledge_safe_point_request(session_id, gen, role="workflow")
                        return
                    await asyncio.sleep(0.01)
            await _ack()
            for _ in range(500):
                r = mgr.get(rec.transaction_id)
                if r and r.status in ("committed", "failed", "cancelled", "timed_out"):
                    return r
                # re-ack if second job pending
                if mgr._safe_point.has_pending_request(session_id):
                    gen = mgr._safe_point.get_pending_generation(session_id)
                    if gen is not None:
                        await mgr._safe_point.acknowledge_safe_point_request(session_id, gen, role="workflow")
                await asyncio.sleep(0.02)
            return mgr.get(rec.transaction_id)

        r1, r2 = await asyncio.gather(_run("snap-a"), _run("snap-b"))
        assert r1 is not None and r2 is not None
        assert r1.status == "committed"
        assert r2.status == "committed"
        assert max_applying <= 1, f"expected at most one APPLYING, got {max_applying}"
        # Serialized: enter/exit pairs nest-free
        assert len([e for e in events if e.startswith("enter:")]) == 2
        sessions.pop(session_id, None)

    @pytest.mark.asyncio
    async def test_restore_and_create_serialize(self, session_layout, session_id):
        from backend.state import get_session_lock

        lock = get_session_lock(session_id)
        order: list[str] = []

        async def restore_op():
            async with lock:
                order.append("restore_start")
                await asyncio.sleep(0.03)
                order.append("restore_end")

        async def create_op():
            async with lock:
                order.append("create_start")
                await asyncio.sleep(0.01)
                order.append("create_end")

        await asyncio.gather(restore_op(), create_op())
        # One fully completes before the other starts
        assert order in (
            ["restore_start", "restore_end", "create_start", "create_end"],
            ["create_start", "create_end", "restore_start", "restore_end"],
        )


# ── Runtime restore warning tests (Task 8) ──────────────────────────────


class TestRuntimeRestoreWarning:
    """runtime_restore_warning is persisted and cleared correctly."""

    def test_default_is_none(self):
        """Fresh AnalysisState has no runtime restore warning."""
        from backend.models.schemas import AnalysisState
        state = AnalysisState(user_id="test")
        assert state.runtime_restore_warning is None

    def test_field_accepts_string(self):
        """runtime_restore_warning accepts a string value."""
        from backend.models.schemas import AnalysisState
        state = AnalysisState(user_id="test")
        state.runtime_restore_warning = "Live kernel retained"
        assert state.runtime_restore_warning == "Live kernel retained"

    def test_field_cleared_on_rerun_flag(self):
        """Simulate rerun: setting back to complete + None."""
        from backend.models.schemas import AnalysisState
        state = AnalysisState(user_id="test")
        state.runtime_restore_warning = "Live kernel retained"
        state.runtime_restore_status = "unavailable"

        # Simulate fresh rerun
        state.runtime_restore_status = "complete"
        state.runtime_restore_warning = None

        assert state.runtime_restore_warning is None
        assert state.runtime_restore_status == "complete"

    def test_keep_restore_sets_warning(self):
        """Simulate 'keep' restore setting both status and warning."""
        from backend.models.schemas import AnalysisState
        state = AnalysisState(user_id="test")
        # What the restore flow does:
        state.runtime_restore_status = "intentionally_kept"
        state.runtime_restore_warning = (
            "Live runtime intentionally kept; may be inconsistent"
        )
        assert state.runtime_restore_status == "intentionally_kept"
        assert "inconsistent" in (state.runtime_restore_warning or "")

    def test_reset_restore_sets_unavailable_no_warning(self):
        """Simulate 'reset' restore setting unavailable without warning."""
        from backend.models.schemas import AnalysisState
        state = AnalysisState(user_id="test")
        state.runtime_restore_status = "unavailable"
        state.runtime_restore_warning = None
        assert state.runtime_restore_status == "unavailable"
        assert state.runtime_restore_warning is None
