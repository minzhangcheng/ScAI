"""Gate A — Batch 1 real capture acceptance.

Proves complete V2 capture: real-kernel round-trip with independent object
verification, unpersistable required failure (no commit + failure gate),
empty complete idle/terminal without kernel, and typed introspection errors.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
        reason="Set SCAI_RUN_KERNEL_TESTS=1 to run Gate A capture acceptance",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_id() -> str:
    return f"gate-a-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def v2_paths(tmp_path):
    class FakePaths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"

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


@pytest.fixture
def workspace_root(tmp_path):
    d = tmp_path / "ws_scope"
    d.mkdir(exist_ok=True)
    fig = d / "figures"
    fig.mkdir(exist_ok=True)
    (fig / "plot.png").write_bytes(b"png-gate-a")
    (d / "analysis.py").write_text("# gate-a analysis\n")
    return d


@pytest_asyncio.fixture
async def sandbox(tmp_path, session_id):
    from backend.utils.sandbox import LocalJupyterSandbox

    sb = LocalJupyterSandbox(
        session_id=session_id,
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    await sb.start()
    assert sb.km is not None, "Kernel failed to start — Gate A BLOCKED"
    yield sb
    await sb.stop()


@pytest.fixture
def service(session_id, v2_paths):
    from backend.services.snapshot_v2.service import SnapshotService

    return SnapshotService(session_id, v2_paths)


def _seed_analysis_state(session_id: str, **kw):
    from backend.models.schemas import AnalysisState
    from backend.state import sessions

    state = AnalysisState(
        id=session_id,
        name="Gate A",
        user_id="gate-a",
        initial_instruction="acceptance",
        **kw,
    )
    sessions[session_id] = state
    return state


def _make_request(session_id: str, *, reason: str = "gate_a", capsule=None, **kw):
    from backend.services.snapshot_v2.models import (
        IdleCapsule,
        SnapshotCaptureRequest,
    )

    if capsule is None:
        capsule = IdleCapsule(graph_version="2.0")
    return SnapshotCaptureRequest(
        session_id=session_id,
        reason=reason,
        phase=reason,
        capsule=capsule,
        **kw,
    )


def _committed_manifest_ids(v2_paths) -> set[str]:
    committed = v2_paths.committed
    if not committed.exists():
        return set()
    return {
        p.stem
        for p in committed.iterdir()
        if p.suffix == ".json" and p.is_file() and not p.stem.startswith("audit-")
    }


def _verify_all_manifest_refs(manifest, object_store) -> None:
    """Independently load and verify every object ref on a committed manifest."""
    object_store.verify(manifest.analysis_state_ref)
    object_store.verify(manifest.notebook_ref)
    if manifest.checkpoint_ref is not None:
        object_store.verify(manifest.checkpoint_ref)

    runtime = manifest.runtime
    assert runtime is not None
    assert runtime.status == "complete"
    for name, ref in (runtime.adata_refs or {}).items():
        object_store.verify(ref)
        raw = object_store.get_bytes(ref.sha256)
        assert len(raw) == ref.size
        assert len(raw) > 0, f"empty adata object for {name}"

    if runtime.residual_ref is not None:
        object_store.verify(runtime.residual_ref)
        raw = object_store.get_bytes(runtime.residual_ref.sha256)
        assert len(raw) == runtime.residual_ref.size
        assert len(raw) > 0

    for entry in manifest.workspace or []:
        from backend.services.snapshot_v2.models import ObjectRef

        object_store.verify(
            ObjectRef(
                sha256=entry.object_hash,
                size=entry.size,
                media_type="application/octet-stream",
            )
        )

    # analysis_state JSON must parse
    state_bytes = object_store.get_bytes(manifest.analysis_state_ref.sha256)
    state_dict = json.loads(state_bytes.decode("utf-8"))
    assert isinstance(state_dict, dict)
    assert state_dict.get("id") or state_dict.get("name") is not None

    nb_bytes = object_store.get_bytes(manifest.notebook_ref.sha256)
    nb = json.loads(nb_bytes.decode("utf-8"))
    assert "cells" in nb


# ---------------------------------------------------------------------------
# Case 1 — real kernel complete capture + independent ref verification
# ---------------------------------------------------------------------------


class TestRealKernelCompleteCapture:
    @pytest.mark.asyncio
    async def test_round_trip_results_adata_and_declared_output(
        self, sandbox, v2_paths, service, session_id, workspace_root,
    ):
        """Canonical results + AnnData + declared persistent output → complete
        commit; every manifest ref independently loadable and verified."""
        from backend.services.snapshot_v2.object_store import ObjectStore, ManifestStore
        from backend.state import sessions

        state = _seed_analysis_state(session_id)
        try:
            setup = """
import anndata as ad
import numpy as np

results = {"n_clusters": 3, "method": "leiden", "scores": [0.1, 0.2, 0.3]}
adata = ad.AnnData(X=np.random.rand(40, 12))
adata.obs["cluster"] = ["A"] * 20 + ["B"] * 20
my_scores = {"cluster_A": 0.91, "cluster_B": 0.87}
print("setup ok", adata.shape, list(results.keys()), list(my_scores.keys()))
"""
            result = await sandbox.execute(
                setup, save_to_history=False, skip_safety_check=True,
            )
            assert result.success, f"kernel setup failed: {result.error}"

            # Declared persistent output is required even if non-canonical name.
            state.sandbox_state = {
                "runtime_dependencies_v2": ["my_scores"],
            }
            sessions[session_id] = state

            req = _make_request(session_id, reason="gate_a_complete")
            record = await service.capture(
                req,
                sandbox=sandbox,
                workspace_root=str(workspace_root),
                workflow_state={
                    "kernel_present": True,
                    "persistent_outputs": ["my_scores"],
                },
                analysis_state=state,
            )

            assert record.failure is None, f"unexpected failure: {record.failure}"
            assert record.manifest is not None
            assert record.snapshot_id in _committed_manifest_ids(v2_paths)

            mstore = ManifestStore(v2_paths)
            loaded = mstore.load(record.snapshot_id)
            runtime = loaded.runtime
            assert runtime.status == "complete"

            # AnnData via h5ad
            assert "adata" in runtime.adata_refs
            # Canonical non-AnnData results + declared my_scores via residual dill
            residual_names = set(runtime.captured_residual_names or [])
            assert "results" in residual_names
            assert "my_scores" in residual_names
            assert runtime.residual_ref is not None

            required = set(runtime.dependencies.required or {})
            assert "adata" in required
            assert "results" in required
            assert "my_scores" in required

            # Independent load/verify of every ref
            obj_store = ObjectStore(v2_paths)
            _verify_all_manifest_refs(loaded, obj_store)

            # Workspace scoped files present
            ws_paths = {e.path for e in loaded.workspace}
            assert "figures/plot.png" in ws_paths
            assert "analysis.py" in ws_paths
        finally:
            sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Case 2 — unpicklable required output: no commit + persistent failure gate
# ---------------------------------------------------------------------------


class TestUnpicklableRequiredOutput:
    @pytest.mark.asyncio
    async def test_unpicklable_required_blocks_commit_and_sets_failure_gate(
        self, sandbox, v2_paths, service, session_id, workspace_root,
    ):
        from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
        from backend.state import sessions

        state = _seed_analysis_state(session_id)
        before = _committed_manifest_ids(v2_paths)
        try:
            setup = """
class _UnpicklableRequired:
    def __getstate__(self):
        raise TypeError("intentionally unpicklable for Gate A")

unpicklable_required = _UnpicklableRequired()
print("unpicklable ready")
"""
            result = await sandbox.execute(
                setup, save_to_history=False, skip_safety_check=True,
            )
            assert result.success, f"setup failed: {result.error}"

            state.sandbox_state = {
                "runtime_dependencies_v2": ["unpicklable_required"],
            }
            sessions[session_id] = state

            req = _make_request(session_id, reason="gate_a_unpicklable")
            with pytest.raises(RequiredSnapshotBlocked) as exc_info:
                await service.capture_required_snapshot(
                    state,
                    req,
                    sandbox=sandbox,
                    workflow_state={
                        "kernel_present": True,
                        "persistent_outputs": ["unpicklable_required"],
                    },
                    workspace_root=str(workspace_root),
                )

            assert "required_runtime_unpersistable" in (
                exc_info.value.code or ""
            ) or "unpersistable" in str(exc_info.value).lower()

            after = _committed_manifest_ids(v2_paths)
            assert after == before, (
                f"manifest committed on unpicklable failure: {after - before}"
            )

            # Persistent failure gate (no Skip)
            assert state.wait_input_reason == "snapshot_failure"
            assert state.snapshot_failure is not None
            actions = set(state.snapshot_failure.get("allowed_actions") or [])
            assert "skip" not in actions
            assert actions == {"retry", "manage", "terminate"}
            assert state.snapshot_failure.get("code")
        finally:
            sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Case 3 — non-kernel idle / terminal: empty complete without runtime refs
# ---------------------------------------------------------------------------


class TestNonKernelEmptyComplete:
    def _mock_sandbox_no_kernel(self, session_id: str):
        mock = MagicMock()
        mock.session_id = session_id
        mock.runtime_revision = 0
        mock.saved_runtime_revision = 0
        mock.km = None
        mock.runtime_dependency_tracker = None
        mock.execute = AsyncMock(
            side_effect=AssertionError("kernel execute must not run when kernel_present=False")
        )
        mock.await_pending_saves = AsyncMock()
        mock._exec_lock = asyncio.Lock()
        return mock

    @pytest.mark.asyncio
    async def test_idle_empty_required_commits_complete_without_runtime_refs(
        self, v2_paths, session_id, workspace_root,
    ):
        from backend.services.snapshot_v2.models import IdleCapsule
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import sessions

        state = _seed_analysis_state(session_id)
        service = SnapshotService(session_id, v2_paths)
        sandbox = self._mock_sandbox_no_kernel(session_id)
        try:
            req = _make_request(
                session_id,
                reason="gate_a_idle",
                capsule=IdleCapsule(graph_version="2.0"),
            )
            record = await service.capture(
                req,
                sandbox=sandbox,
                workspace_root=str(workspace_root),
                workflow_state={"kernel_present": False},
                analysis_state=state,
            )
            assert record.failure is None, f"unexpected failure: {record.failure}"
            runtime = record.manifest.runtime
            assert runtime.status == "complete"
            assert runtime.dependencies.required == {}
            assert runtime.adata_refs == {}
            assert runtime.residual_ref is None
            assert runtime.captured_residual_names == []
            assert record.manifest.continuation.kind == "idle"
            assert record.snapshot_id in _committed_manifest_ids(v2_paths)
        finally:
            sessions.pop(session_id, None)

    @pytest.mark.asyncio
    async def test_terminal_empty_required_commits_complete_without_runtime_refs(
        self, v2_paths, session_id, workspace_root,
    ):
        from backend.services.snapshot_v2.models import TerminalCapsule
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import sessions

        state = _seed_analysis_state(session_id)
        service = SnapshotService(session_id, v2_paths)
        sandbox = self._mock_sandbox_no_kernel(session_id)
        try:
            req = _make_request(
                session_id,
                reason="gate_a_terminal",
                capsule=TerminalCapsule(graph_version="2.0", status="completed"),
            )
            record = await service.capture(
                req,
                sandbox=sandbox,
                workspace_root=str(workspace_root),
                workflow_state={"kernel_present": False},
                analysis_state=state,
            )
            assert record.failure is None, f"unexpected failure: {record.failure}"
            runtime = record.manifest.runtime
            assert runtime.status == "complete"
            assert runtime.dependencies.required == {}
            assert not runtime.adata_refs
            assert runtime.residual_ref is None
            assert record.manifest.continuation.kind == "terminal"
            assert record.snapshot_id in _committed_manifest_ids(v2_paths)
        finally:
            sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Case 4 — failed / partial introspection: no commit + typed error preserved
# ---------------------------------------------------------------------------


class TestIntrospectionFailures:
    def _mock_live_kernel(self, session_id: str, execute_return):
        mock = MagicMock()
        mock.session_id = session_id
        mock.runtime_revision = 1
        mock.saved_runtime_revision = 0
        mock.km = MagicMock()  # live kernel present
        mock.runtime_dependency_tracker = None
        mock.execute = AsyncMock(return_value=execute_return)
        mock.await_pending_saves = AsyncMock()
        mock._exec_lock = asyncio.Lock()
        return mock

    @pytest.mark.asyncio
    async def test_failed_introspection_no_manifest_preserves_typed_error(
        self, v2_paths, session_id, workspace_root,
    ):
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import sessions

        state = _seed_analysis_state(session_id)
        service = SnapshotService(session_id, v2_paths)
        before = _committed_manifest_ids(v2_paths)
        sandbox = self._mock_live_kernel(
            session_id,
            MagicMock(success=False, error="kernel timeout during introspect", output=""),
        )
        try:
            req = _make_request(session_id, reason="gate_a_introspect_fail")
            record = await service.capture(
                req,
                sandbox=sandbox,
                workspace_root=str(workspace_root),
                workflow_state={"kernel_present": True},
                analysis_state=state,
            )
            assert record.failure is not None
            assert record.failure.code == "RUNTIME_INTROSPECT_FAILED"
            assert "timeout" in (record.failure.message or "").lower() or (
                "introspection" in (record.failure.message or "").lower()
            )
            after = _committed_manifest_ids(v2_paths)
            assert after == before
        finally:
            sessions.pop(session_id, None)

    @pytest.mark.asyncio
    async def test_partial_introspection_no_manifest_preserves_parse_error(
        self, v2_paths, session_id, workspace_root,
    ):
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import sessions

        state = _seed_analysis_state(session_id)
        service = SnapshotService(session_id, v2_paths)
        before = _committed_manifest_ids(v2_paths)
        # Begin marker only — missing end / payload
        sandbox = self._mock_live_kernel(
            session_id,
            MagicMock(
                success=True,
                output="__RUNTIME_GLOBALS_BEGIN__",
                error=None,
            ),
        )
        try:
            req = _make_request(session_id, reason="gate_a_introspect_partial")
            record = await service.capture(
                req,
                sandbox=sandbox,
                workspace_root=str(workspace_root),
                workflow_state={"kernel_present": True},
                analysis_state=state,
            )
            assert record.failure is not None
            assert record.failure.code == "RUNTIME_INTROSPECT_PARSE_ERROR"
            after = _committed_manifest_ids(v2_paths)
            assert after == before
        finally:
            sessions.pop(session_id, None)
