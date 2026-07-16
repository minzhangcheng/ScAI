"""Test snapshot V2 capture transaction — inject failures at each step boundary."""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.snapshot_v2.errors import SnapshotCaptureError, SnapshotStorageError


# ---- Fixtures ---------------------------------------------------------------

@pytest.fixture
def v2_paths(tmp_path):
    from backend.services.snapshot_v2.paths import SnapshotV2Paths

    class FakePaths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"

        def ensure_dirs(self):
            for d in [self.root, self.objects, self.staging,
                      self.journals, self.integrity, self.committed]:
                d.mkdir(parents=True, exist_ok=True)

    p = FakePaths()
    p.ensure_dirs()
    return p


def make_analysis_state(**kw):
    from backend.models.schemas import AnalysisState
    return AnalysisState(id="sess-cap", name="Test", user_id="u1",
                         initial_instruction="analyze", **kw)


def make_workflow_state(analysis_state):
    return {
        "analysis_state": analysis_state,
        "current_round": 1,
        "planning_stage": "full",
        "plan_approved": True,
        "messages": [],
        "should_continue": True,
        "error": None,
        "failed_node": None,
        "skip_nodes": [],
        "node_retry_count": {},
        "node_errors": {},
    }


def make_workspace_file(root: Path, rel: str, content: str = "data"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


@pytest.fixture
def workspace_root(tmp_path):
    d = tmp_path / "workspace"
    d.mkdir()
    make_workspace_file(d, "figures/plot.png", "png-data")
    make_workspace_file(d, "analysis.py", "# code")
    return d


@pytest.fixture
def sandbox():
    mock = MagicMock()
    mock.session_id = "test-kernel"
    mock.runtime_revision = 5
    mock.saved_runtime_revision = 3
    # No live kernel: empty complete runtime is valid for capture-tx tests
    # that focus on workspace/publish boundaries (not runtime registry).
    mock.km = None
    mock.runtime_dependency_tracker = None
    mock.execute = AsyncMock(return_value=MagicMock(
        success=True,
        output="__RUNTIME_GLOBALS_BEGIN__\n{}\n__RUNTIME_GLOBALS_END__",
        error=None,
    ))
    mock.await_pending_saves = AsyncMock()
    import asyncio
    mock._exec_lock = asyncio.Lock()
    return mock


# ---- Helper: create a capture request --------------------------------------

def make_capture_request(session_id="sess-cap", reason="manual",
                         capsule=None, **kw):
    from backend.services.snapshot_v2.models import (
        SnapshotCaptureRequest, IdleCapsule,
    )
    if capsule is None:
        capsule = IdleCapsule(graph_version="2.0")
    return SnapshotCaptureRequest(
        session_id=session_id,
        reason=reason,
        capsule=capsule,
        **kw,
    )


def make_paths(tmp_path):
    """Helper to get a FakePaths from a tmp_path."""
    from backend.services.snapshot_v2.paths import SnapshotV2Paths

    class FakePaths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"

        def ensure_dirs(self):
            for d in [self.root, self.objects, self.staging,
                      self.journals, self.integrity, self.committed]:
                d.mkdir(parents=True, exist_ok=True)

    p = FakePaths()
    p.ensure_dirs()
    return p


# ---- Capture Transaction Boundary Tests ------------------------------------

class TestCaptureTransactionBoundary:

    @pytest.fixture(autouse=True)
    def _seed_session(self):
        from backend.state import sessions
        sessions["sess-cap"] = make_analysis_state()
        yield
        sessions.pop("sess-cap", None)

    @pytest.mark.asyncio
    async def test_successful_capture_commits_manifest(self, v2_paths, sandbox, workspace_root):
        """A full successful capture produces a committed manifest and returns
        a record without a failure."""
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.services.snapshot_v2.models import IdleCapsule

        req = make_capture_request()
        service = SnapshotService("sess-cap", v2_paths)
        record = await service.capture(
            req,
            sandbox=sandbox,
            workspace_root=str(workspace_root),
            workflow_state={"kernel_present": False},
        )

        # No failure
        assert record.failure is None, f"Unexpected failure: {record.failure}"

        # Manifest should be committed
        committed_manifest_path = v2_paths.committed / f"{record.snapshot_id}.json"
        assert committed_manifest_path.exists(), "Committed manifest missing"

        # Signature should exist
        sig_path = v2_paths.integrity / f"{record.snapshot_id}.sig"
        assert sig_path.exists(), "Signature missing"

        # Manifest should be loadable and valid
        from backend.services.snapshot_v2.object_store import ManifestStore
        mstore = ManifestStore(v2_paths)
        loaded = mstore.load(record.snapshot_id)
        assert loaded.session_id == "sess-cap"
        assert loaded.continuation.kind == "idle"

        # Workspace entries should be present (scoped)
        paths = {e.path for e in loaded.workspace}
        assert "figures/plot.png" in paths
        assert loaded.analysis_state_ref.size > 0
        assert loaded.notebook_ref.size > 0

    @pytest.mark.asyncio
    async def test_failure_before_workspace_returns_failure_no_manifest(
        self, v2_paths, sandbox, workspace_root,
    ):
        """Failure before workspace capture returns a SnapshotRecordV2 with failure
        set and no committed manifest."""
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.services.snapshot_v2.object_store import ManifestStore

        req = make_capture_request()
        service = SnapshotService("sess-cap", v2_paths)

        # Monkey-patch to inject failure before workspace capture
        original = service._capture_workspace
        async def fail_before(*args, **kw):
            raise SnapshotCaptureError("TEST_FAILURE", "injected before workspace")
        service._capture_workspace = fail_before

        record = await service.capture(
            req, sandbox=sandbox, workspace_root=str(workspace_root),
            workflow_state={"kernel_present": False},
        )

        # Should return a failure record
        assert record.failure is not None, "Expected failure record"
        assert record.failure.code == "TEST_FAILURE"

        # No committed manifest
        mstore = ManifestStore(v2_paths)
        assert len(mstore.list_committed()) == 0, "Expected no committed snapshots"

    @pytest.mark.asyncio
    async def test_failure_after_workspace_before_publish_is_clean(
        self, v2_paths, sandbox, workspace_root,
    ):
        """Failure after workspace/continuation but before manifest publish
        leaves no committed snapshot. The staging directory may contain
        partial artifacts that reconcile() cleans up."""
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.services.snapshot_v2.object_store import ManifestStore

        req = make_capture_request()
        service = SnapshotService("sess-cap", v2_paths)

        # Inject failure during manifest write (just before publish)
        original = service._compute_root_hash
        async def fail_after(*args, **kw):
            raise SnapshotCaptureError("PRE_PUBLISH_FAILURE", "injected before publish")
        service._compute_root_hash = fail_after

        record = await service.capture(
            req, sandbox=sandbox, workspace_root=str(workspace_root),
            workflow_state={"kernel_present": False},
        )

        assert record.failure is not None
        assert record.failure.code == "PRE_PUBLISH_FAILURE"

        # No committed manifest
        mstore = ManifestStore(v2_paths)
        assert len(mstore.list_committed()) == 0

    @pytest.mark.asyncio
    async def test_failure_after_publish_before_session_save(
        self, v2_paths, sandbox, workspace_root,
    ):
        """Failure after manifest publish but before session state save leaves
        a committed manifest. Reconciliation cleans up orphaned staging data."""
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.services.snapshot_v2.object_store import ManifestStore

        req = make_capture_request()
        service = SnapshotService("sess-cap", v2_paths)

        # Inject failure during session state save
        original = service._save_session_state
        async def fail_save(*args, **kw):
            raise SnapshotCaptureError("POST_PUBLISH_FAILURE", "session save failed")
        service._save_session_state = fail_save

        record = await service.capture(
            req, sandbox=sandbox, workspace_root=str(workspace_root),
            workflow_state={"kernel_present": False},
        )

        # Task 7: post-publish session save failure does not uncommit.
        # Capture returns committed=True with session_reconciliation_pending.
        assert record.failure is None
        assert record.committed is True
        assert record.session_reconciliation_pending is True
        assert record.manifest is not None

        # Manifest IS committed despite the failure after publish
        mstore = ManifestStore(v2_paths)
        committed = mstore.list_committed()
        assert len(committed) >= 1

    @pytest.mark.asyncio
    async def test_reconciliation_repairs_staging(self, v2_paths, sandbox, workspace_root):
        """Reconciliation cleans up orphaned staging entries after a failure."""
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.services.snapshot_v2.object_store import ManifestStore, ObjectStore

        # Create some staging artifacts
        staging_file = v2_paths.staging / "orphan.tmp"
        staging_file.write_text("orphan data")
        assert staging_file.exists()

        # Inject failure to ensure partial capture
        req = make_capture_request()
        service = SnapshotService("sess-cap", v2_paths)

        original = service._capture_workspace
        async def fail_before(*args, **kw):
            raise SnapshotCaptureError("PRE_WORKSPACE", "injected")
        service._capture_workspace = fail_before

        await service.capture(
            req, sandbox=sandbox, workspace_root=str(workspace_root),
            workflow_state={"kernel_present": False},
        )

        # Orphan staging file should still exist (failure happened before reconcile)
        # Now run reconcile
        mstore = ManifestStore(v2_paths)
        mstore.reconcile()

        # Orphan staging file should be cleaned up
        assert not staging_file.exists(), "Reconcile should clean up staging orphans"


# ---- SafePoint Tests -------------------------------------------------------

class TestSafePoint:

    @pytest.mark.asyncio
    async def test_acquire_returns_true(self):
        """Acquiring a safe point for a new session returns True."""
        from backend.services.snapshot_v2.safe_point import SafePointManager
        mgr = SafePointManager()
        result = await mgr.acquire("sess-sp-1", {}, owner_token="test-owner")
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_twice_returns_false(self):
        """Acquiring a safe point for an already-paused session returns False."""
        from backend.services.snapshot_v2.safe_point import SafePointManager
        mgr = SafePointManager()
        first = await mgr.acquire("sess-sp-2", {}, owner_token="test-owner")
        assert first is True
        second = await mgr.acquire("sess-sp-2", {}, owner_token="test-owner")
        assert second is False

    @pytest.mark.asyncio
    async def test_release_unblocks(self):
        """Releasing a safe point sets the underlying event."""
        from backend.services.snapshot_v2.safe_point import SafePointManager
        mgr = SafePointManager()
        await mgr.acquire("sess-sp-3", {}, owner_token="test-owner")
        event = mgr._events.get("sess-sp-3")
        assert event is not None
        assert not event.is_set()

        await mgr.release("sess-sp-3", "test-owner")
        assert event.is_set()  # Event was set (not popped)

    @pytest.mark.asyncio
    async def test_release_nonexistent_is_noop(self):
        """Releasing a non-existent session is a no-op."""
        from backend.services.snapshot_v2.safe_point import SafePointManager
        mgr = SafePointManager()
        await mgr.release("does-not-exist", "test-owner")  # Should not raise

    @pytest.mark.asyncio
    async def test_is_paused_true_after_acquire(self):
        """is_paused returns True after acquire, False after release."""
        from backend.services.snapshot_v2.safe_point import SafePointManager
        mgr = SafePointManager()
        assert mgr.is_paused("sess-sp-4") is False
        await mgr.acquire("sess-sp-4", {}, owner_token="test-owner")
        assert mgr.is_paused("sess-sp-4") is True
        await mgr.release("sess-sp-4", "test-owner")
        assert mgr.is_paused("sess-sp-4") is False

    def test_is_paused_nonexistent(self):
        """is_paused returns False for nonexistent sessions."""
        from backend.services.snapshot_v2.safe_point import SafePointManager
        mgr = SafePointManager()
        assert mgr.is_paused("nonexistent") is False

    @pytest.mark.asyncio
    async def test_descriptor_stores_kernel_present(self):
        """acquire stores workflow_state descriptor for kernel_present lookup."""
        from backend.services.snapshot_v2.safe_point import SafePointManager
        mgr = SafePointManager()
        await mgr.acquire("sess-sp-desc", {"kernel_present": False, "checkpoint_dependencies": ["x"]}, owner_token="test-owner")
        desc = mgr.get_descriptor("sess-sp-desc")
        assert desc is not None
        assert desc["kernel_present"] is False
        assert desc["checkpoint_dependencies"] == ["x"]
        await mgr.release("sess-sp-desc", "test-owner")
        assert mgr.get_descriptor("sess-sp-desc") is None


class TestRuntimeCaptureFailClosed:
    """System capture must not publish empty-complete when runtime capture fails."""

    @pytest.fixture(autouse=True)
    def _seed_session(self):
        from backend.state import sessions
        sessions["sess-cap"] = make_analysis_state()
        yield
        sessions.pop("sess-cap", None)

    @pytest.mark.asyncio
    async def test_runtime_capture_error_does_not_publish(
        self, v2_paths, sandbox, workspace_root,
    ):
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.services.snapshot_v2.object_store import ManifestStore

        # Live kernel + empty globals → NO_RUNTIME_VARIABLES
        sandbox.km = MagicMock()
        req = make_capture_request()
        service = SnapshotService("sess-cap", v2_paths)
        record = await service.capture(
            req, sandbox=sandbox, workspace_root=str(workspace_root),
            workflow_state={"kernel_present": True},
        )
        assert record.failure is not None
        assert "NO_RUNTIME_VARIABLES" in str(record.failure.code)
        mstore = ManifestStore(v2_paths)
        assert len(mstore.list_committed()) == 0

    @pytest.mark.asyncio
    async def test_checkpoint_deps_from_workflow_state_fail_closed(
        self, v2_paths, sandbox, workspace_root,
    ):
        """checkpoint_dependencies from workflow_state enter required set."""
        from backend.services.snapshot_v2.service import SnapshotService

        sandbox.km = MagicMock()
        sandbox.execute = AsyncMock(return_value=MagicMock(
            success=True,
            output="__RUNTIME_GLOBALS_BEGIN__\n{}\n__RUNTIME_GLOBALS_END__",
            error=None,
        ))
        req = make_capture_request()
        service = SnapshotService("sess-cap", v2_paths)
        record = await service.capture(
            req,
            sandbox=sandbox,
            workspace_root=str(workspace_root),
            workflow_state={
                "kernel_present": True,
                "checkpoint_dependencies": ["selected_genes"],
            },
        )
        assert record.failure is not None
        # Declared via checkpoint but missing from kernel → unpersistable or NO_RUNTIME
        assert record.failure.code in (
            "required_runtime_unpersistable",
            "NO_RUNTIME_VARIABLES",
        ) or "required_runtime" in record.failure.code or "selected_genes" in record.failure.message

    @pytest.mark.asyncio
    async def test_kernel_present_false_from_descriptor_allows_empty(
        self, v2_paths, sandbox, workspace_root,
    ):
        from backend.services.snapshot_v2.service import SnapshotService

        # km present but descriptor proves kernel_present=False
        sandbox.km = MagicMock()
        req = make_capture_request()
        service = SnapshotService("sess-cap", v2_paths)
        record = await service.capture(
            req,
            sandbox=sandbox,
            workspace_root=str(workspace_root),
            workflow_state={"kernel_present": False},
        )
        assert record.failure is None
        assert record.manifest.runtime.status == "complete"
        assert record.manifest.runtime.dependencies.required == {}

    @pytest.mark.asyncio
    async def test_rehydrate_tracker_from_sandbox_state(
        self, v2_paths, sandbox, workspace_root,
    ):
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.services.snapshot_v2.runtime_registry import RuntimeDependencyTracker
        from backend.state import sessions

        sandbox.km = MagicMock()
        sandbox.runtime_dependency_tracker = None
        state = make_analysis_state()
        state.sandbox_state = {
            "runtime_dependency_tracker_v2": {
                "created": ["embedding"],
                "required": ["embedding"],
            },
            "runtime_dependencies_v2": ["embedding"],
        }
        sessions["sess-cap"] = state
        try:
            sandbox.execute = AsyncMock(return_value=MagicMock(
                success=True,
                output="__RUNTIME_GLOBALS_BEGIN__\n{}\n__RUNTIME_GLOBALS_END__",
                error=None,
            ))
            req = make_capture_request()
            service = SnapshotService("sess-cap", v2_paths)
            record = await service.capture(
                req, sandbox=sandbox, workspace_root=str(workspace_root),
            )
            # `embedding` is an advisory declared dependency (rehydrated from
            # sandbox_state) that is absent from kernel globals. It is soft-skipped
            # (recorded as an incidental skip) rather than failing closed — only
            # continuation-critical checkpoint dependencies fail closed when absent.
            assert record.failure is None
            skip_names = {
                s.name for s in record.manifest.runtime.dependencies.incidental_skips
            }
            assert "embedding" in skip_names
            # Tracker rehydrated onto sandbox (proves sandbox_state rehydration ran).
            assert sandbox.runtime_dependency_tracker is not None
            assert "embedding" in sandbox.runtime_dependency_tracker.required_outputs
        finally:
            sessions.pop("sess-cap", None)


# ---- Task 3: complete content capture --------------------------------------

class TestCaptureRequiredContent:

    @pytest.fixture
    def scoped_workspace(self, tmp_path):
        d = tmp_path / "workspace"
        d.mkdir()
        make_workspace_file(d, "root.txt", "root-data")
        make_workspace_file(d, "figures/a.png", "png-a")
        make_workspace_file(d, "tables/a.csv", "csv-a")
        make_workspace_file(d, "reproducibility/run.json", '{"ok":true}')
        make_workspace_file(d, "scratch/private.txt", "secret")
        return d

    @pytest.fixture
    def analysis_state(self):
        return make_analysis_state()

    @pytest.mark.asyncio
    async def test_capture_commits_all_required_refs_and_preserves_scope(
        self, v2_paths, sandbox, scoped_workspace, analysis_state, tmp_path,
    ):
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import sessions
        import json

        sessions["sess-cap"] = analysis_state
        try:
            states = tmp_path / "states"
            states.mkdir()
            nb = states / "sess-cap.ipynb"
            nb.write_text(json.dumps({"cells": [], "nbformat": 4, "nbformat_minor": 5,
                                      "metadata": {}}))

            req = make_capture_request()
            service = SnapshotService("sess-cap", v2_paths)

            with patch(
                "backend.services.snapshot_v2.service.get_session_states",
                return_value=states,
            ):
                record = await service.capture_required_snapshot(
                    analysis_state,
                    req,
                    sandbox,
                    {"kernel_present": False},
                    workspace_root=str(scoped_workspace),
                )

            assert record.failure is None, f"Unexpected failure: {record.failure}"
            assert record.manifest.analysis_state_ref
            assert record.manifest.analysis_state_ref.size > 0
            assert record.manifest.notebook_ref
            assert record.manifest.notebook_ref.size > 0
            assert record.manifest.runtime.status == "complete"
            assert {e.path for e in record.manifest.workspace} == {
                "root.txt", "figures/a.png", "tables/a.csv", "reproducibility/run.json",
            }
            assert "scratch/private.txt" not in {e.path for e in record.manifest.workspace}

            store = service.object_store
            store.verify(record.manifest.analysis_state_ref)
            store.verify(record.manifest.notebook_ref)
            for we in record.manifest.workspace:
                from backend.services.snapshot_v2.models import ObjectRef
                store.verify(
                    ObjectRef(
                        sha256=we.object_hash, size=we.size, media_type="application/octet-stream",
                    )
                )
        finally:
            sessions.pop("sess-cap", None)

    @pytest.mark.asyncio
    async def test_physical_added_bytes_only_for_new_hashes(
        self, v2_paths, sandbox, scoped_workspace, analysis_state, tmp_path,
    ):
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import sessions
        import json

        sessions["sess-cap"] = analysis_state
        try:
            states = tmp_path / "states"
            states.mkdir()
            (states / "sess-cap.ipynb").write_text(
                json.dumps({"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}})
            )
            service = SnapshotService("sess-cap", v2_paths)
            with patch(
                "backend.services.snapshot_v2.service.get_session_states",
                return_value=states,
            ):
                first = await service.capture_required_snapshot(
                    analysis_state, make_capture_request(), sandbox,
                    {"kernel_present": False},
                    workspace_root=str(scoped_workspace),
                )
                assert first.failure is None
                first_physical = first.manifest.physical_added_bytes
                assert first_physical > 0

                second = await service.capture_required_snapshot(
                    analysis_state, make_capture_request(), sandbox,
                    {"kernel_present": False},
                    workspace_root=str(scoped_workspace),
                )
                assert second.failure is None
                # Second capture reuses identical content for notebook/workspace
                # (state may differ slightly); physical should be strictly less
                # than logical when any hashes already exist.
                assert second.manifest.physical_added_bytes <= second.manifest.logical_size_bytes
                assert second.manifest.physical_added_bytes < first_physical or (
                    second.manifest.physical_added_bytes < second.manifest.logical_size_bytes
                )
        finally:
            sessions.pop("sess-cap", None)

    @pytest.mark.asyncio
    async def test_mark_runtime_committed_after_publish(
        self, v2_paths, sandbox, scoped_workspace, analysis_state, tmp_path,
    ):
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import sessions
        import json

        sessions["sess-cap"] = analysis_state
        sandbox.runtime_revision = 7
        sandbox.saved_runtime_revision = 3
        try:
            states = tmp_path / "states"
            states.mkdir()
            (states / "sess-cap.ipynb").write_text(
                json.dumps({"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}})
            )
            service = SnapshotService("sess-cap", v2_paths)
            with patch(
                "backend.services.snapshot_v2.service.get_session_states",
                return_value=states,
            ):
                record = await service.capture_required_snapshot(
                    analysis_state, make_capture_request(), sandbox,
                    {"kernel_present": False},
                    workspace_root=str(scoped_workspace),
                )
            assert record.failure is None
            assert sandbox.saved_runtime_revision == 7
            ss = analysis_state.sandbox_state or {}
            assert ss.get("saved_runtime_revision") == 7
        finally:
            sessions.pop("sess-cap", None)

    @pytest.mark.asyncio
    async def test_mark_runtime_committed_uses_frozen_revision(
        self, v2_paths, sandbox, scoped_workspace, analysis_state, tmp_path,
    ):
        """After lock release, concurrent execute must not change committed rev."""
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import sessions
        import json

        sessions["sess-cap"] = analysis_state
        sandbox.runtime_revision = 7
        sandbox.saved_runtime_revision = 3

        real_compute = None

        async def bump_then_compute(manifest):
            # Simulate concurrent user execute after exec lock released
            sandbox.runtime_revision = 99
            return await real_compute(manifest)

        try:
            states = tmp_path / "states"
            states.mkdir()
            (states / "sess-cap.ipynb").write_text(
                json.dumps({"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}})
            )
            service = SnapshotService("sess-cap", v2_paths)
            real_compute = service._compute_root_hash
            service._compute_root_hash = bump_then_compute
            with patch(
                "backend.services.snapshot_v2.service.get_session_states",
                return_value=states,
            ):
                record = await service.capture_required_snapshot(
                    analysis_state, make_capture_request(), sandbox,
                    {"kernel_present": False},
                    workspace_root=str(scoped_workspace),
                )
            assert record.failure is None
            assert sandbox.runtime_revision == 99
            assert sandbox.saved_runtime_revision == 7
            ss = analysis_state.sandbox_state or {}
            assert ss.get("saved_runtime_revision") == 7
        finally:
            sessions.pop("sess-cap", None)

    @pytest.mark.asyncio
    async def test_runtime_capture_uses_unlocked_when_exec_lock_held(
        self, v2_paths, sandbox, scoped_workspace, analysis_state, tmp_path,
    ):
        """Live-kernel capture under _exec_lock must call _execute_unlocked."""
        import asyncio
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
        from backend.state import sessions
        import json

        sessions["sess-cap"] = analysis_state
        sandbox.km = MagicMock()
        sandbox._exec_lock = asyncio.Lock()
        unlocked_calls: list[str] = []
        execute_calls: list[str] = []

        async def tracking_unlocked(code, save_to_history=True, skip_safety_check=False):
            unlocked_calls.append(code[:40])
            return MagicMock(
                success=True,
                output="__RUNTIME_GLOBALS_BEGIN__\n{}\n__RUNTIME_GLOBALS_END__",
                error=None,
            )

        async def tracking_execute(code, save_to_history=True, skip_safety_check=False):
            execute_calls.append(code[:40])
            if sandbox._exec_lock.locked():
                raise AssertionError(
                    "sandbox.execute re-entered while _exec_lock held"
                )
            return MagicMock(
                success=True,
                output="__RUNTIME_GLOBALS_BEGIN__\n{}\n__RUNTIME_GLOBALS_END__",
                error=None,
            )

        sandbox._execute_unlocked = tracking_unlocked
        sandbox.execute = tracking_execute
        try:
            states = tmp_path / "states"
            states.mkdir()
            (states / "sess-cap.ipynb").write_text(
                json.dumps({"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}})
            )
            service = SnapshotService("sess-cap", v2_paths)
            with patch(
                "backend.services.snapshot_v2.service.get_session_states",
                return_value=states,
            ):
                with pytest.raises(RequiredSnapshotBlocked) as exc:
                    await service.capture_required_snapshot(
                        analysis_state, make_capture_request(), sandbox,
                        {"kernel_present": True},
                        workspace_root=str(scoped_workspace),
                    )
            # Live empty globals fail closed — but must have used unlocked path
            assert unlocked_calls, "expected _execute_unlocked under held lock"
            assert not execute_calls, "execute must not be used while lock held"
            assert "NO_RUNTIME_VARIABLES" in str(exc.value.code)
        finally:
            sessions.pop("sess-cap", None)


    @pytest.mark.asyncio
    async def test_equal_revision_reuse_for_pre_restore(
        self, v2_paths, sandbox, analysis_state, tmp_path,
    ):
        """Equal revisions reuse exact runtime generation for pre_restore."""
        from backend.services.snapshot_v2.pre_restore import (
            PreRestoreCapture, RestoreCaptureContext,
        )
        from backend.services.snapshot_v2.object_store import ObjectStore, ManifestStore
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.services.snapshot_v2.models import (
            IdleCapsule, SnapshotRecordV2, RuntimeGenerationManifest,
            runtime_manifest_from_generation,
        )
        from backend.tests.snapshot_v2_fixtures import v2_manifest

        object_store = ObjectStore(v2_paths)
        manifest_store = ManifestStore(v2_paths)
        service = SnapshotService("sess-cap", v2_paths)

        adata_ref = object_store.put_bytes(b"reuse-adata", "application/x-hdf5").ref
        gen = RuntimeGenerationManifest(
            capture_command="runtime_capture_v2",
            capture_timestamp="2025-01-01T00:00:00Z",
            kernel_id="test-kernel",
            captured_adata_vars=["adata"],
            captured_adata_refs={"adata": adata_ref},
        )
        rt_manifest = runtime_manifest_from_generation(gen, status="complete")
        prev = v2_manifest(
            session_id="sess-cap",
            snapshot_id="snap-prev-reuse",
            captured_at="2025-01-01T00:00:00Z",
            capsule=IdleCapsule(graph_version="2.0"),
            runtime=rt_manifest,
        )
        staging = manifest_store.write_staging(prev, prefix="test_")
        prev = manifest_store.publish_staging(staging)

        sandbox.runtime_revision = 5
        sandbox.saved_runtime_revision = 5
        analysis_state.sandbox_state = {
            "runtime_revision": 5,
            "saved_runtime_revision": 5,
        }
        ctx = RestoreCaptureContext(
            session_id="sess-cap",
            paths=v2_paths,
            sandbox=sandbox,
            workspace_root=None,
            workflow_state={},
            analysis_state=analysis_state,
            prev_manifest=prev,
        )
        capture = PreRestoreCapture(service, object_store, manifest_store)
        captured_requests = []

        async def tracking_locked(req, **kw):
            captured_requests.append(req)
            return SnapshotRecordV2(
                snapshot_id="snap-pre",
                manifest=v2_manifest(
                    session_id="sess-cap",
                    snapshot_id="snap-pre",
                    captured_at="2025-01-01T00:00:01Z",
                    capsule=IdleCapsule(graph_version="2.0"),
                    runtime=req.runtime,
                ),
                failure=None,
            )

        with patch.object(service, "_capture_locked", tracking_locked):
            await capture.capture_locked(ctx)

        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert req.runtime is not None
        assert req.runtime.status == "complete"
        assert "adata" in req.runtime.adata_refs
        assert req.runtime.adata_refs["adata"].sha256 == adata_ref.sha256

    @pytest.mark.asyncio
    async def test_dirty_runtime_after_execution(
        self, v2_paths, sandbox, analysis_state, tmp_path,
    ):
        """Later successful execution makes dirty; dirty runtime freshly captured."""
        from backend.services.snapshot_v2.pre_restore import (
            PreRestoreCapture, RestoreCaptureContext,
        )
        from backend.services.snapshot_v2.object_store import ObjectStore, ManifestStore
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.services.snapshot_v2.models import (
            IdleCapsule, SnapshotRecordV2, RuntimeGenerationManifest,
            runtime_manifest_from_generation,
        )
        from backend.tests.snapshot_v2_fixtures import v2_manifest

        object_store = ObjectStore(v2_paths)
        manifest_store = ManifestStore(v2_paths)
        service = SnapshotService("sess-cap", v2_paths)

        adata_ref = object_store.put_bytes(b"dirty-adata", "application/x-hdf5").ref
        gen = RuntimeGenerationManifest(
            capture_command="runtime_capture_v2",
            capture_timestamp="2025-01-01T00:00:00Z",
            kernel_id="test-kernel",
            captured_adata_vars=["adata"],
            captured_adata_refs={"adata": adata_ref},
        )
        rt_manifest = runtime_manifest_from_generation(gen, status="complete")
        prev = v2_manifest(
            session_id="sess-cap",
            snapshot_id="snap-prev-dirty",
            captured_at="2025-01-01T00:00:00Z",
            capsule=IdleCapsule(graph_version="2.0"),
            runtime=rt_manifest,
        )
        staging = manifest_store.write_staging(prev, prefix="test_")
        prev = manifest_store.publish_staging(staging)

        # Equal then successful execution dirties runtime
        sandbox.runtime_revision = 5
        sandbox.saved_runtime_revision = 5
        analysis_state.sandbox_state = {
            "runtime_revision": 5,
            "saved_runtime_revision": 5,
        }
        # Simulate successful user execute
        sandbox.runtime_revision = 6
        analysis_state.sandbox_state = {
            "runtime_revision": 6,
            "saved_runtime_revision": 5,
        }
        assert sandbox.runtime_revision != sandbox.saved_runtime_revision

        ctx = RestoreCaptureContext(
            session_id="sess-cap",
            paths=v2_paths,
            sandbox=sandbox,
            workspace_root=None,
            workflow_state={},
            analysis_state=analysis_state,
            prev_manifest=prev,
        )
        capture = PreRestoreCapture(service, object_store, manifest_store)
        assert not capture._can_reuse_runtime(ctx)

        captured_requests = []

        async def tracking_locked(req, **kw):
            captured_requests.append(req)
            return SnapshotRecordV2(
                snapshot_id="snap-fresh",
                manifest=v2_manifest(
                    session_id="sess-cap",
                    snapshot_id="snap-fresh",
                    captured_at="2025-01-01T00:00:01Z",
                    capsule=IdleCapsule(graph_version="2.0"),
                ),
                failure=None,
            )

        with patch.object(service, "_capture_locked", tracking_locked):
            await capture.capture_locked(ctx)

        assert len(captured_requests) == 1
        assert captured_requests[0].runtime is None

    @pytest.mark.asyncio
    async def test_session_save_failure_after_publish_leaves_committed(
        self, v2_paths, sandbox, scoped_workspace, analysis_state, tmp_path,
    ):
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.services.snapshot_v2.object_store import ManifestStore
        from backend.state import sessions
        import json

        sessions["sess-cap"] = analysis_state
        try:
            states = tmp_path / "states"
            states.mkdir()
            (states / "sess-cap.ipynb").write_text(
                json.dumps({"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}})
            )
            service = SnapshotService("sess-cap", v2_paths)

            async def fail_save(*args, **kw):
                raise SnapshotCaptureError("SESSION_SAVE_FAILED", "injected")

            service._save_session_state = fail_save
            with patch(
                "backend.services.snapshot_v2.service.get_session_states",
                return_value=states,
            ):
                record = await service.capture(
                    make_capture_request(),
                    sandbox=sandbox,
                    workspace_root=str(scoped_workspace),
                    workflow_state={"kernel_present": False},
                )
            # Failure after publish is reported, but manifest is committed
            mstore = ManifestStore(v2_paths)
            committed = mstore.list_committed()
            assert len(committed) >= 1
        finally:
            sessions.pop("sess-cap", None)

    @pytest.mark.asyncio
    async def test_lock_order_never_acquires_session_lock_while_holding_exec_lock(
        self, v2_paths, sandbox, scoped_workspace, analysis_state, tmp_path,
    ):
        """Capture must not call get_session_lock while holding sandbox._exec_lock."""
        import asyncio
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import sessions
        import json

        sessions["sess-cap"] = analysis_state
        sandbox._exec_lock = asyncio.Lock()
        sandbox.await_pending_saves = AsyncMock()
        try:
            states = tmp_path / "states"
            states.mkdir()
            (states / "sess-cap.ipynb").write_text(
                json.dumps({"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}})
            )
            service = SnapshotService("sess-cap", v2_paths)
            violations: list[str] = []

            real_get_session_lock = None
            from backend import state as state_mod
            real_get_session_lock = state_mod.get_session_lock

            def guarded_get_session_lock(session_id):
                if sandbox._exec_lock.locked():
                    violations.append(
                        f"get_session_lock({session_id}) while _exec_lock held"
                    )
                return real_get_session_lock(session_id)

            with patch(
                "backend.services.snapshot_v2.service.get_session_states",
                return_value=states,
            ), patch(
                "backend.state.get_session_lock",
                side_effect=guarded_get_session_lock,
            ), patch(
                "backend.services.snapshot_v2.service.get_session_lock",
                side_effect=guarded_get_session_lock,
            ):
                record = await service.capture_required_snapshot(
                    analysis_state, make_capture_request(), sandbox,
                    {"kernel_present": False},
                    workspace_root=str(scoped_workspace),
                )
            assert violations == [], f"Lock order violations: {violations}"
            assert record.failure is None
        finally:
            sessions.pop("sess-cap", None)

    def test_no_catch_and_return_none_on_snapshot_capture_error(self):
        """Structural: capture helpers must not swallow SnapshotCaptureError into None."""
        import inspect
        import ast
        from backend.services.snapshot_v2 import service as svc_mod

        source = inspect.getsource(svc_mod.SnapshotService)
        tree = ast.parse(source)
        # Walk methods that capture runtime
        bad = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in ("_capture_runtime", "_capture_impl", "capture", "capture_required_snapshot"):
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.ExceptHandler):
                    continue
                # Handler that catches SnapshotCaptureError
                catches_sce = False
                if child.type is None:
                    continue
                type_src = ast.unparse(child.type) if hasattr(ast, "unparse") else ""
                if "SnapshotCaptureError" in type_src or type_src in ("Exception", "BaseException"):
                    # Look for return None in handler body
                    for stmt in child.body:
                        if isinstance(stmt, ast.Return) and (
                            stmt.value is None
                            or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None)
                        ):
                            bad.append((node.name, type_src))
                        # Also reject assign runtime = None then return without re-raise
        assert bad == [], f"Capture helpers catch-and-return-None: {bad}"

    def test_quota_config_injected(self, v2_paths):
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.config._config import default_config

        service = SnapshotService("sess-cap", v2_paths)
        assert service.session_quota_bytes == default_config.snapshot_v2_session_quota_bytes
        assert (
            service.pre_restore_quota_grace_bytes
            == default_config.snapshot_v2_pre_restore_quota_grace_bytes
        )
        assert service.free_space_reserve_bytes == default_config.snapshot_v2_free_space_reserve_bytes

    @pytest.mark.asyncio
    async def test_check_quota_when_v2_root_dir_missing(self, tmp_path):
        """Regression: real captures never create the ``v2/`` root dir before the
        quota check, so ``shutil.disk_usage`` must not choke on a missing path.

        Reproduces DISK_USAGE_CHECK_FAILED that blocked the workflow at
        bioinformatics start (v0.33.x)."""
        from types import SimpleNamespace
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.services.snapshot_v2.paths import SnapshotV2Paths

        # Use the REAL paths class (frozen dataclass, no ensure_dirs); the v2
        # root deliberately does not exist on disk.
        real_paths = SnapshotV2Paths(
            root=tmp_path / "v2",
            objects=tmp_path / "objects",
            staging=tmp_path / "staging",
            journals=tmp_path / "journals",
            integrity=tmp_path / "integrity",
            committed=tmp_path / "committed",
        )
        assert not real_paths.root.exists()

        service = SnapshotService("sess-cap", real_paths)
        # _check_quota only reads physical_added_bytes off the manifest.
        manifest = SimpleNamespace(physical_added_bytes=0)

        # Must not raise DISK_USAGE_CHECK_FAILED just because v2/ is absent.
        await service._check_quota(manifest)

    @pytest.mark.asyncio
    async def test_rebuilds_snapshots_list_from_committed_manifests(
        self, v2_paths, sandbox, scoped_workspace, analysis_state, tmp_path,
    ):
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import sessions
        import json

        sessions["sess-cap"] = analysis_state
        analysis_state.snapshots = []
        try:
            states = tmp_path / "states"
            states.mkdir()
            (states / "sess-cap.ipynb").write_text(
                json.dumps({"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}})
            )
            service = SnapshotService("sess-cap", v2_paths)
            with patch(
                "backend.services.snapshot_v2.service.get_session_states",
                return_value=states,
            ):
                record = await service.capture_required_snapshot(
                    analysis_state, make_capture_request(), sandbox,
                    {"kernel_present": False},
                    workspace_root=str(scoped_workspace),
                )
            assert record.failure is None
            assert any(s.id == record.snapshot_id for s in analysis_state.snapshots)
            rebuilt = service.rebuild_snapshots_from_manifests()
            assert any(s.id == record.snapshot_id for s in rebuilt)
        finally:
            sessions.pop("sess-cap", None)


class TestCaptureTransactionEvidenceReuse:
    """Capture must reuse put evidence and not re-verify transaction-local writes."""

    @pytest.mark.asyncio
    async def test_capture_skips_verify_for_transaction_local_puts(
        self, v2_paths, sandbox, workspace_root, tmp_path, monkeypatch,
    ):
        from unittest.mock import Mock

        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import sessions
        import json

        analysis_state = make_analysis_state()
        sessions["sess-cap"] = analysis_state
        try:
            states = tmp_path / "states"
            states.mkdir()
            (states / "sess-cap.ipynb").write_text(
                json.dumps({
                    "cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {},
                })
            )
            service = SnapshotService("sess-cap", v2_paths)
            verify = Mock(side_effect=AssertionError("transaction-local reread"))
            monkeypatch.setattr(service.object_store, "verify", verify)
            with patch(
                "backend.services.snapshot_v2.service.get_session_states",
                return_value=states,
            ):
                record = await service.capture(
                    make_capture_request(),
                    sandbox=sandbox,
                    workspace_root=str(workspace_root),
                    analysis_state=analysis_state,
                )
            assert record.failure is None, f"Unexpected failure: {record.failure}"
            # External-ref verify must not be required for pure transaction puts.
            assert verify.call_count == 0
        finally:
            sessions.pop("sess-cap", None)
