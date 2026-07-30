"""Producer fail-closed tests for Snapshot V2 required capture.

Task 4: every required producer must block forward progress on capture failure,
use a process-wide SafePointManager, and never offer Skip.
"""
from __future__ import annotations
from backend.tests.split_facade import source_for_path

import ast
from pathlib import Path
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Files that are producers of snapshots (they contain snapshot triggers)
PRODUCER_FILES: List[str] = [
    "backend/workflow/core.py",
    "backend/routes/chat.py",
    "backend/workflow/callbacks.py",
    "backend/workflow/reporting.py",
]

# V1 SnapshotManager imports / calls that should NOT appear in producer files
V1_CALL_NAMES = frozenset({
    "create_snapshot",
    "create_and_register_snapshot",
})


# Certifies Snapshot V2-enabled behaviour. The operator gate ships off, so pin it
# on here instead of inheriting it from the environment: these assertions describe
# the enabled lifecycle and are meaningless (and were failing) with it off.
pytestmark = pytest.mark.requires_snapshot_v2

def _read_source(path: str) -> str:
    return source_for_path(path)


def _has_call_to_method(tree: ast.Module, class_name: str, method_name: str) -> bool:
    """Check if any call of the form `class_name.method_name` exists."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == method_name:
                base = node.func.value
                if isinstance(base, ast.Name) and base.id == class_name:
                    return True
                if isinstance(base, ast.Attribute):
                    if base.attr == class_name:
                        return True
    return False


@pytest.mark.parametrize("filepath", PRODUCER_FILES)
def test_no_v1_snapshot_manager_calls(filepath: str):
    """No producer file directly calls SnapshotManager.create_snapshot."""
    source = _read_source(filepath)
    tree = ast.parse(source)

    for method in V1_CALL_NAMES:
        if _has_call_to_method(tree, "SnapshotManager", method):
            pytest.fail(
                f"{filepath} still calls SnapshotManager.{method}() — "
                f"must use SnapshotService.capture_required_snapshot() instead"
            )


def test_core_py_has_v2_capture():
    """core.py must call capture (capture_required_snapshot or capture)."""
    source = _read_source("backend/workflow/core.py")
    tree = ast.parse(source)

    has_v2_call = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("capture", "capture_required_snapshot"):
                has_v2_call = True
                break

    assert has_v2_call, (
        "core.py must call SnapshotService.capture_required_snapshot() "
        "in _create_auto_snapshot"
    )


def test_chat_py_no_v1_and_has_v2():
    """chat.py must use V2 capture and not V1."""
    source = _read_source("backend/routes/chat.py")
    tree = ast.parse(source)

    for method in V1_CALL_NAMES:
        if _has_call_to_method(tree, "SnapshotManager", method):
            pytest.fail(
                f"chat.py still calls SnapshotManager.{method}() — "
                f"must use SnapshotService.capture_required_snapshot() instead"
            )

    has_v2 = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("capture", "capture_required_snapshot"):
                has_v2 = True
                break

    assert has_v2, "chat.py must call SnapshotService capture methods"


def test_callbacks_py_no_v1():
    """callbacks.py must not call SnapshotManager methods."""
    source = _read_source("backend/workflow/callbacks.py")
    tree = ast.parse(source)

    for method in V1_CALL_NAMES:
        if _has_call_to_method(tree, "SnapshotManager", method):
            pytest.fail(
                f"callbacks.py still calls SnapshotManager.{method}() — "
                f"should be removed per migration plan"
            )


def test_reporting_py_no_v1_and_has_v2():
    """reporting.py must use V2 capture, not V1."""
    source = _read_source("backend/workflow/reporting.py")
    tree = ast.parse(source)

    for method in V1_CALL_NAMES:
        if _has_call_to_method(tree, "SnapshotManager", method):
            pytest.fail(
                f"reporting.py still calls SnapshotManager.{method}() — "
                f"must use SnapshotService.capture_required_snapshot() instead"
            )

    has_v2 = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("capture", "capture_required_snapshot"):
                has_v2 = True
                break
    assert has_v2, "reporting.py must call SnapshotService capture methods"


# ── Task 4 behavioral tests ──────────────────────────────────────────────────


def _make_analysis_state(**kw):
    from backend.models.schemas import AnalysisState

    return AnalysisState(
        id=kw.pop("id", "sess-prod"),
        name="Producer Test",
        user_id="u1",
        initial_instruction="analyze",
        **kw,
    )


def _committed_record(snapshot_id: str = "snap-ok-001", session_id: str = "sess-prod"):
    from datetime import datetime, timezone

    from backend.services.snapshot_v2.models import (
        IdleCapsule,
        SnapshotManifestV2,
        SnapshotRecordV2,
        empty_complete_runtime,
    )
    from backend.services.snapshot_v2.models import ObjectRef

    stub = ObjectRef(sha256="a" * 64, size=1, media_type="application/json")
    return SnapshotRecordV2(
        snapshot_id=snapshot_id,
        manifest=SnapshotManifestV2(
            session_id=session_id,
            snapshot_id=snapshot_id,
            name=snapshot_id,
            phase="post_report",
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            analysis_state_ref=stub,
            notebook_ref=stub,
            runtime=empty_complete_runtime(),
            continuation=IdleCapsule(graph_version="2.0"),
            logical_size_bytes=1,
            physical_added_bytes=0,
            root_hash="b" * 64,
        ),
        failure=None,
    )


@pytest.mark.asyncio
async def test_required_boundary_failure_blocks_next_node():
    """Required capture failure must raise RequiredSnapshotBlocked and set no-Skip gate."""
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked

    workflow = MagicMock()
    workflow.session_id = "sess-prod"
    workflow.sandbox = MagicMock()
    workflow._current_graph_state = {"current_round": 1}
    state = _make_analysis_state()
    state.last_snapshot_id = "snap-parent"
    state.restore_branch_id = "branch-a"
    state.restore_branch_label = "Branch A"
    workflow.state = state

    from backend.workflow.core import SingleCellAnalysisWorkflow

    workflow._create_auto_snapshot = SingleCellAnalysisWorkflow._create_auto_snapshot.__get__(
        workflow, SingleCellAnalysisWorkflow
    )

    async def _fail(state_arg, request, *a, **kw):
        assert request.parent_id == "snap-parent"
        assert request.branch_id == "branch-a"
        assert request.branch_label == "Branch A"
        state_arg.wait_input_reason = "snapshot_failure"
        state_arg.snapshot_failure = {
            "code": "required_runtime_unpersistable",
            "message": "results",
            "allowed_actions": ["retry", "manage", "terminate"],
        }
        raise RequiredSnapshotBlocked(
            "required_runtime_unpersistable",
            "results",
        )

    workflow.snapshot_service = MagicMock()
    workflow.snapshot_service.capture_required_snapshot = AsyncMock(side_effect=_fail)
    # Provide a real-enough object_store so freeze can run if called; prefer
    # short-circuit by patching freeze to return a dummy capsule.
    from backend.services.snapshot_v2.models import SemanticGateCapsule, ObjectRef
    stub_ref = ObjectRef(sha256="c" * 64, size=1, media_type="application/json")
    dummy_capsule = SemanticGateCapsule(
        graph_version="2.0",
        gate="round_wait",
        payload_ref=stub_ref,
    )
    with patch(
        "backend.services.snapshot_v2.continuation.ContinuationRegistry.freeze",
        return_value=dummy_capsule,
    ):
        with patch(
            "backend.config.get_session_workspace",
            return_value=Path("/tmp"),
        ):
            with pytest.raises(RequiredSnapshotBlocked):
                await workflow._create_auto_snapshot(state, "pre_round_wait")

    assert state.wait_input_reason == "snapshot_failure"
    assert "skip" not in state.snapshot_failure["allowed_actions"]
    assert set(state.snapshot_failure["allowed_actions"]) == {
        "retry", "manage", "terminate",
    }


def test_normal_v2_producers_use_shared_lineage_resolution():
    """Workflow, chat, and report captures must share one lineage policy."""
    for relative_path in (
        "backend/workflow/core.py",
        "backend/routes/chat.py",
        "backend/workflow/reporting.py",
    ):
        source = _read_source(relative_path)
        assert "capture_lineage_from_state" in source, relative_path


def test_capture_lineage_defaults_and_restored_branch_values():
    from backend.services.snapshot_v2.lineage import capture_lineage_from_state

    state = _make_analysis_state()
    assert capture_lineage_from_state(state) == {
        "parent_id": None,
        "branch_id": "main",
        "branch_label": "main",
    }

    state.last_snapshot_id = "snap-parent"
    state.restore_branch_id = " branch-a "
    state.restore_branch_label = " Branch A "
    assert capture_lineage_from_state(state) == {
        "parent_id": "snap-parent",
        "branch_id": "branch-a",
        "branch_label": "Branch A",
    }


def test_workflow_snapshot_state_accessor_returns_deep_copy():
    from backend.workflow.core import SingleCellAnalysisWorkflow

    workflow = MagicMock()
    workflow._current_graph_state = {
        "node_retry_count": {"pi_synthesis": 2},
        "skip_nodes": ["literature_research"],
    }
    workflow.get_snapshot_workflow_state = (
        SingleCellAnalysisWorkflow.get_snapshot_workflow_state.__get__(
            workflow, SingleCellAnalysisWorkflow,
        )
    )

    frozen = workflow.get_snapshot_workflow_state()
    frozen["node_retry_count"]["pi_synthesis"] = 99
    frozen["skip_nodes"].append("bioinformatics_analysis")

    assert workflow._current_graph_state["node_retry_count"]["pi_synthesis"] == 2
    assert workflow._current_graph_state["skip_nodes"] == ["literature_research"]


def test_restore_service_prefers_explicit_workflow_snapshot_accessor():
    from backend.services.snapshot_v2.service import _workflow_state_for_snapshot

    class Workflow:
        _current_graph_state = {"source": "private"}

        def get_snapshot_workflow_state(self):
            return {"source": "accessor", "nested": {"value": 1}}

    workflow = Workflow()
    frozen = _workflow_state_for_snapshot(workflow)
    frozen["nested"]["value"] = 2

    assert frozen["source"] == "accessor"
    assert workflow.get_snapshot_workflow_state()["nested"]["value"] == 1


def test_active_workflow_requires_snapshot_state_or_safe_point_descriptor():
    from backend.services.snapshot_v2.service import (
        _has_trusted_workflow_snapshot_context,
    )

    workflow = object()
    assert _has_trusted_workflow_snapshot_context(None, {}, {}) is True
    assert _has_trusted_workflow_snapshot_context(workflow, {"next_node": "pi"}, {})
    assert _has_trusted_workflow_snapshot_context(
        workflow, {}, {"kernel_present": False},
    )
    assert _has_trusted_workflow_snapshot_context(workflow, {}, {}) is False


@pytest.mark.asyncio
async def test_run_boundary_catches_required_snapshot_blocked():
    """core.run must catch RequiredSnapshotBlocked and return waiting state, not FAILED."""
    from backend.models.schemas import TaskStatus
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.workflow.core import SingleCellAnalysisWorkflow

    state = _make_analysis_state()
    state.status = TaskStatus.RUNNING
    state.wait_input_reason = "snapshot_failure"
    state.snapshot_failure = {
        "code": "quota",
        "message": "full",
        "allowed_actions": ["retry", "manage", "terminate"],
    }

    wf = MagicMock(spec=SingleCellAnalysisWorkflow)
    wf.session_id = "sess-prod"
    wf._set_status = staticmethod(SingleCellAnalysisWorkflow._set_status).__get__(
        wf, SingleCellAnalysisWorkflow
    )
    # staticmethod: call with (state, status) only
    def _set_status(state, status):
        SingleCellAnalysisWorkflow._set_status(state, status)
    wf._set_status = _set_status
    wf._send_message = AsyncMock()
    wf._send_state_sync = AsyncMock()
    wf.initialize = AsyncMock()
    wf.compiled_graph = MagicMock()
    wf.compiled_graph.ainvoke = AsyncMock(
        side_effect=RequiredSnapshotBlocked("quota", "full")
    )
    wf._make_initial_workflow_state = MagicMock(
        return_value={"analysis_state": state}
    )
    wf.resume_restored_wait_state = AsyncMock()

    run = SingleCellAnalysisWorkflow.run.__get__(wf, SingleCellAnalysisWorkflow)

    with patch(
        "backend.workflow.core.is_forward_resume_blocked",
        return_value=False,
    ):
        result = await run(state, "thread-1")

    assert result is state
    assert state.wait_input_reason == "snapshot_failure"
    assert state.status == TaskStatus.WAITING_INPUT
    assert state.status != TaskStatus.FAILED


@pytest.mark.asyncio
async def test_post_report_retry_only_retries_capture():
    """retry_pending_required_snapshot re-enters real ReportingMixin capture path only."""
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.services.snapshot_v2.models import SnapshotRecordV2
    from backend.workflow.reporting import ReportingMixin

    workflow = MagicMock()
    workflow.session_id = "sess-prod"
    workflow.sandbox = None
    workflow.report_generator = AsyncMock()
    state = _make_analysis_state()
    state.pending_required_snapshot = {
        "report_generation_id": "report-7",
        "phase": "post_report",
        "attempts": 1,
    }
    workflow.state = state
    workflow._current_analysis_state = state

    # Bind real ReportingMixin methods (no fallback stub path).
    workflow.retry_pending_required_snapshot = (
        ReportingMixin.retry_pending_required_snapshot.__get__(
            workflow, type(workflow)
        )
    )
    workflow._capture_required_snapshot = (
        ReportingMixin._capture_required_snapshot.__get__(
            workflow, type(workflow)
        )
    )
    workflow._capture_post_report_snapshot = (
        ReportingMixin._capture_post_report_snapshot.__get__(
            workflow, type(workflow)
        )
    )

    committed = _committed_record()
    call_n = {"n": 0}

    async def _capture(state_arg, request, *a, **kw):
        call_n["n"] += 1
        if call_n["n"] == 1:
            raise RequiredSnapshotBlocked("quota", "full")
        return committed

    workflow.snapshot_service = MagicMock()
    workflow.snapshot_service.object_store = MagicMock()
    workflow.snapshot_service.capture_required_snapshot = AsyncMock(side_effect=_capture)
    # No-sandbox path pre-acquires a trusted safe point; mock must look free.
    sp = MagicMock()
    sp.is_paused.return_value = False
    sp.has_pending_request.return_value = False
    sp.is_eligibility_trusted.return_value = False
    sp.acquire = AsyncMock(return_value=True)
    sp.cancel_safe_point_request = AsyncMock(return_value=True)
    workflow.snapshot_service.safe_point = sp

    with patch(
        "backend.services.snapshot_v2.continuation.ContinuationRegistry.freeze",
        return_value=_committed_record().manifest.continuation,
    ):
        with patch(
            "backend.config.get_session_workspace",
            return_value=Path("/tmp"),
        ):
            # First retry attempt fails closed
            with pytest.raises(RequiredSnapshotBlocked):
                await workflow.retry_pending_required_snapshot()
            # Second attempt succeeds via real mixin path
            await workflow.retry_pending_required_snapshot()

    workflow.report_generator.assert_not_awaited()
    assert state.pending_required_snapshot is None
    assert call_n["n"] == 2
    assert sp.acquire.call_count >= 1
    assert sp.cancel_safe_point_request.call_count >= 1


@pytest.mark.asyncio
async def test_same_post_report_idempotency_key_returns_original_manifest(tmp_path):
    """Idempotent capture by (session_id, phase, idempotency_key) returns original."""
    from backend.services.snapshot_v2.models import (
        IdleCapsule,
        SnapshotCaptureRequest,
    )
    from backend.services.snapshot_v2.paths import SnapshotV2Paths
    from backend.services.snapshot_v2.service import SnapshotService

    class FakePaths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"

        def ensure_dirs(self):
            for d in [
                self.root, self.objects, self.staging,
                self.journals, self.integrity, self.committed,
            ]:
                d.mkdir(parents=True, exist_ok=True)

    paths = FakePaths()
    paths.ensure_dirs()
    session_id = "sess-idy"
    service = SnapshotService(session_id, paths)
    state = _make_analysis_state(id=session_id)

    workspace = tmp_path / "ws"
    workspace.mkdir()

    def _req(key: str):
        return SnapshotCaptureRequest(
            session_id=session_id,
            reason="post_report",
            phase="post_report",
            capsule=IdleCapsule(graph_version="2.0"),
            idempotency_key=key,
        )

    # Trusted idle descriptor (producer path); raw workflow_state alone is not enough.
    await service.safe_point.acquire(
        session_id,
        {"kernel_present": False, "status": "completed"},
        eligibility_trusted=True,
        owner_token="test-owner",
    )
    try:
        with patch.object(service, "_get_session_state", new=AsyncMock(return_value=state)):
            with patch.object(service, "_save_session_state", new=AsyncMock()):
                with patch.object(service, "_send_state_sync", new=AsyncMock()):
                    first = await service.capture_required_snapshot(
                        state, _req("report-7"), sandbox=None,
                        workflow_state={"kernel_present": False},
                        workspace_root=str(workspace),
                        safe_point_owner_token="test-owner",
                    )
                    second = await service.capture_required_snapshot(
                        state, _req("report-7"), sandbox=None,
                        workflow_state={"kernel_present": False},
                        workspace_root=str(workspace),
                        safe_point_owner_token="test-owner",
                    )
    finally:
        await service.safe_point.release(session_id, "test-owner")

    assert first.failure is None
    assert second.failure is None
    assert second.snapshot_id == first.snapshot_id
    assert len(service.list_snapshots()) == 1


@pytest.mark.asyncio
async def test_post_report_no_sandbox_acquires_trusted_safe_point(tmp_path):
    """ReportingMixin no-sandbox path must trust empty-complete (post-teardown).

    After WS teardown the rebuilt workflow has sandbox=None. Capture must
    pre-acquire a trusted no-kernel descriptor so empty-complete is allowed;
    untrusted _create_safe_point seeding alone must not be sufficient.
    """
    from backend.models.schemas import TaskStatus
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.services.snapshot_v2.service import SnapshotService
    from backend.state import sessions, workflows
    from backend.workflow.reporting import ReportingMixin

    session_id = "sess-post-report-no-sandbox"
    workspace = tmp_path / "ws"
    workspace.mkdir()

    class FakePaths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"

        def ensure_dirs(self):
            for d in [
                self.root, self.objects, self.staging,
                self.journals, self.integrity, self.committed,
            ]:
                d.mkdir(parents=True, exist_ok=True)

    paths = FakePaths()
    paths.ensure_dirs()
    service = SnapshotService(session_id, paths)
    state = _make_analysis_state(id=session_id)
    state.status = TaskStatus.GENERATING_REPORT
    state.current_round = 1
    state.pending_required_snapshot = {
        "report_generation_id": "report-ns-1",
        "phase": "post_report",
        "attempts": 1,
    }

    workflow = MagicMock()
    workflow.session_id = session_id
    workflow.sandbox = None  # post-teardown rebuild
    workflow.snapshot_service = service
    workflow.state = state
    workflow._current_analysis_state = state
    workflow._capture_post_report_snapshot = (
        ReportingMixin._capture_post_report_snapshot.__get__(
            workflow, type(workflow),
        )
    )
    workflow._capture_required_snapshot = (
        ReportingMixin._capture_required_snapshot.__get__(
            workflow, type(workflow),
        )
    )
    workflow.retry_pending_required_snapshot = (
        ReportingMixin.retry_pending_required_snapshot.__get__(
            workflow, type(workflow),
        )
    )

    sessions[session_id] = state
    workflows[session_id] = workflow  # active ownership like rebuild path
    try:
        with patch(
            "backend.config.get_session_workspace",
            return_value=workspace,
        ):
            with patch.object(
                service, "_get_session_state", new=AsyncMock(return_value=state),
            ):
                with patch.object(service, "_save_session_state", new=AsyncMock()):
                    with patch.object(service, "_send_state_sync", new=AsyncMock()):
                        # Must NOT raise LIVE_SANDBOX_REQUIRED solely due to
                        # untrusted safe-point seed.
                        record = await workflow._capture_post_report_snapshot(
                            state, "report-ns-1",
                        )
        assert record.failure is None
        assert record.manifest.runtime.status == "complete"
        assert record.manifest.runtime.dependencies.required == {}
        # Producer-owned hold must be released after capture.
        assert not service.safe_point.is_paused(session_id)
        assert not service.safe_point.is_eligibility_trusted(session_id)
    except RequiredSnapshotBlocked as exc:
        pytest.fail(
            f"no-sandbox post_report capture must authorize empty-complete; "
            f"got {exc}"
        )
    finally:
        sessions.pop(session_id, None)
        workflows.pop(session_id, None)
        await service.safe_point.release(session_id, "test-owner")


@pytest.mark.asyncio
async def test_post_report_dead_sandbox_uses_empty_complete_path(tmp_path):
    """A sandbox object without a live kernel is equivalent to no sandbox.

    Regression: report-only resume can retain the LocalJupyterSandbox object
    after its kernel has gone away. Passing that object into SnapshotService
    re-enters runtime capture and treats historical runtime dependencies as
    values that must be read from a nonexistent kernel.
    """
    from backend.models.schemas import TaskStatus
    from backend.services.snapshot_v2.models import TerminalCapsule
    from backend.services.snapshot_v2.safe_point import SafePointManager
    from backend.workflow.reporting import ReportingMixin

    session_id = "sess-post-report-dead-sandbox"
    state = _make_analysis_state(id=session_id)
    state.status = TaskStatus.GENERATING_REPORT
    state.current_round = 2

    service = MagicMock()
    service.safe_point = SafePointManager()
    service.object_store = MagicMock()
    service.capture_required_snapshot = AsyncMock(
        return_value=_committed_record(session_id=session_id),
    )

    workflow = MagicMock()
    workflow.session_id = session_id
    workflow.sandbox = MagicMock(km=None)
    workflow.snapshot_service = service
    workflow._capture_post_report_snapshot = (
        ReportingMixin._capture_post_report_snapshot.__get__(
            workflow, type(workflow),
        )
    )

    with patch(
        "backend.services.snapshot_v2.continuation.ContinuationRegistry.freeze",
        return_value=TerminalCapsule(graph_version="2.0", status="completed"),
    ):
        with patch(
            "backend.config.get_session_workspace",
            return_value=tmp_path,
        ):
            await workflow._capture_post_report_snapshot(state, "report-dead-kernel")

    capture_kwargs = service.capture_required_snapshot.await_args.kwargs
    assert capture_kwargs["sandbox"] is None
    assert capture_kwargs["workflow_state"]["kernel_present"] is False
    assert not service.safe_point.is_paused(session_id)


@pytest.mark.asyncio
async def test_snapshot_failure_not_converted_to_node_retry():
    """RequiredSnapshotBlocked must propagate through with_retry, not become node failure."""
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.workflow.retry import with_retry

    class Host:
        session_id = "sess-prod"

        async def _send_message(self, *a, **kw):
            pass

        @with_retry("pi_synthesis")
        async def _node(self, state):
            raise RequiredSnapshotBlocked("quota", "full")

    host = Host()
    with pytest.raises(RequiredSnapshotBlocked):
        await host._node({"analysis_state": _make_analysis_state(), "skip_nodes": []})


@pytest.mark.asyncio


@pytest.mark.asyncio
async def test_generate_report_catches_required_snapshot_blocked():
    """generate_report must hold WAITING_INPUT on RequiredSnapshotBlocked, not FAILED."""
    from backend.models.schemas import TaskStatus
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    from backend.workflow.core import SingleCellAnalysisWorkflow

    state = _make_analysis_state()
    state.status = TaskStatus.GENERATING_REPORT
    state.current_round = 1

    wf = MagicMock(spec=SingleCellAnalysisWorkflow)
    wf.session_id = "sess-prod"
    wf._state = None
    wf.compiled_graph = MagicMock()
    wf.compiled_graph.aget_state = AsyncMock(return_value=None)
    wf.initialize = AsyncMock()
    wf._send_message = AsyncMock()
    wf._send_state_sync = AsyncMock()
    wf._generate_final_report = AsyncMock(
        side_effect=RequiredSnapshotBlocked("quota", "full")
    )
    wf._generate_graphic_abstract = AsyncMock()
    wf._generate_detailed_report = AsyncMock()

    def _set_status(st, status):
        SingleCellAnalysisWorkflow._set_status(st, status)

    wf._set_status = _set_status

    generate_report = SingleCellAnalysisWorkflow.generate_report.__get__(
        wf, SingleCellAnalysisWorkflow
    )
    result = await generate_report(state, "thread-1")

    assert result is state
    assert state.status == TaskStatus.WAITING_INPUT
    assert state.status != TaskStatus.FAILED
    assert state.wait_input_reason == "snapshot_failure"
    assert state.snapshot_failure is not None
    assert state.snapshot_failure["code"] == "quota"
    # post_report is skippable: reports are already generated, so Skip is offered
    # in addition to retry/manage/terminate.
    assert set(state.snapshot_failure["allowed_actions"]) == {
        "retry", "skip", "manage", "terminate",
    }
    wf._generate_graphic_abstract.assert_not_awaited()
    wf._generate_detailed_report.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_report_entry_guard_skips_regen_when_snapshot_blocked():
    """Blocked/pending post-report snapshot must not re-run report generators."""
    from backend.models.schemas import TaskStatus
    from backend.workflow.core import SingleCellAnalysisWorkflow

    state = _make_analysis_state()
    state.status = TaskStatus.GENERATING_REPORT
    state.current_round = 2
    state.wait_input_reason = "snapshot_failure"
    state.snapshot_failure = {
        "code": "quota",
        "message": "full",
        "allowed_actions": ["retry", "manage", "terminate"],
    }
    state.pending_required_snapshot = {
        "report_generation_id": "report-7",
        "phase": "post_report",
        "attempts": 1,
    }
    state.final_report_requested = True

    wf = MagicMock(spec=SingleCellAnalysisWorkflow)
    wf.session_id = "sess-prod"
    wf._state = None
    wf.compiled_graph = MagicMock()
    wf.compiled_graph.aget_state = AsyncMock(
        side_effect=AssertionError("must not load checkpoint when blocked")
    )
    wf.initialize = AsyncMock(
        side_effect=AssertionError("must not initialize when blocked")
    )
    wf._send_message = AsyncMock()
    wf._send_state_sync = AsyncMock()
    wf._generate_final_report = AsyncMock(
        side_effect=AssertionError("must not re-run report generators")
    )
    wf._generate_graphic_abstract = AsyncMock(
        side_effect=AssertionError("must not re-run report generators")
    )
    wf._generate_detailed_report = AsyncMock(
        side_effect=AssertionError("must not re-run report generators")
    )
    wf._build_graph = MagicMock(
        side_effect=AssertionError("must not build graph when blocked")
    )

    def _set_status(st, status):
        SingleCellAnalysisWorkflow._set_status(st, status)

    wf._set_status = _set_status

    generate_report = SingleCellAnalysisWorkflow.generate_report.__get__(
        wf, SingleCellAnalysisWorkflow
    )
    result = await generate_report(state, "thread-1")

    assert result is state
    assert state.status == TaskStatus.WAITING_INPUT
    assert state.wait_input_reason == "snapshot_failure"
    assert state.snapshot_failure is not None
    assert state.pending_required_snapshot is not None
    # Do not clear finalize flag or re-run generators on reconnect hold
    assert state.final_report_requested is True
    wf.initialize.assert_not_awaited()
    wf._generate_final_report.assert_not_awaited()
    wf._generate_graphic_abstract.assert_not_awaited()
    wf._generate_detailed_report.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_report_entry_guard_pending_only_skips_regen():
    """pending_required_snapshot alone (post-report capture retry) must skip regen."""
    from backend.models.schemas import TaskStatus
    from backend.workflow.core import SingleCellAnalysisWorkflow

    state = _make_analysis_state()
    state.status = TaskStatus.GENERATING_REPORT
    state.pending_required_snapshot = {
        "report_generation_id": "report-8",
        "phase": "post_report",
        "attempts": 0,
    }

    wf = MagicMock(spec=SingleCellAnalysisWorkflow)
    wf.session_id = "sess-prod"
    wf._state = None
    wf.compiled_graph = MagicMock()
    wf.initialize = AsyncMock()
    wf._send_message = AsyncMock()
    wf._send_state_sync = AsyncMock()
    wf._generate_final_report = AsyncMock(
        side_effect=AssertionError("must not re-run report generators")
    )
    wf._generate_graphic_abstract = AsyncMock()
    wf._generate_detailed_report = AsyncMock()

    def _set_status(st, status):
        SingleCellAnalysisWorkflow._set_status(st, status)

    wf._set_status = _set_status

    generate_report = SingleCellAnalysisWorkflow.generate_report.__get__(
        wf, SingleCellAnalysisWorkflow
    )
    result = await generate_report(state, "thread-1")

    assert result is state
    assert state.status == TaskStatus.WAITING_INPUT
    assert state.wait_input_reason == "snapshot_failure"
    assert state.pending_required_snapshot is not None
    wf.initialize.assert_not_awaited()
    wf._generate_final_report.assert_not_awaited()


def test_restore_job_manager_uses_process_wide_safe_point():
    """RestoreJobManager must share the process-wide SafePointManager."""
    from backend.services.snapshot_v2.jobs import RestoreJobManager
    from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager

    class FakePaths:
        journals = Path("/tmp")

    mgr = RestoreJobManager(paths=FakePaths(), session_id="s1")
    assert mgr._safe_point is get_snapshot_safe_point_manager()


def test_process_wide_safe_point_manager_is_shared():
    """get_snapshot_safe_point_manager returns one process-wide instance."""
    from backend.services.snapshot_v2.safe_point import (
        SafePointManager,
        get_snapshot_safe_point_manager,
    )

    a = get_snapshot_safe_point_manager()
    b = get_snapshot_safe_point_manager()
    assert a is b
    assert isinstance(a, SafePointManager)

    # SnapshotService must use the shared manager, not an instance-local one.
    from backend.services.snapshot_v2.service import SnapshotService

    class FakePaths:
        root = Path("/tmp")
        objects = Path("/tmp")
        staging = Path("/tmp")
        journals = Path("/tmp")
        integrity = Path("/tmp")
        committed = Path("/tmp")

    s1 = SnapshotService("s1", FakePaths())
    s2 = SnapshotService("s2", FakePaths())
    assert s1.safe_point is s2.safe_point
    assert s1.safe_point is get_snapshot_safe_point_manager()


def test_pre_major_revision_is_plan_revision_gate():
    from backend.services.snapshot_v2.continuation import (
        _PHASE_TO_GATE,
        _PHASE_TO_NEXT_NODE,
        _SEMANTIC_GATE_PHASES,
    )

    assert "pre_major_revision" in _SEMANTIC_GATE_PHASES
    assert _PHASE_TO_GATE["pre_major_revision"] == "plan_revision"
    assert "pre_major_revision" not in _PHASE_TO_NEXT_NODE


def test_pre_execution_full_maps_to_bioinformatics_analysis():
    from backend.services.snapshot_v2.continuation import _PHASE_TO_NEXT_NODE
    from backend.workflow.restore_targets import SNAPSHOT_RESUME_TARGET

    assert SNAPSHOT_RESUME_TARGET["pre_execution_full"]["node"] == "bioinformatics_analysis"
    assert _PHASE_TO_NEXT_NODE["pre_execution_full"] == "bioinformatics_analysis"


def test_post_chat_pre_restore_not_unconditional_idle():
    from backend.services.snapshot_v2 import continuation as cont

    assert "post_chat" not in cont._IDLE_PHASES
    assert "pre_restore" not in cont._IDLE_PHASES


def test_capture_post_chat_snapshot_exported_from_chat():
    """Post-chat producer must expose capture_post_chat_snapshot coordinator."""
    import backend.routes.chat as chat

    assert callable(getattr(chat, "capture_post_chat_snapshot", None))
    assert callable(getattr(chat, "_create_chat_snapshot", None))


def test_resolve_empty_runtime_eligibility_on_service():
    from backend.services.snapshot_v2.service import SnapshotService

    assert hasattr(SnapshotService, "_resolve_empty_runtime_eligibility")


def test_no_skip_in_snapshot_failure_actions():
    from backend.services.snapshot_v2.errors import SNAPSHOT_FAILURE_ACTIONS

    assert "skip" not in SNAPSHOT_FAILURE_ACTIONS
    assert set(SNAPSHOT_FAILURE_ACTIONS) == {"retry", "manage", "terminate"}


def test_is_snapshot_failure_blocked_helper():
    from backend.services.snapshot_v2.errors import is_snapshot_failure_blocked

    state = _make_analysis_state()
    assert is_snapshot_failure_blocked(state) is False
    assert is_snapshot_failure_blocked(None) is False

    state.wait_input_reason = "snapshot_failure"
    assert is_snapshot_failure_blocked(state) is True

    state.wait_input_reason = None
    state.snapshot_failure = {"code": "QUOTA", "message": "full"}
    assert is_snapshot_failure_blocked(state) is True

    state.snapshot_failure = None
    state.wait_input_reason = "llm_failure"
    assert is_snapshot_failure_blocked(state) is False


@pytest.mark.asyncio
async def test_run_entry_guard_when_snapshot_failure_already_present():
    """core.run must hold and return when snapshot_failure is already set."""
    from backend.models.schemas import TaskStatus
    from backend.workflow.core import SingleCellAnalysisWorkflow

    state = _make_analysis_state()
    state.status = TaskStatus.RUNNING
    state.wait_input_reason = "snapshot_failure"
    state.snapshot_failure = {
        "code": "quota",
        "message": "full",
        "allowed_actions": ["retry", "manage", "terminate"],
    }

    wf = MagicMock(spec=SingleCellAnalysisWorkflow)
    wf.session_id = "sess-prod"
    wf.initialize = AsyncMock()
    wf.compiled_graph = MagicMock()
    wf.compiled_graph.ainvoke = AsyncMock(
        side_effect=AssertionError("graph must not start when gate is set")
    )
    wf._make_initial_workflow_state = MagicMock()
    wf._send_message = AsyncMock()
    wf._send_state_sync = AsyncMock()

    def _set_status(st, status):
        SingleCellAnalysisWorkflow._set_status(st, status)

    wf._set_status = _set_status

    run = SingleCellAnalysisWorkflow.run.__get__(wf, SingleCellAnalysisWorkflow)
    with patch(
        "backend.workflow.core.is_forward_resume_blocked",
        return_value=False,
    ):
        result = await run(state, "thread-1")

    assert result is state
    assert state.status == TaskStatus.WAITING_INPUT
    assert state.wait_input_reason == "snapshot_failure"
    wf.compiled_graph.ainvoke.assert_not_awaited()
    wf.initialize.assert_not_awaited()


def test_pre_restore_uses_freeze_dynamic_not_idle_hardcode():
    """PreRestoreCapture must build capsule via freeze/freeze_dynamic."""
    import inspect
    from backend.services.snapshot_v2 import pre_restore as pr

    source = inspect.getsource(pr.PreRestoreCapture)
    assert "freeze" in source
    assert "ContinuationRegistry" in source or "freeze_dynamic" in source
    # Must not unconditionally construct IdleCapsule in capture_locked
    assert "IdleCapsule(graph_version" not in inspect.getsource(
        pr.PreRestoreCapture.capture_locked
    )
