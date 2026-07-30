"""Regression tests for the second pre-restore edge-case pass (0.36.12).

One test per defect, named for the failure it prevents rather than for the
function it calls. Every one of these was verified to fail against the
unfixed source.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.snapshot_v2.models import (
    IdleCapsule,
    RuntimeGenerationManifest,
    RuntimeManifest,
    SnapshotFailureRecord,
    SnapshotRecordV2,
    runtime_manifest_from_generation,
)
from backend.tests.snapshot_v2_fixtures import v2_manifest

SESSION_ID = "sess-edge2"


# ── shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def v2_paths(tmp_path):
    class _Paths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"

        def ensure_dirs(self):
            for d in (
                self.root, self.objects, self.staging,
                self.journals, self.integrity, self.committed,
            ):
                d.mkdir(parents=True, exist_ok=True)

    paths = _Paths()
    paths.ensure_dirs()
    return paths


@pytest.fixture
def object_store(v2_paths):
    from backend.services.snapshot_v2.object_store import ObjectStore
    return ObjectStore(v2_paths)


@pytest.fixture
def manifest_store(v2_paths):
    from backend.services.snapshot_v2.object_store import ManifestStore
    return ManifestStore(v2_paths)


@pytest.fixture
def service(v2_paths):
    from backend.services.snapshot_v2.service import SnapshotService
    return SnapshotService(SESSION_ID, v2_paths)


@pytest.fixture
def sandbox():
    """Stand-in for a live kernel the capture would have to introspect."""
    mock = MagicMock()
    mock.session_id = "test-kernel"
    mock.execute = AsyncMock()
    return mock


def make_analysis_state(**kw):
    from backend.models.schemas import AnalysisState
    defaults = dict(
        sandbox_state={"runtime_revision": 5, "saved_runtime_revision": 5},
    )
    defaults.update(kw)
    return AnalysisState(
        id=SESSION_ID, name="Test", user_id="u1",
        initial_instruction="analyze", **defaults,
    )


def make_runtime_generation(**kw) -> RuntimeManifest:
    defaults = dict(
        capture_command="runtime_capture_v2",
        capture_timestamp="2025-01-01T00:00:00Z",
        kernel_id="test-kernel",
    )
    defaults.update(kw)
    return runtime_manifest_from_generation(
        RuntimeGenerationManifest(**defaults), status="complete",
    )


def commit_manifest(mstore, manifest):
    staging = mstore.write_staging(manifest, prefix="test_")
    return mstore.publish_staging(staging)


def _record(snapshot_id: str, *, failure_code: str | None = None):
    manifest = v2_manifest(
        session_id=SESSION_ID,
        snapshot_id=snapshot_id,
        captured_at="2025-01-01T00:00:00Z",
        capsule=IdleCapsule(graph_version="2.0"),
    )
    failure = None
    if failure_code is not None:
        failure = SnapshotFailureRecord(
            snapshot_id=snapshot_id,
            session_id=SESSION_ID,
            failed_at="2025-01-01T00:00:00Z",
            code=failure_code,
            message="injected capture failure",
        )
    return SnapshotRecordV2(
        snapshot_id=snapshot_id, manifest=manifest, failure=failure,
    )


# ── 1. The discard_current escape hatch has an exit (F06 / F14) ──────────────
#
# PRE_RESTORE_UNAVAILABLE steers the client into discard_current precisely
# because the live kernel could not be serialized. If the audit bundle re-drives
# that same capture, discard fails with AUDIT_FAILED — a code the panel offers
# no action for — and Retry fails the same way, so no API path completes the
# restore.


def _kernel_hostile_capture(calls: list):
    """Stand in for ``_capture_impl``'s live-kernel branch.

    A request with no ``runtime`` and a live sandbox drives ``_capture_runtime``;
    ``_capture_locked`` turns its SnapshotCaptureError into ``record.failure``.
    Anything that declares its runtime up front never reaches the kernel.
    """

    async def _capture(req, **kw):
        calls.append((req, kw))
        snapshot_id = kw.get("snapshot_id_override") or "snap-captured"
        if kw.get("sandbox") is not None and req.runtime is None:
            return _record(snapshot_id, failure_code="RUNTIME_CAPTURE_FAILED")
        return _record(snapshot_id)

    return _capture


@pytest.mark.asyncio
async def test_discard_audit_survives_the_kernel_that_blocked_pre_restore(
    v2_paths, object_store, manifest_store, service, sandbox,
):
    from backend.services.snapshot_v2.pre_restore import (
        AuditBundle, RestoreCaptureContext,
    )

    ctx = RestoreCaptureContext(
        session_id=SESSION_ID,
        paths=v2_paths,
        sandbox=sandbox,
        workspace_root=None,
        workflow_state={},
        analysis_state=make_analysis_state(),
    )
    calls: list = []

    with patch.object(
        service, "_capture_locked_for_restore", _kernel_hostile_capture(calls),
    ):
        ref = await AuditBundle(service).capture_locked(
            ctx,
            failures=[{
                "code": "DISCARD_CURRENT",
                "message": "User confirmed discard of current state",
            }],
        )

    assert ref is not None, (
        "the audit bundle re-ran the live-kernel capture that made pre_restore "
        "fail, so discard_current — the only exit PRE_RESTORE_UNAVAILABLE "
        "offers — returned AUDIT_FAILED and the restore was unreachable"
    )
    last_req, last_kw = calls[-1]
    assert last_kw["sandbox"] is None
    assert last_req.runtime is not None and last_req.runtime.status == "complete", (
        "sandbox=None alone is not enough: with no runtime on the request the "
        "capture takes the empty-runtime eligibility gate, which refuses "
        "reason='audit_bundle' and trades AUDIT_CAPTURE_FAILED for "
        "LIVE_SANDBOX_REQUIRED"
    )
    codes = [f.get("code") for f in ref.failure.details.get("failures", [])]
    assert "AUDIT_RUNTIME_NOT_CAPTURED" in codes, (
        "the bundle is the only record of the discarded state, so a bundle "
        "published without the kernel must say why"
    )


@pytest.mark.asyncio
async def test_audit_bundle_still_records_a_kernel_that_can_be_captured(
    v2_paths, object_store, manifest_store, service, sandbox,
):
    """The runtime is dropped only when it provably cannot be captured."""
    from backend.services.snapshot_v2.pre_restore import (
        AuditBundle, RestoreCaptureContext,
    )

    ctx = RestoreCaptureContext(
        session_id=SESSION_ID,
        paths=v2_paths,
        sandbox=sandbox,
        workspace_root=None,
        workflow_state={},
        analysis_state=make_analysis_state(),
    )
    calls: list = []

    async def _capture(req, **kw):
        calls.append((req, kw))
        return _record(kw.get("snapshot_id_override") or "snap-captured")

    with patch.object(service, "_capture_locked_for_restore", _capture):
        ref = await AuditBundle(service).capture_locked(ctx, failures=[])

    assert len(calls) == 1
    req, kw = calls[0]
    assert kw["sandbox"] is sandbox
    assert req.runtime is None
    assert ref.failure.details.get("failures") == []


@pytest.mark.asyncio
async def test_audit_bundle_failure_that_is_not_the_kernel_still_raises(
    v2_paths, object_store, manifest_store, service, sandbox,
):
    """Quota / workspace / state failures recur without a kernel and must not
    be laundered into a published bundle by the runtime-free retry."""
    from backend.services.snapshot_v2.errors import SnapshotCaptureError
    from backend.services.snapshot_v2.pre_restore import (
        AuditBundle, RestoreCaptureContext,
    )

    ctx = RestoreCaptureContext(
        session_id=SESSION_ID,
        paths=v2_paths,
        sandbox=sandbox,
        workspace_root=None,
        workflow_state={},
        analysis_state=make_analysis_state(),
    )

    async def _capture(req, **kw):
        return _record(
            kw.get("snapshot_id_override") or "snap-captured",
            failure_code="snapshot_quota_exceeded",
        )

    with patch.object(service, "_capture_locked_for_restore", _capture):
        with pytest.raises(SnapshotCaptureError, match="Audit bundle capture failed"):
            await AuditBundle(service).capture_locked(ctx, failures=[])


# ── 2. pre_restore freezes the live session, not the lineage head (F07) ──────
#
# The revision gate proves the kernel is unchanged. It proves nothing about the
# analysis state, notebook or round, and a session advances a round without ever
# dirtying the kernel (the plan-review gate of round N+1).


def _prev_head_with_runtime(object_store, manifest_store, *, round_number: int):
    adata_ref = object_store.put_bytes(b"anndata content", "application/x-hdf5").ref
    prev = v2_manifest(
        session_id=SESSION_ID,
        snapshot_id="snap-lineage-head",
        captured_at="2025-01-01T00:00:00Z",
        capsule=IdleCapsule(graph_version="2.0"),
        round_number=round_number,
        runtime=make_runtime_generation(
            captured_adata_vars=["adata"],
            captured_adata_refs={"adata": adata_ref},
        ),
    )
    commit_manifest(manifest_store, prev)
    return prev


async def _capture_pre_restore(service, capture, ctx):
    from backend.services.snapshot_v2.pre_restore import PreRestoreCapture

    requests: list = []

    async def tracking(req, **kw):
        requests.append(req)
        return _record("snap-pre-restore")

    with patch.object(service, "_capture_locked", tracking):
        await capture.capture_locked(ctx)
    assert len(requests) == 1
    return requests[0]


@pytest.mark.asyncio
async def test_pre_restore_reuse_does_not_redeclare_the_lineage_head_state(
    v2_paths, object_store, manifest_store, service, sandbox,
):
    from backend.models.schemas import TaskStatus
    from backend.services.snapshot_v2.pre_restore import (
        PreRestoreCapture, RestoreCaptureContext,
    )

    prev = _prev_head_with_runtime(object_store, manifest_store, round_number=1)
    ctx = RestoreCaptureContext(
        session_id=SESSION_ID,
        paths=v2_paths,
        sandbox=sandbox,
        workspace_root=None,
        workflow_state={"current_round": 2},
        analysis_state=make_analysis_state(
            current_round=2, status=TaskStatus.WAITING_PLAN_REVIEW,
        ),
        prev_manifest=prev,
    )
    capture = PreRestoreCapture(service, object_store, manifest_store)
    assert capture._can_reuse_runtime(ctx)

    req = await _capture_pre_restore(service, capture, ctx)

    assert req.runtime is not None and req.runtime.adata_refs == prev.runtime.adata_refs, (
        "the revision gate still qualifies the kernel for reuse"
    )
    assert req.analysis_state_ref is None, (
        "the revision gate proves nothing about the analysis state; aliasing "
        "the lineage head's state makes the pre_restore snapshot silently "
        "discard everything since that head when it is restored"
    )
    assert req.notebook_ref is None
    assert req.checkpoint_ref is None


@pytest.mark.asyncio
async def test_pre_restore_round_matches_the_capsule_it_froze(
    v2_paths, object_store, manifest_store, service, sandbox,
):
    """manifest.round_number and the frozen capsule must name the same round.

    ``activate`` writes the capsule's ``current_round`` onto the restored state
    and ``live_verify`` compares it against ``manifest.round_number``; a split
    between them fails the restore with LIVE_VERIFY_FAILED and leaves the
    session recovery-blocked.
    """
    from backend.models.schemas import TaskStatus
    from backend.services.snapshot_v2.pre_restore import (
        PreRestoreCapture, RestoreCaptureContext,
    )

    prev = _prev_head_with_runtime(object_store, manifest_store, round_number=1)
    ctx = RestoreCaptureContext(
        session_id=SESSION_ID,
        paths=v2_paths,
        sandbox=sandbox,
        workspace_root=None,
        workflow_state={"current_round": 2},
        analysis_state=make_analysis_state(
            current_round=2, status=TaskStatus.WAITING_PLAN_REVIEW,
        ),
        prev_manifest=prev,
    )
    capture = PreRestoreCapture(service, object_store, manifest_store)

    req = await _capture_pre_restore(service, capture, ctx)

    assert req.capsule.kind == "semantic_gate"
    payload = json.loads(object_store.get_bytes(req.capsule.payload_ref.sha256))
    assert req.round_number == 2
    assert payload["current_round"] == req.round_number, (
        "the capsule is frozen from the live session, so a round_number taken "
        "from the lineage head makes the manifest internally inconsistent"
    )


# ── 3. waiting_input is not an unambiguous gate (F30) ────────────────────────


def test_agent_disabled_park_is_not_frozen_as_the_round_gate(object_store):
    from backend.models.schemas import TaskStatus
    from backend.services.snapshot_v2.continuation import ContinuationRegistry

    parked = MagicMock()
    parked.status = TaskStatus.WAITING_INPUT
    parked.wait_input_reason = "agent_disabled"
    parked.agent_disabled_info = {
        "agent": "bioinformatics", "node_name": "bioinformatics_analysis",
    }

    capsule = ContinuationRegistry(object_store).freeze_dynamic(
        {
            "kernel_present": False,
            "current_round": 3,
            "inactive": True,
            "status": "waiting_input",
        },
        analysis_state=parked,
        workflow_state={"current_round": 3},
    )

    assert capsule.kind == "idle", (
        f"the disabled-agent gate blocks inside a node, but it was frozen as "
        f"{getattr(capsule, 'gate', None)!r}; restoring that capsule rewrites "
        f"wait_input_reason to the gate name, so the disabled-agent state_sync "
        f"is never re-emitted and /resume drives the graph forward from a "
        f"round-end gate the session was never at"
    )


def test_plain_waiting_input_still_freezes_the_round_gate(object_store):
    from backend.models.schemas import TaskStatus
    from backend.services.snapshot_v2.continuation import ContinuationRegistry

    parked = MagicMock()
    parked.status = TaskStatus.WAITING_INPUT
    parked.wait_input_reason = None

    capsule = ContinuationRegistry(object_store).freeze_dynamic(
        {
            "kernel_present": False,
            "current_round": 3,
            "inactive": True,
            "status": "waiting_input",
        },
        analysis_state=parked,
        workflow_state={"current_round": 3},
    )

    assert capsule.kind == "semantic_gate"
    assert capsule.gate == "round_wait"
