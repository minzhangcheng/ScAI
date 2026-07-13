"""Test pre_restore capture, revision-qualified reuse, and audit bundle.

Tests for:
- Equal signed revisions → reuse runtime generation
- Dirty revision → capture fresh runtime
- Legacy signed pkl → cannot be reused
- Capture failure → leaves workflow resumable
- Discard requires confirmation
- Discard+keep is rejected
- Audit bundle must succeed before mutation
"""

from backend.tests.snapshot_v2_fixtures import v2_manifest
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.snapshot_v2.errors import SnapshotCaptureError, SnapshotIntegrityError
from backend.services.snapshot_v2.models import (
    IdleCapsule,
    ObjectRef,
    RuntimeDependency,
    RuntimeDependencyManifest,
    RuntimeManifest,
    SnapshotManifestV2,
    SnapshotRecordV2,
    runtime_manifest_from_generation,
    RuntimeGenerationManifest,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


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
            for d in [self.root, self.objects, self.staging,
                      self.journals, self.integrity, self.committed]:
                d.mkdir(parents=True, exist_ok=True)

    p = FakePaths()
    p.ensure_dirs()
    return p


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
    return SnapshotService("sess-pre", v2_paths)


@pytest.fixture
def sandbox():
    mock = MagicMock()
    mock.session_id = "test-kernel"
    mock.execute = AsyncMock(return_value=MagicMock(
        success=True,
        output="__RUNTIME_GLOBALS_BEGIN__\n{}\n__RUNTIME_GLOBALS_END__",
        error=None,
    ))
    return mock


def make_analysis_state(**kw):
    from backend.models.schemas import AnalysisState
    defaults = dict(
        sandbox_state={"runtime_revision": 5, "saved_runtime_revision": 5},
    )
    defaults.update(kw)
    return AnalysisState(
        id="sess-pre", name="Test", user_id="u1",
        initial_instruction="analyze", **defaults,
    )


def make_runtime_generation(**kw) -> RuntimeManifest:
    defaults = dict(
        capture_command="runtime_capture_v2",
        capture_timestamp="2025-01-01T00:00:00Z",
        kernel_id="test-kernel",
    )
    defaults.update(kw)
    gen = RuntimeGenerationManifest(**defaults)
    return runtime_manifest_from_generation(gen, status="complete")


def commit_manifest(mstore, manifest: SnapshotManifestV2) -> SnapshotManifestV2:
    """Write and publish a manifest through the ManifestStore (V2 signed)."""
    staging = mstore.write_staging(manifest, prefix="test_")
    return mstore.publish_staging(staging)


def make_object_ref(object_store: "ObjectStore", content: bytes,
                    media_type: str = "application/x-hdf5") -> ObjectRef:
    """Store *content* in the object store and return its ObjectRef."""
    return object_store.put_bytes(content, media_type).ref


# ── Revision-Qualified Reuse Tests ────────────────────────────────────────────


class TestPreRestoreCaptureReuse:

    @pytest.mark.asyncio
    async def test_equal_revisions_reuses_runtime(
        self, v2_paths, object_store, manifest_store, service, sandbox,
    ):
        """When runtime_revision == saved_runtime_revision the runtime
        generation from the previous snapshot is reused."""
        from backend.services.snapshot_v2.pre_restore import (
            PreRestoreCapture, RestoreCaptureContext,
        )

        # Create runtime objects with real content hashes
        adata_ref = make_object_ref(object_store, b"anndata content")
        rt_manifest = make_runtime_generation(
            captured_adata_vars=["adata"],
            captured_adata_refs={"adata": adata_ref},
        )
        prev = v2_manifest(
            session_id="sess-pre",
            snapshot_id="snap-prev-1",
            captured_at="2025-01-01T00:00:00Z",
            capsule=IdleCapsule(graph_version="2.0"),
            runtime=rt_manifest,
        )

        # Commit it (V2 signed) — objects are already stored above
        commit_manifest(manifest_store, prev)

        # Build context with matching revisions
        analysis_state = make_analysis_state(
            sandbox_state={"runtime_revision": 5, "saved_runtime_revision": 5},
        )
        ctx = RestoreCaptureContext(
            session_id="sess-pre",
            paths=v2_paths,
            sandbox=sandbox,
            workspace_root=None,
            workflow_state={},
            analysis_state=analysis_state,
            prev_manifest=prev,
        )

        capture = PreRestoreCapture(service, object_store, manifest_store)

        # Intercept the request going into _capture_locked
        captured_requests = []

        async def tracking_locked(req, **kw):
            captured_requests.append(req)
            return SnapshotRecordV2(
                snapshot_id="snap-captured",
                manifest=v2_manifest(
                    session_id="sess-pre",
                    snapshot_id="snap-captured",
                    captured_at="2025-01-01T00:00:01Z",
                    capsule=IdleCapsule(graph_version="2.0"),
                    runtime=req.runtime,
                    runtime_dependencies=req.runtime_dependencies,
                ),
                failure=None,
            )

        with patch.object(service, '_capture_locked', tracking_locked):
            record = await capture.capture_locked(ctx)

        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert req.runtime is not None, \
            "Expected runtime to be reused (pre-populated in request)"
        assert req.runtime.status == "complete"
        assert "adata" in req.runtime.adata_refs

    @pytest.mark.asyncio
    async def test_dirty_revision_captures_fresh_runtime(
        self, v2_paths, object_store, manifest_store, service, sandbox,
    ):
        """When runtime_revision != saved_runtime_revision fresh runtime
        capture is performed (not reused)."""
        from backend.services.snapshot_v2.pre_restore import (
            PreRestoreCapture, RestoreCaptureContext,
        )

        adata_ref = make_object_ref(object_store, b"more adata")
        rt_manifest = make_runtime_generation(
            captured_adata_vars=["adata"],
            captured_adata_refs={"adata": adata_ref},
        )
        prev = v2_manifest(
            session_id="sess-pre",
            snapshot_id="snap-prev-2",
            captured_at="2025-01-01T00:00:00Z",
            capsule=IdleCapsule(graph_version="2.0"),
            runtime=rt_manifest,
        )
        commit_manifest(manifest_store, prev)

        # Mismatched revisions: runtime=5, saved=3
        analysis_state = make_analysis_state(
            sandbox_state={"runtime_revision": 5, "saved_runtime_revision": 3},
        )
        ctx = RestoreCaptureContext(
            session_id="sess-pre",
            paths=v2_paths,
            sandbox=sandbox,
            workspace_root=None,
            workflow_state={},
            analysis_state=analysis_state,
            prev_manifest=prev,
        )

        capture = PreRestoreCapture(service, object_store, manifest_store)

        assert not capture._can_reuse_runtime(ctx), \
            "Should NOT reuse runtime for dirty revision"

        captured_requests = []

        async def tracking_locked(req, **kw):
            captured_requests.append(req)
            return SnapshotRecordV2(
                snapshot_id="snap-captured",
                manifest=v2_manifest(
                    session_id="sess-pre",
                    snapshot_id="snap-captured",
                    captured_at="2025-01-01T00:00:01Z",
                    capsule=IdleCapsule(graph_version="2.0"),
                ),
                failure=None,
            )

        with patch.object(service, '_capture_locked', tracking_locked):
            record = await capture.capture_locked(ctx)

        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert req.runtime is None, \
            "Expected fresh runtime capture (not pre-populated)"

    @pytest.mark.asyncio
    async def test_legacy_signed_pkl_not_reused(
        self, v2_paths, object_store, manifest_store, service, sandbox,
    ):
        """A previous snapshot without a valid V2 signature cannot have
        its runtime generation reused."""
        from backend.services.snapshot_v2.pre_restore import (
            PreRestoreCapture, RestoreCaptureContext,
        )

        adata_ref = make_object_ref(object_store, b"legacy adata")
        rt_manifest = make_runtime_generation(
            captured_adata_vars=["adata"],
            captured_adata_refs={"adata": adata_ref},
        )
        prev = v2_manifest(
            session_id="sess-pre",
            snapshot_id="snap-prev-3",
            captured_at="2025-01-01T00:00:00Z",
            capsule=IdleCapsule(graph_version="2.0"),
            runtime=rt_manifest,
        )

        # Write the manifest to committed/ but WITHOUT signing it
        import hashlib
        from backend.services.snapshot_v2.object_store import canonical_json
        raw = canonical_json(prev)
        prev.root_hash = hashlib.sha256(raw).hexdigest()
        manifest_bytes = json.dumps(
            prev.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
        ).encode()
        (v2_paths.committed / f"{prev.snapshot_id}.json").write_bytes(manifest_bytes)
        # Intentionally no .sig file written

        analysis_state = make_analysis_state(
            sandbox_state={"runtime_revision": 5, "saved_runtime_revision": 5},
        )
        ctx = RestoreCaptureContext(
            session_id="sess-pre",
            paths=v2_paths,
            sandbox=sandbox,
            workspace_root=None,
            workflow_state={},
            analysis_state=analysis_state,
            prev_manifest=prev,
        )

        capture = PreRestoreCapture(service, object_store, manifest_store)

        # _can_reuse_runtime should return False because load() fails (no sig)
        assert not capture._can_reuse_runtime(ctx), \
            "Should NOT reuse runtime for unsigned (legacy) manifest"

    @pytest.mark.asyncio
    async def test_capture_failure_returns_failure_record(
        self, v2_paths, object_store, manifest_store, service, sandbox,
    ):
        """If pre_restore capture fails with analysis_state, raise
        RequiredSnapshotBlocked and set the snapshot_failure gate."""
        from backend.services.snapshot_v2.pre_restore import (
            PreRestoreCapture, RestoreCaptureContext,
        )
        from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
        from backend.services.snapshot_v2.models import (
            SnapshotFailureRecord,
            SnapshotRestoreRequest,
            SnapshotRetryContext,
        )

        analysis_state = make_analysis_state()
        ctx = RestoreCaptureContext(
            session_id="sess-pre",
            paths=v2_paths,
            sandbox=sandbox,
            workspace_root=None,
            workflow_state={},
            analysis_state=analysis_state,
            restore_transaction_id="restore-job-original",
            restore_request=SnapshotRestoreRequest(
                snapshot_id="target-snapshot",
                runtime_choice="reset",
                current_state_choice="abort",
            ),
        )

        capture = PreRestoreCapture(service, object_store, manifest_store)

        fail_record = SnapshotRecordV2(
            snapshot_id="snap-fail",
            manifest=v2_manifest(
                session_id="sess-pre",
                snapshot_id="snap-fail",
                captured_at="2025-01-01T00:00:00Z",
                capsule=IdleCapsule(graph_version="2.0"),
            ),
            failure=SnapshotFailureRecord(
                snapshot_id="snap-fail",
                session_id="sess-pre",
                failed_at="2025-01-01T00:00:00Z",
                code="TEST_FAIL",
                message="injected failure",
            ),
        )
        service._capture_locked = AsyncMock(return_value=fail_record)
        with pytest.raises(RequiredSnapshotBlocked) as ei:
            await capture.capture_locked(ctx)

        assert ei.value.code == "TEST_FAIL"
        assert analysis_state.wait_input_reason == "snapshot_failure"
        assert analysis_state.snapshot_failure is not None
        assert analysis_state.snapshot_failure["code"] == "TEST_FAIL"
        retry_context = SnapshotRetryContext.model_validate(
            analysis_state.snapshot_failure["retry_context"]
        )
        assert retry_context.restore_transaction_id == "restore-job-original"
        assert retry_context.restore_request is not None
        assert retry_context.restore_request["snapshot_id"] == "target-snapshot"


# ── Discard / SnapshotRestoreRequest Validation Tests ─────────────────────────


class TestDiscardValidation:

    def test_discard_requires_confirmation(self):
        """discard_current without discard_confirmation=True must raise."""
        from backend.services.snapshot_v2.models import SnapshotRestoreRequest
        with pytest.raises(ValueError, match="discard_confirmation"):
            SnapshotRestoreRequest(
                snapshot_id="snap-1",
                current_state_choice="discard_current",
                discard_confirmation=False,
            )

    def test_discard_plus_keep_is_rejected(self):
        """discard_current + runtime_choice=keep must be rejected."""
        from backend.services.snapshot_v2.models import SnapshotRestoreRequest
        with pytest.raises(ValueError, match="discard_current cannot be combined"):
            SnapshotRestoreRequest(
                snapshot_id="snap-1",
                current_state_choice="discard_current",
                runtime_choice="keep",
                discard_confirmation=True,
            )


# ── Audit Bundle Tests ────────────────────────────────────────────────────────


class TestAuditBundle:

    @pytest.mark.asyncio
    async def test_audit_bundle_must_succeed_before_mutation(
        self, v2_paths, object_store, manifest_store, service, sandbox,
    ):
        """Audit bundle capture must complete successfully before restore
        mutation proceeds."""
        from backend.services.snapshot_v2.pre_restore import (
            AuditBundle, RestoreCaptureContext,
        )
        from backend.services.snapshot_v2.models import SnapshotFailureRecord

        analysis_state = make_analysis_state()
        ctx = RestoreCaptureContext(
            session_id="sess-pre",
            paths=v2_paths,
            sandbox=sandbox,
            workspace_root=None,
            workflow_state={},
            analysis_state=analysis_state,
        )

        bundle = AuditBundle(service)
        failures = [{"step": "runtime_apply", "error": "kernel restart failed"}]

        mock_manifest = v2_manifest(
            session_id="sess-pre",
            snapshot_id="audit-snap-test-1",
            captured_at="2025-01-01T00:00:00Z",
            capsule=IdleCapsule(graph_version="2.0"),
        )
        mock_record = SnapshotRecordV2(
            snapshot_id="audit-snap-test-1",
            manifest=mock_manifest,
            failure=None,
        )

        with patch.object(service, '_capture_locked', AsyncMock(return_value=mock_record)):
            ref = await bundle.capture_locked(ctx, failures)

        assert ref is not None
        assert ref.audit_id.startswith("audit-"), \
            f"Expected audit- prefix, got {ref.audit_id}"
        assert ref.failure.code == "RESTORE_FAILURE"
        assert len(ref.failure.details.get("failures", [])) == 1
        assert ref.failure.details["failures"][0]["step"] == "runtime_apply"

    @pytest.mark.asyncio
    async def test_audit_bundle_failure_raises(
        self, v2_paths, object_store, manifest_store, service, sandbox,
    ):
        """If audit bundle capture itself fails, it raises and does not
        silently swallow the error."""
        from backend.services.snapshot_v2.pre_restore import (
            AuditBundle, RestoreCaptureContext,
        )
        from backend.services.snapshot_v2.models import SnapshotFailureRecord

        analysis_state = make_analysis_state()
        ctx = RestoreCaptureContext(
            session_id="sess-pre",
            paths=v2_paths,
            sandbox=sandbox,
            workspace_root=None,
            workflow_state={},
            analysis_state=analysis_state,
        )

        bundle = AuditBundle(service)

        fail_record = SnapshotRecordV2(
            snapshot_id="snap-audit-fail",
            manifest=v2_manifest(
                session_id="sess-pre",
                snapshot_id="snap-audit-fail",
                captured_at="2025-01-01T00:00:00Z",
                capsule=IdleCapsule(graph_version="2.0"),
            ),
            failure=SnapshotFailureRecord(
                snapshot_id="snap-audit-fail",
                session_id="sess-pre",
                failed_at="2025-01-01T00:00:00Z",
                code="AUDIT_FAIL",
                message="injected audit failure",
            ),
        )

        service._capture_locked = AsyncMock(return_value=fail_record)
        with pytest.raises(SnapshotCaptureError, match="Audit bundle capture failed"):
            await bundle.capture_locked(ctx, [])


class TestManifestStoreFiltering:

    def test_list_committed_filters_audit_bundles(self, manifest_store):
        """ManifestStore.list_committed should skip audit-bundle manifests."""
        # Commit a normal snapshot
        normal = v2_manifest(
            session_id="sess-pre",
            snapshot_id="snap-normal-list",
            captured_at="2025-01-01T00:00:00Z",
            capsule=IdleCapsule(graph_version="2.0"),
        )
        commit_manifest(manifest_store, normal)

        # Commit an audit-bundle manifest
        audit = v2_manifest(
            session_id="sess-pre",
            snapshot_id="audit-snap-list-1",
            captured_at="2025-01-01T00:00:00Z",
            capsule=IdleCapsule(graph_version="2.0"),
        )
        commit_manifest(manifest_store, audit)

        committed = manifest_store.list_committed()
        snap_ids = [m.snapshot_id for m in committed]
        assert "snap-normal-list" in snap_ids, "Normal snapshot should be listed"
        assert "audit-snap-list-1" not in snap_ids, \
            "Audit bundle should not be listed as restorable"
