import pytest
from pydantic import ValidationError

from backend.tests.snapshot_v2_fixtures import (
    complete_runtime,
    dependency_manifest,
    dill_dependency,
    make_snapshot_v2_manifest,
    minimal_manifest_payload,
    runtime_manifest,
    validate_system_commit,
)


def valid_manifest_payload() -> dict:
    """Backward-compatible alias used by older tests in this module."""
    return minimal_manifest_payload()


class TestSnapshotV2Models:
    def test_manifest_requires_format_v2(self):
        from backend.services.snapshot_v2.models import SnapshotManifestV2
        payload = valid_manifest_payload()
        payload["format_version"] = 1
        with pytest.raises(ValidationError) as exc:
            SnapshotManifestV2.model_validate(payload)
        assert any(error["loc"] == ("format_version",) for error in exc.value.errors())

    def test_workspace_entry_rejects_traversal(self):
        from backend.services.snapshot_v2.models import WorkspaceEntry
        with pytest.raises(ValidationError):
            WorkspaceEntry(path="../uploads/raw.h5ad", object_hash="0" * 64, size=1, mode=0o644)

    def test_v2_paths_are_session_confined(self):
        from backend.services.snapshot_v2.paths import SnapshotV2Paths
        paths = SnapshotV2Paths.for_session("sess-v2")
        assert paths.root.name == "v2"
        assert paths.objects.parent == paths.root.parent
        assert paths.staging.parent == paths.root.parent


class TestContinuationCapsuleVariants:
    def test_graph_node_capsule(self):
        from backend.services.snapshot_v2.models import GraphNodeCapsule
        c = GraphNodeCapsule(
            graph_version="1.0",
            next_node="pi_synthesis",
            workflow_state_ref={"sha256": "a" * 64, "size": 1024, "media_type": "application/json"},
            planning_stage="full",
        )
        assert c.kind == "graph_node"
        assert c.next_node == "pi_synthesis"
        assert c.workflow_state_ref.sha256 == "a" * 64
        assert c.planning_stage == "full"

    def test_semantic_gate_capsule(self):
        from backend.services.snapshot_v2.models import SemanticGateCapsule
        c = SemanticGateCapsule(
            graph_version="1.0",
            gate="user_review_gate",
            wait_reason="awaiting_user_approval",
            payload_ref={"sha256": "b" * 64, "size": 512, "media_type": "application/json"},
        )
        assert c.kind == "semantic_gate"
        assert c.gate == "user_review_gate"
        assert c.wait_reason == "awaiting_user_approval"

    def test_terminal_capsule(self):
        from backend.services.snapshot_v2.models import TerminalCapsule
        c = TerminalCapsule(graph_version="1.0", status="completed")
        assert c.kind == "terminal"
        assert c.status == "completed"

    def test_idle_capsule(self):
        from backend.services.snapshot_v2.models import IdleCapsule
        c = IdleCapsule(graph_version="1.0")
        assert c.kind == "idle"

    def test_continuation_capsule_discriminated_union(self):
        from pydantic import TypeAdapter
        from backend.services.snapshot_v2.models import (
            ContinuationCapsule, GraphNodeCapsule, TerminalCapsule,
        )
        adapter = TypeAdapter(ContinuationCapsule)
        data = {
            "kind": "graph_node",
            "graph_version": "1.0",
            "next_node": "check_convergence",
            "workflow_state_ref": {"sha256": "c" * 64, "size": 256, "media_type": "application/json"},
            "planning_stage": "stage2",
        }
        capsule = adapter.validate_python(data)
        assert isinstance(capsule, GraphNodeCapsule)

        data2 = {"kind": "terminal", "graph_version": "1.0", "status": "stopped"}
        capsule2 = adapter.validate_python(data2)
        assert isinstance(capsule2, TerminalCapsule)

    def test_invalid_kind_rejected(self):
        from pydantic import TypeAdapter, ValidationError
        from backend.services.snapshot_v2.models import ContinuationCapsule
        adapter = TypeAdapter(ContinuationCapsule)
        data = {"kind": "unknown", "graph_version": "1.0"}
        with pytest.raises(ValidationError):
            adapter.validate_python(data)


class TestObjectRef:
    def test_frozen(self):
        from backend.services.snapshot_v2.models import ObjectRef
        ref = ObjectRef(sha256="d" * 64, size=0, media_type="text/plain")
        with pytest.raises((ValidationError, TypeError)):
            ref.sha256 = "e" * 64

    def test_rejects_invalid_sha256(self):
        from backend.services.snapshot_v2.models import ObjectRef
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ObjectRef(sha256="not-a-valid-hash", size=0, media_type="text/plain")
        with pytest.raises(ValidationError):
            ObjectRef(sha256="a" * 63, size=0, media_type="text/plain")

    def test_rejects_negative_size(self):
        from backend.services.snapshot_v2.models import ObjectRef
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ObjectRef(sha256="f" * 64, size=-1, media_type="text/plain")


class TestSnapshotManifestV2:
    def test_valid_manifest_accepts(self):
        from backend.services.snapshot_v2.models import SnapshotManifestV2
        payload = valid_manifest_payload()
        manifest = SnapshotManifestV2.model_validate(payload)
        assert manifest.format_version == 2
        assert manifest.session_id == "sess-test"
        assert isinstance(manifest.continuation, object)
        assert manifest.analysis_state_ref.sha256
        assert manifest.notebook_ref.sha256
        assert manifest.runtime.status == "complete"

    def test_missing_required_fields_raises(self):
        from backend.services.snapshot_v2.models import SnapshotManifestV2
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SnapshotManifestV2.model_validate({})

    def test_workspace_empty_dir_kind(self):
        from backend.services.snapshot_v2.models import WorkspaceEntry
        entry = WorkspaceEntry(path="empty_folder", object_hash="a" * 64, size=0, mode=0o755, kind="empty_dir")
        assert entry.kind == "empty_dir"

    def test_workspace_rejects_absolute_path(self):
        from backend.services.snapshot_v2.models import WorkspaceEntry
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            WorkspaceEntry(path="/etc/passwd", object_hash="a" * 64, size=1, mode=0o644)

    def test_manifest_rejects_duplicate_workspace_paths(self):
        """Manifest with duplicate workspace paths must be rejected."""
        from backend.services.snapshot_v2.models import SnapshotManifestV2
        from pydantic import ValidationError
        payload = valid_manifest_payload()
        payload["workspace"] = [
            {"path": "figures/plot.png", "object_hash": "b" * 64, "size": 2048, "mode": 0o644, "kind": "file"},
            {"path": "figures/plot.png", "object_hash": "c" * 64, "size": 1024, "mode": 0o644, "kind": "file"},
        ]
        with pytest.raises(ValidationError, match="Duplicate workspace paths"):
            SnapshotManifestV2.model_validate(payload)

    def test_object_ref_equality(self):
        from backend.services.snapshot_v2.models import ObjectRef
        a = ObjectRef(sha256="a" * 64, size=100, media_type="text/csv")
        b = ObjectRef(sha256="a" * 64, size=100, media_type="text/csv")
        assert a == b

    def test_manifest_requires_state_notebook_and_complete_runtime(self):
        from backend.services.snapshot_v2.models import SnapshotManifestV2
        payload = minimal_manifest_payload()
        payload.pop("analysis_state_ref")
        with pytest.raises(ValidationError):
            SnapshotManifestV2.model_validate(payload)

    def test_complete_runtime_requires_every_required_name(self):
        from backend.services.snapshot_v2.models import RuntimeManifest
        runtime = complete_runtime(required={"results": "dill"}, captured={})
        with pytest.raises(ValidationError, match="results"):
            RuntimeManifest.model_validate(runtime)

    def test_no_live_kernel_with_empty_required_set_is_complete(self):
        from backend.services.snapshot_v2.models import RuntimeManifest
        runtime = RuntimeManifest(
            status="complete",
            dependencies=dependency_manifest(required={}),
            residual_ref=None,
            captured_residual_names=[],
            adata_refs={},
        )
        assert runtime.status == "complete"

    def test_residual_ref_without_names_is_invalid(self):
        from backend.services.snapshot_v2.models import (
            ObjectRef,
            RuntimeDependencyManifest,
            RuntimeManifest,
        )
        object_ref = ObjectRef(
            sha256="d" * 64, size=64, media_type="application/octet-stream",
        )
        with pytest.raises(ValidationError, match="residual_ref requires captured names"):
            RuntimeManifest(
                status="complete",
                dependencies=RuntimeDependencyManifest(required={}),
                residual_ref=object_ref,
                captured_residual_names=[],
            )

    def test_live_kernel_required_value_cannot_use_empty_runtime(self):
        from backend.services.snapshot_v2.models import RuntimeManifest
        with pytest.raises(ValidationError, match="results"):
            RuntimeManifest(
                status="complete",
                dependencies=dependency_manifest(required={"results": dill_dependency()}),
                residual_ref=None,
                captured_residual_names=[],
                adata_refs={},
            )

    def test_system_capture_rejects_non_complete_runtime_status(self):
        for status in ("state_only", "missing", "corrupt"):
            with pytest.raises(ValidationError):
                validate_system_commit(runtime_manifest(status=status))

    def test_factory_builds_valid_manifest(self):
        manifest = make_snapshot_v2_manifest(name="factory-check")
        assert manifest.name == "factory-check"
        assert manifest.branch_id == "main"
        assert not hasattr(manifest, "branch") or "branch" not in SnapshotManifestV2.model_fields  # noqa: F821
        from backend.services.snapshot_v2.models import SnapshotManifestV2
        assert "branch" not in SnapshotManifestV2.model_fields
        assert "generation" not in SnapshotManifestV2.model_fields
        assert "captured_at" not in SnapshotManifestV2.model_fields
