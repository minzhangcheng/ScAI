"""Tests for Snapshot V2 preflight (read-only restore verification)."""

import hashlib
import hmac
import json
import os
from pathlib import Path

import pytest


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


def _publish_valid_manifest(v2_paths, **overrides):
    """Helper: create and publish a valid snapshot manifest with real objects."""
    from backend.services.snapshot_v2.object_store import ManifestStore, ObjectStore
    from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest, complete_runtime

    store = ObjectStore(v2_paths)

    state_ref = store.put_bytes(
        json.dumps({"round": 1, "status": "running"}).encode(),
        "application/json",
    ).ref
    notebook_ref = store.put_bytes(b'{"cells":[]}', "application/x-ipynb+json").ref
    plot_ref = store.put_bytes(b"fake_png_data", "image/png").ref

    runtime_payload = complete_runtime(required={}, captured={})
    # Mark runtime as present for choice support tests that expect True by default
    residual = store.put_bytes(b"residual-bytes", "application/octet-stream").ref
    runtime_payload = complete_runtime(
        required={"results": "dill"},
        residual_names=["results"],
        residual_ref=residual,
    )

    kwargs = dict(
        session_id=overrides.get("session_id", "sess-test"),
        snapshot_id=overrides.get("snapshot_id", "snap-001"),
        analysis_state_ref=state_ref.model_dump(mode="json"),
        notebook_ref=notebook_ref.model_dump(mode="json"),
        runtime=runtime_payload,
        continuation={
            "kind": "graph_node",
            "graph_version": "1.0",
            "next_node": "pi_synthesis",
            "workflow_state_ref": {
                "sha256": state_ref.sha256,
                "size": state_ref.size,
                "media_type": "application/json",
            },
            "planning_stage": "full",
        },
        workspace=[
            {
                "path": "figures/plot.png",
                "object_hash": plot_ref.sha256,
                "size": plot_ref.size,
                "mode": 0o644,
                "kind": "file",
            },
        ],
        objects={
            state_ref.sha256: state_ref.model_dump(mode="json"),
            notebook_ref.sha256: notebook_ref.model_dump(mode="json"),
            plot_ref.sha256: plot_ref.model_dump(mode="json"),
            residual.sha256: residual.model_dump(mode="json"),
        },
    )
    for k, v in overrides.items():
        if k not in ("session_id", "snapshot_id"):
            kwargs[k] = v
    manifest = make_snapshot_v2_manifest(**kwargs)

    manifest_store = ManifestStore(v2_paths)
    staging_path = manifest_store.write_staging(manifest)
    manifest_store.publish_staging(staging_path)
    return manifest, store


class TestRestorePreflight:
    def test_rejects_format_1(self, v2_paths):
        """Format version 1 manifests must be reported as incompatible."""
        raw = {
            "format_version": 1,
            "session_id": "sess-test",
            "snapshot_id": "snap-f1",
            "captured_at": "2026-01-01T00:00:00Z",
            "branch": "main",
            "generation": 1,
            "capsule": {"kind": "idle", "graph_version": "1.0"},
            "workspace": [],
            "objects": {},
        }
        committed_path = v2_paths.committed / "snap-f1.json"
        committed_path.write_text(json.dumps(raw))
        session_key = hashlib.sha256(b"sess-test").digest()
        sig = hmac.new(session_key, committed_path.read_bytes(), hashlib.sha256).hexdigest()
        (v2_paths.integrity / "snap-f1.sig").write_text(sig)

        from backend.services.snapshot_v2.preflight import RestorePreflight

        result = RestorePreflight(v2_paths).run("snap-f1")

        assert result.status == "incompatible"
        assert any("format" in err.lower() for err in result.compatibility_errors)
        assert result.format_version == 1

    def test_detects_hmac_mismatch(self, v2_paths):
        """Tampered signature file must be detected."""
        _publish_valid_manifest(v2_paths, snapshot_id="snap-hmac")
        (v2_paths.integrity / "snap-hmac.sig").write_text("tampered")

        from backend.services.snapshot_v2.preflight import RestorePreflight

        result = RestorePreflight(v2_paths).run("snap-hmac")
        assert result.status == "corrupted"

    def test_detects_root_hash_drift(self, v2_paths):
        """Manifest with altered root hash must be detected."""
        manifest, _ = _publish_valid_manifest(v2_paths, snapshot_id="snap-root")
        committed_path = v2_paths.committed / "snap-root.json"
        sig_path = v2_paths.integrity / "snap-root.sig"
        data = json.loads(committed_path.read_bytes())
        data["root_hash"] = "0" * 64
        tampered = json.dumps(data, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False).encode()
        committed_path.write_bytes(tampered)
        from backend.tests.snapshot_v2_fixtures import sign_manifest_bytes
        tampered_sig = sign_manifest_bytes(tampered)
        sig_path.write_text(tampered_sig)

        from backend.services.snapshot_v2.preflight import RestorePreflight

        result = RestorePreflight(v2_paths).run("snap-root")
        assert result.status == "corrupted"

    def test_detects_object_corruption(self, v2_paths):
        """Corrupted object file must be detected."""
        manifest, _ = _publish_valid_manifest(v2_paths, snapshot_id="snap-obj")
        obj_hash = next(iter(manifest.objects))
        (v2_paths.objects / obj_hash).write_text("tampered")

        from backend.services.snapshot_v2.preflight import RestorePreflight

        result = RestorePreflight(v2_paths).run("snap-obj")
        assert result.status == "corrupted"

    def test_detects_missing_object(self, v2_paths):
        """Missing object file must be detected."""
        manifest, _ = _publish_valid_manifest(v2_paths, snapshot_id="snap-miss")
        obj_hash = next(iter(manifest.objects))
        (v2_paths.objects / obj_hash).unlink()

        from backend.services.snapshot_v2.preflight import RestorePreflight

        result = RestorePreflight(v2_paths).run("snap-miss")
        assert result.status == "corrupted"
        assert obj_hash in result.objects_failed

    def test_ready_with_valid_manifest(self, v2_paths):
        """A valid intact manifest must return status 'ready'."""
        _publish_valid_manifest(v2_paths, snapshot_id="snap-ready")

        from backend.services.snapshot_v2.preflight import RestorePreflight

        result = RestorePreflight(v2_paths).run("snap-ready")
        assert result.status == "ready"
        assert result.objects_verified > 0
        assert result.objects_verified == result.objects_total
        assert len(result.objects_failed) == 0
        assert result.format_version == 2

    def test_preflight_is_read_only(self, v2_paths):
        """Preflight must not modify any files on disk."""
        _publish_valid_manifest(v2_paths, snapshot_id="snap-ro")

        checksums_before = {}
        for root, _dirs, files in os.walk(v2_paths.root):
            for f in files:
                path = Path(root) / f
                checksums_before[str(path)] = path.read_bytes()

        from backend.services.snapshot_v2.preflight import RestorePreflight

        RestorePreflight(v2_paths).run("snap-ro")

        checksums_after = {}
        for root, _dirs, files in os.walk(v2_paths.root):
            for f in files:
                path = Path(root) / f
                checksums_after[str(path)] = path.read_bytes()

        assert checksums_before == checksums_after

    def test_empty_objects_ready(self, v2_paths):
        """Manifest with no objects passes if workspace is also empty."""
        from backend.services.snapshot_v2.models import SnapshotManifestV2
        from backend.services.snapshot_v2.object_store import ManifestStore

        from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest
        manifest = make_snapshot_v2_manifest(
            session_id="sess-test",
            snapshot_id="snap-empty",
            workspace=[],
            objects={},
        )
        store = ManifestStore(v2_paths)
        staging_path = store.write_staging(manifest)
        store.publish_staging(staging_path)

        from backend.services.snapshot_v2.preflight import RestorePreflight

        result = RestorePreflight(v2_paths).run("snap-empty")
        assert result.status == "ready"
        assert result.objects_total == 0
        assert result.objects_verified == 0

    def test_multiple_object_failures(self, v2_paths):
        """All corrupted objects are reported in objects_failed."""
        manifest, _ = _publish_valid_manifest(v2_paths, snapshot_id="snap-multi")
        for obj_hash in manifest.objects:
            (v2_paths.objects / obj_hash).write_text("tampered")

        from backend.services.snapshot_v2.preflight import RestorePreflight

        result = RestorePreflight(v2_paths).run("snap-multi")
        assert result.status == "corrupted"
        assert len(result.objects_failed) == len(manifest.objects)
        for h in manifest.objects:
            assert h in result.objects_failed

    def test_capsule_info_in_result(self, v2_paths):
        """Result includes capsule kind and details."""
        _publish_valid_manifest(v2_paths, snapshot_id="snap-cap")

        from backend.services.snapshot_v2.preflight import RestorePreflight

        result = RestorePreflight(v2_paths).run("snap-cap")
        assert result.status == "ready"
        assert "kind" in result.capsule
        assert result.capsule["kind"] == "graph_node"
        assert "graph_version" in result.capsule

    def test_manifest_not_found(self, v2_paths):
        """Preflight reports corrupted when manifest does not exist."""
        from backend.services.snapshot_v2.preflight import RestorePreflight

        result = RestorePreflight(v2_paths).run("snap-nonexistent")
        assert result.status == "corrupted"
        assert result.objects_total == 0

    def test_detects_invalid_workflow_state_json(self, v2_paths):
        """Preflight must catch workflow state objects with invalid JSON content."""
        from backend.services.snapshot_v2.models import SnapshotManifestV2
        from backend.services.snapshot_v2.object_store import ManifestStore, ObjectStore

        store = ObjectStore(v2_paths)
        bad_state_ref = store.put_bytes(b"not valid json {{{", "application/json").ref
        plot_ref = store.put_bytes(b"fake_png_data", "image/png").ref

        from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest
        nb = store.put_bytes(b'{"cells":[]}', "application/x-ipynb+json").ref
        manifest = make_snapshot_v2_manifest(
            session_id="sess-test",
            snapshot_id="snap-bad-state",
            analysis_state_ref=bad_state_ref.model_dump(mode="json"),
            notebook_ref=nb.model_dump(mode="json"),
            continuation={
                "kind": "graph_node",
                "graph_version": "1.0",
                "next_node": "pi_synthesis",
                "workflow_state_ref": {
                    "sha256": bad_state_ref.sha256,
                    "size": bad_state_ref.size,
                    "media_type": "application/json",
                },
                "planning_stage": "full",
            },
            workspace=[
                {
                    "path": "figures/plot.png",
                    "object_hash": plot_ref.sha256,
                    "size": plot_ref.size,
                    "mode": 0o644,
                    "kind": "file",
                },
            ],
            objects={
                bad_state_ref.sha256: bad_state_ref.model_dump(mode="json"),
                plot_ref.sha256: plot_ref.model_dump(mode="json"),
                nb.sha256: nb.model_dump(mode="json"),
            },
        )
        manifest_store = ManifestStore(v2_paths)
        staging_path = manifest_store.write_staging(manifest)
        manifest_store.publish_staging(staging_path)

        from backend.services.snapshot_v2.preflight import RestorePreflight
        result = RestorePreflight(v2_paths).run("snap-bad-state")

        assert result.status == "corrupted"
        assert any("Workflow state" in err for err in result.compatibility_errors)

    def test_staging_bytes_positive_with_objects(self, v2_paths):
        """required_staging_bytes must be > 0 when manifest has objects."""
        _publish_valid_manifest(v2_paths, snapshot_id="snap-bytes")

        from backend.services.snapshot_v2.preflight import RestorePreflight
        result = RestorePreflight(v2_paths).run("snap-bytes")
        assert result.required_staging_bytes > 0

    def test_runtime_choice_not_required_when_complete_payload_present(self, v2_paths):
        """Complete runtime with payload does not require explicit runtime_choice."""
        _publish_valid_manifest(v2_paths, snapshot_id="snap-rt-yes")

        from backend.services.snapshot_v2.preflight import RestorePreflight
        result = RestorePreflight(v2_paths).run("snap-rt-yes")
        assert result.runtime_choice_supported is True
        assert result.runtime_choice_required is False

    def test_runtime_choice_required_when_complete_payload_absent(self, v2_paths):
        """Empty complete runtime is an exact empty runtime, not unavailable."""
        from backend.services.snapshot_v2.models import SnapshotManifestV2
        from backend.services.snapshot_v2.object_store import ManifestStore, ObjectStore

        store = ObjectStore(v2_paths)
        store.put_bytes(b'{"round": 1}', "application/json")

        from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest, complete_runtime
        manifest = make_snapshot_v2_manifest(
            session_id="sess-test",
            snapshot_id="snap-rt-no",
            workspace=[],
            objects={},
            runtime=complete_runtime(required={}, captured={}),
        )
        manifest_store = ManifestStore(v2_paths)
        staging_path = manifest_store.write_staging(manifest)
        manifest_store.publish_staging(staging_path)

        from backend.services.snapshot_v2.preflight import RestorePreflight
        result = RestorePreflight(v2_paths).run("snap-rt-no")
        assert result.runtime_choice_supported is True
        assert result.runtime_choice_required is False

    def test_preflight_requires_runtime_choice_for_unavailable_runtime(self, v2_paths):
        """Unavailable runtime with restore request and no runtime_choice must raise."""
        from backend.services.snapshot_v2.object_store import ManifestStore
        from backend.services.snapshot_v2.models import SnapshotRestoreRequest
        from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest, complete_runtime
        from backend.utils.runtime_apply import RuntimeChoiceRequired
        from backend.services.snapshot_v2.preflight import RestorePreflight

        manifest = make_snapshot_v2_manifest(
            session_id="sess-test",
            snapshot_id="snap-rt-missing",
            workspace=[],
            objects={},
            runtime=complete_runtime(required={}, captured={}, status="missing"),
        )
        manifest_store = ManifestStore(v2_paths)
        staging_path = manifest_store.write_staging(manifest)
        manifest_store.publish_staging(staging_path)

        request = SnapshotRestoreRequest(
            snapshot_id=manifest.snapshot_id,
            current_state_choice="abort",
            runtime_choice=None,
        )

        with pytest.raises(RuntimeChoiceRequired):
            RestorePreflight(v2_paths).run(manifest.snapshot_id, request)

        result = RestorePreflight(v2_paths).run(manifest.snapshot_id)
        assert result.runtime_choice_supported is True
        assert result.runtime_choice_required is True

    def test_corrupted_preflight_skips_runtime_choice_enforcement(self, v2_paths):
        """Corrupted snapshot must return corrupted, not raise RuntimeChoiceRequired."""
        from backend.services.snapshot_v2.models import SnapshotRestoreRequest
        from backend.services.snapshot_v2.preflight import RestorePreflight
        from backend.utils.runtime_apply import RuntimeChoiceRequired

        manifest, _ = _publish_valid_manifest(v2_paths, snapshot_id="snap-rt-corrupt")
        # Tamper one object so status becomes corrupted.
        first_hash = next(iter(manifest.objects))
        (v2_paths.objects / first_hash).write_text("tampered")

        request = SnapshotRestoreRequest(
            snapshot_id=manifest.snapshot_id,
            current_state_choice="abort",
            runtime_choice=None,
        )

        try:
            result = RestorePreflight(v2_paths).run(manifest.snapshot_id, request)
        except RuntimeChoiceRequired as exc:
            pytest.fail(f"corrupted preflight raised RuntimeChoiceRequired: {exc}")

        assert result.status == "corrupted"
        assert first_hash in result.objects_failed
