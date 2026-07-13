from backend.tests.snapshot_v2_fixtures import v2_manifest
import pytest
from pathlib import Path
from backend.services.snapshot_v2.errors import SnapshotIntegrityError, SnapshotStorageError


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
            for d in [self.root, self.objects, self.staging, self.journals, self.integrity, self.committed]:
                d.mkdir(parents=True, exist_ok=True)

    p = FakePaths()
    p.ensure_dirs()
    return p


@pytest.fixture
def manifest():
    return v2_manifest(
        session_id="sess-test",
        snapshot_id="snap-001",
        continuation={
            "kind": "graph_node",
            "graph_version": "1.0",
            "next_node": "pi_synthesis",
            "workflow_state_ref": {"sha256": "a" * 64, "size": 1024, "media_type": "application/json"},
            "planning_stage": "full",
        },
        workspace=[
            {"path": "figures/plot.png", "object_hash": "b" * 64, "size": 2048, "mode": 0o644, "kind": "file"},
        ],
        objects={
            "a" * 64: {"sha256": "a" * 64, "size": 1024, "media_type": "application/json"},
            "b" * 64: {"sha256": "b" * 64, "size": 2048, "media_type": "image/png"},
        },
    )


class TestCanonicalJson:
    def test_deterministic_serialization(self):
        from backend.services.snapshot_v2.object_store import canonical_json

        a = canonical_json({"b": 2, "a": 1})
        b = canonical_json({"a": 1, "b": 2})
        assert a == b
        assert a == b'{"a":1,"b":2}'

    def test_compact_separators(self):
        from backend.services.snapshot_v2.object_store import canonical_json

        result = canonical_json({"x": [1, 2, 3]})
        assert result == b'{"x":[1,2,3]}'
        assert b" " not in result

    def test_pydantic_model(self):
        from backend.services.snapshot_v2.object_store import canonical_json
        from backend.services.snapshot_v2.models import ObjectRef

        ref = ObjectRef(sha256="a" * 64, size=100, media_type="text/plain")
        result = canonical_json(ref)
        assert b'"sha256"' in result
        assert b'"size"' in result

    def test_root_hash_excluded(self):
        from backend.services.snapshot_v2.object_store import canonical_json
        from backend.services.snapshot_v2.models import SnapshotManifestV2

        from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest
        manifest = make_snapshot_v2_manifest(
            session_id="sess-test",
            snapshot_id="snap-001",
            root_hash="should_be_excluded",
            continuation={
                "kind": "graph_node",
                "graph_version": "1.0",
                "next_node": "pi_synthesis",
                "workflow_state_ref": {"sha256": "a" * 64, "size": 1024, "media_type": "application/json"},
                "planning_stage": "full",
            },
            workspace=[],
            objects={},
        )
        result = canonical_json(manifest)
        assert b"root_hash" not in result
        assert b'"session_id"' in result


class TestObjectStore:
    def test_put_bytes_creates_missing_staging_and_objects_dirs(self, tmp_path):
        """Fresh sessions may not have Snapshot V2 dirs yet; put_bytes must create them."""
        from backend.services.snapshot_v2.object_store import ObjectStore

        class MissingDirsPaths:
            root = tmp_path / "v2"
            objects = tmp_path / "objects"
            staging = tmp_path / "staging"
            journals = tmp_path / "journals"
            integrity = tmp_path / "integrity"
            committed = tmp_path / "committed"

        paths = MissingDirsPaths()
        assert not paths.staging.exists()
        assert not paths.objects.exists()

        store = ObjectStore(paths)
        result = store.put_bytes(b"hello world", "text/plain")

        assert paths.staging.exists()
        assert paths.objects.exists()
        assert (paths.objects / result.ref.sha256).read_bytes() == b"hello world"
        assert result.created is True

    def test_put_file_creates_missing_staging_and_objects_dirs(self, tmp_path):
        from backend.services.snapshot_v2.object_store import ObjectStore

        class MissingDirsPaths:
            root = tmp_path / "v2"
            objects = tmp_path / "objects"
            staging = tmp_path / "staging"
            journals = tmp_path / "journals"
            integrity = tmp_path / "integrity"
            committed = tmp_path / "committed"

        paths = MissingDirsPaths()
        src = tmp_path / "source.bin"
        src.write_bytes(b"file payload")
        assert not paths.staging.exists()
        assert not paths.objects.exists()

        store = ObjectStore(paths)
        result = store.put_file(src, "application/octet-stream")

        assert paths.staging.exists()
        assert paths.objects.exists()
        assert (paths.objects / result.ref.sha256).read_bytes() == b"file payload"
        assert result.created is True

    def test_put_stream_creates_missing_staging_and_objects_dirs(self, tmp_path):
        import hashlib
        import io
        from backend.services.snapshot_v2.object_store import ObjectStore

        class MissingDirsPaths:
            root = tmp_path / "v2"
            objects = tmp_path / "objects"
            staging = tmp_path / "staging"
            journals = tmp_path / "journals"
            integrity = tmp_path / "integrity"
            committed = tmp_path / "committed"

        paths = MissingDirsPaths()
        payload = b"stream payload"
        expected_sha = hashlib.sha256(payload).hexdigest()
        assert not paths.staging.exists()
        assert not paths.objects.exists()

        store = ObjectStore(paths)
        result = store.put_stream(
            io.BytesIO(payload),
            expected_sha256=expected_sha,
            expected_size=len(payload),
            media_type="application/octet-stream",
        )

        assert paths.staging.exists()
        assert paths.objects.exists()
        assert (paths.objects / result.ref.sha256).read_bytes() == payload
        assert result.created is True

    def test_put_bytes_returns_object_ref(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore

        store = ObjectStore(v2_paths)
        result = store.put_bytes(b"hello world", "text/plain")
        ref = result.ref
        assert ref.sha256
        assert len(ref.sha256) == 64
        assert ref.size == len(b"hello world")
        assert ref.media_type == "text/plain"
        assert result.created is True

    def test_put_bytes_creates_file(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore

        store = ObjectStore(v2_paths)
        result = store.put_bytes(b"hello world", "text/plain")
        obj_path = v2_paths.objects / result.ref.sha256
        assert obj_path.exists()
        assert obj_path.read_bytes() == b"hello world"

    def test_identical_bytes_are_deduplicated(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore

        store = ObjectStore(v2_paths)
        one = store.put_bytes(b"same", "application/octet-stream")
        two = store.put_bytes(b"same", "application/octet-stream")
        assert one.ref == two.ref
        assert two.created is False
        assert len(list(v2_paths.objects.iterdir())) == 1

    def test_put_bytes_verify_success(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore

        store = ObjectStore(v2_paths)
        result = store.put_bytes(b"verify me", "text/plain")
        store.verify(result.ref)

    def test_verify_corrupted_hash_raises(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore

        store = ObjectStore(v2_paths)
        result = store.put_bytes(b"original data", "text/plain")
        ref = result.ref
        obj_path = v2_paths.objects / ref.sha256
        # Clear evidence cache to force full streaming verification
        store._evidence.clear()
        obj_path.write_bytes(b"tampered data that is different")
        with pytest.raises(SnapshotIntegrityError):
            store.verify(ref)

    def test_verify_missing_file_raises(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore

        store = ObjectStore(v2_paths)
        from backend.services.snapshot_v2.models import ObjectRef

        ref = ObjectRef(sha256="f" * 64, size=100, media_type="text/plain")
        with pytest.raises(SnapshotIntegrityError):
            store.verify(ref)

    def test_verify_size_mismatch_raises(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore

        store = ObjectStore(v2_paths)
        result = store.put_bytes(b"exact content", "text/plain")
        ref = result.ref
        from backend.services.snapshot_v2.models import ObjectRef

        bad_ref = ObjectRef(sha256=ref.sha256, size=9999, media_type=ref.media_type)
        with pytest.raises(SnapshotIntegrityError):
            store.verify(bad_ref)

    def test_put_file_rejects_symlink(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore

        store = ObjectStore(v2_paths)
        link = v2_paths.staging / "bad_link.txt"
        link.write_text("real content")
        link_path = Path(str(link) + "_sym")
        link_path.symlink_to(link)
        with pytest.raises(SnapshotStorageError):
            store.put_file(link_path, "text/plain")

    def test_put_file_rejects_directory(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore

        store = ObjectStore(v2_paths)
        with pytest.raises(SnapshotStorageError):
            store.put_file(v2_paths.staging, "text/plain")

    def test_put_file_works_on_regular_file(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore

        store = ObjectStore(v2_paths)
        src = v2_paths.staging / "real_file.bin"
        src.write_bytes(b"file content data")
        result = store.put_file(src, "application/octet-stream")
        assert result.ref.size == len(b"file content data")
        assert result.created is True
        obj_path = v2_paths.objects / result.ref.sha256
        assert obj_path.exists()


def test_put_file_dedup_detects_same_size_corruption(tmp_path):
    from backend.services.snapshot_v2.object_store import ObjectStore
    from backend.services.snapshot_v2.errors import SnapshotIntegrityError

    # SnapshotV2Paths.for_session() uses import-time DATA_BASE_DIR; use tmp FakePaths for isolation.
    class FakePaths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"

    paths = FakePaths()
    for d in (paths.root, paths.objects, paths.staging, paths.journals, paths.integrity, paths.committed):
        d.mkdir(parents=True, exist_ok=True)
    store = ObjectStore(paths)

    src = tmp_path / "obj.bin"
    src.write_bytes(b"AAAA")
    ref = store.put_file(src, "application/octet-stream").ref

    # Corrupt the stored object to the SAME length, different content.
    dest = store._objects_dir / ref.sha256
    dest.write_bytes(b"BBBB")

    # A fresh store has no cached evidence, so the dedup path runs.
    fresh = ObjectStore(paths)
    with pytest.raises(SnapshotIntegrityError) as exc:
        fresh.put_file(src, "application/octet-stream")
    assert exc.value.code == "OBJECT_HASH_MISMATCH"


class TestManifestStore:
    def test_write_staging_creates_file(self, v2_paths, manifest):
        from backend.services.snapshot_v2.object_store import ManifestStore

        store = ManifestStore(v2_paths)
        staging_path = store.write_staging(manifest)
        assert staging_path.exists()
        assert staging_path.suffix == ".json"
        assert staging_path.parent == v2_paths.staging

    def test_write_staging_sets_root_hash(self, v2_paths, manifest):
        from backend.services.snapshot_v2.object_store import ManifestStore
        import json

        store = ManifestStore(v2_paths)
        staging_path = store.write_staging(manifest)
        data = json.loads(staging_path.read_bytes())
        assert "root_hash" in data
        assert len(data["root_hash"]) == 64
        assert data["root_hash"] != ""

    def test_manifest_is_invisible_before_atomic_publish(self, v2_paths, manifest):
        from backend.services.snapshot_v2.object_store import ManifestStore

        store = ManifestStore(v2_paths)
        assert store.list_committed() == []
        staging_path = store.write_staging(manifest)
        assert store.list_committed() == []
        store.publish_staging(staging_path)
        committed = store.list_committed()
        assert len(committed) == 1
        assert committed[0].snapshot_id == manifest.snapshot_id

    def test_publish_creates_committed_and_sig(self, v2_paths, manifest):
        from backend.services.snapshot_v2.object_store import ManifestStore

        store = ManifestStore(v2_paths)
        staging_path = store.write_staging(manifest)
        published = store.publish_staging(staging_path)
        assert published.snapshot_id == manifest.snapshot_id
        assert (v2_paths.committed / f"{manifest.snapshot_id}.json").exists()
        assert (v2_paths.integrity / f"{manifest.snapshot_id}.sig").exists()

    def test_load_returns_verified_manifest(self, v2_paths, manifest):
        from backend.services.snapshot_v2.object_store import ManifestStore

        store = ManifestStore(v2_paths)
        staging_path = store.write_staging(manifest)
        store.publish_staging(staging_path)
        loaded = store.load(manifest.snapshot_id)
        assert loaded.snapshot_id == manifest.snapshot_id
        assert loaded.root_hash  # non-empty
        assert loaded.continuation.kind == manifest.continuation.kind

    def test_load_tampered_manifest_raises(self, v2_paths, manifest):
        from backend.services.snapshot_v2.object_store import ManifestStore

        store = ManifestStore(v2_paths)
        staging_path = store.write_staging(manifest)
        store.publish_staging(staging_path)
        committed_path = v2_paths.committed / f"{manifest.snapshot_id}.json"
        committed_path.write_text(committed_path.read_text().replace("graph_node", "terminal"))
        with pytest.raises(SnapshotIntegrityError):
            store.load(manifest.snapshot_id)

    def test_load_tampered_sig_raises(self, v2_paths, manifest):
        from backend.services.snapshot_v2.object_store import ManifestStore

        store = ManifestStore(v2_paths)
        staging_path = store.write_staging(manifest)
        store.publish_staging(staging_path)
        sig_path = v2_paths.integrity / f"{manifest.snapshot_id}.sig"
        sig_path.write_bytes(b"tampered")
        with pytest.raises(SnapshotIntegrityError):
            store.load(manifest.snapshot_id)

    def test_list_committed_empty_on_fresh_paths(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ManifestStore

        store = ManifestStore(v2_paths)
        assert store.list_committed() == []

    def test_reconcile_removes_orphan_staging(self, v2_paths, manifest):
        from backend.services.snapshot_v2.object_store import ManifestStore

        store = ManifestStore(v2_paths)
        staging_path = store.write_staging(manifest)
        assert staging_path.exists()
        store.reconcile()
        assert not staging_path.exists(), "orphan staging should be removed by reconcile"

    def test_reconcile_preserves_published_manifests(self, v2_paths, manifest):
        from backend.services.snapshot_v2.object_store import ManifestStore

        store = ManifestStore(v2_paths)
        staging_path = store.write_staging(manifest)
        store.publish_staging(staging_path)
        result = store.reconcile()
        assert len(store.list_committed()) == 1
        assert manifest.snapshot_id in result.verified_ids
        assert result.corrupt_ids == []

    def test_reconcile_reports_corrupt_manifest(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ManifestStore

        store = ManifestStore(v2_paths)
        bad = v2_paths.committed / "snap-bad.json"
        bad.write_text("not a valid manifest")
        result = store.reconcile()
        assert result.corrupt_ids == ["snap-bad"]

    def test_verify_all_committed_raises_on_corrupt(self, v2_paths, manifest):
        from backend.services.snapshot_v2.object_store import ManifestStore

        store = ManifestStore(v2_paths)
        store.publish_staging(store.write_staging(manifest))
        bad = v2_paths.committed / "snap-bad.json"
        bad.write_text("{not-json")
        with pytest.raises(SnapshotIntegrityError):
            store.verify_all_committed()

    def test_verify_all_committed_returns_verified_only(self, v2_paths, manifest):
        from backend.services.snapshot_v2.object_store import ManifestStore

        store = ManifestStore(v2_paths)
        store.publish_staging(store.write_staging(manifest))
        verified = store.verify_all_committed()
        assert len(verified) == 1
        assert verified[0].snapshot_id == manifest.snapshot_id

    def test_list_committed_with_diagnostics_surfaces_corrupt(self, v2_paths, manifest):
        from backend.services.snapshot_v2.object_store import ManifestStore

        store = ManifestStore(v2_paths)
        store.publish_staging(store.write_staging(manifest))
        bad = v2_paths.committed / "snap-bad.json"
        bad.write_text("corrupt")
        entries = store.list_committed_with_diagnostics()
        statuses = {e.snapshot_id: e.status for e in entries}
        assert statuses[manifest.snapshot_id] == "ok"
        assert statuses["snap-bad"] == "corrupt"

    def test_load_detects_root_hash_drift(self, v2_paths, manifest):
        from backend.services.snapshot_v2.object_store import ManifestStore, canonical_json
        import hashlib, hmac, json
        from backend.utils.crypto import get_integrity_secret

        store = ManifestStore(v2_paths)
        staging_path = store.write_staging(manifest)
        store.publish_staging(staging_path)
        committed_path = v2_paths.committed / f"{manifest.snapshot_id}.json"
        sig_path = v2_paths.integrity / f"{manifest.snapshot_id}.sig"
        data = json.loads(committed_path.read_bytes())
        data["root_hash"] = "0" * 64
        tampered_bytes = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        committed_path.write_bytes(tampered_bytes)
        secret = get_integrity_secret()
        key = secret if isinstance(secret, bytes) else secret.encode()
        tampered_sig = hmac.new(key, tampered_bytes, hashlib.sha256).hexdigest()
        sig_path.write_text(tampered_sig)
        with pytest.raises(SnapshotIntegrityError) as exc:
            store.load(manifest.snapshot_id)
        assert "ROOT_HASH" in str(exc.value.code)

    def test_write_staging_computes_correct_root_hash(self, v2_paths, manifest):
        from backend.services.snapshot_v2.object_store import ManifestStore, canonical_json
        import hashlib

        store = ManifestStore(v2_paths)
        staging_path = store.write_staging(manifest)
        import json
        data = json.loads(staging_path.read_bytes())
        manifest.root_hash = ""
        raw = canonical_json(manifest)
        expected = hashlib.sha256(raw).hexdigest()
        assert data["root_hash"] == expected

    def test_committed_bytes_are_canonical_document(self, v2_paths, manifest):
        from backend.services.snapshot_v2.object_store import (
            ManifestStore, canonical_manifest_body_bytes,
            canonical_manifest_document_bytes,
        )

        store = ManifestStore(v2_paths)
        path = store.write_staging(manifest)
        published = store.publish_staging(path)
        committed = v2_paths.committed / f"{published.snapshot_id}.json"
        assert committed.read_bytes() == canonical_manifest_document_bytes(published)

    def test_crash_before_manifest_commit_never_exposes_manifest(self, v2_paths, manifest, monkeypatch):
        from backend.services.snapshot_v2.object_store import ManifestStore
        from unittest.mock import Mock
        import os

        store = ManifestStore(v2_paths)
        path = store.write_staging(manifest)
        monkeypatch.setattr(os, "replace", Mock(side_effect=OSError("crash")))
        store._sign = Mock(return_value="sig-placeholder")
        with pytest.raises(OSError, match="crash"):
            store.publish_staging(path)
        assert not (v2_paths.committed / f"{manifest.snapshot_id}.json").exists()

    def test_load_rejects_pending_import_manifest(self, v2_paths, manifest):
        from backend.services.snapshot_v2.object_store import ManifestStore

        store = ManifestStore(v2_paths)
        staging = store.write_staging(manifest)
        store.publish_staging(staging)

        # Mark as pending — must become invisible to load().
        ManifestStore.mark_pending_imports(v2_paths.committed, [manifest.snapshot_id])
        with pytest.raises(SnapshotIntegrityError) as exc:
            store.load(manifest.snapshot_id)
        assert exc.value.code == "MANIFEST_NOT_FOUND"

        # Clearing the marker restores visibility.
        ManifestStore.clear_pending_imports(v2_paths.committed, [manifest.snapshot_id])
        assert store.load(manifest.snapshot_id).snapshot_id == manifest.snapshot_id

    def test_mark_pending_imports_creates_committed_dir_when_missing(self, tmp_path):
        """First import may run before committed/ exists; markers must still land."""
        from backend.services.snapshot_v2.object_store import ManifestStore

        committed = tmp_path / "committed"
        assert not committed.exists()
        ManifestStore.mark_pending_imports(committed, ["snap-new"])
        assert committed.is_dir()
        assert (committed / ".pending-import-snap-new").is_file()
        assert ManifestStore.list_pending_import_ids(committed) == {"snap-new"}
