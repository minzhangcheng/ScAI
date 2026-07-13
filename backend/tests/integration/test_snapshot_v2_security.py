"""Security integration tests for Snapshot V2.

Tests HMAC binding, path traversal rejection, forged manifest rejection,
archive safety verification, and V1 cutover.
"""

from backend.tests.snapshot_v2_fixtures import v2_manifest
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from backend.services.snapshot_v2.errors import SnapshotIntegrityError, SnapshotStorageError
from backend.services.snapshot_v2.models import (
    FORMAT_VERSION,
    IdleCapsule,
    ObjectRef,
    SnapshotManifestV2,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _build_object_ref(sha256: str | None = None, size: int = 256,
                       media_type: str = "application/octet-stream") -> ObjectRef:
    if sha256 is None:
        sha256 = hashlib.sha256(f"obj-{size}".encode()).hexdigest()
    return ObjectRef(sha256=sha256, size=size, media_type=media_type)


def _make_manifest(session_id: str, snapshot_id: str,
                   objects: dict | None = None) -> SnapshotManifestV2:
    return v2_manifest(
        format_version=2,
        session_id=session_id,
        snapshot_id=snapshot_id,
        captured_at="2026-07-11T12:00:00Z",
        capsule=IdleCapsule(graph_version="2.0"),
        objects=objects or {},
    )


def _paths_factory(tmp_path: Path, prefix: str):
    class FakePaths:
        root = tmp_path / f"{prefix}_v2"
        objects = tmp_path / f"{prefix}_objects"
        staging = tmp_path / f"{prefix}_staging"
        journals = tmp_path / f"{prefix}_journals"
        integrity = tmp_path / f"{prefix}_integrity"
        committed = tmp_path / f"{prefix}_committed"

        def ensure_dirs(self):
            for d in [self.root, self.objects, self.staging,
                      self.journals, self.integrity, self.committed]:
                d.mkdir(parents=True, exist_ok=True)

    p = FakePaths()
    p.ensure_dirs()
    return p


# ── Tests ─────────────────────────────────────────────────────────────────


class TestHMACIntegrity:
    """Verify HMAC binding prevents tampered manifests from loading."""

    def test_manifest_hmac_mismatch_raises(self, tmp_path):
        paths = _paths_factory(tmp_path, "hmac")
        session_id = "hmac-test"

        from backend.services.snapshot_v2.object_store import ObjectStore, ManifestStore
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError

        store = ObjectStore(paths)
        mstore = ManifestStore(paths)

        manifest = _make_manifest(session_id, "snap-hmac-1")
        staging_path = mstore.write_staging(manifest)
        published = mstore.publish_staging(staging_path)

        sig_path = paths.integrity / f"{published.snapshot_id}.sig"
        assert sig_path.exists()

        sig_path.write_text("tampered-signature")

        with pytest.raises(SnapshotIntegrityError) as exc:
            mstore.load(published.snapshot_id)
        assert any(kw in str(exc.value) for kw in ["HMAC", "signature", "mismatch"])

    def test_manifest_root_hash_mismatch_raises(self, tmp_path):
        paths = _paths_factory(tmp_path, "rhash")
        session_id = "rhash-test"

        from backend.services.snapshot_v2.object_store import ObjectStore, ManifestStore
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError

        store = ObjectStore(paths)
        mstore = ManifestStore(paths)

        manifest = _make_manifest(session_id, "snap-rhash-1")
        staging_path = mstore.write_staging(manifest)
        mstore.publish_staging(staging_path)

        committed_path = paths.committed / f"{manifest.snapshot_id}.json"
        manifest_dict = json.loads(committed_path.read_text())
        manifest_dict["root_hash"] = "0" * 64
        committed_path.write_text(json.dumps(manifest_dict, sort_keys=True))

        with pytest.raises(SnapshotIntegrityError) as exc:
            mstore.load(manifest.snapshot_id)
        assert any(kw in str(exc.value) for kw in ["HMAC", "MISMATCH", "root_hash"])

    def test_object_hash_mismatch_raises(self, tmp_path):
        paths = _paths_factory(tmp_path, "obj")
        session_id = "obj-test"

        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError

        store = ObjectStore(paths)
        ref = store.put_bytes(b"original-content", "text/plain").ref

        tampered_path = paths.objects / ref.sha256
        tampered_path.write_bytes(b"tampered-content")
        # Clear evidence cache to force full streaming verification
        store._evidence.clear()

        with pytest.raises(SnapshotIntegrityError) as exc:
            store.verify(ref)
        error_msg = str(exc.value)
        assert any(kw in error_msg for kw in ["hash mismatch", "MISMATCH"])

    def test_missing_object_raises(self, tmp_path):
        paths = _paths_factory(tmp_path, "mobj")

        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError

        store = ObjectStore(paths)
        ref = _build_object_ref(sha256="a" * 64, size=100)

        with pytest.raises(SnapshotIntegrityError) as exc:
            store.verify(ref)
        assert "not found" in str(exc.value).lower()

    def test_object_size_mismatch_raises(self, tmp_path):
        paths = _paths_factory(tmp_path, "osz")

        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError

        store = ObjectStore(paths)
        data = b"correct-size-content"
        ref = store.put_bytes(data, "text/plain").ref

        ref_with_bad_size = ObjectRef(
            sha256=ref.sha256,
            size=ref.size + 100,
            media_type=ref.media_type,
        )

        with pytest.raises(SnapshotIntegrityError) as exc:
            store.verify(ref_with_bad_size)
        assert "size" in str(exc.value).lower()


class TestArchiveSafety:
    """Reject archives with symlinks, path traversal, or forged manifests."""

    def test_archive_with_symlink_rejected(self, tmp_path):
        from backend.services.snapshot_v2.importer import SnapshotImporter
        from backend.services.snapshot_v2.errors import SnapshotStorageError

        dest_paths = _paths_factory(tmp_path, "safe")
        importer = SnapshotImporter("safe-session", dest_paths)

        class FakeSymlinkMember:
            def issym(self): return True
            def islnk(self): return False
            def isdev(self): return False
            def ischr(self): return False
            def isfifo(self): return False
            name = "objects/evil_link"

        with pytest.raises(SnapshotStorageError) as exc:
            importer._verify_archive_safety([FakeSymlinkMember()])
        assert exc.value.code == "ARCHIVE_UNSAFE"

    def test_archive_safety_check_rejects_symlinks(self, tmp_path):
        from backend.services.snapshot_v2.importer import SnapshotImporter
        from backend.services.snapshot_v2.errors import SnapshotStorageError

        dest_paths = _paths_factory(tmp_path, "safe2")

        class FakeSymlinkMember:
            def issym(self): return True
            def islnk(self): return False
            def isdev(self): return False
            def ischr(self): return False
            def isfifo(self): return False
            name = "objects/evil"

        class FakeHardlinkMember:
            def issym(self): return False
            def islnk(self): return True
            def isdev(self): return False
            def ischr(self): return False
            def isfifo(self): return False
            name = "objects/hard"

        importer = SnapshotImporter("safe-session", dest_paths)
        with pytest.raises(SnapshotStorageError) as exc:
            importer._verify_archive_safety([FakeSymlinkMember(), FakeHardlinkMember()])
        assert exc.value.code == "ARCHIVE_UNSAFE"

    def test_path_traversal_in_manifest_rejected(self, tmp_path):
        dest_paths = _paths_factory(tmp_path, "traverse")

        from backend.services.snapshot_v2.importer import SnapshotImporter
        from backend.services.snapshot_v2.errors import SnapshotStorageError

        archive = tmp_path / "traverse.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            info = tarfile.TarInfo(name="manifests/../../../etc/passwd.json")
            info.size = 4
            tf.addfile(info, io.BytesIO(b"data"))
            meta = {"format_version": 2, "session_id": "traverse-src",
                    "export_id": "exp-trav", "exported_at": "now",
                    "runtime_included": True,
                    "included_snapshot_ids": ["traverse-snap"]}
            meta_bytes = json.dumps(meta, sort_keys=True, ensure_ascii=False).encode()
            meta_info = tarfile.TarInfo(name="snapshot-v2-export.json")
            meta_info.size = len(meta_bytes)
            tf.addfile(meta_info, io.BytesIO(meta_bytes))

        importer = SnapshotImporter("traverse-session", dest_paths)
        result = importer.import_bundle(archive)
        assert not result.success
        assert len(result.errors) > 0

    def test_forged_manifest_hash_rejected(self, tmp_path):
        dest_paths = _paths_factory(tmp_path, "forged")

        from backend.services.snapshot_v2.importer import SnapshotImporter
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError

        manifest = _make_manifest("forged-src", "forged-snap")
        manifest.root_hash = "0" * 64

        from backend.services.snapshot_v2.object_store import ObjectStore, ManifestStore
        src_paths = _paths_factory(tmp_path, "forged_src")
        ObjectStore(src_paths).put_bytes(b"dummy", "application/octet-stream")

        mbytes = json.dumps(
            manifest.model_dump(mode="json"),
            sort_keys=True, ensure_ascii=False,
        ).encode()

        archive = tmp_path / "forged.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            m_info = tarfile.TarInfo(name=f"manifests/{manifest.snapshot_id}.json")
            m_info.size = len(mbytes)
            tf.addfile(m_info, io.BytesIO(mbytes))
            meta = {"format_version": 2, "session_id": "forged-src",
                    "export_id": "exp-forged", "exported_at": "now",
                    "runtime_included": True,
                    "included_snapshot_ids": ["forged-snap"]}
            meta_bytes = json.dumps(meta, sort_keys=True, ensure_ascii=False).encode()
            meta_info = tarfile.TarInfo(name="snapshot-v2-export.json")
            meta_info.size = len(meta_bytes)
            tf.addfile(meta_info, io.BytesIO(meta_bytes))

        importer = SnapshotImporter("forged-dest", dest_paths)
        result = importer.import_bundle(archive, trust_runtime=True)

        assert not result.success
        assert len(result.errors) > 0


class TestV1Cutover:
    """V1 snapshots appear in inventory with version warning; V2 cutover works."""

    def test_v1_inventory_built_from_analysis_state(self):
        from backend.services.snapshot_v2.legacy import build_legacy_inventory
        from backend.models.schemas import AnalysisState

        state = AnalysisState(id="v1-test", name="v1", user_id="u1", initial_instruction="x")
        inventory = build_legacy_inventory(state, warning_version="0.32.0")

        assert inventory.snapshot_count == 0
        assert inventory.warning_version == "0.32.0"

    def test_v1_restore_blocked_returns_message(self):
        from backend.services.snapshot_v2.legacy import should_reject_v1_restore

        msg = should_reject_v1_restore(cutover=True, deprecation_version="0.32.0")
        assert msg is not None
        assert "V1" in msg

    def test_v1_restore_rejected_even_when_cutover_flag_false(self):
        """Batch 4: no config combination re-enables V1 restore."""
        from backend.services.snapshot_v2.legacy import should_reject_v1_restore

        msg = should_reject_v1_restore(cutover=False)
        assert msg is not None
        assert "V1" in msg
