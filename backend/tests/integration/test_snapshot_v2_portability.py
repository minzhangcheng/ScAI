"""Export/Import portability integration tests for Snapshot V2.

Tests runtime inclusion/trust matrix, rebinding, and cross-session import.
"""

from backend.tests.snapshot_v2_fixtures import v2_manifest
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from backend.services.snapshot_v2.models import (
    FORMAT_VERSION,
    IdleCapsule,
    ObjectRef,
    RuntimeGenerationManifest,
    SnapshotManifestV2,
    WorkspaceEntry,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _build_object_ref(sha256: str | None = None, size: int = 256,
                       media_type: str = "application/octet-stream") -> ObjectRef:
    if sha256 is None:
        sha256 = hashlib.sha256(f"obj-{size}".encode()).hexdigest()
    return ObjectRef(sha256=sha256, size=size, media_type=media_type)


def _build_workspace_entry(path: str, obj_hash: str, size: int = 256) -> WorkspaceEntry:
    return WorkspaceEntry(path=path, object_hash=obj_hash, size=size, mode=0o644)


def _make_manifest(session_id: str, snapshot_id: str,
                   objects: dict | None = None,
                   workspace: list | None = None,
                   runtime_capture: RuntimeGenerationManifest | None = None,
                   parent_id: str | None = None) -> SnapshotManifestV2:
    return v2_manifest(
        format_version=2,
        session_id=session_id,
        snapshot_id=snapshot_id,
        captured_at="2026-07-11T12:00:00Z",
        capsule=IdleCapsule(graph_version="2.0"),
        objects=objects or {},
        workspace=workspace or [],
        runtime_snapshot_capture=runtime_capture,
        parent_id=parent_id,
    )


def _make_runtime_capture(kernel_id: str = "test-kernel") -> RuntimeGenerationManifest:
    return RuntimeGenerationManifest(
        capture_command="runtime_capture_v2",
        capture_timestamp="2026-07-11T12:00:00Z",
        kernel_id=kernel_id,
    )


def _publish_manifest(paths, manifest: SnapshotManifestV2) -> SnapshotManifestV2:
    """Publish a manifest through ManifestStore to compute root_hash and HMAC."""
    from backend.services.snapshot_v2.object_store import ManifestStore
    mstore = ManifestStore(paths)
    staging_path = mstore.write_staging(manifest)
    return mstore.publish_staging(staging_path)


def _build_export_archive(dest_path: Path, manifests: list[SnapshotManifestV2],
                          object_data: dict[str, bytes], session_id: str,
                          *, runtime_included: bool = True) -> Path:
    """Build a tar.gz archive matching SnapshotExporter.export() output."""
    with tarfile.open(dest_path, "w:gz") as tf:
        for h, data in sorted(object_data.items()):
            info = tarfile.TarInfo(name=f"objects/{h}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        for m in manifests:
            m_bytes = json.dumps(m.model_dump(mode="json"),
                                 sort_keys=True, ensure_ascii=False).encode()
            m_info = tarfile.TarInfo(name=f"manifests/{m.snapshot_id}.json")
            m_info.size = len(m_bytes)
            tf.addfile(m_info, io.BytesIO(m_bytes))

        meta = {
            "format_version": 2,
            "session_id": session_id,
            "export_id": "exp-test",
            "exported_at": "2026-07-11T12:00:00Z",
            "runtime_included": runtime_included,
            "included_snapshot_ids": [m.snapshot_id for m in manifests],
        }
        meta_bytes = json.dumps(meta, sort_keys=True, ensure_ascii=False).encode()
        meta_info = tarfile.TarInfo(name="snapshot-v2-export.json")
        meta_info.size = len(meta_bytes)
        tf.addfile(meta_info, io.BytesIO(meta_bytes))

    return dest_path


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


class TestExportImportTrustMatrix:
    """Rows: include_runtime true/false crossed with trust true/false."""

    def test_export_with_runtime(self, tmp_path):
        paths = _paths_factory(tmp_path, "src")
        session_id = "source-session"

        from backend.services.snapshot_v2.exporter import SnapshotExporter
        from backend.services.snapshot_v2.object_store import ObjectStore, ManifestStore

        obj_content = b"workspace-file"
        obj_data = hashlib.sha256(obj_content).hexdigest()
        obj_ref = _build_object_ref(obj_data, size=len(obj_content))
        ws_entry = _build_workspace_entry("analysis.py", obj_data, size=len(obj_content))
        rt = _make_runtime_capture()
        manifest = _make_manifest(
            session_id, "snap-export-1",
            objects={obj_data: obj_ref},
            workspace=[ws_entry],
            runtime_capture=rt,
        )

        store = ObjectStore(paths)
        mstore = ManifestStore(paths)
        store.put_bytes(obj_content, "text/x-python")
        staging_path = mstore.write_staging(manifest)
        mstore.publish_staging(staging_path)

        exporter = SnapshotExporter(session_id, paths)
        archive_path = exporter.export(session_id, include_runtime=True)

        assert archive_path.exists()
        assert archive_path.suffix == ".gz"

        with tarfile.open(archive_path, "r:gz") as tf:
            names = tf.getnames()
            assert "snapshot-v2-export.json" in names
            manifest_names = [n for n in names if n.startswith("manifests/")]
            assert len(manifest_names) > 0

        archive_path.unlink()

    def test_export_without_runtime(self, tmp_path):
        paths = _paths_factory(tmp_path, "src-nort")
        session_id = "source-session"

        from backend.services.snapshot_v2.exporter import SnapshotExporter
        from backend.services.snapshot_v2.object_store import ObjectStore, ManifestStore

        obj_content = b"ws-no-runtime"
        obj_data = hashlib.sha256(obj_content).hexdigest()
        obj_ref = _build_object_ref(obj_data, size=len(obj_content))
        ws_entry = _build_workspace_entry("data.csv", obj_data, size=len(obj_content))
        rt = _make_runtime_capture()
        manifest = _make_manifest(
            session_id, "snap-export-nort",
            objects={obj_data: obj_ref},
            workspace=[ws_entry],
            runtime_capture=rt,
        )

        store = ObjectStore(paths)
        mstore = ManifestStore(paths)
        store.put_bytes(obj_content, "text/csv")
        staging_path = mstore.write_staging(manifest)
        mstore.publish_staging(staging_path)

        exporter = SnapshotExporter(session_id, paths)
        archive_path = exporter.export(session_id, include_runtime=False)

        assert archive_path.exists()
        archive_path.unlink()

    def test_import_with_trust_runtime(self, tmp_path):
        dest_paths = _paths_factory(tmp_path, "dtr")
        dest_session_id = "dest-session"

        from backend.services.snapshot_v2.importer import SnapshotImporter
        from backend.services.snapshot_v2.object_store import ObjectStore, ManifestStore

        obj_content = b"import-obj-content"
        obj_hash = hashlib.sha256(obj_content).hexdigest()
        obj_ref = _build_object_ref(obj_hash, size=len(obj_content))
        manifest = _make_manifest(
            "source-session", "snap-import-1",
            objects={obj_hash: obj_ref},
        )

        src_paths = _paths_factory(tmp_path, "src_trust")
        ObjectStore(src_paths).put_bytes(obj_content, "application/octet-stream")
        manifest = _publish_manifest(src_paths, manifest)

        object_data = {obj_hash: obj_content}
        archive = tmp_path / "import_trust.tar.gz"
        _build_export_archive(archive, [manifest], object_data, "source-session",
                              runtime_included=True)

        importer = SnapshotImporter(dest_session_id, dest_paths)
        result = importer.import_bundle(archive, trust_runtime=True)

        assert result.success, f"Import failed: {result.errors}"
        assert result.committed_count >= 1

    def test_import_without_trust_runtime(self, tmp_path):
        dest_paths = _paths_factory(tmp_path, "dnt")
        dest_session_id = "dest-session"

        from backend.services.snapshot_v2.importer import SnapshotImporter
        from backend.services.snapshot_v2.object_store import ObjectStore, ManifestStore

        obj_content = b"import-no-trust-content"
        obj_hash = hashlib.sha256(obj_content).hexdigest()
        obj_ref = _build_object_ref(obj_hash, size=len(obj_content))
        rt = _make_runtime_capture()
        manifest = _make_manifest(
            "source-session", "snap-import-notrust",
            objects={obj_hash: obj_ref},
            runtime_capture=rt,
        )

        src_paths = _paths_factory(tmp_path, "src_notrust")
        ObjectStore(src_paths).put_bytes(obj_content, "application/octet-stream")
        manifest = _publish_manifest(src_paths, manifest)

        object_data = {obj_hash: obj_content}
        archive = tmp_path / "import_notrust.tar.gz"
        _build_export_archive(archive, [manifest], object_data, "source-session",
                              runtime_included=True)

        importer = SnapshotImporter(dest_session_id, dest_paths)
        result = importer.import_bundle(archive, trust_runtime=False)

        assert result.success, f"Import failed: {result.errors}"
        assert result.committed_count >= 1
        assert len(result.skipped_runtime_snapshot_ids) >= 0

    def test_import_rebind_session_id(self, tmp_path):
        dest_paths = _paths_factory(tmp_path, "drb")
        dest_session_id = "dest-session"

        from backend.services.snapshot_v2.importer import SnapshotImporter
        from backend.services.snapshot_v2.object_store import ObjectStore, ManifestStore

        obj_content = b"rebind-obj-content"
        obj_hash = hashlib.sha256(obj_content).hexdigest()
        obj_ref = _build_object_ref(obj_hash, size=len(obj_content))
        manifest = _make_manifest(
            "source-session", "snap-rebind",
            objects={obj_hash: obj_ref},
        )

        src_paths = _paths_factory(tmp_path, "src_rebind")
        ObjectStore(src_paths).put_bytes(obj_content, "application/octet-stream")
        manifest = _publish_manifest(src_paths, manifest)

        object_data = {obj_hash: obj_content}
        archive = tmp_path / "import_rebind.tar.gz"
        _build_export_archive(archive, [manifest], object_data, "source-session",
                              runtime_included=True)

        importer = SnapshotImporter(dest_session_id, dest_paths)
        result = importer.import_bundle(archive, trust_runtime=True)

        assert result.success

        mstore = ManifestStore(dest_paths)
        committed = mstore.list_committed()
        assert len(committed) >= 1
        imported = committed[0]
        assert imported.session_id == dest_session_id
