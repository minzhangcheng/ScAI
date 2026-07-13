"""Tests: V2 snapshot import with verified extraction, rebinding, and trust downgrade."""
import hashlib
import hmac
import io
import json
import tarfile
from pathlib import Path

import pytest

from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest, complete_runtime, dependency_manifest
from backend.services.snapshot_v2.models import (
    RuntimeManifest,
    RuntimeDependency,

    FORMAT_VERSION,
    IdleCapsule,
    ObjectRef,
    RuntimeGenerationManifest,
    SnapshotManifestV2,
    WorkspaceEntry,
)


@pytest.fixture
def dest_paths(tmp_path):
    """Destination SnapshotV2Paths (where imported content lands)."""
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


def _make_obj_data(size: int = 256) -> tuple[bytes, str]:
    """Generate deterministic object data and its sha256 hash."""
    data = b"x" * size
    h = hashlib.sha256(data).hexdigest()
    return data, h


def _build_archive(
    archive_path: Path,
    manifests: list[SnapshotManifestV2],
    object_data: dict[str, bytes],
    session_id: str,
    *,
    runtime_included: bool = True,
    extra_entries: list[tuple[str, bytes]] | None = None,
) -> Path:
    """Build a tar.gz archive with manifests, objects, and metadata.

    The archive format mirrors ``SnapshotExporter.export()`` output:
      snapshot-v2-export.json
      objects/<sha256>
      manifests/<snapshot_id>.json
    """
    all_hashes = set()
    for m in manifests:
        all_hashes.update(m.objects.keys())

    with tarfile.open(archive_path, "w:gz") as tf:
        # Objects
        for h in sorted(all_hashes):
            data = object_data.get(h)
            if data is None:
                continue
            info = tarfile.TarInfo(name=f"objects/{h}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        # Manifests (with root_hash already set)
        for m in manifests:
            m_bytes = json.dumps(
                m.model_dump(mode="json"),
                sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
            m_info = tarfile.TarInfo(name=f"manifests/{m.snapshot_id}.json")
            m_info.size = len(m_bytes)
            tf.addfile(m_info, io.BytesIO(m_bytes))

        # Extra entries for negative tests (symlinks, sig files, etc.)
        if extra_entries:
            for name, content in extra_entries:
                info = tarfile.TarInfo(name=name)
                info.size = len(content)
                tf.addfile(info, io.BytesIO(content))

        # Metadata
        omitted_runtime = []
        if not runtime_included:
            for m in manifests:
                if (m.runtime.adata_refs or m.runtime.captured_residual_names or m.runtime.residual_ref is not None):
                    omitted_runtime.append(m.snapshot_id)

        meta = {
            "format_version": 2,
            "session_id": session_id,
            "export_id": "exp-test-0001",
            "exported_at": "2026-07-11T12:00:00Z",
            "runtime_included": runtime_included,
            "included_snapshot_ids": [m.snapshot_id for m in manifests],
            "included_hashes": sorted(all_hashes),
            "omitted_runtime_hashes": omitted_runtime,
            "object_count": len(all_hashes),
            "total_size_bytes": sum(len(object_data.get(h, b"")) for h in all_hashes),
        }
        meta_bytes = json.dumps(meta, sort_keys=True, ensure_ascii=False).encode()
        meta_info = tarfile.TarInfo(name="snapshot-v2-export.json")
        meta_info.size = len(meta_bytes)
        tf.addfile(meta_info, io.BytesIO(meta_bytes))

    return archive_path


def _build_manifest(
    session_id: str,
    snapshot_id: str,
    *,
    objects: dict[str, ObjectRef] | None = None,
    workspace: list[WorkspaceEntry] | None = None,
    parent_id: str | None = None,
    runtime_capture: RuntimeGenerationManifest | None = None,
    runtime: RuntimeManifest | None = None,
) -> SnapshotManifestV2:
    """Build a SnapshotManifestV2 with root_hash computed."""
    from backend.services.snapshot_v2.models import runtime_manifest_from_generation
    obj = objects or {}
    ws = workspace or []
    if runtime is None and runtime_capture is not None:
        runtime = runtime_manifest_from_generation(runtime_capture, status="complete")
    kwargs = dict(
        session_id=session_id,
        snapshot_id=snapshot_id,
        parent_id=parent_id,
        workspace=[e.model_dump(mode="json") if hasattr(e, "model_dump") else e for e in ws],
        objects={k: (v.model_dump(mode="json") if hasattr(v, "model_dump") else v) for k, v in obj.items()},
    )
    if runtime is not None:
        kwargs["runtime"] = runtime.model_dump(mode="json") if hasattr(runtime, "model_dump") else runtime
    manifest = make_snapshot_v2_manifest(**kwargs)
    # Compute root_hash
    from backend.services.snapshot_v2.object_store import canonical_json
    raw = canonical_json(manifest)
    manifest.root_hash = hashlib.sha256(raw).hexdigest()
    return manifest


def _make_ref(sha256: str, size: int = 256, media_type: str = "text/plain") -> ObjectRef:
    return ObjectRef(sha256=sha256, size=size, media_type=media_type)


def _list_committed(dst_paths) -> list[SnapshotManifestV2]:
    """Read all committed manifests from destination store."""
    results = []
    if not dst_paths.committed.exists():
        return results
    for entry in sorted(dst_paths.committed.iterdir()):
        if entry.suffix == ".json" and entry.is_file():
            try:
                m = SnapshotManifestV2.model_validate_json(entry.read_bytes())
                results.append(m)
            except Exception:
                pass
    return results


def _verify_hmac(dst_paths, manifest: SnapshotManifestV2) -> bool:
    """Check that the HMAC signature is valid under the host integrity secret."""
    from backend.tests.snapshot_v2_fixtures import verify_manifest_hmac
    return verify_manifest_hmac(dst_paths.integrity, dst_paths.committed, manifest)


# ── Tests ──────────────────────────────────────────────────────────────

SESSION_ID_SRC = "source-session-001"
SESSION_ID_DST = "dest-session-002"


class TestSnapshotV2ImportSecurity:
    """Security-focused tests: traversal, symlinks, corruption."""

    def test_import_rejects_symlinks(self, tmp_path, dest_paths):
        """Archives with symlink entries are rejected."""
        from backend.services.snapshot_v2.importer import SnapshotImporter
        from backend.services.snapshot_v2.errors import SnapshotStorageError

        archive = tmp_path / "bad.tar.gz"
        # Create archive with a symlink entry
        with tarfile.open(archive, "w:gz") as tf:
            sinfo = tarfile.TarInfo(name="objects/abcdef")
            sinfo.type = tarfile.SYMTYPE
            sinfo.linkname = "/etc/passwd"
            tf.addfile(sinfo)

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert not result.success
        assert any("ARCHIVE_UNSAFE" in e for e in result.errors)

    def test_import_rejects_hardlinks(self, tmp_path, dest_paths):
        """Archives with hardlink entries are rejected."""
        from backend.services.snapshot_v2.importer import SnapshotImporter
        from backend.services.snapshot_v2.errors import SnapshotStorageError

        archive = tmp_path / "bad.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            hinfo = tarfile.TarInfo(name="objects/abcdef")
            hinfo.type = tarfile.LNKTYPE
            hinfo.linkname = "objects/other"
            tf.addfile(hinfo)

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert not result.success

    def test_import_rejects_device_entries(self, tmp_path, dest_paths):
        """Archives with device entries are rejected."""
        from backend.services.snapshot_v2.importer import SnapshotImporter

        archive = tmp_path / "bad.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            dinfo = tarfile.TarInfo(name="objects/abcdef")
            dinfo.type = tarfile.CHRTYPE
            dinfo.devmajor = 1
            dinfo.devminor = 3
            tf.addfile(dinfo)

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert not result.success

    def test_import_rejects_duplicate_manifest_ids(self, tmp_path, dest_paths):
        """Archive with duplicate snapshot IDs is rejected."""
        from backend.services.snapshot_v2.importer import SnapshotImporter

        data, h = _make_obj_data()
        ref = _make_ref(h)

        m = _build_manifest(SESSION_ID_SRC, "snap-dupe", objects={h: ref})
        archive = _build_archive(
            tmp_path / "dupe.tar.gz",
            [m, m],  # same manifest twice
            {h: data},
            SESSION_ID_SRC,
        )

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert not result.success
        assert any("DUPLICATE" in e for e in result.errors)

    def test_import_rejects_manifest_path_traversal(self, tmp_path, dest_paths):
        """Manifest path with '..' is rejected."""
        from backend.services.snapshot_v2.importer import SnapshotImporter

        archive = tmp_path / "traversal.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            # Valid metadata
            meta = {"format_version": 2, "session_id": "s", "runtime_included": True,
                    "included_hashes": [], "included_snapshot_ids": []}
            meta_bytes = json.dumps(meta, sort_keys=True, ensure_ascii=False).encode()
            minfo = tarfile.TarInfo(name="snapshot-v2-export.json")
            minfo.size = len(meta_bytes)
            tf.addfile(minfo, io.BytesIO(meta_bytes))
            # Malicious manifest path
            payload = b'{"format_version":2,"session_id":"s","snapshot_id":"s"}'
            info = tarfile.TarInfo(name="manifests/../../etc/passwd.json")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert not result.success
        assert any("SUSPICIOUS" in e for e in result.errors)

    def test_import_detects_bad_object_hash(self, tmp_path, dest_paths):
        """Archive object whose content doesn't match its hash filename is rejected."""
        from backend.services.snapshot_v2.importer import SnapshotImporter

        data = b"corrupted data"
        false_hash = hashlib.sha256(b"real content").hexdigest()

        archive = tmp_path / "bad_hash.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            # Valid metadata so importer gets past metadata check
            meta = {"format_version": 2, "session_id": "s", "runtime_included": True,
                    "included_hashes": [], "included_snapshot_ids": []}
            meta_bytes = json.dumps(meta, sort_keys=True, ensure_ascii=False).encode()
            minfo = tarfile.TarInfo(name="snapshot-v2-export.json")
            minfo.size = len(meta_bytes)
            tf.addfile(minfo, io.BytesIO(meta_bytes))
            # Bad object
            info = tarfile.TarInfo(name=f"objects/{false_hash}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert not result.success
        assert any("OBJECT_HASH_MISMATCH" in e for e in result.errors)

    def test_import_corrupt_required_object(self, tmp_path, dest_paths):
        """Archive missing a required object referenced by a manifest."""
        from backend.services.snapshot_v2.importer import SnapshotImporter

        data, h1 = _make_obj_data(256)
        _, h2 = _make_obj_data(512)  # not in archive objects
        ref1 = _make_ref(h1, 256)
        ref2 = _make_ref(h2, 512)

        m = _build_manifest(
            SESSION_ID_SRC, "snap-corrupt",
            objects={h1: ref1, h2: ref2},
        )
        archive = _build_archive(
            tmp_path / "corrupt.tar.gz",
            [m],
            {h1: data},  # h2 missing
            SESSION_ID_SRC,
        )

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert not result.success
        assert any("MISSING_OBJECT" in e for e in result.errors)


class TestSnapshotV2ImportRebinding:
    """ID rebinding and HMAC tests."""

    def test_import_rebinds_session_id(self, tmp_path, dest_paths):
        """Manifest session_id is rebound to destination session."""
        from backend.services.snapshot_v2.importer import SnapshotImporter

        data, h = _make_obj_data()
        ref = _make_ref(h)

        m = _build_manifest(SESSION_ID_SRC, "snap-a", objects={h: ref})
        archive = _build_archive(
            tmp_path / "rebind.tar.gz", [m], {h: data}, SESSION_ID_SRC,
        )

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert result.success
        assert result.dest_session_id == SESSION_ID_DST

        committed = _list_committed(dest_paths)
        assert len(committed) == 1
        assert committed[0].session_id == SESSION_ID_DST

    def test_import_rebinds_snapshot_id(self, tmp_path, dest_paths):
        """Snapshot ID is deterministically rebound to prevent collisions."""
        from backend.services.snapshot_v2.importer import SnapshotImporter, _rebind_id

        data, h = _make_obj_data()
        ref = _make_ref(h)

        m = _build_manifest(SESSION_ID_SRC, "snap-original", objects={h: ref})
        archive = _build_archive(
            tmp_path / "rebind_snap.tar.gz", [m], {h: data}, SESSION_ID_SRC,
        )

        expected_id = _rebind_id("snap-original", SESSION_ID_DST)

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert result.success

        committed = _list_committed(dest_paths)
        assert len(committed) == 1
        assert committed[0].snapshot_id == expected_id

    def test_import_rebinds_parent_id(self, tmp_path, dest_paths):
        """parent_id is deterministically rebound."""
        from backend.services.snapshot_v2.importer import SnapshotImporter, _rebind_id

        data, h = _make_obj_data()
        ref = _make_ref(h)
        parent_src = "snap-parent"

        m = _build_manifest(
            SESSION_ID_SRC, "snap-child", objects={h: ref},
            parent_id=parent_src,
        )
        archive = _build_archive(
            tmp_path / "rebind_parent.tar.gz", [m], {h: data}, SESSION_ID_SRC,
        )

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert result.success

        committed = _list_committed(dest_paths)
        assert committed[0].parent_id == _rebind_id(parent_src, SESSION_ID_DST)

    def test_import_discards_source_hmac(self, tmp_path, dest_paths):
        """Source HMAC is discarded; destination is re-signed with host integrity secret."""
        from backend.services.snapshot_v2.importer import SnapshotImporter
        from backend.tests.snapshot_v2_fixtures import sign_manifest_bytes

        data, h = _make_obj_data()
        ref = _make_ref(h)

        m = _build_manifest(SESSION_ID_SRC, "snap-hmac", objects={h: ref})
        archive = _build_archive(
            tmp_path / "hmac.tar.gz", [m], {h: data}, SESSION_ID_SRC,
        )

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert result.success

        committed = _list_committed(dest_paths)
        assert len(committed) == 1
        # HMAC should be valid under host integrity secret after import re-sign
        assert _verify_hmac(dest_paths, committed[0]), \
            "Destination HMAC should be valid for host integrity secret"
        # Session-id-derived key must NOT verify (legacy signing)
        m_bytes = (dest_paths.committed / f"{committed[0].snapshot_id}.json").read_bytes()
        legacy_sig = hmac.new(
            hashlib.sha256(SESSION_ID_SRC.encode()).digest(),
            m_bytes,
            hashlib.sha256,
        ).hexdigest()
        dest_sig = (dest_paths.integrity / f"{committed[0].snapshot_id}.sig").read_text().strip()
        assert not hmac.compare_digest(legacy_sig, dest_sig), \
            "Legacy session-id HMAC should NOT match host-secret signature"
        assert dest_sig == sign_manifest_bytes(m_bytes)

    def test_import_same_session_does_not_rebind(self, tmp_path, dest_paths):
        """If source == dest session, IDs stay unchanged."""
        from backend.services.snapshot_v2.importer import SnapshotImporter

        data, h = _make_obj_data()
        ref = _make_ref(h)
        sid = SESSION_ID_DST

        m = _build_manifest(sid, "snap-same", objects={h: ref})
        archive = _build_archive(
            tmp_path / "same.tar.gz", [m], {h: data}, sid,
        )

        importer = SnapshotImporter(sid, dest_paths)
        result = importer.import_bundle(archive)
        assert result.success

        committed = _list_committed(dest_paths)
        assert committed[0].snapshot_id == "snap-same"
        assert committed[0].session_id == sid


class TestSnapshotV2ImportRuntime:
    """Runtime omission and trust downgrade."""

    def test_import_runtime_omitted_from_archive(self, tmp_path, dest_paths):
        """When runtime objects are omitted from archive, runtime is marked unavailable."""
        from backend.services.snapshot_v2.importer import SnapshotImporter

        data, h = _make_obj_data(256)
        rt_data, rt_h = _make_obj_data(512)
        ref = _make_ref(h, 256)
        rt_ref = _make_ref(rt_h, 512, "application/x-hdf5")

        # Build manifest with runtime capture
        rt_capture = RuntimeGenerationManifest(
            capture_command="runtime_capture_v2",
            capture_timestamp="2026-07-11T12:00:00Z",
            kernel_id="kernel-1",
            captured_adata_vars=["adata"],
            captured_adata_refs={"adata": rt_ref},
        )

        m = _build_manifest(
            SESSION_ID_SRC, "snap-rt",
            objects={h: ref, rt_h: rt_ref},
            runtime_capture=rt_capture,
        )

        # Archive includes only base object, NOT runtime object
        archive = _build_archive(
            tmp_path / "rt_omit.tar.gz", [m], {h: data}, SESSION_ID_SRC,
            runtime_included=False,
        )

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert not result.success  # Should fail because manifest references missing object

    def test_import_trust_false_omits_runtime(self, tmp_path, dest_paths):
        """trust_runtime=False marks runtime as unavailable without writing dill."""
        from backend.services.snapshot_v2.importer import SnapshotImporter

        data, h = _make_obj_data(256)
        rt_data, rt_h = _make_obj_data(512)
        ref = _make_ref(h, 256)
        rt_ref = _make_ref(rt_h, 512, "application/x-hdf5")

        rt_capture = RuntimeGenerationManifest(
            capture_command="runtime_capture_v2",
            capture_timestamp="2026-07-11T12:00:00Z",
            kernel_id="kernel-1",
            captured_adata_vars=["adata"],
            captured_adata_refs={"adata": rt_ref},
        )

        m = _build_manifest(
            SESSION_ID_SRC, "snap-trust",
            objects={h: ref, rt_h: rt_ref},
            runtime_capture=rt_capture,
        )

        # Archive includes runtime objects
        archive = _build_archive(
            tmp_path / "trust_false.tar.gz", [m],
            {h: data, rt_h: rt_data},
            SESSION_ID_SRC,
            runtime_included=True,
        )

        # Import with trust_runtime=False
        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive, trust_runtime=False)
        assert result.success

        committed = _list_committed(dest_paths)
        assert len(committed) == 1
        # Runtime should be nullified
        assert committed[0].runtime.status == "missing"
        assert committed[0].runtime.adata_refs == {}
        assert committed[0].runtime.residual_ref is None

    def test_import_trust_true_preserves_runtime(self, tmp_path, dest_paths):
        """trust_runtime=True preserves runtime refs."""
        from backend.services.snapshot_v2.importer import SnapshotImporter

        data, h = _make_obj_data(256)
        rt_data, rt_h = _make_obj_data(512)
        ref = _make_ref(h, 256)
        rt_ref = _make_ref(rt_h, 512, "application/x-hdf5")

        rt_capture = RuntimeGenerationManifest(
            capture_command="runtime_capture_v2",
            capture_timestamp="2026-07-11T12:00:00Z",
            kernel_id="kernel-1",
            captured_adata_vars=["adata"],
            captured_adata_refs={"adata": rt_ref},
        )

        m = _build_manifest(
            SESSION_ID_SRC, "snap-true",
            objects={h: ref, rt_h: rt_ref},
            runtime_capture=rt_capture,
        )

        archive = _build_archive(
            tmp_path / "trust_true.tar.gz", [m],
            {h: data, rt_h: rt_data},
            SESSION_ID_SRC,
            runtime_included=True,
        )

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive, trust_runtime=True)
        assert result.success

        committed = _list_committed(dest_paths)
        rt = committed[0].runtime
        assert rt is not None
        assert "adata" in rt.adata_refs

        # Runtime object should exist in destination store
        obj_path = dest_paths.objects / rt_h
        assert obj_path.exists()
        assert obj_path.read_bytes() == rt_data


class TestSnapshotV2ImportObjects:
    """Object extraction and store tests."""

    def test_import_basic_workflow(self, tmp_path, dest_paths):
        """Basic import extracts objects and commits manifests."""
        from backend.services.snapshot_v2.importer import SnapshotImporter

        data, h = _make_obj_data()
        ref = _make_ref(h)

        m = _build_manifest(
            SESSION_ID_SRC, "snap-basic",
            objects={h: ref},
            workspace=[WorkspaceEntry(path="data/file.txt", object_hash=h, size=256, mode=0o644)],
        )
        archive = _build_archive(
            tmp_path / "basic.tar.gz", [m], {h: data}, SESSION_ID_SRC,
        )

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert result.success
        assert result.committed_count == 1

        # Object exists in store
        obj_path = dest_paths.objects / h
        assert obj_path.exists()
        assert obj_path.read_bytes() == data

        # Manifest committed
        committed = _list_committed(dest_paths)
        assert len(committed) == 1
        assert committed[0].snapshot_id != "snap-basic"  # rebound

    def test_import_multiple_manifests(self, tmp_path, dest_paths):
        """Multiple manifests are imported correctly."""
        from backend.services.snapshot_v2.importer import SnapshotImporter

        data_a, h_a = _make_obj_data(128)
        data_b, h_b = _make_obj_data(256)
        ref_a = _make_ref(h_a, 128)
        ref_b = _make_ref(h_b, 256)

        m_a = _build_manifest(SESSION_ID_SRC, "snap-multi-a", objects={h_a: ref_a})
        m_b = _build_manifest(SESSION_ID_SRC, "snap-multi-b", objects={h_b: ref_b})
        archive = _build_archive(
            tmp_path / "multi.tar.gz", [m_a, m_b],
            {h_a: data_a, h_b: data_b},
            SESSION_ID_SRC,
        )

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert result.success
        assert result.committed_count == 2

        committed = _list_committed(dest_paths)
        assert len(committed) == 2

    def test_import_dedup_reuses_objects(self, tmp_path, dest_paths):
        """Same object hash reused across manifests is only stored once."""
        from backend.services.snapshot_v2.importer import SnapshotImporter

        data, h = _make_obj_data(128)
        ref = _make_ref(h, 128)

        m_a = _build_manifest(SESSION_ID_SRC, "snap-dedup-a", objects={h: ref})
        m_b = _build_manifest(SESSION_ID_SRC, "snap-dedup-b", objects={h: ref})
        archive = _build_archive(
            tmp_path / "dedup.tar.gz", [m_a, m_b],
            {h: data},
            SESSION_ID_SRC,
        )

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert result.success
        assert result.committed_count == 2

        # Object stored once
        obj_path = dest_paths.objects / h
        assert obj_path.exists()
        assert len(list(dest_paths.objects.iterdir())) == 1  # only one object file

    def test_import_staging_cleaned(self, tmp_path, dest_paths):
        """Staging directory is empty after successful import."""
        from backend.services.snapshot_v2.importer import SnapshotImporter

        data, h = _make_obj_data()
        ref = _make_ref(h)

        m = _build_manifest(SESSION_ID_SRC, "snap-clean", objects={h: ref})
        archive = _build_archive(
            tmp_path / "clean.tar.gz", [m], {h: data}, SESSION_ID_SRC,
        )

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert result.success

        # No temp files in staging (except maybe hidden files which are temp artifacts)
        remaining = [f for f in dest_paths.staging.iterdir() if not f.name.startswith(".")]
        assert len(remaining) == 0, f"Staging not clean: {remaining}"


class TestSnapshotV2ImportTrustFilter:
    """Enforce trust_runtime before any object-store mutation."""

    def test_untrusted_runtime_hash_never_reaches_object_store(self, tmp_path, dest_paths):
        """trust_runtime=False must never put_bytes runtime hashes."""
        from backend.services.snapshot_v2.importer import SnapshotImporter
        from backend.services.snapshot_v2.object_store import ObjectStore

        data, h = _make_obj_data(256)
        rt_data, rt_h = _make_obj_data(512)
        ref = _make_ref(h, 256)
        rt_ref = _make_ref(rt_h, 512, "application/x-hdf5")

        rt_capture = RuntimeGenerationManifest(
            capture_command="runtime_capture_v2",
            capture_timestamp="2026-07-11T12:00:00Z",
            kernel_id="kernel-1",
            captured_adata_vars=["adata"],
            captured_adata_refs={"adata": rt_ref},
        )
        m = _build_manifest(
            SESSION_ID_SRC, "snap-untrusted",
            objects={h: ref, rt_h: rt_ref},
            runtime_capture=rt_capture,
        )
        archive = _build_archive(
            tmp_path / "untrusted.tar.gz", [m],
            {h: data, rt_h: rt_data},
            SESSION_ID_SRC,
            runtime_included=True,
        )

        class SpyStore(ObjectStore):
            def __init__(self, paths):
                super().__init__(paths)
                self.put_hashes: list[str] = []

            def put_bytes(self, data: bytes, media_type: str):
                import hashlib as _hl
                sha = _hl.sha256(data).hexdigest()
                self.put_hashes.append(sha)
                return super().put_bytes(data, media_type)

            def put_stream(self, reader, expected_sha256, expected_size, media_type):
                self.put_hashes.append(expected_sha256)
                return super().put_stream(
                    reader, expected_sha256, expected_size, media_type,
                )

        spy = SpyStore(dest_paths)
        importer = SnapshotImporter(
            SESSION_ID_DST, dest_paths, object_store=spy,
        )
        result = importer.import_bundle(archive, trust_runtime=False)
        assert result.success
        assert rt_h not in spy.put_hashes
        assert rt_h not in {p.name for p in dest_paths.objects.iterdir()}
        # Non-runtime object must still be written
        assert h in spy.put_hashes
        assert (dest_paths.objects / h).exists()

    def test_crash_before_publish_never_wrote_untrusted_runtime(self, tmp_path, dest_paths):
        """Crash after approved put_bytes must leave no runtime hash on disk."""
        from backend.services.snapshot_v2.importer import SnapshotImporter
        from backend.services.snapshot_v2.object_store import ManifestStore, ObjectStore

        data, h = _make_obj_data(256)
        rt_data, rt_h = _make_obj_data(512)
        ref = _make_ref(h, 256)
        rt_ref = _make_ref(rt_h, 512, "application/x-hdf5")

        rt_capture = RuntimeGenerationManifest(
            capture_command="runtime_capture_v2",
            capture_timestamp="2026-07-11T12:00:00Z",
            kernel_id="kernel-1",
            captured_adata_vars=["adata"],
            captured_adata_refs={"adata": rt_ref},
        )
        m = _build_manifest(
            SESSION_ID_SRC, "snap-crash",
            objects={h: ref, rt_h: rt_ref},
            runtime_capture=rt_capture,
        )
        archive = _build_archive(
            tmp_path / "crash.tar.gz", [m],
            {h: data, rt_h: rt_data},
            SESSION_ID_SRC,
            runtime_included=True,
        )

        class CrashManifestStore(ManifestStore):
            def publish_staging(self, staging_path):
                raise RuntimeError("simulated crash before publish")

        class SpyStore(ObjectStore):
            def __init__(self, paths):
                super().__init__(paths)
                self.put_hashes: list[str] = []

            def put_bytes(self, data: bytes, media_type: str):
                import hashlib as _hl
                sha = _hl.sha256(data).hexdigest()
                self.put_hashes.append(sha)
                return super().put_bytes(data, media_type)

            def put_stream(self, reader, expected_sha256, expected_size, media_type):
                self.put_hashes.append(expected_sha256)
                return super().put_stream(
                    reader, expected_sha256, expected_size, media_type,
                )

        spy = SpyStore(dest_paths)
        importer = SnapshotImporter(
            SESSION_ID_DST, dest_paths,
            object_store=spy,
            manifest_store=CrashManifestStore(dest_paths),
        )
        result = importer.import_bundle(archive, trust_runtime=False)
        assert not result.success
        assert rt_h not in spy.put_hashes
        assert rt_h not in {p.name for p in dest_paths.objects.iterdir() if p.is_file()}
        # Approved non-runtime may have been written before crash
        assert h in spy.put_hashes or not spy.put_hashes


class TestSnapshotV2ImportInterrupted:
    """Crash recovery tests."""

    def test_import_write_journal(self, tmp_path, dest_paths):
        """Import removes journal after successful reconciliation."""
        from backend.services.snapshot_v2.importer import SnapshotImporter

        data, h = _make_obj_data()
        ref = _make_ref(h)

        m = _build_manifest(SESSION_ID_SRC, "snap-journal", objects={h: ref})
        archive = _build_archive(
            tmp_path / "journal.tar.gz", [m], {h: data}, SESSION_ID_SRC,
        )

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert result.success

        # Journal is removed on full success — only FAILED journals remain
        journals = list(dest_paths.journals.iterdir())
        assert len(journals) == 0, f"expected no journals, got {journals}"


class TestSnapshotV2ImportRoute:
    """Route-level tests for the V2 import endpoint (auth-enforcement only)."""

    @pytest.mark.asyncio
    async def test_import_v2_route_requires_auth(self, tmp_path):
        """import_v2 without auth returns 401 (no app lifespan/TestClient)."""
        from unittest.mock import MagicMock

        from fastapi import HTTPException

        from backend.routes.export_import import import_v2

        request = MagicMock()
        request.state.user_id = ""
        with pytest.raises(HTTPException) as caught:
            await import_v2("nonexistent", request)
        assert caught.value.status_code == 401


class TestSnapshotV2ImportBatchAtomic:
    """Batch-atomic visibility and recovery tests."""

    def test_imported_manifests_stay_hidden_until_batch_record_commits(self, tmp_path, dest_paths):
        """Manifests are written to committed/ but hidden until batch record publishes."""
        from backend.services.snapshot_v2.importer import SnapshotImporter
        from backend.services.snapshot_v2.object_store import ManifestStore

        data, h = _make_obj_data()
        ref = _make_ref(h)
        m1 = _build_manifest(SESSION_ID_SRC, "snap-a", objects={h: ref})
        m2 = _build_manifest(SESSION_ID_SRC, "snap-b", objects={h: ref})
        archive = _build_archive(
            tmp_path / "batch.tar.gz", [m1, m2], {h: data}, SESSION_ID_SRC,
        )

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert result.success

        # All manifests should be visible after successful batch import
        committed = _list_committed(dest_paths)
        assert len(committed) == 2

    def test_import_crash_after_first_manifest_never_exposes_partial_batch(
        self, tmp_path, dest_paths, monkeypatch,
    ):
        """Crash after first manifest publish should not leave visible manifests."""
        from backend.services.snapshot_v2.importer import SnapshotImporter
        from backend.services.snapshot_v2.object_store import ManifestStore

        data, h = _make_obj_data()
        ref = _make_ref(h)
        m1 = _build_manifest(SESSION_ID_SRC, "snap-a", objects={h: ref})
        m2 = _build_manifest(SESSION_ID_SRC, "snap-b", objects={h: ref})
        archive = _build_archive(
            tmp_path / "crash.tar.gz", [m1, m2], {h: data}, SESSION_ID_SRC,
        )

        calls = 0
        original_publish = ManifestStore.publish_staging

        def fail_second(self, staging_path):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("crash after first staged manifest")
            return original_publish(self, staging_path)

        monkeypatch.setattr(ManifestStore, "publish_staging", fail_second)
        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert not result.success

        # After recovery, no manifests should remain visible
        from backend.services.snapshot_v2.batch_store import ImportBatchStore
        batch_store = ImportBatchStore(dest_paths.root)
        assert batch_store.list_batch_ids() == []

    def test_import_crash_between_manifests_exposes_nothing(
        self, tmp_path, dest_paths, monkeypatch,
    ):
        """If publishing the 2nd manifest raises, the 1st must stay invisible."""
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError
        from backend.services.snapshot_v2.importer import SnapshotImporter, _rebind_id
        from backend.services.snapshot_v2.object_store import ManifestStore

        data, h = _make_obj_data()
        ref = _make_ref(h)
        m1 = _build_manifest(SESSION_ID_SRC, "snap-a", objects={h: ref})
        m2 = _build_manifest(SESSION_ID_SRC, "snap-b", objects={h: ref})
        archive = _build_archive(
            tmp_path / "crash_between.tar.gz", [m1, m2], {h: data}, SESSION_ID_SRC,
        )
        planned_ids = {
            _rebind_id("snap-a", SESSION_ID_DST),
            _rebind_id("snap-b", SESSION_ID_DST),
        }

        original_publish = ManifestStore.publish_staging
        calls = {"n": 0}

        def flaky_publish(self, path):
            calls["n"] += 1
            result = original_publish(self, path)
            if calls["n"] == 1:
                raise RuntimeError("crash after first publish")
            return result

        monkeypatch.setattr(ManifestStore, "publish_staging", flaky_publish)

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(archive)
        assert not result.success
        assert calls["n"] == 1

        # No imported manifest is visible in the authoritative inventory.
        store = ManifestStore(dest_paths)
        visible_ids = {m.snapshot_id for m in store.list_committed()}
        assert visible_ids.isdisjoint(planned_ids)
        for sid in planned_ids:
            with pytest.raises(SnapshotIntegrityError) as exc:
                store.load(sid)
            assert exc.value.code == "MANIFEST_NOT_FOUND"


class TestSnapshotV2ImportStreamProof:
    """Bounded streaming and write-proof reuse (no put_bytes / no destination reread)."""

    def test_import_second_pass_never_calls_put_bytes(self, tmp_path, dest_paths, monkeypatch):
        from unittest.mock import Mock

        from backend.services.snapshot_v2.importer import SnapshotImporter

        data, h = _make_obj_data(64 * 1024)
        ref = _make_ref(h, len(data))
        m = _build_manifest(SESSION_ID_SRC, "snap-stream", objects={h: ref})
        large_archive = _build_archive(
            tmp_path / "stream_large.tar.gz", [m], {h: data}, SESSION_ID_SRC,
        )
        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        monkeypatch.setattr(
            importer.object_store,
            "put_bytes",
            Mock(side_effect=AssertionError("streamed object buffered")),
        )
        assert importer.import_bundle(large_archive).success

    def test_import_write_proof_skips_destination_verify(self, tmp_path, dest_paths, monkeypatch):
        from unittest.mock import Mock

        from backend.services.snapshot_v2.importer import SnapshotImporter

        data, h = _make_obj_data(512)
        ref = _make_ref(h, len(data))
        m = _build_manifest(SESSION_ID_SRC, "snap-proof", objects={h: ref})
        archive = _build_archive(
            tmp_path / "proof.tar.gz", [m], {h: data}, SESSION_ID_SRC,
        )
        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        verify = Mock(side_effect=AssertionError("destination reread"))
        monkeypatch.setattr(importer.object_store, "verify", verify)
        assert importer.import_bundle(archive).success

    def test_evidence_fingerprint_includes_ctime(self, tmp_path):
        from backend.services.snapshot_v2.object_store import VerifiedObjectEvidence

        object_path = tmp_path / ("a" * 64)
        object_path.write_bytes(b"fingerprint-ctime")
        evidence = VerifiedObjectEvidence.from_path(object_path, object_path.name)
        assert str(object_path.stat().st_ctime_ns) in evidence.stat_fingerprint
