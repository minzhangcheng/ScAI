"""Gate C — Batch 3 storage safety acceptance.

Proves destructive-path lifecycle safety:
- shared-object leaf deletion and last-reference deletion
- corrupted survivor aborts without object removal
- failure after manifest removal keeps APPLYING journal
- repeated startup recovery is idempotent
- active restore_target / pre_restore protection
- untrusted import with crash injection (no runtime bytes)
- include-runtime export/import round-trip
- destination re-sign verification

Every failure case records object directory bytes before and after.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import tarfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from backend.services.snapshot_v2.errors import (
    SnapshotDeleteConflict,
    SnapshotIntegrityError,
)
from backend.services.snapshot_v2.models import (
    ObjectRef,
    RuntimeGenerationManifest,
    SnapshotManifestV2,
    WorkspaceEntry,
)
from backend.tests.snapshot_v2_fixtures import (
    make_snapshot_v2_manifest,
    sign_manifest_bytes,
    verify_manifest_hmac,
    write_manifest_to_store,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def paths(tmp_path):
    class FakePaths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"

        def ensure_dirs(self):
            for d in [
                self.root,
                self.objects,
                self.staging,
                self.journals,
                self.integrity,
                self.committed,
            ]:
                d.mkdir(parents=True, exist_ok=True)

    p = FakePaths()
    p.ensure_dirs()
    return p


SESSION_ID = "gate-c-session"
SESSION_ID_SRC = "gate-c-src"
SESSION_ID_DST = "gate-c-dst"


def build_object_ref(
    sha256: str | None = None,
    size: int = 1024,
    media_type: str = "application/octet-stream",
) -> ObjectRef:
    if sha256 is None:
        sha256 = hashlib.sha256(f"obj-{size}".encode()).hexdigest()
    if not all(c in "0123456789abcdef" for c in sha256):
        sha256 = hashlib.sha256(sha256.encode()).hexdigest()
    return ObjectRef(sha256=sha256, size=size, media_type=media_type)


def write_object(paths, sha256: str, size: int = 1024, data: bytes | None = None) -> None:
    obj_path = paths.objects / sha256
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_bytes(data if data is not None else (b"x" * size))


def object_bytes(paths) -> dict[str, bytes]:
    """Snapshot of all hash-named object file contents."""
    result: dict[str, bytes] = {}
    if not paths.objects.exists():
        return result
    for entry in paths.objects.iterdir():
        if entry.is_file() and len(entry.name) == 64:
            result[entry.name] = entry.read_bytes()
    return result


def object_names(paths) -> set[str]:
    return set(object_bytes(paths))


def build_manifest(
    session_id: str,
    snapshot_id: str,
    *,
    objects: dict[str, ObjectRef] | None = None,
    workspace: list[WorkspaceEntry] | None = None,
    parent_id: str | None = None,
    runtime: dict | None = None,
):
    obj = objects or {}
    ws = workspace or []
    kwargs = dict(
        session_id=session_id,
        snapshot_id=snapshot_id,
        parent_id=parent_id,
        workspace=[
            e.model_dump(mode="json") if hasattr(e, "model_dump") else e for e in ws
        ],
        objects={
            k: (v.model_dump(mode="json") if hasattr(v, "model_dump") else v)
            for k, v in obj.items()
        },
    )
    if runtime is not None:
        kwargs["runtime"] = runtime
    return make_snapshot_v2_manifest(**kwargs)


def corrupt_surviving_manifest(paths, snapshot_id: str = "snap-survivor") -> None:
    committed_path = paths.committed / f"{snapshot_id}.json"
    data = json.loads(committed_path.read_bytes())
    data["created_at"] = "2099-01-01T00:00:00Z"
    committed_path.write_bytes(
        json.dumps(
            data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    )


def setup_shared_object_lineage(paths, session_id: str = SESSION_ID):
    """Parent + child leaf sharing one object; child has unique object + orphan."""
    shared = hashlib.sha256(b"shared-gate-c").hexdigest()
    unique = hashlib.sha256(b"unique-child-gate-c").hexdigest()
    orphan = hashlib.sha256(b"orphan-gate-c").hexdigest()
    ref_shared = build_object_ref(sha256=shared, size=300)
    ref_unique = build_object_ref(sha256=unique, size=150)
    write_object(paths, shared, 300)
    write_object(paths, unique, 150)
    write_object(paths, orphan, 80)

    parent = build_manifest(session_id, "snap-parent", objects={shared: ref_shared})
    child = build_manifest(
        session_id,
        "snap-child",
        objects={shared: ref_shared, unique: ref_unique},
        parent_id="snap-parent",
    )
    write_manifest_to_store(parent, paths.committed, paths.integrity)
    write_manifest_to_store(child, paths.committed, paths.integrity)
    return shared, unique, orphan


def setup_two_leaf_snapshots(paths, session_id: str = SESSION_ID):
    """Two independent leaves + orphan for last-ref / corrupt / recovery tests."""
    hex_a = hashlib.sha256(b"obj-a-gate-c").hexdigest()
    hex_b = hashlib.sha256(b"obj-b-gate-c").hexdigest()
    hex_orphan = hashlib.sha256(b"orphan-two-gate-c").hexdigest()
    ref_a = build_object_ref(sha256=hex_a, size=100)
    ref_b = build_object_ref(sha256=hex_b, size=200)
    write_object(paths, hex_a, 100)
    write_object(paths, hex_b, 200)
    write_object(paths, hex_orphan, 50)

    m_a = build_manifest(session_id, "snap-a", objects={hex_a: ref_a})
    m_b = build_manifest(session_id, "snap-survivor", objects={hex_b: ref_b})
    write_manifest_to_store(m_a, paths.committed, paths.integrity)
    write_manifest_to_store(m_b, paths.committed, paths.integrity)
    return hex_a, hex_b, hex_orphan


def list_delete_journals(paths) -> list[Path]:
    return list(paths.journals.glob("del-*.json"))


def load_delete_journal(paths):
    from backend.services.snapshot_v2.deletion import read_delete_journal

    return read_delete_journal(paths)


def _make_obj_data(size: int = 256, seed: bytes = b"x") -> tuple[bytes, str]:
    data = seed * size if len(seed) == 1 else (seed * ((size // len(seed)) + 1))[:size]
    return data, hashlib.sha256(data).hexdigest()


def _make_ref(sha256: str, size: int = 256, media_type: str = "text/plain") -> ObjectRef:
    return ObjectRef(sha256=sha256, size=size, media_type=media_type)


def _build_import_manifest(
    session_id: str,
    snapshot_id: str,
    *,
    objects: dict[str, ObjectRef] | None = None,
    workspace: list[WorkspaceEntry] | None = None,
    parent_id: str | None = None,
    runtime_capture: RuntimeGenerationManifest | None = None,
) -> SnapshotManifestV2:
    from backend.services.snapshot_v2.models import runtime_manifest_from_generation
    from backend.services.snapshot_v2.object_store import canonical_json

    obj = objects or {}
    ws = workspace or []
    kwargs = dict(
        session_id=session_id,
        snapshot_id=snapshot_id,
        parent_id=parent_id,
        workspace=[
            e.model_dump(mode="json") if hasattr(e, "model_dump") else e for e in ws
        ],
        objects={
            k: (v.model_dump(mode="json") if hasattr(v, "model_dump") else v)
            for k, v in obj.items()
        },
    )
    if runtime_capture is not None:
        rt = runtime_manifest_from_generation(runtime_capture, status="complete")
        kwargs["runtime"] = rt.model_dump(mode="json")
    manifest = make_snapshot_v2_manifest(**kwargs)
    raw = canonical_json(manifest)
    manifest.root_hash = hashlib.sha256(raw).hexdigest()
    return manifest


def _build_archive(
    archive_path: Path,
    manifests: list[SnapshotManifestV2],
    object_data: dict[str, bytes],
    session_id: str,
    *,
    runtime_included: bool = True,
) -> Path:
    all_hashes = set()
    for m in manifests:
        all_hashes.update(m.objects.keys())

    with tarfile.open(archive_path, "w:gz") as tf:
        for h in sorted(all_hashes):
            data = object_data.get(h)
            if data is None:
                continue
            info = tarfile.TarInfo(name=f"objects/{h}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        for m in manifests:
            m_bytes = json.dumps(
                m.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
            m_info = tarfile.TarInfo(name=f"manifests/{m.snapshot_id}.json")
            m_info.size = len(m_bytes)
            tf.addfile(m_info, io.BytesIO(m_bytes))

        omitted_runtime = []
        if not runtime_included:
            for m in manifests:
                if (
                    m.runtime.adata_refs
                    or m.runtime.captured_residual_names
                    or m.runtime.residual_ref is not None
                ):
                    omitted_runtime.append(m.snapshot_id)

        meta = {
            "format_version": 2,
            "session_id": session_id,
            "export_id": "exp-gate-c-0001",
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


def _list_committed(dst_paths) -> list[SnapshotManifestV2]:
    results = []
    if not dst_paths.committed.exists():
        return results
    for entry in sorted(dst_paths.committed.iterdir()):
        if entry.suffix == ".json" and entry.is_file():
            try:
                results.append(SnapshotManifestV2.model_validate_json(entry.read_bytes()))
            except Exception:
                pass
    return results


@pytest.fixture
def deletion_service(paths):
    from backend.services.snapshot_v2.deletion import DeletionService

    return DeletionService(SESSION_ID, paths)


# ---------------------------------------------------------------------------
# Shared-object / last-reference deletion
# ---------------------------------------------------------------------------


class TestSharedAndLastReferenceDeletion:
    def test_shared_object_leaf_deletion_preserves_shared(
        self, paths, deletion_service
    ):
        shared, unique, orphan = setup_shared_object_lineage(paths)
        before = object_bytes(paths)

        result = deletion_service.delete_leaf("snap-child")

        assert "snap-child" in result.deleted_ids
        after = object_bytes(paths)
        assert shared in after and after[shared] == before[shared]
        assert unique not in after
        assert orphan not in after
        assert (paths.committed / "snap-parent.json").exists()
        assert not (paths.committed / "snap-child.json").exists()

    def test_last_reference_deletion_removes_unique_object(
        self, paths, deletion_service
    ):
        hex_a, hex_b, hex_orphan = setup_two_leaf_snapshots(paths)
        before = object_bytes(paths)
        assert hex_a in before

        result = deletion_service.delete_leaf("snap-a")

        assert "snap-a" in result.deleted_ids
        after = object_bytes(paths)
        assert hex_a not in after
        assert hex_b in after and after[hex_b] == before[hex_b]
        assert hex_orphan not in after
        assert not list_delete_journals(paths)


# ---------------------------------------------------------------------------
# Corrupted survivor / failure after manifest removal
# ---------------------------------------------------------------------------


class TestFailurePathsRecordObjectBytes:
    def test_corrupted_survivor_aborts_without_object_removal(
        self, paths, deletion_service
    ):
        hex_a, hex_b, hex_orphan = setup_two_leaf_snapshots(paths)
        corrupt_surviving_manifest(paths, "snap-survivor")
        before = object_bytes(paths)

        with pytest.raises(SnapshotIntegrityError):
            deletion_service.delete_leaf("snap-a")

        after = object_bytes(paths)
        assert after == before
        assert hex_a in after and hex_b in after and hex_orphan in after

    def test_failure_after_manifest_removal_keeps_journal_and_objects(
        self, paths, deletion_service, monkeypatch
    ):
        hex_a, hex_b, hex_orphan = setup_two_leaf_snapshots(paths)
        monkeypatch.setattr(
            deletion_service.gc, "run", Mock(side_effect=OSError("fault"))
        )
        before = object_bytes(paths)

        with pytest.raises(OSError, match="fault"):
            deletion_service.delete_leaf("snap-a")

        after = object_bytes(paths)
        # Manifest removed, but GC fault means objects still present
        assert not (paths.committed / "snap-a.json").exists()
        assert after == before
        journal = load_delete_journal(paths)
        assert journal.status == "APPLYING"
        assert "snap-a" in journal.delete_set


# ---------------------------------------------------------------------------
# Repeated startup recovery
# ---------------------------------------------------------------------------


class TestRepeatedStartupRecovery:
    def test_repeated_startup_recovery_is_idempotent(self, paths, deletion_service):
        from backend.services.snapshot_v2.deletion import (
            _write_journal,
            recover_delete_transaction,
        )
        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions

        hex_a, hex_b, hex_orphan = setup_two_leaf_snapshots(paths)
        (paths.committed / "snap-a.json").unlink()
        (paths.integrity / "snap-a.sig").unlink(missing_ok=True)

        tx_id = "del-gate-c-recover"
        record = {
            "transaction_id": tx_id,
            "session_id": SESSION_ID,
            "delete_set": ["snap-a"],
            "surviving_manifest_ids": ["snap-survivor"],
            "previous_lineage_head": "snap-a",
            "new_lineage_head": None,
            "status": "APPLYING",
        }
        _write_journal(paths, tx_id, record)
        before_recover = object_bytes(paths)
        assert hex_a in before_recover

        # First recovery via public API
        first = recover_delete_transaction(
            record, paths=paths, session_id=SESSION_ID
        )
        assert "snap-a" in first.deleted_ids
        assert not (paths.objects / hex_a).exists()
        assert (paths.objects / hex_b).exists()
        assert not (paths.journals / f"{tx_id}.json").exists()
        mid = object_bytes(paths)

        # Simulate re-appearing APPLYING journal (crash before journal cleanup)
        _write_journal(paths, tx_id, {**record, "status": "APPLYING"})
        second = recover_delete_transaction(
            record, paths=paths, session_id=SESSION_ID
        )
        assert first.deleted_ids == second.deleted_ids
        after = object_bytes(paths)
        assert after == mid
        assert hex_a not in after
        assert hex_b in after
        assert not (paths.journals / f"{tx_id}.json").exists()

        # Startup path: no-op when no journals remain
        import asyncio

        async def _startup_twice():
            r1 = await recover_snapshot_transactions(
                SESSION_ID,
                paths,
                session_dir=str(paths.root),
                workspace_root=str(paths.root / "ws"),
                states_root=str(paths.root / "st"),
            )
            r2 = await recover_snapshot_transactions(
                SESSION_ID,
                paths,
                session_dir=str(paths.root),
                workspace_root=str(paths.root / "ws"),
                states_root=str(paths.root / "st"),
            )
            return r1, r2

        asyncio.run(_startup_twice())
        assert object_bytes(paths) == after
        assert hex_orphan not in after


# ---------------------------------------------------------------------------
# Active target / pre_restore protection
# ---------------------------------------------------------------------------


class TestActiveProtection:
    def test_restore_target_cannot_be_deleted(self, paths):
        from backend.services.snapshot_v2.jobs import RestoreJobHandle
        from backend.services.snapshot_v2.models import SnapshotTransactionRecord
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.state import snapshot_restore_jobs

        setup_two_leaf_snapshots(paths)
        sid = "snap-a"
        record = SnapshotTransactionRecord(
            transaction_id="restore-gate-c",
            session_id=SESSION_ID,
            snapshot_id=sid,
            status="staging",
        )
        handle = RestoreJobHandle(
            record=record,
            task=None,
            cancel_event=Mock(),
            paths=paths,
        )
        snapshot_restore_jobs["restore-gate-c"] = handle
        try:
            service = SnapshotService(session_id=SESSION_ID, paths=paths)
            before = object_bytes(paths)
            with pytest.raises(SnapshotDeleteConflict, match="restore_target"):
                service.delete_snapshot(sid)
            assert object_bytes(paths) == before
            assert (paths.committed / f"{sid}.json").exists()
        finally:
            snapshot_restore_jobs.pop("restore-gate-c", None)

    def test_pre_restore_ref_cannot_be_deleted(self, paths):
        from backend.services.snapshot_v2.service import SnapshotService

        setup_two_leaf_snapshots(paths)
        sid = "snap-a"
        journal = {
            "transaction_id": "restore-preref-gate-c",
            "target_snapshot_id": "snap-survivor",
            "session_id": SESSION_ID,
            "recovery_mode": "rollback",
            "status": "STAGED",
            "pre_restore_ref": sid,
            "applied_step_ledger": [],
        }
        (paths.journals / "restore-preref-gate-c.json").write_text(
            json.dumps(journal, sort_keys=True)
        )
        service = SnapshotService(session_id=SESSION_ID, paths=paths)
        before = object_bytes(paths)
        with pytest.raises(SnapshotDeleteConflict, match="pre_restore"):
            service.delete_snapshot(sid)
        assert object_bytes(paths) == before
        assert (paths.committed / f"{sid}.json").exists()


# ---------------------------------------------------------------------------
# Untrusted import with crash injection
# ---------------------------------------------------------------------------


class TestUntrustedImportCrashInjection:
    def test_untrusted_import_with_crash_never_writes_runtime(self, tmp_path, paths):
        from backend.services.snapshot_v2.importer import SnapshotImporter
        from backend.services.snapshot_v2.object_store import ManifestStore, ObjectStore

        data, h = _make_obj_data(256, b"a")
        rt_data, rt_h = _make_obj_data(512, b"r")
        ref = _make_ref(h, 256)
        rt_ref = _make_ref(rt_h, 512, "application/x-hdf5")

        rt_capture = RuntimeGenerationManifest(
            capture_command="runtime_capture_v2",
            capture_timestamp="2026-07-11T12:00:00Z",
            kernel_id="kernel-1",
            captured_adata_vars=["adata"],
            captured_adata_refs={"adata": rt_ref},
        )
        m = _build_import_manifest(
            SESSION_ID_SRC,
            "snap-crash",
            objects={h: ref, rt_h: rt_ref},
            runtime_capture=rt_capture,
        )
        archive = _build_archive(
            tmp_path / "crash-gate-c.tar.gz",
            [m],
            {h: data, rt_h: rt_data},
            SESSION_ID_SRC,
            runtime_included=True,
        )

        class CrashManifestStore(ManifestStore):
            def publish_staging(self, staging_path):
                raise RuntimeError("simulated crash before publish")

        class SpyStore(ObjectStore):
            def __init__(self, p):
                super().__init__(p)
                self.put_hashes: list[str] = []

            def put_bytes(self, data: bytes, media_type: str):
                sha = hashlib.sha256(data).hexdigest()
                self.put_hashes.append(sha)
                return super().put_bytes(data, media_type)

        before = object_bytes(paths)
        spy = SpyStore(paths)
        importer = SnapshotImporter(
            SESSION_ID_DST,
            paths,
            object_store=spy,
            manifest_store=CrashManifestStore(paths),
        )
        result = importer.import_bundle(archive, trust_runtime=False)
        after = object_bytes(paths)

        assert not result.success
        assert rt_h not in spy.put_hashes
        assert rt_h not in object_names(paths)
        # Runtime bytes must never appear even if non-runtime was written
        assert rt_h not in after
        assert before.get(rt_h) is None


# ---------------------------------------------------------------------------
# include-runtime export/import + destination re-sign
# ---------------------------------------------------------------------------


class TestExportImportRuntimeAndResign:
    def test_include_runtime_export_import_round_trip(self, tmp_path, paths):
        from backend.services.snapshot_v2.exporter import SnapshotExporter
        from backend.services.snapshot_v2.importer import SnapshotImporter
        from backend.services.snapshot_v2.paths import SnapshotV2Paths

        # Content must match write_object's b"x" * size so export hashes match files
        ws_hash = hashlib.sha256(b"x" * 256).hexdigest()
        rt_hash = hashlib.sha256(b"x" * 512).hexdigest()
        write_object(paths, ws_hash, 256)
        write_object(paths, rt_hash, 512)

        rt_capture = RuntimeGenerationManifest(
            capture_command="capture",
            capture_timestamp="2026-07-11T12:00:00Z",
            kernel_id="kernel-1",
            captured_adata_refs={
                "adata": ObjectRef(
                    sha256=rt_hash, size=512, media_type="application/x-hdf5"
                ),
            },
        )
        from backend.services.snapshot_v2.models import runtime_manifest_from_generation

        rt = runtime_manifest_from_generation(rt_capture, status="complete")
        manifest = build_manifest(
            SESSION_ID,
            "snap-export",
            objects={
                ws_hash: ObjectRef(
                    sha256=ws_hash, size=256, media_type="application/octet-stream"
                ),
                rt_hash: ObjectRef(
                    sha256=rt_hash, size=512, media_type="application/x-hdf5"
                ),
            },
            workspace=[
                WorkspaceEntry(
                    path="data/file.txt", object_hash=ws_hash, size=256, mode=0o644
                ),
            ],
            runtime=rt.model_dump(mode="json"),
        )
        write_manifest_to_store(manifest, paths.committed, paths.integrity)

        sp = SnapshotV2Paths(
            root=paths.root,
            objects=paths.objects,
            staging=paths.staging,
            journals=paths.journals,
            integrity=paths.integrity,
            committed=paths.committed,
        )
        exporter = SnapshotExporter(SESSION_ID, sp)
        export_path = exporter.export(SESSION_ID, include_runtime=True)

        with tarfile.open(export_path, "r:gz") as tf:
            meta = json.loads(tf.extractfile("snapshot-v2-export.json").read())
            names = tf.getnames()
        assert meta["runtime_included"] is True
        assert rt_hash in meta["included_hashes"]
        assert f"objects/{rt_hash}" in names
        assert f"objects/{ws_hash}" in names

        # Import into a fresh destination
        dest = tmp_path / "dest"
        dest_paths = type(paths)()
        # rebuild FakePaths-like structure
        class DestPaths:
            root = dest / "v2"
            objects = dest / "objects"
            staging = dest / "staging"
            journals = dest / "journals"
            integrity = dest / "integrity"
            committed = dest / "committed"

            def ensure_dirs(self):
                for d in [
                    self.root,
                    self.objects,
                    self.staging,
                    self.journals,
                    self.integrity,
                    self.committed,
                ]:
                    d.mkdir(parents=True, exist_ok=True)

        dest_paths = DestPaths()
        dest_paths.ensure_dirs()

        importer = SnapshotImporter(SESSION_ID_DST, dest_paths)
        result = importer.import_bundle(export_path, trust_runtime=True)
        assert result.success

        committed = _list_committed(dest_paths)
        assert len(committed) == 1
        assert (dest_paths.objects / rt_hash).exists()
        assert (dest_paths.objects / ws_hash).exists()
        assert "adata" in committed[0].runtime.adata_refs

    def test_destination_re_sign_verification(self, tmp_path, paths):
        from backend.services.snapshot_v2.importer import SnapshotImporter

        data, h = _make_obj_data(128, b"s")
        ref = _make_ref(h, 128)
        m = _build_import_manifest(
            SESSION_ID_SRC, "snap-resign", objects={h: ref}
        )
        archive = _build_archive(
            tmp_path / "resign-gate-c.tar.gz",
            [m],
            {h: data},
            SESSION_ID_SRC,
        )

        importer = SnapshotImporter(SESSION_ID_DST, paths)
        result = importer.import_bundle(archive)
        assert result.success

        committed = _list_committed(paths)
        assert len(committed) == 1
        assert verify_manifest_hmac(paths.integrity, paths.committed, committed[0])

        m_bytes = (paths.committed / f"{committed[0].snapshot_id}.json").read_bytes()
        dest_sig = (
            paths.integrity / f"{committed[0].snapshot_id}.sig"
        ).read_text().strip()
        assert dest_sig == sign_manifest_bytes(m_bytes)

        # Source-session-derived key must NOT match destination signature
        legacy_sig = hmac.new(
            hashlib.sha256(SESSION_ID_SRC.encode()).digest(),
            m_bytes,
            hashlib.sha256,
        ).hexdigest()
        assert not hmac.compare_digest(legacy_sig, dest_sig)
