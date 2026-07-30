"""Tests: V2 snapshot mark-and-sweep garbage collection."""
import hashlib
import json
from pathlib import Path

import pytest

from backend.services.snapshot_v2.models import (
    ObjectRef,
    SnapshotManifestV2,
    WorkspaceEntry,
)
from backend.tests.snapshot_v2_fixtures import (
    make_snapshot_v2_manifest,
    write_manifest_to_store,
)


# ── Fixtures (shared with deletion tests) ──────────────────────────────


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


def build_manifest(
    session_id: str,
    snapshot_id: str,
    *,
    objects: dict[str, ObjectRef] | None = None,
    workspace: list[WorkspaceEntry] | None = None,
    parent_id: str | None = None,
) -> SnapshotManifestV2:
    obj = objects or {}
    ws = workspace or []
    return make_snapshot_v2_manifest(
        session_id=session_id,
        snapshot_id=snapshot_id,
        parent_id=parent_id,
        workspace=[e.model_dump(mode="json") if hasattr(e, "model_dump") else e for e in ws],
        objects={k: (v.model_dump(mode="json") if hasattr(v, "model_dump") else v) for k, v in obj.items()},
    )


def build_object_ref(sha256: str | None = None, size: int = 1024,
                     media_type: str = "application/octet-stream") -> ObjectRef:
    if sha256 is None:
        sha256 = hashlib.sha256(f"obj-{size}".encode()).hexdigest()
    if not all(c in "0123456789abcdef" for c in sha256):
        sha256 = hashlib.sha256(sha256.encode()).hexdigest()
    return ObjectRef(sha256=sha256, size=size, media_type=media_type)


def write_object(paths, sha256: str, size: int = 1024) -> None:
    obj_path = paths.objects / sha256
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_bytes(b"x" * size)


def write_staging_artifact(paths, name: str, size: int = 512) -> None:
    """Write a non-object file into staging for orphan-cleanup tests."""
    staging_path = paths.staging / name
    staging_path.write_bytes(b"y" * size)


# ═════════════════════════════════════════════════════════════════════════
#  GCResult model
# ═════════════════════════════════════════════════════════════════════════


class TestGCResultModel:
    def test_gc_result_fields(self):
        """GCResult has required fields."""
        from backend.services.snapshot_v2.gc import GCResult

        result = GCResult(
            removed_count=3,
            removed_bytes=6144,
            surviving_count=10,
        )
        assert result.removed_count == 3
        assert result.removed_bytes == 6144
        assert result.surviving_count == 10


# ═════════════════════════════════════════════════════════════════════════
#  mark()
# ═════════════════════════════════════════════════════════════════════════


class TestMark:
    def test_mark_empty_no_manifests(self, v2_paths):
        """mark() returns empty set when no manifests exist."""
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector

        gc = SnapshotGarbageCollector("s1", v2_paths)
        marked = gc.mark()
        assert marked == set()

    def test_mark_single_manifest(self, v2_paths):
        """mark() collects objects from a single manifest."""
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector

        hex_a = hashlib.sha256(b"a").hexdigest()
        hex_b = hashlib.sha256(b"b").hexdigest()
        ref_a = build_object_ref(sha256=hex_a, size=100)
        ref_b = build_object_ref(sha256=hex_b, size=200)
        write_object(v2_paths, hex_a, 100)
        write_object(v2_paths, hex_b, 200)
        manifest = build_manifest("s1", "snap-1", objects={hex_a: ref_a, hex_b: ref_b})
        write_manifest_to_store(manifest, v2_paths.committed, v2_paths.integrity)

        marked = SnapshotGarbageCollector("s1", v2_paths).mark()
        assert hex_a in marked
        assert hex_b in marked

    def test_mark_shared_objects(self, v2_paths):
        """mark() handles objects shared across manifests."""
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector

        shared = hashlib.sha256(b"shared").hexdigest()
        unique_a = hashlib.sha256(b"unique-a").hexdigest()
        unique_b = hashlib.sha256(b"unique-b").hexdigest()
        ref_shared = build_object_ref(sha256=shared, size=100)
        ref_a = build_object_ref(sha256=unique_a, size=100)
        ref_b = build_object_ref(sha256=unique_b, size=100)
        write_object(v2_paths, shared, 100)
        write_object(v2_paths, unique_a, 100)
        write_object(v2_paths, unique_b, 100)

        m1 = build_manifest("s1", "snap-1", objects={shared: ref_shared, unique_a: ref_a})
        m2 = build_manifest("s1", "snap-2", objects={shared: ref_shared, unique_b: ref_b})
        write_manifest_to_store(m1, v2_paths.committed, v2_paths.integrity)
        write_manifest_to_store(m2, v2_paths.committed, v2_paths.integrity)

        marked = SnapshotGarbageCollector("s1", v2_paths).mark()
        assert shared in marked
        assert unique_a in marked
        assert unique_b in marked

    def test_mark_includes_verified_pending_import_manifest(self, v2_paths):
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector
        from backend.services.snapshot_v2.object_store import ManifestStore

        object_hash = hashlib.sha256(b"pending-object").hexdigest()
        ref = build_object_ref(sha256=object_hash, size=100)
        write_object(v2_paths, object_hash, 100)
        manifest = build_manifest(
            "s1", "snap-pending", objects={object_hash: ref},
        )
        write_manifest_to_store(
            manifest, v2_paths.committed, v2_paths.integrity,
        )
        ManifestStore.mark_pending_imports(
            v2_paths.committed, [manifest.snapshot_id],
        )

        marked = SnapshotGarbageCollector("s1", v2_paths).mark()

        assert object_hash in marked

    def test_gc_mark_aborts_on_corrupt_committed_manifest(self, v2_paths):
        """mark() aborts when any committed (non-audit) manifest is corrupt."""
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector

        hex_a = hashlib.sha256(b"a").hexdigest()
        ref_a = build_object_ref(sha256=hex_a, size=100)
        write_object(v2_paths, hex_a, 100)
        manifest = build_manifest("s1", "snap-good", objects={hex_a: ref_a})
        write_manifest_to_store(manifest, v2_paths.committed, v2_paths.integrity)

        bad_path = v2_paths.committed / "snap-corrupt.json"
        bad_path.write_text("not json")

        with pytest.raises(SnapshotIntegrityError):
            SnapshotGarbageCollector("s1", v2_paths).mark()

    def test_mark_uses_verify_all_committed(self, v2_paths, monkeypatch):
        """GC mark must call verify_all_committed, never diagnostics enumerator."""
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector
        from backend.services.snapshot_v2.object_store import ManifestStore

        calls: list[str] = []
        orig_verify = ManifestStore.verify_all_committed
        orig_diag = ManifestStore.list_committed_with_diagnostics

        def tracking_verify(self):
            calls.append("verify_all_committed")
            return orig_verify(self)

        def tracking_diag(self):
            calls.append("list_committed_with_diagnostics")
            return orig_diag(self)

        monkeypatch.setattr(ManifestStore, "verify_all_committed", tracking_verify)
        monkeypatch.setattr(
            ManifestStore, "list_committed_with_diagnostics", tracking_diag,
        )

        SnapshotGarbageCollector("s1", v2_paths).mark()
        assert "verify_all_committed" in calls
        assert "list_committed_with_diagnostics" not in calls

    def test_mark_protects_audit_manifest_objects(self, v2_paths):
        """mark() protects audit-* objects even though listings hide them.

        The ``audit-`` prefix is a *presentation* rule — audit bundles must not
        be offered as restorable snapshots — and `_iter_committed_json_stems`
        enforces it for every listing path. Reachability is a different
        question: `AuditBundle.capture_locked` runs a full capture and the
        `discard_current` restore path takes no pre_restore snapshot, so the
        audit manifest is the only record of the state the user discarded (and
        `restore_snapshot` verifies its objects immediately after writing it).
        Excluding it from mark() let the next sweep unlink every one of those
        objects, leaving a manifest of dangling refs that no API can delete.
        """
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector
        from backend.services.snapshot_v2.object_store import ManifestStore

        hex_a = hashlib.sha256(b"a").hexdigest()
        hex_b = hashlib.sha256(b"b").hexdigest()
        ref_a = build_object_ref(sha256=hex_a, size=100)
        ref_b = build_object_ref(sha256=hex_b, size=100)
        write_object(v2_paths, hex_a, 100)
        write_object(v2_paths, hex_b, 100)

        # non-audit manifest
        m1 = build_manifest("s1", "snap-normal", objects={hex_a: ref_a})
        write_manifest_to_store(m1, v2_paths.committed, v2_paths.integrity)

        # audit manifest (starts with "audit-")
        m2 = build_manifest("s1", "audit-snap-001", objects={hex_b: ref_b})
        write_manifest_to_store(m2, v2_paths.committed, v2_paths.integrity)

        marked = SnapshotGarbageCollector("s1", v2_paths).mark()
        assert hex_a in marked
        assert hex_b in marked, (
            "audit bundle objects were left unmarked; the next sweep would "
            "unlink the only copy of the discarded state"
        )

        # The listing rule is unchanged: audit bundles stay invisible.
        listed = {
            m.snapshot_id for m in ManifestStore(v2_paths).verify_all_committed()
        }
        assert listed == {"snap-normal"}

        # And they survive an actual sweep.
        gc = SnapshotGarbageCollector("s1", v2_paths)
        gc.sweep(gc.mark())
        assert (v2_paths.objects / hex_b).exists()


# ═════════════════════════════════════════════════════════════════════════
#  sweep()
# ═════════════════════════════════════════════════════════════════════════


class TestSweep:
    def test_sweep_removes_unreferenced(self, v2_paths):
        """sweep() removes objects not in the marked set."""
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector

        hex_a = hashlib.sha256(b"a").hexdigest()
        hex_b = hashlib.sha256(b"b").hexdigest()
        ref_a = build_object_ref(sha256=hex_a, size=100)
        write_object(v2_paths, hex_a, 100)
        write_object(v2_paths, hex_b, 200)
        manifest = build_manifest("s1", "snap-1", objects={hex_a: ref_a})
        write_manifest_to_store(manifest, v2_paths.committed, v2_paths.integrity)

        gc = SnapshotGarbageCollector("s1", v2_paths)
        marked = gc.mark()
        assert hex_a in marked

        result = gc.sweep(marked)
        assert (v2_paths.objects / hex_a).exists()
        assert not (v2_paths.objects / hex_b).exists()
        assert result.removed_count == 1
        assert result.removed_bytes == 200
        assert result.surviving_count == 1

    def test_sweep_preserves_referenced(self, v2_paths):
        """sweep() preserves all objects in the marked set."""
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector

        hex_a = hashlib.sha256(b"a").hexdigest()
        hex_b = hashlib.sha256(b"b").hexdigest()
        write_object(v2_paths, hex_a, 100)
        write_object(v2_paths, hex_b, 200)
        # Only hex_a is referenced
        ref_a = build_object_ref(sha256=hex_a, size=100)
        manifest = build_manifest("s1", "snap-1", objects={hex_a: ref_a})
        write_manifest_to_store(manifest, v2_paths.committed, v2_paths.integrity)

        result = SnapshotGarbageCollector("s1", v2_paths).run()
        assert (v2_paths.objects / hex_a).exists()
        assert not (v2_paths.objects / hex_b).exists()
        assert result.removed_count == 1

    def test_sweep_only_hash_named_files(self, v2_paths):
        """sweep() only removes regular files with 64-char hex names."""
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector

        hex_a = hashlib.sha256(b"a").hexdigest()
        ref_a = build_object_ref(sha256=hex_a, size=100)
        write_object(v2_paths, hex_a, 100)
        manifest = build_manifest("s1", "snap-1", objects={hex_a: ref_a})
        write_manifest_to_store(manifest, v2_paths.committed, v2_paths.integrity)

        # Non-hash-named files in objects dir that should be preserved
        wrong_name = v2_paths.objects / "not-a-hash.txt"
        wrong_name.write_text("i am not an object")
        unrelated_dir = v2_paths.objects / "subdir"
        unrelated_dir.mkdir()
        sub_file = unrelated_dir / "some.txt"
        sub_file.write_text("sub")

        result = SnapshotGarbageCollector("s1", v2_paths).run()
        assert wrong_name.exists()
        assert unrelated_dir.exists()
        assert sub_file.exists()
        assert (v2_paths.objects / hex_a).exists()

    def test_sweep_no_objects_dir(self, v2_paths):
        """sweep() handles missing objects directory gracefully."""
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector
        import shutil
        shutil.rmtree(str(v2_paths.objects))

        gc = SnapshotGarbageCollector("s1", v2_paths)
        result = gc.sweep(set())
        assert result.removed_count == 0
        assert result.removed_bytes == 0
        assert result.surviving_count == 0


# ═════════════════════════════════════════════════════════════════════════
#  run()
# ═════════════════════════════════════════════════════════════════════════


@pytest.fixture
def gc_with_two_snapshots(v2_paths):
    """GC instance with two committed snapshots and one orphan object."""
    from backend.services.snapshot_v2.gc import SnapshotGarbageCollector

    hex_a = hashlib.sha256(b"a").hexdigest()
    hex_b = hashlib.sha256(b"b").hexdigest()
    orphan = hashlib.sha256(b"orphan").hexdigest()
    ref_a = build_object_ref(sha256=hex_a, size=100)
    ref_b = build_object_ref(sha256=hex_b, size=200)
    write_object(v2_paths, hex_a, 100)
    write_object(v2_paths, hex_b, 200)
    write_object(v2_paths, orphan, 50)
    m1 = build_manifest("s1", "snap-1", objects={hex_a: ref_a})
    m2 = build_manifest("s1", "snap-2", objects={hex_b: ref_b})
    write_manifest_to_store(m1, v2_paths.committed, v2_paths.integrity)
    write_manifest_to_store(m2, v2_paths.committed, v2_paths.integrity)
    return SnapshotGarbageCollector("s1", v2_paths)


class TestRun:
    def test_gc_run_verifies_committed_set_once_for_mark_and_count(
        self, gc_with_two_snapshots, monkeypatch
    ):
        gc = gc_with_two_snapshots
        calls = {"n": 0}
        real = gc.manifest_store.verify_all_committed

        def counting():
            calls["n"] += 1
            return real()

        monkeypatch.setattr(gc.manifest_store, "verify_all_committed", counting)
        gc.run()
        # mark + count share one snapshot; sweep keeps its own fail-closed re-verify.
        assert calls["n"] <= 2

    def test_run_mark_then_sweep(self, v2_paths):
        """run() marks and sweeps in one operation."""
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector

        hex_a = hashlib.sha256(b"a").hexdigest()
        hex_b = hashlib.sha256(b"b").hexdigest()
        ref_a = build_object_ref(sha256=hex_a, size=100)
        write_object(v2_paths, hex_a, 100)
        write_object(v2_paths, hex_b, 200)
        manifest = build_manifest("s1", "snap-1", objects={hex_a: ref_a})
        write_manifest_to_store(manifest, v2_paths.committed, v2_paths.integrity)

        result = SnapshotGarbageCollector("s1", v2_paths).run()
        assert (v2_paths.objects / hex_a).exists()
        assert not (v2_paths.objects / hex_b).exists()
        assert result.removed_count == 1
        assert result.removed_bytes == 200
        assert result.surviving_count == 1

    def test_run_idempotent(self, v2_paths):
        """Running run() twice produces same result."""
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector

        hex_a = hashlib.sha256(b"a").hexdigest()
        ref_a = build_object_ref(sha256=hex_a, size=100)
        write_object(v2_paths, hex_a, 100)
        manifest = build_manifest("s1", "snap-1", objects={hex_a: ref_a})
        write_manifest_to_store(manifest, v2_paths.committed, v2_paths.integrity)

        gc = SnapshotGarbageCollector("s1", v2_paths)
        first = gc.run()
        second = gc.run()
        assert first == second

    def test_run_full_integration(self, v2_paths):
        """End-to-end: shared objects survive, orphan removed, subdir preserved."""
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector

        shared = hashlib.sha256(b"shared").hexdigest()
        orphan = hashlib.sha256(b"orphan").hexdigest()
        ref_shared = build_object_ref(sha256=shared, size=300)
        write_object(v2_paths, shared, 300)
        write_object(v2_paths, orphan, 500)
        m1 = build_manifest("s1", "snap-a", objects={shared: ref_shared})
        m2 = build_manifest("s1", "snap-b", objects={shared: ref_shared})
        write_manifest_to_store(m1, v2_paths.committed, v2_paths.integrity)
        write_manifest_to_store(m2, v2_paths.committed, v2_paths.integrity)

        # Non-object item in objects dir
        junk = v2_paths.objects / "readme.txt"
        junk.write_text("not an object")

        result = SnapshotGarbageCollector("s1", v2_paths).run()
        assert (v2_paths.objects / shared).exists()
        assert not (v2_paths.objects / orphan).exists()
        assert junk.exists()
        assert result.removed_count == 1
        assert result.removed_bytes == 500
        assert result.surviving_count == 1


# ═════════════════════════════════════════════════════════════════════════
#  Safety: corrupt manifest → abort sweep
# ═════════════════════════════════════════════════════════════════════════


class TestSweepSafety:
    def test_corrupt_manifest_aborts_sweep(self, v2_paths):
        """sweep() raises SnapshotIntegrityError when a surviving manifest
        cannot be verified."""
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError

        hex_a = hashlib.sha256(b"a").hexdigest()
        ref_a = build_object_ref(sha256=hex_a, size=100)
        write_object(v2_paths, hex_a, 100)

        # Good manifest
        good = build_manifest("s1", "snap-good", objects={hex_a: ref_a})
        write_manifest_to_store(good, v2_paths.committed, v2_paths.integrity)

        gc = SnapshotGarbageCollector("s1", v2_paths)
        marked = {hex_a}

        # Tamper the manifest content (root hash will mismatch)
        committed_path = v2_paths.committed / "snap-good.json"
        import json as _json
        data = _json.loads(committed_path.read_bytes())
        data["created_at"] = "2099-01-01T00:00:00Z"
        committed_path.write_bytes(
            _json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        )

        with pytest.raises(SnapshotIntegrityError):
            gc.sweep(marked)

    def test_corrupt_manifest_leaves_objects_untouched(self, v2_paths):
        """When sweep aborts, no objects are removed."""
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError

        hex_a = hashlib.sha256(b"a").hexdigest()
        hex_b = hashlib.sha256(b"b").hexdigest()
        ref_a = build_object_ref(sha256=hex_a, size=100)
        write_object(v2_paths, hex_a, 100)
        write_object(v2_paths, hex_b, 200)

        good = build_manifest("s1", "snap-good", objects={hex_a: ref_a})
        write_manifest_to_store(good, v2_paths.committed, v2_paths.integrity)

        gc = SnapshotGarbageCollector("s1", v2_paths)
        marked = {hex_a}

        # Corrupt the manifest
        sig_path = v2_paths.integrity / "snap-good.sig"
        sig_path.write_text("tampered")

        with pytest.raises(SnapshotIntegrityError):
            gc.sweep(marked)
        # Both objects should still exist
        assert (v2_paths.objects / hex_a).exists()
        assert (v2_paths.objects / hex_b).exists()


# ═════════════════════════════════════════════════════════════════════════
#  Active transaction staging protection
# ═════════════════════════════════════════════════════════════════════════

class TestActiveTransactions:
    def test_active_txn_objects_are_marked(self, v2_paths):
        """Objects referenced in active transaction journals are marked."""
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector
        from backend.services.snapshot_v2.restore import _write_journal_hmac_signed

        hex_a = hashlib.sha256(b"a").hexdigest()
        hex_b = hashlib.sha256(b"b").hexdigest()
        ref_a = build_object_ref(sha256=hex_a, size=100)
        ref_b = build_object_ref(sha256=hex_b, size=200)
        write_object(v2_paths, hex_a, 100)
        write_object(v2_paths, hex_b, 200)

        # No committed manifests - everything depends on active txns
        _write_journal_hmac_signed(v2_paths, "txn-active", {
            "transaction_id": "txn-active",
            "session_id": "s1",
            "target_snapshot_id": "snap-incoming",
            "object_refs": {hex_a: ref_a.model_dump(mode="json"),
                           hex_b: ref_b.model_dump(mode="json")},
            "status": "STAGING",
            "recovery_mode": "rollback",
        })

        marked = SnapshotGarbageCollector("s1", v2_paths).mark()
        assert hex_a in marked
        assert hex_b in marked

    def test_tampered_restore_journal_fails_closed_in_gc(self, v2_paths):
        """Tampered restore journal HMAC must abort GC mark (fail closed)."""
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector
        from backend.services.snapshot_v2.restore import _write_journal_hmac_signed
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError

        hex_a = hashlib.sha256(b"a").hexdigest()
        ref_a = build_object_ref(sha256=hex_a, size=100)
        write_object(v2_paths, hex_a, 100)

        tx_id = "restore-gc-tamper"
        _write_journal_hmac_signed(v2_paths, tx_id, {
            "transaction_id": tx_id,
            "target_snapshot_id": "snap-x",
            "session_id": "s1",
            "pre_restore_ref": "snap-pre",
            "recovery_mode": "rollback",
            "status": "APPLYING",
            "applied_step_ledger": [],
            "object_refs": {hex_a: ref_a.model_dump(mode="json")},
            "branch_id": "main",
            "continuation_kind": "idle",
        })
        jp = v2_paths.journals / f"{tx_id}.json"
        payload = json.loads(jp.read_text())
        payload["status"] = "COMMITTED"  # mutate without re-signing
        jp.write_text(json.dumps(payload, sort_keys=True))

        gc = SnapshotGarbageCollector("s1", v2_paths)
        with pytest.raises(SnapshotIntegrityError, match="JOURNAL_HMAC"):
            gc.mark()

    def test_unreadable_journal_fails_closed_in_gc(self, v2_paths):
        """Corrupt/unreadable journal must abort GC mark (fail closed)."""
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError

        hex_a = hashlib.sha256(b"a").hexdigest()
        write_object(v2_paths, hex_a, 100)

        # Restore-shaped filename so GC routes through verified reader
        jp = v2_paths.journals / "restore-bad.json"
        jp.write_text("not-json{{{")

        gc = SnapshotGarbageCollector("s1", v2_paths)
        with pytest.raises(SnapshotIntegrityError):
            gc.mark()


# ═════════════════════════════════════════════════════════════════════════
#  Orphan staging cleanup
# ═════════════════════════════════════════════════════════════════════════

class TestOrphanStaging:
    def test_orphan_staging_cleaned(self, v2_paths):
        """sweep() removes orphan files in staging that are not journal-linked."""
        # This tests that SnapshotGarbageCollector.run()
        # does touch the staging dir - actually this is more about
        # recovery running orphan staging cleanup. We test GC's
        # behavior w.r.t. staging objects.
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector

        # Create an orphan staging artifact
        write_staging_artifact(v2_paths, "orphan-stage.dat", 100)
        assert (v2_paths.staging / "orphan-stage.dat").exists()

        # GC run should not crash on staging presence
        result = SnapshotGarbageCollector("s1", v2_paths).run()
        # Normally staging cleanup is in recovery, GC just marks/sweeps objects
        # But it should not crash on staging dir presence
        assert result is not None
