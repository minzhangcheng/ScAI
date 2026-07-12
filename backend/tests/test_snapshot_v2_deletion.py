"""Tests: V2 snapshot deletion journal durability and fail-closed GC."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import pytest

from backend.services.snapshot_v2.errors import (
    SnapshotDeleteConflict,
    SnapshotIntegrityError,
    SnapshotStorageError,
)
from backend.services.snapshot_v2.models import ObjectRef, WorkspaceEntry
from backend.tests.snapshot_v2_fixtures import (
    make_snapshot_v2_manifest,
    write_manifest_to_store,
)


# ── Fixtures ────────────────────────────────────────────────────────────


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


def build_manifest(
    session_id: str,
    snapshot_id: str,
    *,
    objects: dict[str, ObjectRef] | None = None,
    workspace: list[WorkspaceEntry] | None = None,
    parent_id: str | None = None,
):
    obj = objects or {}
    ws = workspace or []
    return make_snapshot_v2_manifest(
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


def write_object(paths, sha256: str, size: int = 1024) -> None:
    obj_path = paths.objects / sha256
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_bytes(b"x" * size)


def object_bytes(paths) -> dict[str, bytes]:
    """Snapshot of all hash-named object file contents."""
    result: dict[str, bytes] = {}
    if not paths.objects.exists():
        return result
    for entry in paths.objects.iterdir():
        if entry.is_file() and len(entry.name) == 64:
            result[entry.name] = entry.read_bytes()
    return result


def load_delete_journal(paths):
    """Load the single active delete journal (del-*.json)."""
    from backend.services.snapshot_v2.deletion import read_delete_journal

    return read_delete_journal(paths)


def list_delete_journals(paths) -> list[Path]:
    return list(paths.journals.glob("del-*.json"))


def corrupt_surviving_manifest(paths, snapshot_id: str = "snap-survivor") -> None:
    """Tamper a committed manifest so HMAC/root verification fails."""
    committed_path = paths.committed / f"{snapshot_id}.json"
    data = json.loads(committed_path.read_bytes())
    data["created_at"] = "2099-01-01T00:00:00Z"
    committed_path.write_bytes(
        json.dumps(
            data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    )


def setup_two_leaf_snapshots(paths, session_id: str = "s1"):
    """Two independent leaf snapshots sharing no objects (plus one orphan)."""
    hex_a = hashlib.sha256(b"obj-a").hexdigest()
    hex_b = hashlib.sha256(b"obj-b").hexdigest()
    hex_orphan = hashlib.sha256(b"orphan").hexdigest()
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


def setup_shared_object_lineage(paths, session_id: str = "s1"):
    """Parent + child leaf sharing one object; child has unique object."""
    shared = hashlib.sha256(b"shared").hexdigest()
    unique = hashlib.sha256(b"unique-child").hexdigest()
    orphan = hashlib.sha256(b"orphan-shared").hexdigest()
    ref_shared = build_object_ref(sha256=shared, size=300)
    ref_unique = build_object_ref(sha256=unique, size=150)
    write_object(paths, shared, 300)
    write_object(paths, unique, 150)
    write_object(paths, orphan, 80)

    parent = build_manifest(
        session_id, "snap-parent", objects={shared: ref_shared}
    )
    child = build_manifest(
        session_id,
        "snap-child",
        objects={shared: ref_shared, unique: ref_unique},
        parent_id="snap-parent",
    )
    write_manifest_to_store(parent, paths.committed, paths.integrity)
    write_manifest_to_store(child, paths.committed, paths.integrity)
    return shared, unique, orphan


@pytest.fixture
def service(paths):
    from backend.services.snapshot_v2.deletion import DeletionService

    return DeletionService("s1", paths)


# ═════════════════════════════════════════════════════════════════════════
#  Happy path deletion
# ═════════════════════════════════════════════════════════════════════════


class TestDeleteLeaf:
    def test_delete_leaf_removes_manifest_and_unique_objects(self, paths, service):
        hex_a, hex_b, hex_orphan = setup_two_leaf_snapshots(paths)

        result = service.delete_leaf("snap-a")

        assert "snap-a" in result.deleted_ids
        assert not (paths.committed / "snap-a.json").exists()
        assert (paths.committed / "snap-survivor.json").exists()
        assert not (paths.objects / hex_a).exists()
        assert (paths.objects / hex_b).exists()
        assert not (paths.objects / hex_orphan).exists()
        assert not list_delete_journals(paths)

    def test_delete_leaf_with_children_raises_conflict(self, paths, service):
        setup_shared_object_lineage(paths)
        with pytest.raises(SnapshotDeleteConflict, match="HAS_CHILDREN|children"):
            service.delete_leaf("snap-parent")

    def test_delete_missing_raises(self, paths, service):
        with pytest.raises(SnapshotStorageError, match="NOT_FOUND|not found"):
            service.delete_leaf("snap-missing")

    def test_delete_uses_snapshot_garbage_collector(self, paths, service):
        setup_two_leaf_snapshots(paths)
        assert hasattr(service, "gc")
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector

        assert isinstance(service.gc, SnapshotGarbageCollector)

    def test_delete_shared_objects_preserved(self, paths, service):
        shared, unique, orphan = setup_shared_object_lineage(paths)

        result = service.delete_leaf("snap-child")

        assert "snap-child" in result.deleted_ids
        assert (paths.objects / shared).exists()
        assert not (paths.objects / unique).exists()
        assert not (paths.objects / orphan).exists()
        assert result.freed_bytes >= 150



# ═════════════════════════════════════════════════════════════════════════
#  Protected live lineage and active transactions
# ═════════════════════════════════════════════════════════════════════════


@dataclass
class ActiveProtectionContext:
    snapshot_id: str
    reason: str


@pytest.fixture
def active_context(request, paths, monkeypatch):
    """Establish one active protection reason against snap-a."""
    reason = request.param
    setup_two_leaf_snapshots(paths)
    sid = "snap-a"
    session_id = "s1"

    if reason == "live_head":
        from backend.state import sessions
        state = Mock()
        state.last_snapshot_id = sid
        state.last_restored_snapshot_id = None
        sessions[session_id] = state
    elif reason == "continuation_source":
        from backend.state import sessions
        state = Mock()
        state.last_snapshot_id = "snap-survivor"
        state.last_restored_snapshot_id = sid
        sessions[session_id] = state
    elif reason == "restore_target":
        from backend.services.snapshot_v2.jobs import RestoreJobHandle
        from backend.services.snapshot_v2.models import SnapshotTransactionRecord
        from backend.state import snapshot_restore_jobs
        record = SnapshotTransactionRecord(
            transaction_id="restore-protect-test",
            session_id=session_id,
            snapshot_id=sid,
            status="staging",
        )
        handle = RestoreJobHandle(
            record=record,
            task=None,
            cancel_event=Mock(),
            paths=paths,
        )
        snapshot_restore_jobs["restore-protect-test"] = handle
        yield ActiveProtectionContext(snapshot_id=sid, reason=reason)
        snapshot_restore_jobs.pop("restore-protect-test", None)
        return
    elif reason == "pre_restore":
        journal = {
            "transaction_id": "restore-preref",
            "target_snapshot_id": "snap-survivor",
            "session_id": session_id,
            "recovery_mode": "rollback",
            "status": "STAGED",
            "pre_restore_ref": sid,
            "applied_step_ledger": [],
        }
        (paths.journals / "restore-preref.json").write_text(
            json.dumps(journal, sort_keys=True)
        )
    elif reason == "import":
        from backend.services.snapshot_v2.importer import _atomic_write_json
        _atomic_write_json(
            paths.journals / "import-protect.json",
            {
                "transaction_id": "import-protect",
                "kind": "import",
                "session_id": session_id,
                "status": "STAGED",
                "protected_snapshot_ids": [sid],
            },
        )
    elif reason == "export":
        from backend.services.snapshot_v2.exporter import _write_export_protection
        _write_export_protection(paths, "exp-protect", session_id, [sid])
    else:
        raise AssertionError(f"unknown reason {reason}")

    yield ActiveProtectionContext(snapshot_id=sid, reason=reason)

    try:
        from backend.state import sessions
        sessions.pop(session_id, None)
    except Exception:
        pass


@pytest.mark.parametrize(
    "active_context",
    ["live_head", "continuation_source", "restore_target", "pre_restore", "import", "export"],
    indirect=True,
)
def test_protected_snapshot_cannot_be_deleted(active_context, paths):
    from backend.services.snapshot_v2.service import SnapshotService

    service = SnapshotService(session_id="s1", paths=paths)
    with pytest.raises(SnapshotDeleteConflict, match=active_context.reason):
        service.delete_snapshot(active_context.snapshot_id)


def test_compute_protected_snapshot_ids_by_reason(paths):
    """compute_protected_snapshot_ids returns dict keyed by reason."""
    from backend.services.snapshot_v2.service import SnapshotService
    from backend.services.snapshot_v2.exporter import _write_export_protection
    from backend.state import sessions

    setup_two_leaf_snapshots(paths)
    state = Mock()
    state.last_snapshot_id = "snap-a"
    state.last_restored_snapshot_id = "snap-survivor"
    sessions["s1"] = state
    _write_export_protection(paths, "exp-1", "s1", ["snap-a"])

    try:
        service = SnapshotService(session_id="s1", paths=paths)
        protected = service.compute_protected_snapshot_ids()
        assert isinstance(protected, dict)
        assert "live_head" in protected
        assert "snap-a" in protected["live_head"]
        assert "snap-survivor" in protected["continuation_source"]
        assert "snap-a" in protected["export"]
    finally:
        sessions.pop("s1", None)


def test_delete_conflict_details_include_reason(paths):
    from backend.services.snapshot_v2.service import SnapshotService
    from backend.services.snapshot_v2.exporter import _write_export_protection

    setup_two_leaf_snapshots(paths)
    _write_export_protection(paths, "exp-d", "s1", ["snap-a"])
    service = SnapshotService(session_id="s1", paths=paths)

    with pytest.raises(SnapshotDeleteConflict) as ei:
        service.delete_snapshot("snap-a")
    assert "export" in (ei.value.details.get("reason") or str(ei.value))


# ═════════════════════════════════════════════════════════════════════════
#  Failure durability: journals survive non-COMMITTED failures
# ═════════════════════════════════════════════════════════════════════════


class TestDeleteFailureKeepsJournal:
    def test_delete_failure_keeps_applying_journal(self, paths, service, monkeypatch):
        """On GC failure mid-delete, APPLYING journal must remain for recovery."""
        setup_two_leaf_snapshots(paths)
        monkeypatch.setattr(service.gc, "run", Mock(side_effect=OSError("fault")))

        with pytest.raises(OSError, match="fault"):
            service.delete_leaf("snap-a")

        journals = list_delete_journals(paths)
        assert len(journals) == 1
        journal = load_delete_journal(paths)
        assert journal.status == "APPLYING"
        assert "snap-a" in journal.delete_set

    def test_delete_failure_after_manifest_removal_keeps_journal(
        self, paths, service, monkeypatch
    ):
        """Manifest may already be gone; journal still survives for recovery."""
        setup_two_leaf_snapshots(paths)
        monkeypatch.setattr(service.gc, "run", Mock(side_effect=OSError("fault")))

        with pytest.raises(OSError):
            service.delete_leaf("snap-a")

        assert not (paths.committed / "snap-a.json").exists()
        assert load_delete_journal(paths).status == "APPLYING"


# ═════════════════════════════════════════════════════════════════════════
#  Fail closed: corrupt survivor aborts before object removal
# ═════════════════════════════════════════════════════════════════════════


class TestCorruptSurvivorAborts:
    def test_unreadable_survivor_aborts_without_object_removal(self, paths, service):
        """Corrupt surviving manifest must abort delete/GC before object removal."""
        hex_a, hex_b, hex_orphan = setup_two_leaf_snapshots(paths)
        corrupt_surviving_manifest(paths, "snap-survivor")
        before = object_bytes(paths)

        with pytest.raises(SnapshotIntegrityError):
            service.delete_leaf("snap-a")

        assert object_bytes(paths) == before
        # Unique objects of deleted leaf must not be swept either
        assert (paths.objects / hex_a).exists()
        assert (paths.objects / hex_b).exists()
        assert (paths.objects / hex_orphan).exists()

    def test_unreadable_survivor_leaves_applying_or_aborts_before_commit(
        self, paths, service
    ):
        """After integrity abort, either no journal or non-COMMITTED journal remains."""
        setup_two_leaf_snapshots(paths)
        corrupt_surviving_manifest(paths, "snap-survivor")

        with pytest.raises(SnapshotIntegrityError):
            service.delete_leaf("snap-a")

        journals = list_delete_journals(paths)
        if journals:
            data = json.loads(journals[0].read_text())
            assert data["status"] != "COMMITTED"


# ═════════════════════════════════════════════════════════════════════════
#  Recovery: idempotent recover_delete_transaction
# ═════════════════════════════════════════════════════════════════════════


class TestRecoverDeleteTransaction:
    def test_recover_applying_completes_gc_and_removes_journal(self, paths, service):
        from backend.services.snapshot_v2.deletion import (
            recover_delete_transaction,
            _write_journal,
        )

        hex_a, hex_b, hex_orphan = setup_two_leaf_snapshots(paths)
        # Simulate crash after manifest removal, before GC/COMMITTED
        (paths.committed / "snap-a.json").unlink()
        (paths.integrity / "snap-a.sig").unlink(missing_ok=True)

        tx_id = "del-recover-1"
        record = {
            "transaction_id": tx_id,
            "session_id": "s1",
            "delete_set": ["snap-a"],
            "surviving_manifest_ids": ["snap-survivor"],
            "previous_lineage_head": "snap-a",
            "new_lineage_head": None,
            "status": "APPLYING",
        }
        _write_journal(paths, tx_id, record)

        result = recover_delete_transaction(record, paths=paths, session_id="s1")

        assert "snap-a" in result.deleted_ids
        assert not (paths.objects / hex_a).exists()
        assert (paths.objects / hex_b).exists()
        assert not (paths.objects / hex_orphan).exists()
        assert not (paths.journals / f"{tx_id}.json").exists()

    def test_recover_delete_idempotent(self, paths, service):
        from backend.services.snapshot_v2.deletion import (
            recover_delete_transaction,
            _write_journal,
        )

        hex_a, hex_b, hex_orphan = setup_two_leaf_snapshots(paths)
        (paths.committed / "snap-a.json").unlink()
        (paths.integrity / "snap-a.sig").unlink(missing_ok=True)

        tx_id = "del-idempotent"
        record = {
            "transaction_id": tx_id,
            "session_id": "s1",
            "delete_set": ["snap-a"],
            "surviving_manifest_ids": ["snap-survivor"],
            "previous_lineage_head": "snap-a",
            "new_lineage_head": None,
            "status": "APPLYING",
        }
        _write_journal(paths, tx_id, record)

        first = recover_delete_transaction(record, paths=paths, session_id="s1")
        # Re-write journal as if recovery was re-run before cleanup
        _write_journal(paths, tx_id, {**record, "status": "APPLYING"})
        second = recover_delete_transaction(record, paths=paths, session_id="s1")

        assert first.deleted_ids == second.deleted_ids
        assert not (paths.objects / hex_a).exists()
        assert (paths.objects / hex_b).exists()
        assert not (paths.journals / f"{tx_id}.json").exists()

    def test_recover_staged_only_removes_journal(self, paths):
        from backend.services.snapshot_v2.deletion import (
            recover_delete_transaction,
            _write_journal,
        )

        hex_a, hex_b, _ = setup_two_leaf_snapshots(paths)
        tx_id = "del-staged"
        record = {
            "transaction_id": tx_id,
            "session_id": "s1",
            "delete_set": ["snap-a"],
            "surviving_manifest_ids": ["snap-survivor"],
            "previous_lineage_head": "snap-a",
            "new_lineage_head": None,
            "status": "STAGED",
        }
        _write_journal(paths, tx_id, record)

        result = recover_delete_transaction(record, paths=paths, session_id="s1")

        assert result.deleted_ids == []
        assert (paths.committed / "snap-a.json").exists()
        assert (paths.objects / hex_a).exists()
        assert not (paths.journals / f"{tx_id}.json").exists()

    def test_recover_applying_with_corrupt_survivor_fails_closed(self, paths):
        from backend.services.snapshot_v2.deletion import (
            recover_delete_transaction,
            _write_journal,
        )

        hex_a, hex_b, hex_orphan = setup_two_leaf_snapshots(paths)
        (paths.committed / "snap-a.json").unlink()
        corrupt_surviving_manifest(paths, "snap-survivor")
        before = object_bytes(paths)

        tx_id = "del-corrupt-recover"
        record = {
            "transaction_id": tx_id,
            "session_id": "s1",
            "delete_set": ["snap-a"],
            "surviving_manifest_ids": ["snap-survivor"],
            "previous_lineage_head": "snap-a",
            "new_lineage_head": None,
            "status": "APPLYING",
        }
        _write_journal(paths, tx_id, record)

        with pytest.raises(SnapshotIntegrityError):
            recover_delete_transaction(record, paths=paths, session_id="s1")

        assert object_bytes(paths) == before
        # Journal remains for operator/retry
        assert (paths.journals / f"{tx_id}.json").exists()

    def test_service_recover_delete_delegates(self, paths, service):
        from backend.services.snapshot_v2.deletion import _write_journal

        setup_two_leaf_snapshots(paths)
        (paths.committed / "snap-a.json").unlink()
        (paths.integrity / "snap-a.sig").unlink(missing_ok=True)
        tx_id = "del-service-recover"
        record = {
            "transaction_id": tx_id,
            "session_id": "s1",
            "delete_set": ["snap-a"],
            "surviving_manifest_ids": ["snap-survivor"],
            "previous_lineage_head": "snap-a",
            "new_lineage_head": None,
            "status": "APPLYING",
        }
        _write_journal(paths, tx_id, record)

        result = service.recover_delete(tx_id)
        assert result is not None
        assert not (paths.journals / f"{tx_id}.json").exists()


# ═════════════════════════════════════════════════════════════════════════
#  GC integration via DeletionService
# ═════════════════════════════════════════════════════════════════════════


class TestDeletionGCIntegration:
    def test_delete_gc_idempotent_after_success(self, paths, service):
        setup_two_leaf_snapshots(paths)
        service.delete_leaf("snap-a")
        # Second GC run is a no-op
        second = service.gc.run()
        assert second.removed_count == 0

    def test_corrupt_manifest_aborts_gc_run(self, paths):
        from backend.services.snapshot_v2.gc import SnapshotGarbageCollector

        hex_a, hex_b, _ = setup_two_leaf_snapshots(paths)
        corrupt_surviving_manifest(paths, "snap-survivor")
        before = object_bytes(paths)

        gc = SnapshotGarbageCollector("s1", paths)
        with pytest.raises(SnapshotIntegrityError):
            gc.run()

        assert object_bytes(paths) == before
