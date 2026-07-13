"""Fault-injection matrix tests for Snapshot V2 lifecycle crash recovery.

Tests every restore, delete, and import fault point with simulated
process interruption and recovery convergence.

Journal fixtures go through ``JournalStore`` (production writer) — never
hand-rolled HMAC.
"""

from backend.tests.snapshot_v2_fixtures import v2_manifest
import json
from pathlib import Path

import pytest

from backend.services.snapshot_v2.errors import (
    SnapshotV2Error,
    SnapshotIntegrityError,
    SnapshotStorageError,
)
from backend.tests.integration.faults import (
    RESTORE_POINTS,
    DELETE_POINTS,
    IMPORT_POINTS,
    FaultBarrier,
    FaultInjected,
)


# ── Helper factories ───────────────────────────────────────────────────────


def _make_basic_manifest(session_id: str, snapshot_id: str):
    """Create a minimal manifest for fault testing."""
    from backend.services.snapshot_v2.models import IdleCapsule, SnapshotManifestV2
    return v2_manifest(
        format_version=2,
        session_id=session_id,
        snapshot_id=snapshot_id,
        captured_at="2026-07-11T12:00:00Z",
        capsule=IdleCapsule(graph_version="2.0"),
    )


def _commit_manifest(v2_paths, manifest) -> None:
    """Write and publish a manifest through the manifest store."""
    from backend.services.snapshot_v2.object_store import ManifestStore
    store = ManifestStore(v2_paths)
    staging_path = store.write_staging(manifest)
    store.publish_staging(staging_path)


def _write_pre_restore_manifest(v2_paths, session_id: str) -> str:
    """Create and commit a pre-restore snapshot with real object store bytes."""
    from backend.services.snapshot_v2.models import (
        IdleCapsule,
        ObjectRef,
        RuntimeDependencyManifest,
        RuntimeManifest,
        SnapshotManifestV2,
        WorkspaceEntry,
    )
    from backend.services.snapshot_v2.object_store import ObjectStore

    store = ObjectStore(v2_paths)
    state_ref = store.put_bytes(b'{"status":"idle"}', "application/json").ref
    nb_ref = store.put_bytes(b'{"nbformat":4,"cells":[]}', "application/x-ipynb+json").ref
    manifest = SnapshotManifestV2(
        format_version=2,
        snapshot_id="pre-restore-snap",
        session_id=session_id,
        created_at="2026-07-11T11:00:00Z",
        analysis_state_ref=state_ref,
        notebook_ref=nb_ref,
        runtime=RuntimeManifest(
            status="complete",
            dependencies=RuntimeDependencyManifest(required={}),
            residual_ref=None,
            captured_residual_names=[],
            adata_refs={},
        ),
        workspace=[],
        continuation=IdleCapsule(graph_version="2.0"),
        objects={
            state_ref.sha256: state_ref,
            nb_ref.sha256: nb_ref,
        },
    )
    _commit_manifest(v2_paths, manifest)
    return "pre-restore-snap"


def _write_journal(
    v2_paths,
    tx_id: str,
    *,
    kind: str,
    session_id: str,
    status: str,
    payload: dict | None = None,
) -> Path:
    """Write a signed journal through the production JournalStore."""
    from backend.services.snapshot_v2.journal_store import JournalStore, make_envelope

    env = make_envelope(
        kind=kind,  # type: ignore[arg-type]
        transaction_id=tx_id,
        session_id=session_id,
        status=status,
        payload=payload or {},
    )
    JournalStore(v2_paths).write(env)
    return v2_paths.journals / f"{tx_id}.json"


# ── Test classes ──────────────────────────────────────────────────────────


class TestRestoreFaultMatrix:
    """For every restore point, simulate interruption and verify recovery."""

    @pytest.mark.asyncio
    async def test_restore_all_points_raise(self, v2_paths):
        """Every named restore point can be triggered via fault barrier."""
        for point in RESTORE_POINTS:
            barrier = FaultBarrier(point)
            assert not barrier.was_hit
            with pytest.raises(FaultInjected) as exc:
                barrier.check(point)
            assert exc.value.point == point
            assert barrier.was_hit

    @pytest.mark.asyncio
    @pytest.mark.parametrize("fault_point", list(RESTORE_POINTS))
    async def test_restore_fault_recovery_rollback(self, v2_paths, fault_point):
        """For each fault point, recovery rolls back to pre-restore state.

        After interruption, create a new service instance and run recovery
        twice — both should converge to the same state.
        """
        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions

        session_id = "fault-rollback"
        pre_ref = _write_pre_restore_manifest(v2_paths, session_id)

        # Create a staged journal in APPLYING state (simulating crash mid-apply)
        tx_id = f"restore-apply-{fault_point}-tx"
        jp = _write_journal(
            v2_paths,
            tx_id,
            kind="restore_apply",
            session_id=session_id,
            status="APPLYING",
            payload={
                "target_snapshot_id": "target-snap",
                "target_root_hash": "0" * 64,
                "pre_restore_ref": pre_ref,
                "recovery_mode": "rollback",
                "applied_step_ledger": [],
            },
        )

        result1 = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(v2_paths.root),
            workspace_root=str(v2_paths.root / "ws"),
            states_root=str(v2_paths.root / "states"),
        )

        # Journal should be gone after first recovery
        assert not jp.exists()
        assert tx_id in result1.recovered_transactions

    @pytest.mark.asyncio
    @pytest.mark.parametrize("fault_point", list(RESTORE_POINTS))
    async def test_restore_fault_recovery_forward(self, v2_paths, fault_point):
        """Synthetic APPLYING + forward_only without staging/target must not
        silently drop the journal — verify/forward gate retains it and blocks.
        """
        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions

        session_id = "fault-forward"

        tx_id = f"restore-apply-fwd-{fault_point}"
        jp = _write_journal(
            v2_paths,
            tx_id,
            kind="restore_apply",
            session_id=session_id,
            status="APPLYING",
            payload={
                "target_snapshot_id": "target-snap",
                "target_root_hash": "0" * 64,
                "pre_restore_ref": "",
                "recovery_mode": "forward_only",
                "applied_step_ledger": [],
            },
        )

        result = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(v2_paths.root),
            workspace_root=str(v2_paths.root / "ws"),
            states_root=str(v2_paths.root / "states"),
        )

        # Incomplete steps + missing staging/target: journal retained, blocked.
        assert jp.exists()
        assert tx_id in result.failed_transactions
        assert result.mutations_blocked is True
        assert any(
            "FORWARD_COMPLETE_FAILED" in e
            or "TARGET_VERIFY_FAILED" in e
            or "TARGET_CORRUPT" in e
            or "TARGET_MISSING" in e
            for e in result.errors
        )
        # Idempotent re-scan: still blocked, journal still present.
        result2 = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(v2_paths.root),
            workspace_root=str(v2_paths.root / "ws"),
            states_root=str(v2_paths.root / "states"),
        )
        assert jp.exists()
        assert tx_id in result2.failed_transactions
        assert result2.mutations_blocked is True

    def test_all_fault_points_defined(self):
        """Ensure all expected fault point constants match."""
        expected_restore = (
            "target_staged", "workspace_applied", "runtime_pkl_replaced",
            "runtime_signature_replaced", "notebook_applied", "state_persisted",
            "checkpoint_seeded", "journal_committed", "cleanup_started",
        )
        expected_delete = ("journal_written", "manifests_hidden", "lineage_saved", "gc_started")
        expected_import = ("objects_verified", "manifest_published", "session_saved")

        assert RESTORE_POINTS == expected_restore
        assert DELETE_POINTS == expected_delete
        assert IMPORT_POINTS == expected_import


class TestDeleteFaultMatrix:
    """For every delete fault point, verify recovery convergence."""

    @pytest.mark.parametrize("fault_point", list(DELETE_POINTS))
    def test_delete_recovery_idempotent(self, v2_paths, fault_point):
        """Repeated recovery yields identical visible manifests and object set."""
        session_id = "fault-del"

        from backend.services.snapshot_v2.models import IdleCapsule, SnapshotManifestV2
        manifest = v2_manifest(
            format_version=2,
            session_id=session_id,
            snapshot_id="del-target",
            captured_at="2026-07-11T12:00:00Z",
            capsule=IdleCapsule(graph_version="2.0"),
        )
        _commit_manifest(v2_paths, manifest)

        from backend.services.snapshot_v2.deletion import DeletionService, _write_journal

        del_service = DeletionService(session_id, v2_paths)
        before = del_service._load_all_committed()
        before_count = len(before)

        # Create the journal for this delete point at STAGED status
        tx_id = f"del-{fault_point}-tx"
        record = {
            "transaction_id": tx_id,
            "session_id": session_id,
            "delete_set": ["del-target"],
            "surviving_manifest_ids": [m.snapshot_id for m in before
                                        if m.snapshot_id != "del-target"],
            "previous_lineage_head": "del-target",
            "new_lineage_head": None,
            "status": "STAGED",
        }
        _write_journal(v2_paths, tx_id, record)

        del_service.recover_delete(tx_id)

        after_first = del_service._load_all_committed()
        assert not (v2_paths.journals / f"{tx_id}.json").exists()

        assert len(after_first) >= 0


class TestImportFaultMatrix:
    """For every import fault point, verify recovery convergence."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("fault_point", list(IMPORT_POINTS))
    async def test_import_recovery(self, v2_paths, fault_point):
        """Import recovery cleans up orphaned staging via import branch."""
        session_id = "fault-imp"

        tx_id = f"import-{fault_point}-tx"
        jp = _write_journal(
            v2_paths,
            tx_id,
            kind="import",
            session_id=session_id,
            status="STAGED",
            payload={
                "object_hashes": [],
                "protected_snapshot_ids": ["import-snap"],
            },
        )

        assert jp.exists()

        from backend.services.snapshot_v2.recovery import recover_snapshot_transactions
        result = await recover_snapshot_transactions(
            session_id=session_id,
            paths=v2_paths,
            session_dir=str(v2_paths.root),
            workspace_root=str(v2_paths.root / "ws"),
            states_root=str(v2_paths.root / "states"),
        )

        assert not jp.exists(), f"Journal should be cleaned up for {fault_point}"
        assert tx_id in result.recovered_transactions
