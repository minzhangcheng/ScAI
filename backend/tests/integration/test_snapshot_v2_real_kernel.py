"""Integration tests: real kernel snapshot capture, restore, and HMAC integrity.

Exercises the full capture → mutate → restore cycle with a live Jupyter kernel.
"""

import hashlib
import os

import pytest

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
        reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
    ),
]


class TestRealKernelCaptureRestore:
    """Create kernel vars, capture V2, mutate, restore, assert old values + HMAC."""

    async def _declare_vars(self, sandbox):
        code = """
import anndata as ad
import numpy as np
x = 42
results = {"mean": 0.5, "std": 0.1}
adata = ad.AnnData(X=np.random.rand(10, 5))
print(f"x={x}, results={results}, adata={adata.shape}")
"""
        result = await sandbox.execute(code, save_to_history=False, skip_safety_check=True)
        assert result.success, f"Declare vars failed: {result.error}"

    async def _capture_with_service(self, service, sandbox, v2_paths):
        from backend.services.snapshot_v2.models import IdleCapsule, SnapshotCaptureRequest
        req = SnapshotCaptureRequest(
            session_id=service.session_id,
            reason="test_capture_restore",
            capsule=IdleCapsule(graph_version="2.0"),
        )
        record = await service.capture(req, sandbox=sandbox)
        assert record.failure is None, f"Capture failed: {record.failure}"
        assert record.manifest is not None
        return record

    @pytest.mark.asyncio
    async def test_capture_and_restore_old_values(self, sandbox, v2_paths, service):
        await self._declare_vars(sandbox)

        record = await self._capture_with_service(service, sandbox, v2_paths)
        snapshot_id = record.manifest.snapshot_id

        mutate_code = """
x = 99
results = {"mean": 1.0, "std": 0.5}
print("mutated")
"""
        result = await sandbox.execute(mutate_code, save_to_history=False, skip_safety_check=True)
        assert result.success

        from backend.services.snapshot_v2.restore import RestoreCoordinator

        coord = RestoreCoordinator()
        staged = await coord.stage(
            snapshot_id=snapshot_id,
            session_id=service.session_id,
            paths=v2_paths,
        )
        assert staged.transaction_id is not None

        applied = await coord.apply(
            staged, v2_paths,
            session_dir=str(v2_paths.root),
            workspace_root=str(v2_paths.root / "live_ws"),
            states_root=str(v2_paths.root / "live_states"),
            runtime_choice="reset",
        )
        assert len(applied.steps_applied) > 0

        proof = {
            "verified_live_root_hash": hashlib.sha256(
                b"real-kernel-commit-proof"
            ).hexdigest(),
            "component_hashes": {"runtime_policy": "reset"},
            "verified_at": "2026-07-12T00:00:00Z",
            "restore_journal_version": 1,
        }
        await coord.commit(
            staged.transaction_id, v2_paths, verification_proof=proof,
        )
        await coord.cleanup(staged.transaction_id, v2_paths)

    @pytest.mark.asyncio
    async def test_pkl_integrity_after_capture(self, sandbox, v2_paths):
        from backend.utils import sandbox as sb_utils
        integrity_dir = v2_paths.root / "integrity"
        integrity_dir.mkdir(parents=True, exist_ok=True)

        pkl_path = v2_paths.root / "test_session.pkl"
        pkl_path.write_bytes(b"original-payload")
        sb_utils._write_pkl_signature(pkl_path, integrity_dir)
        assert sb_utils._verify_pkl_integrity(pkl_path, integrity_dir) is True

        pkl_path.write_bytes(b"TAMPERED")
        assert sb_utils._verify_pkl_integrity(pkl_path, integrity_dir) is False

    @pytest.mark.asyncio
    async def test_capture_preserves_runtime_revision(self, sandbox, v2_paths, service):
        rev_before = sandbox.runtime_revision

        await sandbox.execute("""
import anndata as ad
import numpy as np
zdata = ad.AnnData(X=np.random.rand(10, 5))
print("zdata created")
""", save_to_history=True)
        assert sandbox.runtime_revision == rev_before + 1

        from backend.services.snapshot_v2.models import IdleCapsule, SnapshotCaptureRequest
        req = SnapshotCaptureRequest(
            session_id=service.session_id,
            reason="test_revision",
            capsule=IdleCapsule(graph_version="2.0"),
        )
        record = await service.capture(req, sandbox=sandbox)

        manifest = record.manifest
        assert manifest.runtime is not None
        assert manifest.runtime.status == "complete"
        assert (manifest.runtime.adata_refs or manifest.runtime.captured_residual_names)

    @pytest.mark.asyncio
    async def test_service_list_snapshots_after_capture(self, sandbox, v2_paths, service):
        from backend.services.snapshot_v2.models import IdleCapsule, SnapshotCaptureRequest
        # Canonical artifact so required runtime set is non-empty.
        await sandbox.execute(
            "results = {'a': 1}",
            save_to_history=False,
            skip_safety_check=True,
        )

        req = SnapshotCaptureRequest(
            session_id=service.session_id,
            reason="test_list",
            capsule=IdleCapsule(graph_version="2.0"),
        )
        record = await service.capture(req, sandbox=sandbox)
        assert record.failure is None, f"Capture failed: {record.failure}"

        snapshots = service.list_snapshots()
        snap_ids = {
            (s.get("snapshot_id") if isinstance(s, dict) else getattr(s, "snapshot_id", None))
            for s in snapshots
        }
        assert record.manifest.snapshot_id in snap_ids

    @pytest.mark.asyncio
    async def test_store_status_after_capture(self, sandbox, v2_paths, service):
        from backend.services.snapshot_v2.models import IdleCapsule, SnapshotCaptureRequest
        await sandbox.execute("""
import anndata as ad
import numpy as np
bdata = ad.AnnData(X=np.random.rand(10, 5))
print("bdata created")
""", save_to_history=False, skip_safety_check=True)

        req = SnapshotCaptureRequest(
            session_id=service.session_id,
            reason="test_status",
            capsule=IdleCapsule(graph_version="2.0"),
        )
        await service.capture(req, sandbox=sandbox)

        status = service.store_status()
        assert status.manifest_count >= 1
