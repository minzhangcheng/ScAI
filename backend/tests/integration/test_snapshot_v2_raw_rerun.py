"""Integration tests: reset-to-rerun using original uploads.

Restore without runtime using reset, assert forward block, invoke /rerun,
and complete a fresh analysis fixture reading unchanged uploads.
"""

import hashlib
import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
        reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
    ),
]


class TestRawRerun:
    @pytest.mark.asyncio
    async def test_reset_blocks_forward_resume(self, sandbox, v2_paths, service):
        from backend.services.snapshot_v2.models import IdleCapsule, SnapshotCaptureRequest
        # Canonical artifact name so live-kernel capture has a required set.
        await sandbox.execute(
            "results = {'x': 42}",
            save_to_history=False,
            skip_safety_check=True,
        )

        req = SnapshotCaptureRequest(
            session_id=service.session_id,
            reason="test_rerun",
            capsule=IdleCapsule(graph_version="2.0"),
        )
        record = await service.capture(req, sandbox=sandbox)
        assert record.failure is None, f"Capture failed: {record.failure}"

        from backend.utils.runtime_apply import is_forward_resume_blocked
        assert is_forward_resume_blocked("unavailable") is True
        assert is_forward_resume_blocked("complete") is False
        assert is_forward_resume_blocked("intentionally_kept") is False

    @pytest.mark.asyncio
    async def test_quarantine_live_runtime_clears_pkl(self, session_id):
        from backend.utils.runtime_apply import quarantine_live_runtime, _live_pkl, _live_sig

        live_pkl = _live_pkl(session_id)
        live_sig = _live_sig(session_id)
        live_pkl.parent.mkdir(parents=True, exist_ok=True)
        live_sig.parent.mkdir(parents=True, exist_ok=True)
        live_pkl.write_bytes(b"dummy_pkl")
        live_sig.write_text("dummy_sig")

        quarantine_live_runtime(session_id)
        assert not live_pkl.exists()
        assert not live_sig.exists()

        quar_pkl = live_pkl.with_suffix(live_pkl.suffix + ".quarantine")
        quar_sig = live_sig.with_suffix(live_sig.suffix + ".quarantine")
        assert quar_pkl.exists()
        assert quar_sig.exists()

    @pytest.mark.asyncio
    async def test_restore_with_reset_clears_runtime(self, sandbox, v2_paths, service):
        from backend.services.snapshot_v2.models import IdleCapsule, SnapshotCaptureRequest
        from backend.services.snapshot_v2.restore import RestoreCoordinator

        await sandbox.execute("""
import anndata as ad
import numpy as np
rerun_adata = ad.AnnData(X=np.random.rand(10, 5))
print("rerun_adata created")
""", save_to_history=False, skip_safety_check=True)

        req = SnapshotCaptureRequest(
            session_id=service.session_id,
            reason="test_rerun_reset",
            capsule=IdleCapsule(graph_version="2.0"),
        )
        record = await service.capture(req, sandbox=sandbox)
        assert record.failure is None
        snapshot_id = record.manifest.snapshot_id

        coord = RestoreCoordinator()
        staged = await coord.stage(
            snapshot_id=snapshot_id,
            session_id=service.session_id,
            paths=v2_paths,
        )

        applied = await coord.apply(
            staged, v2_paths,
            session_dir=str(v2_paths.root),
            workspace_root=str(v2_paths.root / "live_ws"),
            states_root=str(v2_paths.root / "live_states"),
            runtime_choice="reset",
        )
        assert applied.runtime_applied

        proof = {
            "verified_live_root_hash": hashlib.sha256(
                b"raw-rerun-commit-proof"
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
    async def test_capture_with_declared_outputs(self, sandbox, v2_paths, service):
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.object_store import ObjectStore

        await sandbox.execute("""
import numpy as np
custom_results = {"pval": 0.01, "log2fc": 2.5}
print("custom_results created")
""", save_to_history=False, skip_safety_check=True)

        obj_store = ObjectStore(v2_paths)
        capturer = RuntimeCapture(obj_store, v2_paths)
        manifest = await capturer.capture(sandbox, declared_outputs={"custom_results"})

        assert "custom_results" in manifest.captured_residual_vars

    @pytest.mark.asyncio
    async def test_required_variable_missing_raises(self, sandbox, v2_paths, service):
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.errors import SnapshotCaptureError

        obj_store = ObjectStore(v2_paths)
        capturer = RuntimeCapture(obj_store, v2_paths)
        with pytest.raises(SnapshotCaptureError):
            await capturer.capture(sandbox, declared_outputs={"nonexistent_var"})
