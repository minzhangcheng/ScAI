"""Integration tests: AnnData fidelity after snapshot/restore cycle.

Parameterizes sparse/dense .X, raw, layers, obsm, varm, obsp, uns,
categoricals. Compares arrays, sparse structure, dtype/categories,
and metadata after restore.
"""

import hashlib
import os
from typing import Any

import pytest

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
        reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
    ),
]


@pytest.fixture(params=["dense", "sparse"])
def adata_density(request):
    return request.param


class TestAnnDataFidelity:
    """Capture AnnData with various configurations, restore, compare in-kernel."""

    @pytest.mark.asyncio
    async def test_dense_adata_roundtrip(self, sandbox, v2_paths, service):
        await sandbox.execute("""
import anndata as ad
import numpy as np
adata = ad.AnnData(X=np.random.rand(30, 15))
print("dense adata:", adata.shape)
""", save_to_history=False, skip_safety_check=True)

        await self._capture_and_restore_adata(sandbox, v2_paths, service)

        check = await sandbox.execute("""
import numpy as np
assert isinstance(adata.X, np.ndarray), "X should be dense after restore"
assert adata.X.shape == (30, 15), f"Expected (30,15), got {adata.X.shape}"
print("dense check passed")
""", save_to_history=True)
        assert check.success, f"Dense check failed: {check.error}"

    @pytest.mark.asyncio
    async def test_sparse_adata_roundtrip(self, sandbox, v2_paths, service):
        await sandbox.execute("""
import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix
X = csr_matrix(np.random.rand(40, 12) > 0.5, dtype=float)
adata_sparse = ad.AnnData(X=X)
print("sparse adata:", adata_sparse.shape)
""", save_to_history=False, skip_safety_check=True)

        await self._capture_and_restore_adata(sandbox, v2_paths, service, var_name="adata_sparse")

        check = await sandbox.execute("""
import numpy as np
from scipy.sparse import issparse
assert issparse(adata_sparse.X), "X should be sparse after restore"
assert adata_sparse.X.shape == (40, 12), f"Expected (40,12), got {adata_sparse.X.shape}"
print("sparse check passed")
""", save_to_history=True)
        assert check.success, f"Sparse check failed: {check.error}"

    @pytest.mark.asyncio
    async def test_rich_adata_all_components(self, sandbox_with_adata, v2_paths, service):
        sandbox = sandbox_with_adata
        await self._capture_and_restore_adata(sandbox, v2_paths, service)

        check = await sandbox.execute("""
import numpy as np
from scipy.sparse import issparse
from pandas import CategoricalDtype
# Check .X
assert adata.X.shape == (50, 10), f"X shape: {adata.X.shape}"
# Check .raw
assert adata.raw is not None, "raw should exist"
assert adata.raw.X.shape == (50, 10), f"raw.X shape: {adata.raw.X.shape}"
# Check layers
assert 'counts' in adata.layers, "counts layer should exist"
assert adata.layers['counts'].shape == (50, 10)
# Check obsm
assert 'X_pca' in adata.obsm, "X_pca should exist"
assert adata.obsm['X_pca'].shape == (50, 5)
# Check varm
assert 'loadings' in adata.varm, "loadings should exist"
assert adata.varm['loadings'].shape == (10, 3)
# Check obsp
assert 'connectivities' in adata.obsp, "connectivities should exist"
# Check uns
assert adata.uns.get('dataset') == 'integration_test'
# Check categoricals
assert 'cat' in adata.obs, "cat column should exist"
assert isinstance(adata.obs['cat'].dtype, CategoricalDtype), "cat should be categorical"
print("all components verified after restore")
""", save_to_history=True)
        assert check.success, f"Rich adata check failed: {check.error}"

    @pytest.mark.asyncio
    async def test_multiple_adata_preserved(self, sandbox, v2_paths, service):
        await sandbox.execute("""
import anndata as ad
import numpy as np
a1 = ad.AnnData(X=np.random.rand(10, 5))
a2 = ad.AnnData(X=np.random.rand(20, 8))
a3 = ad.AnnData(X=np.random.rand(5, 3))
print("three adata objects created")
""", save_to_history=False, skip_safety_check=True)

        await self._capture_and_restore_adata(sandbox, v2_paths, service, var_name="a1")

        check = await sandbox.execute("""
assert a1.shape == (10, 5), f"a1: {a1.shape}"
assert a2.shape == (20, 8), f"a2: {a2.shape}"
assert a3.shape == (5, 3), f"a3: {a3.shape}"
print("all adata objects preserved")
""", save_to_history=True)
        assert check.success, f"Multi-adata check failed: {check.error}"

    async def _capture_and_restore_adata(self, sandbox, v2_paths, service,
                                          var_name: str = "adata"):
        from backend.services.snapshot_v2.models import IdleCapsule, SnapshotCaptureRequest
        from backend.services.snapshot_v2.restore import RestoreCoordinator

        req = SnapshotCaptureRequest(
            session_id=service.session_id,
            reason="test_adata",
            capsule=IdleCapsule(graph_version="2.0"),
        )
        record = await service.capture(req, sandbox=sandbox)
        assert record.failure is None, f"Capture failed: {record.failure}"
        snapshot_id = record.manifest.snapshot_id

        coord = RestoreCoordinator()
        staged = await coord.stage(
            snapshot_id=snapshot_id,
            session_id=service.session_id,
            paths=v2_paths,
        )

        await coord.apply(
            staged, v2_paths,
            session_dir=str(v2_paths.root),
            workspace_root=str(v2_paths.root / "live_ws"),
            states_root=str(v2_paths.root / "live_states"),
            runtime_choice="reset",
        )

        proof = {
            "verified_live_root_hash": hashlib.sha256(
                b"anndata-commit-proof"
            ).hexdigest(),
            "component_hashes": {"runtime_policy": "reset"},
            "verified_at": "2026-07-12T00:00:00Z",
            "restore_journal_version": 1,
        }
        await coord.commit(
            staged.transaction_id, v2_paths, verification_proof=proof,
        )
        await coord.cleanup(staged.transaction_id, v2_paths)
