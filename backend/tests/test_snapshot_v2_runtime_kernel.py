import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio


pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
        reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
    ),
]


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
            for d in [self.root, self.objects, self.staging, self.journals, self.integrity, self.committed]:
                d.mkdir(parents=True, exist_ok=True)

    p = FakePaths()
    p.ensure_dirs()
    return p


@pytest_asyncio.fixture
async def sandbox(tmp_path):
    from backend.utils.sandbox import LocalJupyterSandbox
    session_id = f"test-rt-{uuid.uuid4().hex[:8]}"
    state_dir = str(tmp_path / "states")
    working_dir = str(tmp_path / "workspace")
    sb = LocalJupyterSandbox(
        session_id=session_id,
        state_dir=state_dir,
        working_dir=working_dir,
    )
    await sb.start()
    yield sb
    await sb.stop()


class TestAnnDataCapture:
    async def _ensure_anndata(self, sandbox):
        code = """
import anndata as ad
import numpy as np
adata = ad.AnnData(X=np.random.rand(50, 10))
adata.obs['cluster'] = 'A'
adata.var['gene'] = [f'g{i}' for i in range(10)]
print("adata created:", adata.shape)
"""
        result = await sandbox.execute(code, save_to_history=False, skip_safety_check=True)
        assert result.success, f"Failed to create adata: {result.error}"
        return result

    @pytest.mark.asyncio
    async def test_anndata_h5ad_fidelity(self, sandbox, v2_paths):
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.object_store import ObjectStore

        await self._ensure_anndata(sandbox)

        obj_store = ObjectStore(v2_paths)
        capturer = RuntimeCapture(obj_store, v2_paths)
        manifest = await capturer.capture(sandbox, declared_outputs={"adata"})

        assert manifest.capture_command == "runtime_capture_v2"
        assert manifest.kernel_id == sandbox.session_id
        assert "adata" in manifest.captured_adata_vars

    @pytest.mark.asyncio
    async def test_anndata_with_raw_roundtrip(self, sandbox, v2_paths):
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.object_store import ObjectStore

        code = """
import anndata as ad
import numpy as np
import pandas as pd
adata = ad.AnnData(X=np.random.rand(50, 10))
adata.raw = adata.copy()
adata.layers['counts'] = adata.X.copy()
adata.obsm['X_pca'] = np.random.rand(50, 5)
adata.varm['loadings'] = np.random.rand(10, 3)
adata.obsp['connectivities'] = np.eye(50)
adata.uns['dataset'] = 'test'
adata.obs['cat'] = pd.Categorical(['a','b']*25)
print("adata with raw/layers/obsm/varm/obsp/uns/categoricals created")
"""
        result = await sandbox.execute(code, save_to_history=False, skip_safety_check=True)
        assert result.success, f"Failed to create rich adata: {result.error}"

        obj_store = ObjectStore(v2_paths)
        capturer = RuntimeCapture(obj_store, v2_paths)
        manifest = await capturer.capture(sandbox, declared_outputs={"adata"})

        assert "adata" in manifest.captured_adata_vars
        assert len(manifest.captured_adata_refs) == 1
        ref = manifest.captured_adata_refs["adata"]
        assert ref.sha256 != "0" * 64
        assert ref.size > 0

        obj_store.verify(ref)

    @pytest.mark.asyncio
    async def test_multiple_anndata_captured(self, sandbox, v2_paths):
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.object_store import ObjectStore

        code = """
import anndata as ad
import numpy as np
adata1 = ad.AnnData(X=np.random.rand(20, 5))
adata2 = ad.AnnData(X=np.random.rand(30, 8))
adata3 = ad.AnnData(X=np.random.rand(10, 3))
print("three adata objects created")
"""
        result = await sandbox.execute(code, save_to_history=False, skip_safety_check=True)
        assert result.success

        obj_store = ObjectStore(v2_paths)
        capturer = RuntimeCapture(obj_store, v2_paths)
        manifest = await capturer.capture(sandbox, declared_outputs=set())

        assert "adata1" in manifest.captured_adata_vars
        assert "adata2" in manifest.captured_adata_vars
        assert "adata3" in manifest.captured_adata_vars

    @pytest.mark.asyncio
    async def test_cross_cell_promoted_variable(self, sandbox, v2_paths):
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.object_store import ObjectStore

        code = """
import anndata as ad
import numpy as np
adata = ad.AnnData(X=np.random.rand(20, 5))
shared_results = {"pca": "done", "leiden": "done"}
print("adata + shared_results created")
"""
        result = await sandbox.execute(code, save_to_history=False, skip_safety_check=True)
        assert result.success

        obj_store = ObjectStore(v2_paths)
        capturer = RuntimeCapture(obj_store, v2_paths)
        manifest = await capturer.capture(sandbox, declared_outputs={"shared_results"})

        assert "adata" in manifest.captured_adata_vars
        assert "shared_results" in manifest.captured_residual_vars or "shared_results" in manifest.captured_adata_vars

    @pytest.mark.asyncio
    async def test_required_variable_failure_raises_capture_error(self, sandbox, v2_paths):
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.errors import SnapshotCaptureError

        obj_store = ObjectStore(v2_paths)
        capturer = RuntimeCapture(obj_store, v2_paths)
        with pytest.raises(SnapshotCaptureError):
            await capturer.capture(sandbox, declared_outputs={"nonexistent_var"})

    @pytest.mark.asyncio
    async def test_sparse_anndata_h5ad(self, sandbox, v2_paths):
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.object_store import ObjectStore

        code = """
import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix
X = csr_matrix(np.random.rand(40, 12) > 0.5, dtype=float)
adata_sparse = ad.AnnData(X=X)
print("sparse adata created")
"""
        result = await sandbox.execute(code, save_to_history=False, skip_safety_check=True)
        assert result.success, f"sparse adata creation failed: {result.error}"

        obj_store = ObjectStore(v2_paths)
        capturer = RuntimeCapture(obj_store, v2_paths)
        manifest = await capturer.capture(sandbox, declared_outputs=set())

        assert "adata_sparse" in manifest.captured_adata_vars

    @pytest.mark.asyncio
    async def test_dense_anndata_h5ad(self, sandbox, v2_paths):
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.object_store import ObjectStore

        code = """
import anndata as ad
import numpy as np
adata_dense = ad.AnnData(X=np.random.rand(30, 15))
print("dense adata created")
"""
        result = await sandbox.execute(code, save_to_history=False, skip_safety_check=True)
        assert result.success

        obj_store = ObjectStore(v2_paths)
        capturer = RuntimeCapture(obj_store, v2_paths)
        manifest = await capturer.capture(sandbox, declared_outputs=set())
        assert "adata_dense" in manifest.captured_adata_vars


class TestRuntimeFreeSpaceGate:
    @pytest.mark.asyncio
    async def test_anndata_with_unwritable_obsm_is_captured(self, sandbox, v2_paths):
        """AnnData with non-h5ad-native obsm values must still capture via sanitize."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.object_store import ObjectStore
        import pytest as _pytest

        # formulaic may not be installed in all test envs
        code_probe = "import formulaic\nprint('ok')"
        probe = await sandbox.execute(code_probe, save_to_history=False, skip_safety_check=True)
        if not probe.success:
            _pytest.skip("formulaic not installed in sandbox")

        code = """
import anndata as ad
import numpy as np
import pandas as pd
from formulaic import model_matrix
dds = ad.AnnData(X=np.random.rand(4, 3))
dds.obs['condition'] = pd.Categorical(['A','A','B','B'])
# ModelMatrix is not natively writable to h5ad under obsm
dds.obsm['design_matrix'] = model_matrix('~ condition', dds.obs)
print('dds ready', dds.shape)
"""
        result = await sandbox.execute(code, save_to_history=False, skip_safety_check=True)
        assert result.success, result.error

        capturer = RuntimeCapture(ObjectStore(v2_paths), v2_paths)
        runtime = await capturer.capture(sandbox, declared_outputs={"dds"})
        assert runtime.status == "complete"
        assert "dds" in runtime.adata_refs

    @pytest.mark.asyncio
    async def test_anndata_with_unwritable_uns_series_is_captured(self, sandbox, v2_paths):
        """AnnData whose uns holds a bare pd.Series (PyDESeq2 trend_coeffs) must
        still capture: the kernel template recursively sanitizes uns."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.object_store import ObjectStore

        code = """
import anndata as ad
import numpy as np
import pandas as pd
dds = ad.AnnData(X=np.random.rand(6, 8))
# Bare Series in uns has no registered h5ad writer under /uns.
dds.uns['trend_coeffs'] = pd.Series([0.1, 0.2, 0.3])
dds.uns['fit'] = {'coeffs': pd.Series([1.0, 2.0]), 'n': 6}
dds.uns['neighbors'] = {'params': {'n_neighbors': 15}}
print('dds ready', dds.shape)
"""
        result = await sandbox.execute(code, save_to_history=False, skip_safety_check=True)
        assert result.success, result.error

        capturer = RuntimeCapture(ObjectStore(v2_paths), v2_paths)
        runtime = await capturer.capture(sandbox, declared_outputs={"dds"})
        assert runtime.status == "complete"
        assert "dds" in runtime.adata_refs

    @pytest.mark.asyncio
    async def test_anndata_with_raw_x_obsm_uns_edge_cases_is_captured(self, sandbox, v2_paths):
        """End-to-end: raw.var object col, object-dtype X, obsm DataFrame with a
        bad column, and uns list[dict] must all be sanitized so capture succeeds."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.object_store import ObjectStore

        code = """
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
adata = ad.AnnData(X=sp.csr_matrix(np.random.rand(6, 10)))
adata.var_names = [f'g{i}' for i in range(10)]
adata.var['flag'] = pd.Series([True, None, 'x'] + [False]*7, index=adata.var_names, dtype=object)
adata.raw = adata.copy()
adata = adata[:, :4].copy()
adata.X = np.asarray(adata.X.toarray(), dtype=object)
adata.obsm['meta'] = pd.DataFrame({'b': pd.Series([True, None]*3, dtype=object)}, index=adata.obs_names)
adata.uns['markers'] = ['CD3', 'CD4']
adata.uns['weird'] = [{'k': 1}, {'k': 2}]
print('adata ready', adata.shape)
"""
        result = await sandbox.execute(code, save_to_history=False, skip_safety_check=True)
        assert result.success, result.error

        capturer = RuntimeCapture(ObjectStore(v2_paths), v2_paths)
        runtime = await capturer.capture(sandbox, declared_outputs={"adata"})
        assert runtime.status == "complete"
        assert "adata" in runtime.adata_refs

    @pytest.mark.asyncio
    async def test_anndata_with_unwritable_obs_column_is_captured(self, sandbox, v2_paths):
        """AnnData whose obs has an object column of non-string objects
        (scrublet 'predicted_doublet' bool/None) must still capture."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.object_store import ObjectStore

        code = """
import anndata as ad
import numpy as np
import pandas as pd
adata = ad.AnnData(X=np.random.rand(4, 3))
adata.obs['predicted_doublet'] = pd.Series(
    [True, False, None, True], index=adata.obs_names, dtype=object)
print('adata ready', adata.shape)
"""
        result = await sandbox.execute(code, save_to_history=False, skip_safety_check=True)
        assert result.success, result.error

        capturer = RuntimeCapture(ObjectStore(v2_paths), v2_paths)
        runtime = await capturer.capture(sandbox, declared_outputs={"adata"})
        assert runtime.status == "complete"
        assert "adata" in runtime.adata_refs

    @pytest.mark.asyncio
    async def test_capture_with_ample_free_space_succeeds(self, sandbox, v2_paths):
        """Normal-sized data with ample free space succeeds."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.object_store import ObjectStore

        code = """
import anndata as ad
import numpy as np
adata = ad.AnnData(X=np.random.rand(10, 5))
print("adata created")
"""
        result = await sandbox.execute(code, save_to_history=False, skip_safety_check=True)
        assert result.success

        obj_store = ObjectStore(v2_paths)
        capturer = RuntimeCapture(obj_store, v2_paths)
        manifest = await capturer.capture(sandbox, declared_outputs=set())
        assert "adata" in manifest.captured_adata_vars

    @pytest.mark.asyncio
    async def test_disk_full_raises_error(self, sandbox, v2_paths):
        """Capture raises when post-write free is below reserve."""
        from backend.services.snapshot_v2.runtime import RuntimeCapture
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.errors import SnapshotCaptureError
        from unittest.mock import patch

        code = """
import anndata as ad
import numpy as np
adata = ad.AnnData(X=np.random.rand(10, 5))
print("adata created")
"""
        result = await sandbox.execute(code, save_to_history=False, skip_safety_check=True)
        assert result.success

        obj_store = ObjectStore(v2_paths)
        capturer = RuntimeCapture(obj_store, v2_paths)
        capturer.FREE_SPACE_RESERVE_BYTES = 10**15  # force free < reserve after write

        fake_usage = type("U", (), {"total": 10**12, "used": 0, "free": 10**9})()
        with patch("shutil.disk_usage", return_value=fake_usage):
            with pytest.raises(SnapshotCaptureError) as exc_info:
                await capturer.capture(sandbox, declared_outputs=set())
        assert "RUNTIME_CAPTURE_DISK_FULL" in exc_info.value.code


class TestSandboxRuntimeRevision:
    @pytest.mark.asyncio
    async def test_runtime_revision_starts_at_zero(self, sandbox):
        assert sandbox.runtime_revision == 0

    @pytest.mark.asyncio
    async def test_runtime_revision_increments_on_execute(self, sandbox):
        rev_before = sandbox.runtime_revision
        result = await sandbox.execute("x = 1", save_to_history=True)
        assert result.success
        assert sandbox.runtime_revision == rev_before + 1

    @pytest.mark.asyncio
    async def test_runtime_revision_not_incremented_on_internal_code(self, sandbox):
        rev_before = sandbox.runtime_revision
        result = await sandbox.execute("x = 1", save_to_history=False, skip_safety_check=True)
        assert result.success
        assert sandbox.runtime_revision == rev_before

    @pytest.mark.asyncio
    async def test_runtime_revision_not_incremented_on_failure(self, sandbox):
        rev_before = sandbox.runtime_revision
        result = await sandbox.execute("1/0", save_to_history=True)
        assert not result.success
        assert sandbox.runtime_revision == rev_before

    @pytest.mark.asyncio
    async def test_saved_runtime_revision_starts_at_zero(self, sandbox):
        assert sandbox.saved_runtime_revision == 0

    @pytest.mark.asyncio
    async def test_mark_runtime_committed_updates_saved_revision(self, sandbox):
        await sandbox.execute("x = 1", save_to_history=True)
        assert sandbox.runtime_revision == 1
        assert sandbox.saved_runtime_revision == 0
        sandbox.mark_runtime_committed()
        assert sandbox.saved_runtime_revision == 1
