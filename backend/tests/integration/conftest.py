"""Integration test fixtures for Snapshot V2 production verification.

One kernel per module, bounded by timeout, always stopped in finalizer.
"""

import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio

os.environ.setdefault("SCAI_DATA_DIR", "/tmp/scai_v2_integration")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")


pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
        reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
    ),
]


@pytest.fixture
def session_id() -> str:
    return f"int-{uuid.uuid4().hex[:12]}"


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



@pytest_asyncio.fixture
async def sandbox(tmp_path, session_id):
    """One real kernel per test module, bounded by 120s timeout."""
    from backend.utils.sandbox import LocalJupyterSandbox
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


@pytest_asyncio.fixture
async def sandbox_with_adata(sandbox):
    """Kernel with a basic AnnData object created."""
    code = """
import anndata as ad
import numpy as np
import pandas as pd
adata = ad.AnnData(X=np.random.rand(50, 10))
adata.obs['cluster'] = 'A'
adata.var['gene'] = [f'g{i}' for i in range(10)]
adata.raw = adata.copy()
adata.layers['counts'] = adata.X.copy()
adata.obsm['X_pca'] = np.random.rand(50, 5)
adata.varm['loadings'] = np.random.rand(10, 3)
adata.obsp['connectivities'] = np.eye(50)
adata.uns['dataset'] = 'integration_test'
from pandas import CategoricalDtype
adata.obs['cat'] = pd.Categorical(['a','b']*25)
print("adata:", adata.shape)
"""
    result = await sandbox.execute(code, save_to_history=False, skip_safety_check=True)
    assert result.success, f"Failed to create adata: {result.error}"
    return sandbox


@pytest.fixture
def analysis_state(session_id):
    """Minimal AnalysisState registered in the global sessions dict.

    Registered so code that resolves a session from the global registry
    finds one during an integration run.
    """
    from backend.models.schemas import AnalysisState
    from backend.state import sessions

    state = AnalysisState(
        id=session_id,
        name="integration",
        user_id="integration",
        initial_instruction="integration test",
    )
    sessions[session_id] = state
    yield state
    sessions.pop(session_id, None)
