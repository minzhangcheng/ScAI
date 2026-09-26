import contextlib
import io
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from types import SimpleNamespace
from backend.agents.bioinformatics_agent import BioinformaticsAgent


class _FakeSandbox:
    def __init__(self, output):
        self._output = output
        self.code = ""

    async def execute(
        self, code, records_history=True, skip_safety_check=False, **_kwargs
    ):
        self.code = code
        return SimpleNamespace(success=True, output=self._output, error=None)


class _ExecutingSandbox:
    def __init__(self, adata):
        self.namespace = {"adata": adata}

    async def execute(self, code, **_kwargs):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(code, self.namespace)
        return SimpleNamespace(success=True, output=stream.getvalue(), error=None)


def _agent(sandbox):
    a = BioinformaticsAgent.__new__(BioinformaticsAgent)
    a.sandbox = sandbox
    a.session_id = "s"
    return a


@pytest.mark.asyncio
async def test_capture_signature_hashes_sparse_storage_content():
    sandbox = _FakeSandbox("{}")

    await _agent(sandbox).capture_main_object_signature("adata", None)

    assert "content_sha256" in sandbox.code
    assert "('indices', 'indptr', 'row', 'col')" in sandbox.code
    assert "memoryview(_arr).cast('B')" in sandbox.code


@pytest.mark.asyncio
async def test_capture_signature_detects_sparse_value_reordering():
    adata = SimpleNamespace(
        X=sparse.csr_matrix(np.array([[1.0, 2.0], [3.0, 0.0]])),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
        raw=None,
        layers={}, obsm={}, varm={}, obsp={}, uns={},
        n_obs=2, n_vars=2,
    )
    agent = _agent(_ExecutingSandbox(adata))

    before = await agent.capture_main_object_signature("adata", None)
    adata.X.data[:] = adata.X.data[::-1]
    after = await agent.capture_main_object_signature("adata", None)

    assert before["x_sum"] == after["x_sum"]
    assert before["x_nnz"] == after["x_nnz"]
    assert before["x_fingerprint"] != after["x_fingerprint"]
