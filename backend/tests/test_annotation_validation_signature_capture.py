import json
import pytest
from types import SimpleNamespace
from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.workflow.annotation_validation import MAIN_OBJECT_SIGNATURE_KEYS


class _FakeSandbox:
    def __init__(self, output):
        self._output = output

    async def execute(
        self, code, save_to_history=True, skip_safety_check=False, **_kwargs
    ):
        return SimpleNamespace(success=True, output=self._output, error=None)


def _agent(sandbox):
    a = BioinformaticsAgent.__new__(BioinformaticsAgent)
    a.sandbox = sandbox
    a.session_id = "s"
    return a


@pytest.mark.asyncio
async def test_capture_signature_has_all_keys():
    sig = {k: (1 if k not in ("obs_columns", "layers", "cluster_counts", "obs_hash", "var_hash")
               else ([] if k in ("obs_columns", "layers") else ({} if k == "cluster_counts" else "h")))
           for k in MAIN_OBJECT_SIGNATURE_KEYS}
    out = await _agent(_FakeSandbox(json.dumps(sig))).capture_main_object_signature("adata", "leiden")
    assert set(out.keys()) == set(MAIN_OBJECT_SIGNATURE_KEYS)


@pytest.mark.asyncio
async def test_capture_signature_empty_on_failure():
    assert await _agent(_FakeSandbox("oops")).capture_main_object_signature("adata", "leiden") == {}
