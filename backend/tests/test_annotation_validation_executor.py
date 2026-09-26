import json
import asyncio
import pytest
from types import SimpleNamespace
from backend.agents.bioinformatics_agent import BioinformaticsAgent

_SIG = {"id": 1, "n_obs": 100, "n_vars": 50, "obs_columns": ["leiden"],
        "cluster_counts": {"0": 60, "1": 40}, "layers": ["counts"], "x_id": 2,
        "x_sum": 10.0, "x_nnz": 80, "obs_hash": "h", "var_hash": "v",
        "layers_fingerprint": {"counts": {"sum": 10.0, "nnz": 80}}}
_META = {"var": "adata", "cluster_key": "leiden", "obs_columns": ["leiden"],
         "qc_like": [], "condition_like": [], "layers": ["counts"], "n_obs": 100, "n_vars": 50}


def _sig_for(code, mutated=False):
    s = dict(_SIG)
    if mutated:
        s["x_sum"] = 999.0
    return json.dumps(s)


class _Sandbox:
    def __init__(self, mutate_after=None):
        self.executed = []
        self._sig_calls = 0
        self._mutate_after = mutate_after  # iteration index after which signature changes

    async def execute(
        self, code, records_history=True, skip_safety_check=False, **_kwargs
    ):
        self.executed.append(code)
        if "'var'" in code:                       # introspection
            return SimpleNamespace(success=True, output=json.dumps(_META), error=None)
        if "_sig =" in code:                      # signature capture
            self._sig_calls += 1
            mutated = self._mutate_after is not None and self._sig_calls > self._mutate_after
            return SimpleNamespace(success=True, output=_sig_for(code, mutated), error=None)
        return SimpleNamespace(success=True, output="ran", error=None)  # validation code


def _agent(sandbox, chat_responses):
    a = BioinformaticsAgent.__new__(BioinformaticsAgent)
    a.sandbox = sandbox; a.session_id = "s"
    a._q = list(chat_responses)

    async def fake_chat(prompt, temperature=None):
        return a._q.pop(0)
    a.chat = fake_chat
    return a


@pytest.mark.asyncio
async def test_executor_aborts_on_first_iteration_mutation():
    # Signature changes after the first validation execution -> abort immediately.
    chat = [json.dumps({"code": "adata.X.data[:] = 0", "done": False}),
            json.dumps({"code": "print('should not run')", "done": False})]
    a = _agent(_Sandbox(mutate_after=1), chat)  # pre-sig is call 1; post first exec is call 2 -> mutated
    out = await a.validate_annotations_with_plan(
        validation_plan={"tasks": []}, annotations={"0": {}}, cluster_markers={"0": ["A"]},
        tissue="liver", organism="mouse", context=_META, max_iterations=5, timeout_seconds=30)
    assert out["status"] == "failed"
    assert any("invariant" in w.lower() for w in out["warnings"])
    assert not any("should not run" in c for c in a.sandbox.executed)  # loop aborted


@pytest.mark.asyncio
async def test_executor_timeout_releases_validation_copy(monkeypatch):
    sandbox = _Sandbox()
    a = _agent(sandbox, [])

    async def slow_chat(prompt, temperature=None):
        await asyncio.sleep(1)

    a.chat = slow_chat
    original_wait_for = asyncio.wait_for
    observed_timeouts = []

    async def expire_request(awaitable, *, timeout):
        observed_timeouts.append(timeout)
        return await original_wait_for(awaitable, timeout=0.01)

    monkeypatch.setattr(asyncio, "wait_for", expire_request)
    out = await a.validate_annotations_with_plan(
        validation_plan={"tasks": []}, annotations={"0": {}}, cluster_markers={"0": ["A"]},
        tissue="liver", organism="mouse", context=_META,
        max_iterations=3, timeout_seconds=0.01,
    )

    assert out["status"] == "partial"
    assert out["warnings"] == ["timeout"]
    assert observed_timeouts == [600]
    assert "adata_val = adata.copy()" in sandbox.executed
    assert "globals().pop('adata_val', None)" in sandbox.executed
