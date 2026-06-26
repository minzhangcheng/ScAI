import json
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

    async def execute(self, code, save_to_history=True, skip_safety_check=False):
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
async def test_executor_completes_on_copy():
    final = {"status": "completed", "summary": "done",
             "cluster_results": {"0": {"support_level": "supported", "qc_flags": ["ok"]}},
             "suggested_annotation_changes": {}, "artifacts": [], "warnings": []}
    chat = [json.dumps({"code": "print('check ran')", "done": False}),
            json.dumps({"done": True, "validation_result": final})]
    a = _agent(_Sandbox(), chat)
    out = await a.validate_annotations_with_plan(
        validation_plan={"tasks": []}, annotations={"0": {"cell_type": "X"}},
        cluster_markers={"0": ["A"]}, tissue="liver", organism="mouse", context=_META,
        max_iterations=5, timeout_seconds=30)
    assert out["status"] == "completed"
    val = [c for c in a.sandbox.executed if "check ran" in c]
    assert val and "adata_val = adata.copy()" in val[0]


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
async def test_executor_stops_at_max_iterations():
    chat = [json.dumps({"code": "print('x')", "done": False})] * 20
    a = _agent(_Sandbox(), chat)
    out = await a.validate_annotations_with_plan(
        validation_plan={"tasks": []}, annotations={"0": {}}, cluster_markers={"0": ["A"]},
        tissue="liver", organism="mouse", context=_META, max_iterations=3, timeout_seconds=30)
    assert out["status"] in ("partial", "failed")
    assert len([c for c in a.sandbox.executed if "print('x')" in c]) <= 3


@pytest.mark.asyncio
async def test_executor_rejects_denylisted_code():
    chat = [json.dumps({"code": "sc.tl.leiden(adata_val)", "done": False}),
            json.dumps({"done": True, "validation_result": {"status": "completed", "summary": "ok",
                        "cluster_results": {"0": {"qc_flags": ["x"]}}, "artifacts": [], "warnings": []}})]
    a = _agent(_Sandbox(), chat)
    out = await a.validate_annotations_with_plan(
        validation_plan={"tasks": []}, annotations={"0": {}}, cluster_markers={"0": ["A"]},
        tissue="liver", organism="mouse", context=_META, max_iterations=5, timeout_seconds=30)
    # denylisted code must not be executed as a validation cell
    assert not any("sc.tl.leiden(adata_val)" in c for c in a.sandbox.executed)


@pytest.mark.asyncio
async def test_executor_fails_fast_when_pre_signature_unavailable():
    # Signature capture returns {} (e.g. kernel error) -> must NOT execute any validation code.
    class _NoSigSandbox(_Sandbox):
        async def execute(self, code, save_to_history=True, skip_safety_check=False):
            self.executed.append(code)
            if "'var'" in code:
                return SimpleNamespace(success=True, output=json.dumps(_META), error=None)
            if "_sig =" in code:
                return SimpleNamespace(success=True, output="", error="kernel boom")  # capture fails -> {}
            return SimpleNamespace(success=True, output="ran", error=None)

    chat = [json.dumps({"code": "print('should not run')", "done": False})]
    a = _agent(_NoSigSandbox(), chat)
    out = await a.validate_annotations_with_plan(
        validation_plan={"tasks": []}, annotations={"0": {}}, cluster_markers={"0": ["A"]},
        tissue="liver", organism="mouse", context=_META, max_iterations=5, timeout_seconds=30)
    assert out["status"] == "failed"
    assert any("signature" in w.lower() for w in out["warnings"])
    # No validation code (only the signature-capture probe) should have executed.
    assert not any("should not run" in c for c in a.sandbox.executed)


@pytest.mark.asyncio
async def test_executor_creates_and_passes_absolute_artifact_dir(tmp_path):
    captured = {}

    class _WsSandbox(_Sandbox):
        def __init__(self, working_dir):
            super().__init__()
            self.working_dir = working_dir

    sandbox = _WsSandbox(str(tmp_path))
    final = {"status": "completed", "summary": "ok",
             "cluster_results": {"0": {"qc_flags": ["ok"]}},
             "suggested_annotation_changes": {}, "artifacts": [], "warnings": []}
    chat = [json.dumps({"done": True, "validation_result": final})]
    a = _agent(sandbox, chat)

    # Capture the prompt the executor builds.
    orig = a._build_validation_executor_prompt
    a._build_validation_executor_prompt = lambda *args, **kw: captured.setdefault("prompt", orig(*args, **kw)) or captured["prompt"]

    out = await a.validate_annotations_with_plan(
        validation_plan={"tasks": []}, annotations={"0": {}}, cluster_markers={"0": ["A"]},
        tissue="liver", organism="mouse", context={"var": "adata", "cluster_key": "leiden"},
        max_iterations=5, timeout_seconds=30)
    assert out["status"] == "completed"
    # The annotation_validation dir was created under the session workspace.
    assert (tmp_path / "annotation_validation").is_dir()
    # The absolute artifact dir was surfaced into the prompt.
    assert str(tmp_path / "annotation_validation") in captured["prompt"]
