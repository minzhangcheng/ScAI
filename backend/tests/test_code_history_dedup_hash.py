"""Code-history dedup hash must be stable across process restarts."""
import os
import subprocess
import sys


def _run_with_seeds(*seeds: str) -> list:
    """Probe every seed in its own interpreter, all started together.

    What matters is that two independent interpreters agree, not that they
    run one after the other, so the second no longer waits on the first's
    backend import.
    """
    code = (
        "from backend.workflow.agents import _stable_code_hash; "
        "print(_stable_code_hash('import scanpy as sc\\nsc.pp.pca(adata)'))"
    )
    processes = []
    for seed in seeds:
        env = {**os.environ, "PYTHONHASHSEED": seed}
        processes.append(subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        ))

    outputs = []
    for seed, proc in zip(seeds, processes):
        stdout, stderr = proc.communicate(timeout=300)
        assert proc.returncode == 0, f"PYTHONHASHSEED={seed}: {stderr}"
        outputs.append(stdout.strip())
    return outputs


def test_stable_hash_helper_matches_across_processes():
    out1, out2 = _run_with_seeds("1", "9999")
    assert out1 == out2, f"Hash differs across PYTHONHASHSEED: {out1!r} vs {out2!r}"
    assert out1


def test_stable_hash_differs_for_different_code():
    from backend.workflow.agents import _stable_code_hash

    assert _stable_code_hash("a = 1") != _stable_code_hash("a = 2")


def test_stable_hash_handles_unicode():
    from backend.workflow.agents import _stable_code_hash

    assert _stable_code_hash("# 注释\na=1") == _stable_code_hash("# 注释\na=1")