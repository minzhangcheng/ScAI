"""Code-history dedup hash must be stable across process restarts."""
import os
import subprocess
import sys


def _run_with_seed(seed: str) -> str:
    code = (
        "from backend.workflow.agents import _stable_code_hash; "
        "print(_stable_code_hash('import scanpy as sc\\nsc.pp.pca(adata)'))"
    )
    env = {**os.environ, "PYTHONPATH": "/workspace/cmz/ScAI/scai", "PYTHONHASHSEED": seed}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, check=True,
    )
    return proc.stdout.strip()


def test_stable_hash_helper_matches_across_processes():
    out1 = _run_with_seed("1")
    out2 = _run_with_seed("9999")
    assert out1 == out2, f"Hash differs across PYTHONHASHSEED: {out1!r} vs {out2!r}"
    assert out1


def test_stable_hash_differs_for_different_code():
    from backend.workflow.agents import _stable_code_hash

    assert _stable_code_hash("a = 1") != _stable_code_hash("a = 2")


def test_stable_hash_handles_unicode():
    from backend.workflow.agents import _stable_code_hash

    assert _stable_code_hash("# 注释\na=1") == _stable_code_hash("# 注释\na=1")