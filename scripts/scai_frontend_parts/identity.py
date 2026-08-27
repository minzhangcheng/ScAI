"""Source and input identity checks for frontend test runs."""

from __future__ import annotations

import importlib.util
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from .artifacts import fail, lstat_or_none, require_regular, sha256_bytes, sha256_file

def load_tree_helper(repo: Path):
    path = repo / "scripts" / "scai_test_tree.py"
    spec = importlib.util.spec_from_file_location("scai_frontend_tree", path)
    if spec is None or spec.loader is None:
        fail("source manifest", "cannot load scripts/scai_test_tree.py")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return module


def resolve_run_start_head(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "HEAD^{commit}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode or not completed.stdout.endswith(b"\n"):
        fail("source manifest", "cannot resolve one terminal-LF run-start HEAD")
    raw = completed.stdout[:-1]
    if re.fullmatch(rb"[0-9a-f]{40}", raw) is None:
        fail("source manifest", "run-start HEAD is not one lowercase 40-hex commit")
    return raw.decode("ascii")


def source_snapshot(repo: Path, tree: Any, base_commit: str) -> dict[str, Any]:
    try:
        return tree.build_worktree_manifest(repo, base_commit)
    except Exception as exc:
        fail("source manifest", str(exc))


def source_node_modules_state(frontend: Path) -> tuple[str, str | None]:
    path = frontend / "node_modules"
    info = lstat_or_none(path)
    if info is None:
        return "absent", None
    if stat.S_ISLNK(info.st_mode):
        return "symlink", os.readlink(path)
    if stat.S_ISDIR(info.st_mode):
        fail("source node_modules", "real frontend/node_modules directory is forbidden")
    fail("source node_modules", "frontend/node_modules is a special file")


def input_hashes(frontend: Path) -> tuple[str, str, str]:
    package = frontend / "package.json"
    lock = frontend / "package-lock.json"
    require_regular(package, "frontend inputs")
    require_regular(lock, "frontend inputs")
    package_hash = sha256_file(package)
    lock_hash = sha256_file(lock)
    input_key = sha256_bytes(f"package.json {package_hash}\npackage-lock.json {lock_hash}\n".encode())
    return package_hash, lock_hash, input_key
