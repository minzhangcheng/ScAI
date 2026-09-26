#!/usr/bin/env python3
"""Measure first-party application and test code, including test infrastructure."""

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".css", ".sh"}


def category(name):
    path = Path(name)
    if path.suffix not in CODE_SUFFIXES and not name.startswith("scripts/"):
        return None
    if (
        "tests" in path.parts or "__tests__" in path.parts
        or ".test." in name or ".spec." in name
        or path.name in {"conftest.py", "test-setup.ts", "test-helpers.ts"}
    ):
        return "tests"
    if name.startswith("scripts/") and any(
        word in name for word in ("test", "frontend", "evidence")
    ):
        return "test_infrastructure"
    if name.startswith(("backend/", "frontend/src/", "clients/python/scai_client/")):
        return "application"
    return None


def measure(ref=None):
    command = ["git", "ls-tree", "-r", "--name-only", ref] if ref else [
        "git", "ls-files", "--cached", "--others", "--exclude-standard",
    ]
    paths = subprocess.check_output(command, cwd=ROOT, text=True).splitlines()
    totals = {key: {"files": 0, "lines": 0, "nonblank_lines": 0}
              for key in ("application", "tests", "test_infrastructure")}
    for name in sorted(set(paths)):
        group = category(name)
        if group is None or (not ref and not (ROOT / name).is_file()):
            continue
        source = subprocess.check_output(
            ["git", "show", f"{ref}:{name}"], cwd=ROOT, text=True,
        ) if ref else (ROOT / name).read_text()
        lines = source.splitlines()
        totals[group]["files"] += 1
        totals[group]["lines"] += len(lines)
        totals[group]["nonblank_lines"] += sum(bool(line.strip()) for line in lines)
    application = totals["application"]["lines"]
    tests = totals["tests"]["lines"] + totals["test_infrastructure"]["lines"]
    return {"ref": ref or "worktree", "counts": totals,
            "test_lines_including_infrastructure": tests,
            "test_to_application_percent": round(100 * tests / application, 4),
            "within_30_percent": tests * 10 <= application * 3}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="Measure a Git revision instead of the worktree")
    print(json.dumps(measure(parser.parse_args().ref), indent=2))
