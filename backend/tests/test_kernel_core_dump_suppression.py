"""Contracts for launcher core limits and kernel death diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import uuid

import pytest

from backend.utils.sandbox import LocalJupyterSandbox


def _sandbox(tmp_path: Path) -> LocalJupyterSandbox:
    return LocalJupyterSandbox(
        session_id=f"kernel-core-dump-{uuid.uuid4().hex[:8]}",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )


def _launcher() -> Path:
    return Path(__file__).resolve().parents[1] / "utils" / "kernel_startup" / "launcher.py"


def _launcher_env(workspace: Path, allow_core_dump: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "SCAI_KERNEL_WRITE_ROOTS": json.dumps([str(workspace)]),
        "SCAI_KERNEL_REQUIRE_LANDLOCK": "0",
    })
    if allow_core_dump is None:
        env.pop("SCAI_KERNEL_ALLOW_CORE_DUMP", None)
    else:
        env["SCAI_KERNEL_ALLOW_CORE_DUMP"] = allow_core_dump
    return env


def _run_launcher(workspace: Path, code: str, *, allow_core_dump: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(_launcher()),
            sys.executable,
            "-c",
            code,
        ],
        cwd=workspace,
        env=_launcher_env(workspace, allow_core_dump),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _core_limit_after_launch(tmp_path: Path, allow_core_dump: str | None = None) -> tuple[int, int]:
    workspace = tmp_path / "session"
    workspace.mkdir()
    completed = _run_launcher(
        workspace,
        "import resource; print(resource.getrlimit(resource.RLIMIT_CORE))",
        allow_core_dump=allow_core_dump,
    )
    assert completed.returncode == 0, completed.stderr
    return eval(completed.stdout.strip())  # noqa: S307 - launcher emits a tuple literal


def test_launched_kernel_inherits_zero_core_limit_by_default(tmp_path):
    assert _core_limit_after_launch(tmp_path) == (0, 0)


@pytest.mark.skipif(
    resource.getrlimit(resource.RLIMIT_CORE) == (0, 0),
    reason="this environment already forbids core dumps, so there is no contrast",
)
def test_developer_opt_in_preserves_inherited_core_limit(tmp_path):
    inherited = resource.getrlimit(resource.RLIMIT_CORE)
    assert _core_limit_after_launch(tmp_path, allow_core_dump="1") == inherited


def test_false_opt_in_value_still_suppresses_core(tmp_path):
    assert _core_limit_after_launch(tmp_path, allow_core_dump="0") == (0, 0)


def test_launcher_abort_leaves_no_workspace_core(tmp_path):
    workspace = tmp_path / "session"
    workspace.mkdir()
    completed = _run_launcher(workspace, "import os; os.abort()")
    assert completed.returncode == -int(signal.SIGABRT)
    assert list(workspace.glob("core*")) == []
