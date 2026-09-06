"""Contracts for launcher core limits and kernel death diagnostics."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import resource
from queue import Empty as QueueEmpty
import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock
import uuid

import pytest

import backend.utils.sandbox_parts.execution as execution_module
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


@pytest.mark.parametrize("signum", [signal.SIGKILL, signal.SIGSEGV, signal.SIGABRT])
def test_describe_kernel_exit_names_signal(signum):
    manager = SimpleNamespace(
        provisioner=SimpleNamespace(
            process=SimpleNamespace(poll=lambda: -int(signum))
        )
    )
    assert signum.name in execution_module.describe_kernel_exit(manager)


def test_core_signal_reports_rlimit_suppression():
    manager = SimpleNamespace(
        provisioner=SimpleNamespace(
            process=SimpleNamespace(poll=lambda: -int(signal.SIGSEGV))
        )
    )
    assert "RLIMIT_CORE" in execution_module.describe_kernel_exit(manager)


def test_ordinary_and_unavailable_exit_statuses_are_safe():
    assert execution_module.describe_kernel_exit(None).startswith(
        "exit status unavailable"
    )
    assert execution_module.describe_kernel_exit(SimpleNamespace()).startswith(
        "exit status unavailable"
    )
    still_running = SimpleNamespace(
        provisioner=SimpleNamespace(
            process=SimpleNamespace(poll=lambda: None)
        )
    )
    assert "no exit status yet" in execution_module.describe_kernel_exit(still_running)
    ordinary = SimpleNamespace(
        provisioner=SimpleNamespace(
            process=SimpleNamespace(poll=lambda: 3)
        )
    )
    assert execution_module.describe_kernel_exit(ordinary) == "exited with status 3"


def test_provisioner_async_poll_is_never_called():
    class _Provisioner:
        async def poll(self):  # pragma: no cover - must never be called
            raise AssertionError("the coroutine poll() was used")

        process = SimpleNamespace(poll=lambda: -int(signal.SIGSEGV))

    manager = SimpleNamespace(provisioner=_Provisioner())
    assert "SIGSEGV" in execution_module.describe_kernel_exit(manager)


@pytest.mark.asyncio
async def test_pre_execute_death_log_uses_pre_restart_reason(tmp_path, caplog):
    sandbox = _sandbox(tmp_path)
    manager = SimpleNamespace(
        provisioner=SimpleNamespace(
            process=SimpleNamespace(poll=lambda: -int(signal.SIGKILL))
        ),
        is_alive=lambda: False,
    )
    sandbox.km = manager
    sandbox.kc = SimpleNamespace()

    async def failed_restart():
        sandbox._kernel_dead = True

    sandbox._restart_kernel = AsyncMock(side_effect=failed_restart)
    caplog.set_level(logging.WARNING, logger=execution_module.logger.name)

    result = await sandbox._execute_unlocked(
        "pass", records_history=False, skip_safety_check=True
    )

    assert result.success is False
    assert "SIGKILL" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("restart_failed", "outcome_text"),
    [
        (True, "restart failed"),
        (False, "Restarted successfully"),
    ],
)
async def test_iopub_death_error_uses_pre_restart_reason(
    tmp_path, restart_failed, outcome_text
):
    sandbox = _sandbox(tmp_path)
    alive = True

    def is_alive() -> bool:
        return alive

    def get_iopub_msg(*, timeout):
        nonlocal alive
        alive = False
        raise QueueEmpty

    manager = SimpleNamespace(
        provisioner=SimpleNamespace(
            process=SimpleNamespace(poll=lambda: -int(signal.SIGSEGV))
        ),
        is_alive=is_alive,
    )
    client = SimpleNamespace(execute=lambda code: "execution-id", get_iopub_msg=get_iopub_msg)
    sandbox.km = manager
    sandbox.kc = client

    async def restart():
        sandbox._kernel_dead = restart_failed

    sandbox._restart_kernel = AsyncMock(side_effect=restart)

    result = await sandbox._execute_unlocked(
        "pass", records_history=False, skip_safety_check=True
    )

    assert result.success is False
    assert result.error is not None
    assert "SIGSEGV" in result.error
    assert outcome_text in result.error
