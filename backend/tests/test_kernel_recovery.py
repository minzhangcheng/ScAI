"""Kernel must recover from ZMQ signature corruption during normal execution,
not just from timeout-path drain failures."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.utils.sandbox import LocalJupyterSandbox, _is_kernel_protocol_error


def test_protocol_error_classifier():
    assert _is_kernel_protocol_error(ValueError("Invalid Signature: b'abc'"))
    assert _is_kernel_protocol_error(ValueError("b'<IDS|MSG>' is not in list"))
    assert not _is_kernel_protocol_error(ValueError("some unrelated error"))
    assert not _is_kernel_protocol_error(KeyError("x"))


@pytest.mark.asyncio
async def test_execute_restarts_kernel_on_signature_error(monkeypatch):
    sb = object.__new__(LocalJupyterSandbox)        # bypass heavy __init__
    sb.session_id = "s"
    sb._kernel_dead = False
    sb.km = MagicMock()
    sb.kc = MagicMock()
    sb.kc.execute = MagicMock(side_effect=ValueError("Invalid Signature: b'zz'"))

    restarted = {"n": 0}

    async def fake_restart():
        restarted["n"] += 1

    sb._restart_kernel = fake_restart

    result = await sb.execute("print(1)", skip_safety_check=True)
    assert result.success is False
    assert restarted["n"] == 1  # signature error triggered a restart attempt


def _bare_sandbox():
    """A LocalJupyterSandbox with __init__ bypassed and just enough attributes
    wired for _restart_kernel."""
    sb = object.__new__(LocalJupyterSandbox)
    sb.session_id = "s"
    sb._kernel_dead = False
    sb._initialized = True
    sb._restarting = False
    sb.km = MagicMock()
    sb.kc = MagicMock()
    sb._wait_for_ready = AsyncMock()
    sb._initialize_kernel = AsyncMock()
    sb._restore_session = AsyncMock()
    return sb


@pytest.mark.asyncio
async def test_successful_restart_clears_kernel_dead():
    sb = _bare_sandbox()
    sb._kernel_dead = True  # was bricked by a prior failed restart
    await sb._restart_kernel()
    assert sb._kernel_dead is False  # healthy kernel is usable again


@pytest.mark.asyncio
async def test_restore_failure_keeps_kernel_alive():
    sb = _bare_sandbox()
    sb._restore_session = AsyncMock(side_effect=RuntimeError("corrupt dill dump"))
    await sb._restart_kernel()
    # Kernel itself restarted fine; only variable restore failed -> not dead.
    assert sb._kernel_dead is False


@pytest.mark.asyncio
async def test_kernel_restart_failure_sets_dead():
    sb = _bare_sandbox()
    sb._wait_for_ready = AsyncMock(side_effect=RuntimeError("kernel never ready"))
    await sb._restart_kernel()
    assert sb._kernel_dead is True  # genuine kernel-level failure


@pytest.mark.asyncio
async def test_restart_reentrancy_guard():
    sb = _bare_sandbox()
    sb._restarting = True  # a restart is already in progress
    await sb._restart_kernel()
    sb.km.restart_kernel.assert_not_called()  # nested restart is skipped