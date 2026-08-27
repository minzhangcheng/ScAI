"""Kernel must recover from ZMQ signature corruption during normal execution,
not just from timeout-path drain failures."""
import asyncio
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock

from backend.utils.sandbox import LocalJupyterSandbox, _is_kernel_protocol_error


def test_protocol_error_classifier():
    assert _is_kernel_protocol_error(ValueError("Invalid Signature: b'abc'"))
    assert _is_kernel_protocol_error(ValueError("b'<IDS|MSG>' is not in list"))
    assert not _is_kernel_protocol_error(ValueError("some unrelated error"))
    assert not _is_kernel_protocol_error(KeyError("x"))


@pytest.mark.asyncio
async def test_execute_restarts_kernel_on_signature_error(monkeypatch, tmp_path):
    sb = object.__new__(LocalJupyterSandbox)        # bypass heavy __init__
    sb.session_id = "s"
    sb._exec_lock = asyncio.Lock()
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


def _bare_sandbox(tmp_path):
    """A real LocalJupyterSandbox with only its kernel collaborators faked.

    __init__ is cheap -- it starts no kernel, it only reads the lifecycle
    record and creates its directories -- so it is called for real here.
    Bypassing it with object.__new__ and hand-wiring attributes is what made
    these tests drift: every attribute __init__ later grew was missing from the
    fixture, and _restart_kernel's failure path calls the real stop(), which
    then raised into the `except Exception` that path deliberately swallows, so
    the teardown these tests assert never ran.
    """
    sb = LocalJupyterSandbox(
        session_id=f"kernel-recovery-{uuid.uuid4().hex[:8]}",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    sb._kernel_dead = False
    sb._initialized = True
    sb._restarting = False
    # _restart_kernel_manager / _shutdown_kernel_manager prefer a native
    # coroutine when `getattr(manager, "_async_restart_kernel", None)` is
    # callable. A bare MagicMock auto-creates that attribute, so it claims to
    # have the fast path and then returns a non-awaitable -- every restart in
    # this file died on `await native()` before reaching the behaviour under
    # test. Wire the two natives explicitly.
    km = MagicMock()
    km._async_restart_kernel = AsyncMock()
    km._async_shutdown_kernel = AsyncMock()
    sb.km = km
    sb.kc = MagicMock()
    sb._wait_for_ready = AsyncMock()
    sb._initialize_kernel = AsyncMock()
    sb._restore_session = AsyncMock()
    return sb


@pytest.mark.asyncio
async def test_successful_restart_clears_kernel_dead(tmp_path):
    sb = _bare_sandbox(tmp_path)
    sb._kernel_dead = True  # was bricked by a prior failed restart
    await sb._restart_kernel()
    assert sb._kernel_dead is False  # healthy kernel is usable again


@pytest.mark.asyncio
async def test_restore_failure_discards_partial_namespace(tmp_path):
    sb = _bare_sandbox(tmp_path)
    sb._restore_session = AsyncMock(side_effect=RuntimeError("corrupt dill dump"))
    await sb._restart_kernel()
    assert sb._kernel_dead is True
    assert sb._initialized is False
    assert sb.km is None
    assert sb.kc is None


@pytest.mark.asyncio
async def test_kernel_restart_failure_sets_dead(tmp_path):
    sb = _bare_sandbox(tmp_path)
    sb._wait_for_ready = AsyncMock(side_effect=RuntimeError("kernel never ready"))
    await sb._restart_kernel()
    assert sb._kernel_dead is True
    assert sb._initialized is False
    assert sb.km is None
    assert sb.kc is None


@pytest.mark.asyncio
async def test_restart_reentrancy_guard(tmp_path):
    sb = _bare_sandbox(tmp_path)
    sb._restarting = True  # a restart is already in progress
    await sb._restart_kernel()
    sb.km.restart_kernel.assert_not_called()  # nested restart is skipped
