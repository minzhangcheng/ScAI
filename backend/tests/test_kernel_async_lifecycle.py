"""Native asynchronous kernel lifecycle contracts."""

from __future__ import annotations

import asyncio
import logging
import os
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock

from backend.tests.test_kernel_bootstrap import _FakeKernelManager, _sandbox


class _AsyncKernelManager(_FakeKernelManager):
    """Fake manager whose native lifecycle calls can be held or failed."""

    def __init__(self, *, kernel_name):
        super().__init__(kernel_name=kernel_name)
        self.provisioner = SimpleNamespace(process=object())
        self.start_entered = asyncio.Event()
        self.start_exited = asyncio.Event()
        self.start_release = asyncio.Event()
        self.restart_entered = asyncio.Event()
        self.restart_release = asyncio.Event()
        self.shutdown_entered = asyncio.Event()
        self.shutdown_release = asyncio.Event()
        self.start_mode = "wait"
        self.restart_mode = "wait"
        self.shutdown_mode = "success"
        self.shutdown_calls = 0
        self.shutdown_saw_start_exited: bool | None = None

    async def _async_start_kernel(self, *, cwd, env):
        self.cwd = cwd
        self.env = env
        self.start_entered.set()
        try:
            if self.start_mode == "wait":
                await self.start_release.wait()
            elif self.start_mode == "error":
                raise RuntimeError("native start failed")
        finally:
            self.start_exited.set()

    async def _async_restart_kernel(self, **_kwargs):
        self.restart_entered.set()
        if self.restart_mode == "wait":
            await self.restart_release.wait()
        elif self.restart_mode == "error":
            raise RuntimeError("native restart failed")

    async def _async_shutdown_kernel(self, *, now):
        assert now is True
        self.shutdown_calls += 1
        self.shutdown_saw_start_exited = self.start_exited.is_set()
        self.shutdown_entered.set()
        if self.shutdown_mode == "wait":
            await self.shutdown_release.wait()
        elif self.shutdown_mode == "error":
            raise RuntimeError("native shutdown failed")
        self.shutdown_called = True


@pytest.mark.asyncio
async def test_start_cancellation_drains_and_cleans_partial_manager(
    tmp_path, monkeypatch
):
    import backend.utils.sandbox_parts.kernel as kernel_module

    sandbox = _sandbox(tmp_path)
    manager = _AsyncKernelManager(kernel_name="python3")
    monkeypatch.setattr(kernel_module, "KernelManager", lambda **_kwargs: manager)
    sandbox._wait_for_ready = AsyncMock(return_value=None)
    sandbox._initialize_kernel = AsyncMock(return_value=None)
    sandbox._restore_session = AsyncMock(return_value=None)

    start_task = asyncio.create_task(sandbox.start())
    await asyncio.wait_for(manager.start_entered.wait(), timeout=1)
    start_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start_task

    assert start_task.done() is True
    assert sandbox.km is None
    assert sandbox.kc is None
    assert sandbox._kernel_dead is True
    assert manager.shutdown_called is True
    assert manager.start_exited.is_set()
    assert manager.shutdown_saw_start_exited is True


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["timeout", "cancellation"])
async def test_native_shutdown_interruption_propagates_and_retries_same_identity(
    tmp_path, monkeypatch, caplog, interruption,
):
    import backend.utils.sandbox_parts.kernel as kernel_module

    sandbox = _sandbox(tmp_path)
    manager = _AsyncKernelManager(kernel_name="python3")
    sandbox.km = manager
    sandbox.kc = manager.client()
    sandbox._initialized = True
    manager.shutdown_mode = "wait"
    caplog.set_level(logging.INFO, logger=kernel_module.logger.name)

    process = manager.provisioner.process
    if interruption == "timeout":
        monkeypatch.setattr(kernel_module, "_KERNEL_MANAGER_SHUTDOWN_TIMEOUT", 0.01)
        with pytest.raises(asyncio.TimeoutError):
            await sandbox.stop(save=False)
    else:
        stop_task = asyncio.create_task(sandbox.stop(save=False))
        await asyncio.wait_for(manager.shutdown_entered.wait(), timeout=1)
        stop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stop_task
        assert stop_task.done() is True

    assert sandbox._kernel_dead is True
    assert sandbox.km is manager
    assert sandbox.km.provisioner.process is process
    assert sandbox.kc is None
    assert sandbox._initialized is False
    assert "Kernel stopped" not in caplog.text

    manager.shutdown_mode = "success"
    if interruption == "timeout":
        manager.shutdown_entered.clear()
    await sandbox.stop(save=False)
    assert manager.shutdown_calls == 2
    assert sandbox.km is None
    assert "Kernel stopped" in caplog.text


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
    reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
)
async def test_real_kernel_native_lifecycle_and_zero_core_limit(tmp_path):
    sandbox = _sandbox(tmp_path)
    heartbeat = 0

    async def beat():
        nonlocal heartbeat
        while True:
            heartbeat += 1
            await asyncio.sleep(0)

    heartbeat_task = asyncio.create_task(beat())
    manager = None
    process = None
    shutdown_release: asyncio.Event | None = None
    stop_task: asyncio.Task | None = None
    try:
        await sandbox.start()
        manager = sandbox.km
        process = manager.provisioner.process
        probe = await sandbox.execute(
            "import resource\nprint(resource.getrlimit(resource.RLIMIT_CORE))",
            records_history=False,
            skip_safety_check=True,
        )
        assert probe.success, probe.error
        assert "(0, 0)" in probe.output
        assert heartbeat > 0
        native_shutdown = manager._async_shutdown_kernel
        shutdown_entered = asyncio.Event()
        shutdown_release = asyncio.Event()

        async def held_native_shutdown(**kwargs):
            shutdown_entered.set()
            await shutdown_release.wait()
            return await native_shutdown(**kwargs)

        manager._async_shutdown_kernel = held_native_shutdown
        stop_task = asyncio.create_task(sandbox.stop(save=False))
        await asyncio.wait_for(shutdown_entered.wait(), timeout=1)
        before_stop_heartbeat = heartbeat
        await asyncio.wait_for(asyncio.sleep(0), timeout=1)
        assert heartbeat > before_stop_heartbeat
        shutdown_release.set()
        await asyncio.wait_for(stop_task, timeout=10)
    finally:
        if shutdown_release is not None:
            shutdown_release.set()
        if stop_task is not None and not stop_task.done():
            try:
                await asyncio.wait_for(stop_task, timeout=10)
            except BaseException:
                pass
        try:
            if sandbox.km is not None:
                await sandbox.stop(save=False)
        finally:
            heartbeat_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await heartbeat_task

    assert manager is not None
    assert process is not None
    assert process.poll() is not None
    assert sandbox.km is None
    assert sandbox.kc is None
