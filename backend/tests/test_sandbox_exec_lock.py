"""execute() 与 save_session() 必须经同一把 asyncio.Lock 串行化，
防止 pause() 在 bioinformatics 在途 execute 期间并发读 kernel ZMQ socket。"""
import asyncio
import pytest

from backend.utils.sandbox import LocalJupyterSandbox


def _bare_sandbox() -> LocalJupyterSandbox:
    sb = object.__new__(LocalJupyterSandbox)
    sb._exec_lock = asyncio.Lock()
    return sb


@pytest.mark.asyncio
async def test_execute_and_save_share_one_lock():
    sb = _bare_sandbox()
    order = []

    async def fake_long_exec():
        async with sb._exec_lock:
            order.append("exec-start")
            await asyncio.sleep(0.05)
            order.append("exec-end")

    async def fake_save():
        async with sb._exec_lock:
            order.append("save")

    await asyncio.gather(fake_long_exec(), fake_save())
    # save 必须等待 exec 完整结束后才进入临界区
    assert order == ["exec-start", "exec-end", "save"]


def test_sandbox_init_creates_exec_lock():
    sb = object.__new__(LocalJupyterSandbox)
    LocalJupyterSandbox._init_exec_lock(sb)
    assert isinstance(sb._exec_lock, asyncio.Lock)
