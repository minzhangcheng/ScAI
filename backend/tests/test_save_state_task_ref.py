"""save_state 必须追踪 loop.create_task 返回的 Task 引用并等待完成"""
import asyncio
import types
import pytest

from backend.utils.sandbox import LocalJupyterSandbox


class _FakeNotebookState:
    def save(self):
        pass


@pytest.mark.asyncio
async def test_save_state_tracks_task():
    sb = object.__new__(LocalJupyterSandbox)
    sb._pending_save_tasks = set()
    sb._exec_lock = asyncio.Lock()
    sb.notebook_state = _FakeNotebookState()

    async def fake_save(self):
        await asyncio.sleep(0.01)

    sb.save_session = types.MethodType(fake_save, sb)

    sb.save_state()
    # Before fix: _pending_save_tasks is empty → assertion fails
    assert sb._pending_save_tasks, "save_state 应当向 _pending_save_tasks 添加 task"
    assert len(sb._pending_save_tasks) == 1

    await sb.await_pending_saves()
    assert not sb._pending_save_tasks, "await_pending_saves 后 tasks 应已清空"


@pytest.mark.asyncio
async def test_same_revision_sync_async_requests_share_task_and_later_revision_resaves():
    sb = object.__new__(LocalJupyterSandbox)
    sb._pending_save_tasks = set()
    sb._pending_save_task = None
    sb._exec_lock = asyncio.Lock()
    sb.runtime_revision = 1
    sb.notebook_state = _FakeNotebookState()

    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def fake_save(self):
        calls.append(self.runtime_revision)
        started.set()
        await release.wait()
        return True

    sb.save_session = types.MethodType(fake_save, sb)
    sb.save_state()
    await started.wait()

    waiter = asyncio.create_task(sb.save_state_async())
    sb.save_state()
    await asyncio.sleep(0)
    assert calls == [1]
    assert len(sb._pending_save_tasks) == 1
    assert sb._pending_save_task is not None

    release.set()
    await waiter
    await sb.await_pending_saves()
    assert calls == [1]
    assert sb._pending_save_task is None

    sb.runtime_revision = 2
    release = asyncio.Event()
    sb.save_state()
    await asyncio.sleep(0)
    assert calls == [1, 2]
    release.set()
    await sb.await_pending_saves()
    assert calls == [1, 2]
    assert sb._pending_save_task is None


@pytest.mark.asyncio
async def test_cancelled_save_state_waiter_does_not_cancel_shared_generation():
    sb = object.__new__(LocalJupyterSandbox)
    sb._pending_save_tasks = set()
    sb._pending_save_task = None
    sb._exec_lock = asyncio.Lock()
    sb.runtime_revision = 1
    sb.notebook_state = _FakeNotebookState()

    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def fake_save(self):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return True

    sb.save_session = types.MethodType(fake_save, sb)
    sb.save_state()
    await started.wait()
    shared = next(iter(sb._pending_save_tasks))

    waiter = asyncio.create_task(sb.save_state_async())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert not shared.done()
    assert calls == 1
    release.set()
    await sb.await_pending_saves()
    assert calls == 1
    assert sb._pending_save_task is None
