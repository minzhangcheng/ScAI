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
