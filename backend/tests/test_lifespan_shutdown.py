import asyncio
import pytest
from backend.main import _cancel_and_await


@pytest.mark.asyncio
async def test_cancel_and_await_finishes_tasks():
    started = asyncio.Event()
    async def runner():
        started.set()
        await asyncio.sleep(3600)
    t = asyncio.create_task(runner())
    await started.wait()
    await _cancel_and_await(t)
    assert t.done()
