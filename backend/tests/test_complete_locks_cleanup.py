"""Test that _complete_locks entries are cleaned up after use."""

import asyncio
import pytest


@pytest.mark.asyncio
async def test_complete_lock_serializes_and_cleans_up():
    from backend.services import upload as up

    up._complete_locks.clear()
    up._complete_lock_refs.clear()

    order = []

    async def worker(tag):
        lock, _ = await up._acquire_complete_lock("uid")
        async with lock:
            order.append(f"{tag}-in")
            await asyncio.sleep(0.01)
            order.append(f"{tag}-out")
        up._release_complete_lock("uid")

    await asyncio.gather(worker("A"), worker("B"), worker("C"))

    for i in range(0, 6, 2):
        assert order[i].endswith("-in") and order[i + 1].endswith("-out")
        assert order[i][0] == order[i + 1][0]

    assert "uid" not in up._complete_locks
    assert "uid" not in up._complete_lock_refs
