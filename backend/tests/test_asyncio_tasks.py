"""Performance and cancellation contracts for owned blocking work."""

from __future__ import annotations

import asyncio
import threading

import pytest


@pytest.mark.asyncio
async def test_completed_owned_future_uses_immediate_fast_path(monkeypatch):
    from backend.utils import asyncio_tasks

    future = asyncio.get_running_loop().create_future()
    future.set_result("ready")

    async def forbidden_wait(*_args, **_kwargs):
        raise AssertionError("a completed future must not enter the drain wait")

    monkeypatch.setattr(asyncio_tasks.asyncio, "wait", forbidden_wait)

    assert await asyncio_tasks.await_with_deferred_cancellation(future) == "ready"


@pytest.mark.asyncio
async def test_repeated_chunk_sized_calls_reuse_pool_without_fixed_sleep(
    monkeypatch,
):
    """A large upload loop must not create a pool or sleep for every MiB."""
    from backend.utils import asyncio_tasks

    executor = asyncio_tasks._get_owned_thread_executor()
    worker_ids: set[int] = set()

    async def forbidden_sleep(*_args, **_kwargs):
        raise AssertionError("owned-thread completion must not poll with sleep")

    monkeypatch.setattr(asyncio_tasks.asyncio, "sleep", forbidden_sleep)

    one_mib = 1024 * 1024
    for _ in range(64):
        result = await asyncio_tasks.run_in_owned_thread(
            lambda size: (worker_ids.add(threading.get_ident()), size)[1],
            one_mib,
        )
        assert result == one_mib
        assert asyncio_tasks._get_owned_thread_executor() is executor

    assert 1 <= len(worker_ids) <= asyncio_tasks._OWNED_THREAD_MAX_WORKERS


@pytest.mark.asyncio
async def test_owned_thread_still_defers_cancellation_until_worker_finishes():
    from backend.utils.asyncio_tasks import run_in_owned_thread

    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_work() -> None:
        entered.set()
        assert release.wait(timeout=2)
        finished.set()

    owner = asyncio.create_task(run_in_owned_thread(blocking_work))
    try:
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()

        owner.cancel()
        await asyncio.sleep(0)
        assert not owner.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(owner, timeout=2)
        assert finished.is_set()
    finally:
        release.set()
        if not owner.done():
            owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)


@pytest.mark.asyncio
async def test_bulk_workers_cannot_starve_latency_sensitive_owned_work():
    from backend.utils import asyncio_tasks

    release = threading.Event()
    bulk_executor = asyncio_tasks._get_owned_bulk_thread_executor()
    entered = [
        threading.Event()
        for _ in range(asyncio_tasks._OWNED_BULK_THREAD_MAX_WORKERS)
    ]

    def bulk_work(index: int) -> None:
        entered[index].set()
        assert release.wait(timeout=2)

    bulk_tasks = [
        asyncio.create_task(
            asyncio_tasks.run_in_owned_bulk_thread(bulk_work, index)
        )
        for index in range(asyncio_tasks._OWNED_BULK_THREAD_MAX_WORKERS)
    ]
    try:
        for _ in range(200):
            if all(event.is_set() for event in entered):
                break
            await asyncio.sleep(0.005)
        assert all(event.is_set() for event in entered)
        result = await asyncio.wait_for(
            asyncio_tasks.run_in_owned_thread(lambda: "responsive"),
            timeout=0.5,
        )
        assert result == "responsive"
        assert (
            asyncio_tasks._get_owned_bulk_thread_executor()
            is not asyncio_tasks._get_owned_thread_executor()
        )
    finally:
        release.set()
        await asyncio.gather(*bulk_tasks, return_exceptions=True)
        bulk_executor.shutdown(wait=True)
        asyncio_tasks._owned_bulk_thread_executor = None
        asyncio_tasks._owned_bulk_thread_executor_pid = 0
