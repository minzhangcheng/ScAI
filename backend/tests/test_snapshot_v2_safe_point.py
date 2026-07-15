import asyncio

import pytest

from backend.services.snapshot_v2.safe_point import SafePointManager


@pytest.mark.asyncio
async def test_pending_reservation_blocks_direct_acquire():
    manager = SafePointManager()
    request = await manager.request_safe_point("session-1", "restore:first")
    assert request is not False

    acquired = await manager.acquire(
        "session-1", {"source": "capture"}, owner_token="capture:first"
    )

    assert acquired is False
    assert manager.get_pending_observation("session-1") == request
    assert await manager.cancel_safe_point_request("session-1", "restore:first")


@pytest.mark.asyncio
async def test_acknowledge_does_not_signal_when_conversion_fails(monkeypatch):
    manager = SafePointManager()
    request = await manager.request_safe_point("session-1", "restore:first")
    assert request is not False
    monkeypatch.setattr(manager, "_acquire_unlocked", lambda *args, **kwargs: False)

    acknowledged = await manager.acknowledge_safe_point_request(
        "session-1", request.generation, {"source": "workflow"}
    )

    assert acknowledged is False
    assert request.event.is_set() is False
    assert manager.get_pending_observation("session-1") == request


@pytest.mark.asyncio
async def test_waiter_reserves_only_after_current_owner_releases():
    manager = SafePointManager()
    first = await manager.request_safe_point("session-1", "restore:first")
    assert first is not False

    task = asyncio.create_task(
        manager.request_safe_point_when_available(
            "session-1", "restore:second", timeout_seconds=0.5
        )
    )
    await asyncio.sleep(0)
    assert not task.done()
    assert manager.get_pending_observation("session-1") == first

    assert await manager.cancel_safe_point_request("session-1", "restore:first")
    second = await asyncio.wait_for(task, timeout=0.5)
    assert second is not None
    assert second.owner_token == "restore:second"
    assert second.generation > first.generation
    assert manager.get_pending_observation("session-1") == second
    assert await manager.cancel_safe_point_request("session-1", "restore:second")


@pytest.mark.asyncio
async def test_waiter_timeout_preserves_current_owner():
    manager = SafePointManager()
    first = await manager.request_safe_point("session-1", "restore:first")
    assert first is not False

    second = await manager.request_safe_point_when_available(
        "session-1", "restore:second", timeout_seconds=0.01
    )
    assert second is None
    assert manager.get_pending_observation("session-1") == first
    assert await manager.cancel_safe_point_request("session-1", "restore:first")


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_consume_later_availability():
    manager = SafePointManager()
    first = await manager.request_safe_point("session-1", "restore:first")
    assert first is not False
    task = asyncio.create_task(
        manager.request_safe_point_when_available(
            "session-1", "restore:cancelled", timeout_seconds=1.0
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await manager.cancel_safe_point_request("session-1", "restore:first")
    replacement = await manager.request_safe_point_when_available(
        "session-1", "restore:replacement", timeout_seconds=0.1
    )
    assert replacement is not None
    assert replacement.owner_token == "restore:replacement"
    assert await manager.cancel_safe_point_request(
        "session-1", "restore:replacement"
    )
