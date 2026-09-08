"""The generation check must not stall the event loop.

`_capture_descriptor` reads the whole dump twice -- the HMAC and the SHA-256 the
manifest must match. Measured at 0.92 s for a 973 MiB dump from the A2-10 runs,
and it scales with the session, so on the loop it stalls every other HTTP
request and WebSocket message in the process.
"""
from __future__ import annotations

import asyncio

import pytest


def test_both_capture_points_hand_the_hashing_to_a_thread():
    from backend.tests.split_facade import source_for_path

    source = source_for_path("backend/services/resumable_export.py")
    frozen = source.split("async def frozen_resumable_export(", 1)[1]
    frozen = frozen.split("\ndef ", 1)[0]

    yields = [line.strip() for line in frozen.splitlines() if "yield " in line]
    assert yields, "the freeze no longer yields a descriptor"
    for line in yields:
        assert "_captured_off_the_loop(" in line, (
            f"a capture point still hashes the dump on the event loop: {line}"
        )

    handoff = source.split("async def _captured_off_the_loop(", 1)[1]
    handoff = handoff.split("\n@", 1)[0]
    assert "asyncio.to_thread(_capture_descriptor" in handoff, (
        "the hand-off no longer reaches a worker thread"
    )
    assert "await_with_deferred_cancellation" in handoff, (
        "a cancelled capture would abandon its thread and unwind the freeze "
        "while two full passes over the dump are still running"
    )


@pytest.mark.asyncio
async def test_the_loop_keeps_running_while_the_descriptor_is_captured(monkeypatch):
    """Behavioural, and through the production call site.

    A source guard cannot tell a thread hop from a comment about one. Driving
    `asyncio.to_thread` directly cannot either -- that only proves the standard
    library works, and stays green with the offload removed. So this enters the
    real freeze on its cold path with a slow stand-in bound where the capture
    is looked up, and checks the loop serviced another task meanwhile.
    """
    from backend.services import resumable_export as export

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    def slow_capture(session_id, state):
        # Stands in for two passes over a multi-gigabyte dump.
        import time

        time.sleep(0.3)
        return {"pairing": None, "manifest": {}, "notebooks": {}}

    monkeypatch.setattr(export, "_capture_descriptor", slow_capture)

    beat = asyncio.create_task(heartbeat())
    try:
        await asyncio.sleep(0.02)
        before = ticks
        async with export.frozen_resumable_export(
            "offload-probe-session", None
        ) as descriptor:
            after = ticks
    finally:
        beat.cancel()

    assert descriptor == {"pairing": None, "manifest": {}, "notebooks": {}}

    assert after - before >= 5, (
        f"the loop only advanced {after - before} ticks during a 0.3 s capture; "
        "the hashing is running on the event loop"
    )


def test_the_freeze_still_spans_the_capture():
    """Offloading must not narrow the exclusion window.

    The lifecycle claim and, on the live path, the execution lock have to be
    held across the capture, or a save can commit a new generation between the
    flush and the check that the archive matches it.
    """
    from backend.tests.split_facade import source_for_path

    source = source_for_path("backend/services/resumable_export.py")
    frozen = source.split("async def frozen_resumable_export(", 1)[1]
    frozen = frozen.split("\ndef ", 1)[0]

    live = frozen.split("async with sandbox._exec_lock:", 1)[1]
    live = live.split("finally:", 1)[0]
    assert "_flush_locked" in live
    assert "_captured_off_the_loop" in live, (
        "the capture moved outside the execution lock"
    )


@pytest.mark.asyncio
async def test_a_cancelled_capture_is_drained_not_abandoned(monkeypatch):
    """Cancelling `asyncio.to_thread` abandons the thread; it does not stop it.

    Unwinding there would release the lifecycle claim and the execution lock
    while two full passes over a multi-gigabyte dump are still running, so a
    save could commit a new generation under the reader and a session deletion
    could finish while a thread still holds the file open.
    """
    import threading
    import time

    from backend.services import resumable_export as export

    started = threading.Event()
    finished = []

    def slow_capture(session_id, state):
        started.set()
        time.sleep(0.3)
        finished.append(session_id)
        return {"pairing": None, "manifest": {}, "notebooks": {}}

    monkeypatch.setattr(export, "_capture_descriptor", slow_capture)

    async def freeze():
        async with export.frozen_resumable_export("cancel-probe-session", None):
            pass

    task = asyncio.create_task(freeze())
    await asyncio.to_thread(started.wait, 5.0)
    assert started.is_set(), "the capture never reached its worker"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished, (
        "the freeze unwound while its worker was still reading the dump"
    )
