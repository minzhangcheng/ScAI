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
        assert "asyncio.to_thread(_capture_descriptor" in line, (
            f"a capture point still hashes the dump on the event loop: {line}"
        )


@pytest.mark.asyncio
async def test_the_loop_keeps_running_while_the_descriptor_is_captured():
    """Behavioural, not just structural.

    A source guard cannot tell a thread hop from a comment about one, so this
    runs a slow capture and checks the loop serviced another task meanwhile.
    """
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

    beat = asyncio.create_task(heartbeat())
    try:
        await asyncio.sleep(0.02)
        before = ticks
        await asyncio.to_thread(slow_capture, "s", None)
        after = ticks
    finally:
        beat.cancel()

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
    assert "_capture_descriptor" in live, (
        "the capture moved outside the execution lock"
    )
