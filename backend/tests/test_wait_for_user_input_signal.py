"""Manual-mode _wait_for_user_input must not lose signals received during supplementary handling."""
import asyncio
import pytest


@pytest.mark.asyncio
async def test_signal_during_supplementary_not_lost():
    from backend.workflow.synthesis import _consume_signal_then_clear

    event = asyncio.Event()
    event.set()
    await event.wait()
    _consume_signal_then_clear(event)
    assert not event.is_set()

    event.set()

    try:
        await asyncio.wait_for(event.wait(), timeout=0.2)
    except asyncio.TimeoutError:
        pytest.fail("Signal received during handling was lost — wait() blocked")