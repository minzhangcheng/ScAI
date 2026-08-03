"""L1: wall-clock deadline helper for the sandbox iopub loop. The loop must time
out on elapsed wall-clock time, not reset on every incoming message."""
import pytest

from backend.utils.sandbox import compute_remaining_timeout
from backend.utils.sandbox_parts import execution as execution_module


def test_remaining_positive_before_deadline():
    # deadline = 100, now = 40 → 60s left
    assert compute_remaining_timeout(100.0, 40.0) == 60.0


def test_remaining_zero_at_deadline():
    assert compute_remaining_timeout(100.0, 100.0) == 0.0


def test_remaining_negative_after_deadline():
    assert compute_remaining_timeout(100.0, 130.0) == -30.0


def test_expired_is_non_positive():
    # the loop treats remaining <= 0 as timed out
    assert compute_remaining_timeout(100.0, 100.0) <= 0
    assert compute_remaining_timeout(100.0, 101.0) <= 0


@pytest.mark.asyncio
async def test_interrupt_drain_has_overall_deadline_under_continuous_output(
    monkeypatch,
):
    calls = 0

    class ContinuousClient:
        def get_iopub_msg(self, *, timeout):
            nonlocal calls
            calls += 1
            return {"msg_type": "stream", "content": {"text": "still busy"}}

    async def direct_call(func, *args, **kwargs):
        return func(*args, **kwargs)

    ticks = iter((0.0, 0.1, 0.4, 0.8, 1.1))
    monkeypatch.setattr(execution_module, "_to_thread_drained", direct_call)

    returned_idle = await execution_module._drain_iopub_until_idle(
        ContinuousClient(),
        timeout_seconds=1.0,
        monotonic=lambda: next(ticks),
    )

    assert returned_idle is False
    assert calls == 3
