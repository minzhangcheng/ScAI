"""L1: wall-clock deadline helper for the sandbox iopub loop. The loop must time
out on elapsed wall-clock time, not reset on every incoming message."""
from backend.utils.sandbox import compute_remaining_timeout


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