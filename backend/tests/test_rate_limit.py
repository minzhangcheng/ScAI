"""S2: sliding-window limiter for login brute-force protection."""
from backend.utils.rate_limit import SlidingWindowRateLimiter


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_allows_up_to_max_then_blocks():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(max_attempts=3, window_seconds=60, clock=clock)
    for _ in range(3):
        assert limiter.is_allowed("k") is True
        limiter.record_failure("k")
    assert limiter.is_allowed("k") is False


def test_window_slides_and_unblocks():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(max_attempts=2, window_seconds=60, clock=clock)
    limiter.record_failure("k")
    limiter.record_failure("k")
    assert limiter.is_allowed("k") is False
    clock.t = 61.0  # past the window
    assert limiter.is_allowed("k") is True


def test_reset_clears_failures():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(max_attempts=2, window_seconds=60, clock=clock)
    limiter.record_failure("k")
    limiter.record_failure("k")
    assert limiter.is_allowed("k") is False
    limiter.reset("k")
    assert limiter.is_allowed("k") is True


def test_keys_are_independent():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(max_attempts=1, window_seconds=60, clock=clock)
    limiter.record_failure("a")
    assert limiter.is_allowed("a") is False
    assert limiter.is_allowed("b") is True
