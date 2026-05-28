"""In-process sliding-window rate limiter (no external dependency).

Used to throttle login brute-force. State is per-process, consistent with the
app's in-memory session model.
"""
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Callable, Deque, Dict


class SlidingWindowRateLimiter:
    def __init__(
        self,
        max_attempts: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._clock = clock
        self._events: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def _prune(self, key: str, now: float) -> Deque[float]:
        dq = self._events[key]
        while dq and now - dq[0] > self.window_seconds:
            dq.popleft()
        return dq

    def is_allowed(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            dq = self._prune(key, now)
            return len(dq) < self.max_attempts

    def record_failure(self, key: str) -> None:
        now = self._clock()
        with self._lock:
            self._prune(key, now).append(now)

    def reset(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)
