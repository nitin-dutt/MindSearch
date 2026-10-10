"""Tiny in-memory sliding-window rate limiter (alignment_plan SEC-09), used to slow password guessing.
Per-process state: with several API workers each has its own window (a shared store such as Redis is
the production answer)."""
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, max_attempts=5, window_seconds=60, clock=time.monotonic):
        self.max_attempts, self.window, self.clock = max_attempts, window_seconds, clock
        self._hits = defaultdict(deque)

    def allow(self, key) -> bool:
        """Record an attempt for `key`; False if it already used its allowance inside the window."""
        now = self.clock()
        hits = self._hits[key]
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        if len(hits) >= self.max_attempts:
            return False
        hits.append(now)
        return True

    def reset(self, key):
        self._hits.pop(key, None)
