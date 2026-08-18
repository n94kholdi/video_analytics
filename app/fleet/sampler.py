"""Time-based 0.5 FPS gate: process one frame every two seconds."""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic


class SampleInterval:
    """Return True at most once per interval, dropping missed slots instead of catching up."""

    def __init__(
        self,
        interval_seconds: float,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = interval_seconds
        self._clock = clock
        self._next: float | None = None

    def due(self) -> bool:
        now = self._clock()
        if self._next is None:
            self._next = now + self.interval_seconds
            return True
        if now < self._next:
            return False
        while self._next <= now:
            self._next += self.interval_seconds
        return True
