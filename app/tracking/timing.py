"""Shared wall-clock timing helpers for tracker adapters and benchmarks."""

from __future__ import annotations

from time import perf_counter
from typing import Iterator
from contextlib import contextmanager


class Stopwatch:
    """Accumulate elapsed milliseconds for named pipeline stages."""

    def __init__(self) -> None:
        self._started = perf_counter()
        self.stages_ms: dict[str, float] = {}

    def elapsed_ms(self) -> float:
        return (perf_counter() - self._started) * 1000.0

    def record(self, name: str, milliseconds: float) -> None:
        self.stages_ms[name] = self.stages_ms.get(name, 0.0) + milliseconds

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.record(name, (perf_counter() - started) * 1000.0)
