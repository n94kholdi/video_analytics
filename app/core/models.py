"""Shared data representations used across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class Detection:
    """One image-space detection with an ``xyxy`` bounding box."""

    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    class_name: str | None = "person"

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.xyxy
        values = (*self.xyxy, self.confidence)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("detection coordinates and confidence must be finite")
        if x2 < x1 or y2 < y1:
            raise ValueError("detection xyxy coordinates must be ordered")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection confidence must be between 0 and 1")

