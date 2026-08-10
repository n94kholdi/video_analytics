"""Shared data representations used across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


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


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    """One timestamped foot-point sample retained for a person track."""

    timestamp: float
    frame_index: int
    position: tuple[float, float]
    smoothed_position: tuple[float, float]


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """Shared, tracker-independent representation of a person in one frame."""

    camera_id: str
    track_id: int
    timestamp: float
    frame_index: int
    xyxy: tuple[float, float, float, float]
    foot_point: tuple[float, float]
    detection_confidence: float
    confirmed: bool
    trajectory: tuple[TrajectoryPoint, ...]
    ground_position: tuple[float, float] | None = None
    smoothed_velocity: tuple[float, float] | None = None
    speed_pixels_per_second: float | None = None
    speed_metres_per_second: float | None = None
    speed_unavailable_reason: str | None = None
    ground_smoothed_velocity: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class Event:
    """Shared analytics event representation for downstream storage and APIs."""

    event_id: str
    event_type: str
    camera_id: str
    timestamp: float
    track_id: int | None = None
    zone_id: str | None = None
    line_id: str | None = None
    payload: Mapping[str, Any] | None = None
