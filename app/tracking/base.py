"""Tracker-independent interfaces, normalized outputs, and update metadata."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from app.core.models import Detection, TrackObservation


@dataclass(frozen=True, slots=True)
class NormalizedTrack:
    """Minimal tracker output shared by adapters, benchmarks, and the dashboard."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    confirmed: bool = True


@dataclass(frozen=True, slots=True)
class TrackingResult:
    """Person observations and lifecycle information for one source frame."""

    observations: tuple[TrackObservation, ...]
    expired_track_ids: tuple[int, ...]
    tracking_ms: float
    reid_ms: float = 0.0
    tracker_name: str = ""

    def normalized(self) -> tuple[NormalizedTrack, ...]:
        return tuple(
            NormalizedTrack(
                track_id=item.track_id,
                bbox=item.xyxy,
                confidence=item.detection_confidence,
                class_id=item.class_id,
                confirmed=item.confirmed,
            )
            for item in self.observations
        )


class BaseTracker(ABC):
    """Detector-independent person tracker with timestamped updates."""

    name: str = "tracker"

    @abstractmethod
    def update(
        self,
        detections: Sequence[Detection],
        *,
        camera_id: str,
        timestamp: float,
        frame_index: int,
        frame: np.ndarray | None = None,
        intermediate_frame: np.ndarray | None = None,
    ) -> TrackingResult:
        """Advance tracking state by one processed source frame.

        ``timestamp`` is elapsed source time in seconds, not a frame counter.
        ``intermediate_frame`` is optional visual-tracking context between the
        previous and current processed frames; adapters must tolerate ``None``.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all tracks before processing another source."""

    @property
    def reid_enabled(self) -> bool:
        return False

    @property
    def retained_track_count(self) -> int:
        return 0


class PersonTracker(Protocol):
    """Structural interface implemented by detector-independent person trackers."""

    def update(
        self,
        detections: Sequence[Detection],
        *,
        camera_id: str,
        timestamp: float,
        frame_index: int,
        frame: np.ndarray | None = None,
        intermediate_frame: np.ndarray | None = None,
    ) -> TrackingResult:
        """Advance tracking state by one source frame."""

    def reset(self) -> None:
        """Clear all tracks before processing another source."""
