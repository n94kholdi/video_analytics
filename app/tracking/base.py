"""Tracker-independent interfaces and update metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from app.core.models import Detection, TrackObservation


@dataclass(frozen=True, slots=True)
class TrackingResult:
    """Person observations and lifecycle information for one source frame."""

    observations: tuple[TrackObservation, ...]
    expired_track_ids: tuple[int, ...]
    tracking_ms: float


class PersonTracker(Protocol):
    """Interface implemented by detector-independent person trackers."""

    def update(
        self,
        detections: Sequence[Detection],
        *,
        camera_id: str,
        timestamp: float,
        frame_index: int,
    ) -> TrackingResult:
        """Advance tracking state by one source frame."""

    def reset(self) -> None:
        """Clear all tracks before processing another source."""
