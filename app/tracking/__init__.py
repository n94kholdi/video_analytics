"""Multi-object tracking package boundary; implementation begins after Phase 1."""
"""Multi-object tracking interfaces and ByteTrack implementation."""

from app.tracking.base import PersonTracker, TrackingResult
from app.tracking.bytetrack import ByteTrackAdapter, detections_to_supervision, foot_point

__all__ = [
    "ByteTrackAdapter",
    "PersonTracker",
    "TrackingResult",
    "detections_to_supervision",
    "foot_point",
]
