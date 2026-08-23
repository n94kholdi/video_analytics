"""Multi-object tracking interfaces, factory, and adapters."""

from app.tracking.base import BaseTracker, NormalizedTrack, PersonTracker, TrackingResult
from app.tracking.botsort_adapter import BoTSortAdapter
from app.tracking.bytetrack import ByteTrackAdapter, detections_to_supervision, foot_point
from app.tracking.conversion import detections_to_supervision as convert_detections
from app.tracking.deepocsort_adapter import DeepOCSortAdapter
from app.tracking.factory import available_trackers, create_tracker, public_tracker_catalog
from app.tracking.stabletrack_adapter import StableTrackAdapter
from app.tracking.ucmctrack_adapter import UCMCTrackAdapter

__all__ = [
    "BaseTracker",
    "BoTSortAdapter",
    "ByteTrackAdapter",
    "DeepOCSortAdapter",
    "NormalizedTrack",
    "PersonTracker",
    "StableTrackAdapter",
    "TrackingResult",
    "UCMCTrackAdapter",
    "available_trackers",
    "convert_detections",
    "create_tracker",
    "detections_to_supervision",
    "foot_point",
    "public_tracker_catalog",
]
