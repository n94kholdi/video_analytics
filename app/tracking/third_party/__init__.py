"""Isolated StableTrack implementation (paper-faithful, no official public code).

StableTrack: Stabilizing Multi-Object Tracking on Low-Frequency Detections
Shelukhan, Mamedov, Kvanchiani. arXiv:2511.20418. No public license or
repository was released with the paper; this package is an independent
reference implementation kept separate from application adapters.
"""

from app.tracking.third_party.stabletrack.tracker import StableTrack, StableTrackConfig, TrackState

__all__ = ["StableTrack", "StableTrackConfig", "TrackState"]
