"""BoT-SORT 8-D Kalman filter (xywh + velocities).

BoT-SORT replaced ByteTrack's aspect-ratio state with
``[x, y, w, h, vx, vy, vw, vh]``. Deep OC-SORT reused that filter. This module
re-exports the timestamp-aware implementation already in the repo so 0.5 FPS
gaps do not explode velocity.
"""

from app.tracking.third_party.deepocsort.kalman import (
    DeepOCSortKalman as BoTSortKalman,
    xywh_to_xyxy,
    xyxy_to_xywh,
)

__all__ = ["BoTSortKalman", "xywh_to_xyxy", "xyxy_to_xywh"]
