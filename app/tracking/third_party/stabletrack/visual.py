"""Optional CamShift visual tracking used as an ASMS-like stand-in.

The paper uses ASMS (Vojir et al., 2014). OpenCV does not ship ASMS; CamShift
is the closest built-in scale-adaptive mean-shift tracker. When intermediate
frames are unavailable (0.5 FPS processed-only streams), callers skip this
module and rely on timestamped Kalman prediction + BBD.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def camshift_predict(
    source_frame: NDArray[np.uint8] | None,
    target_frame: NDArray[np.uint8] | None,
    xyxy: Sequence[float],
) -> tuple[float, float, float, float] | None:
    """Propagate ``xyxy`` from ``source_frame`` to ``target_frame``."""

    if source_frame is None or target_frame is None:
        return None
    if source_frame.shape[:2] != target_frame.shape[:2]:
        return None
    try:
        import cv2
    except ImportError:
        return None
    height, width = source_frame.shape[:2]
    x1, y1, x2, y2 = (int(round(float(value))) for value in xyxy)
    x1 = max(0, min(width - 2, x1))
    y1 = max(0, min(height - 2, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    roi = source_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, (0, 30, 32), (180, 255, 255))
    histogram = cv2.calcHist([hsv_roi], [0], mask, [16], [0, 180])
    if float(np.sum(histogram)) <= 1e-6:
        return None
    cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
    hsv_target = cv2.cvtColor(target_frame, cv2.COLOR_BGR2HSV)
    back_project = cv2.calcBackProject([hsv_target], [0], histogram, [0, 180], 1)
    window = (x1, y1, x2 - x1, y2 - y1)
    _ok, tracked = cv2.CamShift(
        back_project,
        window,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 1),
    )
    tx, ty, tw, th = tracked
    if tw <= 1 or th <= 1:
        return None
    return (float(tx), float(ty), float(tx + tw), float(ty + th))


def displacement(from_xyxy: Sequence[float], to_xyxy: Sequence[float], dt: float) -> tuple[float, float]:
    dt = max(float(dt), 1e-6)
    fx1, fy1, fx2, fy2 = (float(value) for value in from_xyxy)
    tx1, ty1, tx2, ty2 = (float(value) for value in to_xyxy)
    return (
        ((tx1 + tx2) / 2.0 - (fx1 + fx2) / 2.0) / dt,
        ((ty1 + ty2) / 2.0 - (fy1 + fy2) / 2.0) / dt,
    )
