"""Optional camera-motion compensation via sparse optical flow + RANSAC.

Official Deep-OC-SORT uses OpenCV affine estimation (sparse flow or SIFT). MOT
file caches are omitted; this copy estimates a similarity transform between the
previous and current BGR frames, masking current detections like the authors.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class SparseFlowCMC:
    """Estimate a 2×3 affine transform from consecutive grayscale frames."""

    def __init__(self, *, minimum_features: int = 10) -> None:
        self.minimum_features = minimum_features
        self._previous: NDArray[np.uint8] | None = None
        self._previous_points: NDArray[np.float32] | None = None

    def reset(self) -> None:
        self._previous = None
        self._previous_points = None

    def compute_affine(self, frame: NDArray[np.uint8], boxes: NDArray[np.floating]) -> NDArray[np.float64]:
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        mask = np.ones_like(gray, dtype=np.uint8)
        if len(boxes):
            for box in np.round(boxes).astype(int):
                x1, y1, x2, y2 = (int(value) for value in box[:4])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = max(x1 + 1, x2)
                y2 = max(y1 + 1, y2)
                mask[y1:y2, x1:x2] = 0
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=3000,
            qualityLevel=0.01,
            minDistance=1,
            blockSize=3,
            mask=mask,
        )
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        if self._previous is None or self._previous_points is None or points is None:
            self._previous = gray
            self._previous_points = points
            return identity
        matched, status, _error = cv2.calcOpticalFlowPyrLK(self._previous, gray, self._previous_points, None)
        if matched is None or status is None:
            self._previous = gray
            self._previous_points = points
            return identity
        previous = self._previous_points.reshape(-1, 2)[status.reshape(-1) == 1]
        current = matched.reshape(-1, 2)[status.reshape(-1) == 1]
        transform = identity
        if previous.shape[0] > self.minimum_features:
            estimated, _ = cv2.estimateAffinePartial2D(previous, current, method=cv2.RANSAC)
            if estimated is not None:
                transform = np.asarray(estimated, dtype=np.float64)
        self._previous = gray
        self._previous_points = points
        return transform


def apply_affine_to_xyxy(box: Sequence[float], transform: NDArray[np.floating]) -> tuple[float, float, float, float]:
    linear = np.asarray(transform, dtype=np.float64).reshape(2, 3)
    matrix, shift = linear[:, :2], linear[:, 2]
    x1, y1, x2, y2 = (float(value) for value in box[:4])
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
    warped = corners @ matrix.T + shift
    xs, ys = warped[:, 0], warped[:, 1]
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
