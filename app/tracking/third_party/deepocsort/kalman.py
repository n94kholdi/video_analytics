"""8-D constant-velocity Kalman filter used by Deep OC-SORT (new_kf).

Official Deep-OC-SORT stores ``[x, y, w, h, vx, vy, vw, vh]`` and assumes a
unit frame step. This copy uses a real-valued ``dt`` so 0.5 FPS / 2 s gaps do
not explode velocity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def xyxy_to_xywh(box: np.ndarray | tuple[float, float, float, float]) -> NDArray[np.float64]:
    x1, y1, x2, y2 = (float(value) for value in box[:4])
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, width, height], dtype=np.float64)


def xywh_to_xyxy(xywh: NDArray[np.floating]) -> tuple[float, float, float, float]:
    xc, yc, width, height = (float(value) for value in xywh[:4])
    width = max(width, 1.0)
    height = max(height, 1.0)
    return (xc - width / 2.0, yc - height / 2.0, xc + width / 2.0, yc + height / 2.0)


def _process_noise(width: float, height: float, dt: float, *, p: float = 1.0 / 20.0, v: float = 1.0 / 160.0) -> NDArray[np.float64]:
    width = max(abs(width), 1.0)
    height = max(abs(height), 1.0)
    dt = max(float(dt), 1e-6)
    std = np.array(
        [
            p * width,
            p * height,
            p * width,
            p * height,
            v * width,
            v * height,
            v * width,
            v * height,
        ],
        dtype=np.float64,
    )
    return np.diag(np.square(std)) * max(dt, 1.0)


def _measurement_noise(width: float, height: float, *, m: float = 1.0 / 20.0) -> NDArray[np.float64]:
    width = max(abs(width), 1.0)
    height = max(abs(height), 1.0)
    return np.diag(np.square(np.array([m * width, m * height, m * width, m * height], dtype=np.float64)))


class DeepOCSortKalman:
    """Official new_kf dynamics with a timestamp-aware transition matrix."""

    def __init__(self) -> None:
        self.mean = np.zeros(8, dtype=np.float64)
        self.covariance = np.eye(8, dtype=np.float64)

    def initiate(self, measurement_xywh: NDArray[np.floating]) -> None:
        xc, yc, width, height = (float(value) for value in measurement_xywh[:4])
        self.mean = np.array([xc, yc, width, height, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        noise = _process_noise(width, height, 1.0)
        noise[:4, :4] *= 4.0
        noise[4:, 4:] *= 100.0
        self.covariance = noise

    def predict(self, dt: float, *, freeze_scale_velocity: bool = False) -> None:
        dt = max(float(dt), 1e-6)
        if freeze_scale_velocity:
            self.mean[6] = 0.0
            self.mean[7] = 0.0
        if self.mean[2] + self.mean[6] * dt <= 0:
            self.mean[6] = 0.0
        if self.mean[3] + self.mean[7] * dt <= 0:
            self.mean[7] = 0.0
        motion = np.eye(8, dtype=np.float64)
        motion[0, 4] = dt
        motion[1, 5] = dt
        motion[2, 6] = dt
        motion[3, 7] = dt
        process = _process_noise(float(self.mean[2]), float(self.mean[3]), dt)
        self.mean = motion @ self.mean
        self.covariance = motion @ self.covariance @ motion.T + process

    def update(self, measurement_xywh: NDArray[np.floating]) -> None:
        observed = np.asarray(measurement_xywh[:4], dtype=np.float64)
        projector = np.zeros((4, 8), dtype=np.float64)
        projector[0, 0] = 1.0
        projector[1, 1] = 1.0
        projector[2, 2] = 1.0
        projector[3, 3] = 1.0
        projected_mean = projector @ self.mean
        projected_cov = projector @ self.covariance @ projector.T
        noise = _measurement_noise(float(self.mean[2]), float(self.mean[3]))
        innovation_cov = projected_cov + noise
        gain = self.covariance @ projector.T @ np.linalg.pinv(innovation_cov)
        self.mean = self.mean + gain @ (observed - projected_mean)
        self.covariance = self.covariance - gain @ projector @ self.covariance

    def apply_affine(self, matrix: NDArray[np.floating], translation: NDArray[np.floating]) -> None:
        linear = np.asarray(matrix, dtype=np.float64).reshape(2, 2)
        shift = np.asarray(translation, dtype=np.float64).reshape(2)
        center = linear @ self.mean[:2] + shift
        velocity = linear @ self.mean[4:6]
        scale = float(np.sqrt(max(np.linalg.det(linear), 1e-12)))
        self.mean[0] = center[0]
        self.mean[1] = center[1]
        self.mean[2] = max(self.mean[2] * scale, 1.0)
        self.mean[3] = max(self.mean[3] * scale, 1.0)
        self.mean[4] = velocity[0]
        self.mean[5] = velocity[1]
        self.mean[6] *= scale
        self.mean[7] *= scale

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return xywh_to_xyxy(self.mean)
