"""Constant-velocity Kalman filter with optional visual-tracking velocity observations.

State: ``[xc, yc, w, h, vx, vy, vw, vh]``.
Observation: ``[xc, yc, w, h]`` or ``[xc, yc, w, h, vx, vy]`` when Forward VT
supplies a displacement (paper supplementary, arXiv:2511.20418).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class StableKalmanFilter:
    """8-D constant-velocity filter using a real-valued time step."""

    def __init__(self, std_weight_position: float = 1.0 / 20.0, std_weight_velocity: float = 1.0 / 160.0) -> None:
        self._std_pos = std_weight_position
        self._std_vel = std_weight_velocity
        self.mean = np.zeros(8, dtype=np.float64)
        self.covariance = np.eye(8, dtype=np.float64)

    def initiate(self, measurement_xywh: NDArray[np.floating]) -> None:
        xc, yc, w, h = (float(value) for value in measurement_xywh)
        self.mean = np.array([xc, yc, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        std = [
            2 * self._std_pos * w,
            2 * self._std_pos * h,
            2 * self._std_pos * w,
            2 * self._std_pos * h,
            10 * self._std_vel * w,
            10 * self._std_vel * h,
            10 * self._std_vel * w,
            10 * self._std_vel * h,
        ]
        self.covariance = np.diag(np.square(std))

    def predict(self, dt: float) -> None:
        dt = max(float(dt), 1e-6)
        motion = np.eye(8, dtype=np.float64)
        motion[0, 4] = dt
        motion[1, 5] = dt
        motion[2, 6] = dt
        motion[3, 7] = dt
        w = max(abs(self.mean[2]), 1.0)
        h = max(abs(self.mean[3]), 1.0)
        std = [
            self._std_pos * w,
            self._std_pos * h,
            self._std_pos * w,
            self._std_pos * h,
            self._std_vel * w,
            self._std_vel * h,
            self._std_vel * w,
            self._std_vel * h,
        ]
        motion_cov = np.diag(np.square(std)) * max(dt, 1.0)
        self.mean = motion @ self.mean
        self.covariance = motion @ self.covariance @ motion.T + motion_cov

    def update(self, measurement: NDArray[np.floating], *, velocity: tuple[float, float] | None = None) -> None:
        if velocity is None:
            observed = np.asarray(measurement[:4], dtype=np.float64)
            projector = np.zeros((4, 8), dtype=np.float64)
            projector[0, 0] = 1.0
            projector[1, 1] = 1.0
            projector[2, 2] = 1.0
            projector[3, 3] = 1.0
        else:
            observed = np.array(
                [*measurement[:4], float(velocity[0]), float(velocity[1])],
                dtype=np.float64,
            )
            projector = np.zeros((6, 8), dtype=np.float64)
            for index in range(6):
                projector[index, index] = 1.0
        projected_mean = projector @ self.mean
        projected_cov = projector @ self.covariance @ projector.T
        w = max(abs(self.mean[2]), 1.0)
        h = max(abs(self.mean[3]), 1.0)
        noise = np.diag(
            np.square(
                [
                    self._std_pos * w,
                    self._std_pos * h,
                    self._std_pos * w,
                    self._std_pos * h,
                    *(() if velocity is None else (self._std_vel * w, self._std_vel * h)),
                ]
            )
        )
        innovation_cov = projected_cov + noise
        gain = self.covariance @ projector.T @ np.linalg.pinv(innovation_cov)
        innovation = observed - projected_mean
        self.mean = self.mean + gain @ innovation
        self.covariance = self.covariance - gain @ projector @ self.covariance

    def to_xyxy(self) -> tuple[float, float, float, float]:
        xc, yc, w, h = (float(value) for value in self.mean[:4])
        w = max(w, 1.0)
        h = max(h, 1.0)
        return (xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0)
