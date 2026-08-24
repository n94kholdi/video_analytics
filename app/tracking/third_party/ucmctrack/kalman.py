"""Ground-plane constant-velocity Kalman filter used by UCMCTrack.

Official ``tracker/kalman.py`` stores ``[x, vx, y, vy]`` on the mapped plane
and assumes a unit frame step. This copy uses elapsed seconds so 0.5 FPS / 2 s
gaps remain well-defined. Process-noise compensation follows the paper's G Q0 Gᵀ
construction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class GroundKalman:
    """2-D CV filter with timestamp-aware transition and mapped measurement R."""

    def __init__(self, measurement: NDArray[np.floating], covariance: NDArray[np.floating], *, wx: float, wy: float, vmax: float) -> None:
        xy = np.asarray(measurement, dtype=np.float64).reshape(2)
        self.wx = float(wx)
        self.wy = float(wy)
        self.mean = np.array([xy[0], 0.0, xy[1], 0.0], dtype=np.float64)
        self.covariance = np.diag(
            np.array([1.0, (vmax**2) / 3.0, 1.0, (vmax**2) / 3.0], dtype=np.float64)
        )
        self._observe(xy, covariance)

    def predict(self, dt: float) -> None:
        dt = max(float(dt), 1e-6)
        motion = np.eye(4, dtype=np.float64)
        motion[0, 1] = dt
        motion[2, 3] = dt
        gain = np.zeros((4, 2), dtype=np.float64)
        gain[0, 0] = 0.5 * dt * dt
        gain[1, 0] = dt
        gain[2, 1] = 0.5 * dt * dt
        gain[3, 1] = dt
        process = np.array([[self.wx, 0.0], [0.0, self.wy]], dtype=np.float64)
        noise = gain @ process @ gain.T
        self.mean = motion @ self.mean
        self.covariance = motion @ self.covariance @ motion.T + noise

    def update(self, measurement: NDArray[np.floating], covariance: NDArray[np.floating]) -> None:
        self._observe(np.asarray(measurement, dtype=np.float64).reshape(2), covariance)

    def association_scores(
        self, measurement: NDArray[np.floating], covariance: NDArray[np.floating]
    ) -> tuple[float, float]:
        """Return (Mahalanobis, MMD) for gating and ranking.

        Paper Eq. 8: MMD = εᵀ S⁻¹ ε + ln|S|. Official MOT17 thresholds assume
        metric-scale S with ln|S| around O(1). At 0.5 FPS in pixel space the
        velocity prior inflates S, so ln|S| alone can exceed that gate. Gate on
        Mahalanobis (chi-square, scale-stable); rank with full MMD.
        """

        observed = np.asarray(measurement, dtype=np.float64).reshape(2)
        predicted = self._measurement_matrix() @ self.mean
        innovation = observed - predicted
        residual = self._residual_covariance(covariance)
        try:
            inverse = np.linalg.inv(residual)
        except np.linalg.LinAlgError:
            return 1e6, 1e6
        mahalanobis = float(innovation @ inverse @ innovation)
        sign, logdet = np.linalg.slogdet(residual)
        if sign <= 0:
            return 1e6, 1e6
        return mahalanobis, mahalanobis + float(logdet)

    def distance(self, measurement: NDArray[np.floating], covariance: NDArray[np.floating]) -> float:
        """Mapped Mahalanobis distance plus log-det (official ``distance``)."""

        return self.association_scores(measurement, covariance)[1]

    def position(self) -> tuple[float, float]:
        return (float(self.mean[0]), float(self.mean[2]))

    def limit_speed(self, vmax: float) -> None:
        speed = float(np.hypot(self.mean[1], self.mean[3]))
        if vmax > 0.0 and speed > vmax:
            scale = vmax / speed
            self.mean[1] *= scale
            self.mean[3] *= scale

    def dampen_velocity(self, factor: float) -> None:
        self.mean[1] *= float(factor)
        self.mean[3] *= float(factor)

    def _observe(self, measurement: NDArray[np.float64], covariance: NDArray[np.floating]) -> None:
        projector = self._measurement_matrix()
        residual = self._residual_covariance(covariance)
        try:
            gain = self.covariance @ projector.T @ np.linalg.inv(residual)
        except np.linalg.LinAlgError:
            return
        innovation = measurement - projector @ self.mean
        self.mean = self.mean + gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        self.covariance = (identity - gain @ projector) @ self.covariance

    def _residual_covariance(self, measurement_noise: NDArray[np.floating]) -> NDArray[np.float64]:
        projector = self._measurement_matrix()
        noise = np.asarray(measurement_noise, dtype=np.float64).reshape(2, 2)
        noise = 0.5 * (noise + noise.T) + 1e-9 * np.eye(2)
        residual = projector @ self.covariance @ projector.T + noise
        return 0.5 * (residual + residual.T)

    @staticmethod
    def _measurement_matrix() -> NDArray[np.float64]:
        projector = np.zeros((2, 4), dtype=np.float64)
        projector[0, 0] = 1.0
        projector[1, 2] = 1.0
        return projector
