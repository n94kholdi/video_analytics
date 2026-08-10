"""Optional image-to-ground homography projection."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from app.geometry.config import CalibrationConfig


@dataclass(frozen=True, slots=True)
class GroundProjection:
    """One projection result, including an explicit unavailable state."""

    available: bool
    point: tuple[float, float] | None
    unit: str | None
    reason: str | None = None


class ImageToGroundProjector:
    """Project image pixels through a validated fixed-resolution homography."""

    def __init__(
        self,
        matrix: np.ndarray | None,
        frame_size: tuple[int, int],
        unit: str | None,
        unavailable_reason: str | None = None,
    ) -> None:
        self._matrix = matrix
        self.frame_size = frame_size
        self.unit = unit
        self.unavailable_reason = unavailable_reason

    @classmethod
    def from_calibration(
        cls,
        calibration: CalibrationConfig | None,
        frame_size: tuple[int, int],
    ) -> "ImageToGroundProjector":
        """Build a projector or a usable unavailable sentinel when absent."""

        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("frame width and height must be positive")
        if calibration is None:
            return cls(None, frame_size, None, "camera calibration is not configured")

        image_points = np.asarray(
            [point.to_pixels(frame_size) for point in calibration.image_points],
            dtype=np.float64,
        )
        ground_points = np.asarray(calibration.ground_points, dtype=np.float64)
        matrix = _solve_homography(image_points, ground_points)
        return cls(matrix, frame_size, calibration.ground_unit)

    @property
    def available(self) -> bool:
        return self._matrix is not None

    def project(self, image_point: tuple[float, float]) -> GroundProjection:
        """Project an image-space pixel coordinate to configured ground units."""

        if self._matrix is None:
            return GroundProjection(False, None, None, self.unavailable_reason)
        if not all(math.isfinite(value) for value in image_point):
            raise ValueError("image point coordinates must be finite")
        homogeneous = self._matrix @ np.asarray(
            [image_point[0], image_point[1], 1.0], dtype=np.float64
        )
        if abs(homogeneous[2]) <= 1e-12:
            return GroundProjection(
                False,
                None,
                self.unit,
                "image point projects to infinity",
            )
        point = (homogeneous[:2] / homogeneous[2]).tolist()
        if not all(math.isfinite(value) for value in point):
            return GroundProjection(False, None, self.unit, "projection is not finite")
        return GroundProjection(True, (point[0], point[1]), self.unit)


def _solve_homography(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(source, destination):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    design = np.asarray(rows, dtype=np.float64)
    _, _, vectors = np.linalg.svd(design)
    matrix = vectors[-1].reshape(3, 3)
    if abs(matrix[2, 2]) > 1e-12:
        matrix /= matrix[2, 2]
    residuals = []
    for point, expected in zip(source, destination):
        projected = matrix @ np.asarray([point[0], point[1], 1.0])
        if abs(projected[2]) <= 1e-12:
            raise ValueError("calibration produced an unstable homography")
        residuals.append(np.linalg.norm(projected[:2] / projected[2] - expected))
    if not np.all(np.isfinite(matrix)) or max(residuals, default=0.0) > 1e-5:
        raise ValueError("calibration correspondences do not define a stable homography")
    return matrix
