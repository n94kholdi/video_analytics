"""Camera geometry for UCMCTrack ground-plane mapping.

UCMCTrack associates in world/ground coordinates. Calibration is optional:

- no geometry → image-plane foot-point mapping (any camera, no survey)
- per-camera YAML or official ``cam_para`` file → metric ground-plane mapping
- existing homography ``CalibrationConfig`` → mapping from image↔ground points

Add a camera later by dropping a YAML/txt file in a catalog directory (the
filename stem or ``camera_id`` field is the key) or by calling
``CameraGeometryCatalog.register``. Tracker code never hardcodes camera values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
import yaml

from app.geometry.calibration import ImageToGroundProjector
from app.geometry.config import CalibrationConfig


@dataclass(frozen=True, slots=True)
class MappedMeasurement:
    """Ground-plane measurement and correlated covariance (CMD)."""

    xy: NDArray[np.float64]
    covariance: NDArray[np.float64]
    calibrated: bool


class GroundPlaneMapper:
    """Map a box foot-point to a 2-D plane and propagate UV error (CMD).

    Calibrated mode follows ``detector/mapper.py`` in corfyi/UCMCTrack
    (Ki, Ko → homography A at ground height z0). Uncalibrated mode uses the
    image plane so the tracker can run before any camera is surveyed.
    """

    def __init__(
        self,
        matrix: NDArray[np.float64] | None,
        inverse: NDArray[np.float64] | None,
        *,
        calibrated: bool,
        unit: str | None,
    ) -> None:
        self._matrix = None if matrix is None else np.asarray(matrix, dtype=np.float64)
        self._inverse = None if inverse is None else np.asarray(inverse, dtype=np.float64)
        self.calibrated = calibrated
        self.unit = unit

    @classmethod
    def uncalibrated(cls) -> GroundPlaneMapper:
        identity = np.eye(3, dtype=np.float64)
        return cls(identity, identity, calibrated=False, unit="pixels")

    @classmethod
    def from_intrinsics_extrinsics(
        cls,
        intrinsic: NDArray[np.floating],
        rotation: NDArray[np.floating],
        translation: NDArray[np.floating],
        *,
        ground_z: float = 0.0,
        unit: str = "metres",
    ) -> GroundPlaneMapper:
        ki = _as_intrinsic_3x4(intrinsic)
        ko = np.eye(4, dtype=np.float64)
        ko[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        ko[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
        ki_ko = ki @ ko
        matrix = np.zeros((3, 3), dtype=np.float64)
        matrix[:, :2] = ki_ko[:, :2]
        matrix[:, 2] = float(ground_z) * ki_ko[:, 2] + ki_ko[:, 3]
        return cls(matrix, np.linalg.inv(matrix), calibrated=True, unit=unit)

    @classmethod
    def from_homography(
        cls,
        matrix: NDArray[np.floating],
        *,
        unit: str | None = "metres",
    ) -> GroundPlaneMapper:
        """``matrix`` maps image pixels → ground (same convention as CalibrationConfig)."""

        image_to_ground = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        try:
            ground_to_image = np.linalg.inv(image_to_ground)
        except np.linalg.LinAlgError as exc:
            raise ValueError("homography must be invertible") from exc
        return cls(ground_to_image, image_to_ground, calibrated=True, unit=unit)

    def map_box(self, xyxy: tuple[float, float, float, float]) -> MappedMeasurement | None:
        x1, _y1, x2, y2 = (float(value) for value in xyxy)
        width = max(x2 - x1, 1e-6)
        height = max(y2 - _y1, 1e-6)
        uv = np.array([[(x1 + x2) / 2.0], [y2]], dtype=np.float64)
        u_err, v_err = _uv_error(width, height)
        sigma_uv = np.diag([u_err * u_err, v_err * v_err])
        xy, covariance = self._uv_to_xy(uv, sigma_uv)
        if xy is None or covariance is None:
            return None
        return MappedMeasurement(xy.reshape(2), covariance, self.calibrated)

    def _uv_to_xy(
        self,
        uv: NDArray[np.float64],
        sigma_uv: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
        if self._inverse is None:
            return uv.reshape(2, 1), sigma_uv.copy()
        uv1 = np.array([[float(uv[0, 0])], [float(uv[1, 0])], [1.0]], dtype=np.float64)
        mapped = self._inverse @ uv1
        if abs(float(mapped[2, 0])) <= 1e-12:
            return None, None
        gamma = 1.0 / float(mapped[2, 0])
        xy = mapped[:2] * gamma
        if not np.all(np.isfinite(xy)):
            return None, None
        jacobian = gamma * self._inverse[:2, :2] - (gamma**2) * mapped[:2] * self._inverse[2, :2]
        covariance = jacobian @ sigma_uv @ jacobian.T
        covariance = 0.5 * (covariance + covariance.T) + 1e-9 * np.eye(2)
        return xy, covariance


@dataclass(frozen=True, slots=True)
class CameraGeometry:
    """One camera's UCMCTrack projection, keyed by ``camera_id`` at runtime."""

    camera_id: str
    mapper: GroundPlaneMapper
    wx: float | None = None
    wy: float | None = None
    vmax: float | None = None
    assignment_threshold: float | None = None

    @property
    def calibrated(self) -> bool:
        return self.mapper.calibrated


class CameraGeometryCatalog:
    """Lookup table of camera_id → geometry. Unknown IDs stay uncalibrated."""

    def __init__(self, cameras: Mapping[str, CameraGeometry] | None = None) -> None:
        self._cameras = dict(cameras or {})
        self._uncalibrated = GroundPlaneMapper.uncalibrated()

    def __len__(self) -> int:
        return len(self._cameras)

    def __contains__(self, camera_id: str) -> bool:
        return camera_id in self._cameras

    def register(self, geometry: CameraGeometry) -> None:
        if not geometry.camera_id.strip():
            raise ValueError("camera_id must be non-empty")
        self._cameras[geometry.camera_id] = geometry

    def geometry_for(self, camera_id: str) -> CameraGeometry | None:
        return self._cameras.get(camera_id)

    def mapper_for(self, camera_id: str) -> GroundPlaneMapper:
        geometry = self._cameras.get(camera_id)
        return geometry.mapper if geometry is not None else self._uncalibrated

    def parameters_for(
        self,
        camera_id: str,
        *,
        wx: float,
        wy: float,
        vmax: float,
        assignment_threshold: float,
    ) -> tuple[float, float, float, float]:
        geometry = self._cameras.get(camera_id)
        if geometry is None:
            return wx, wy, vmax, assignment_threshold
        return (
            float(geometry.wx) if geometry.wx is not None else wx,
            float(geometry.wy) if geometry.wy is not None else wy,
            float(geometry.vmax) if geometry.vmax is not None else vmax,
            (
                float(geometry.assignment_threshold)
                if geometry.assignment_threshold is not None
                else assignment_threshold
            ),
        )

    @classmethod
    def from_directory(cls, directory: str | Path) -> CameraGeometryCatalog:
        path = Path(directory)
        catalog = cls()
        if not path.is_dir():
            raise FileNotFoundError(f"camera geometry directory not found: {path}")
        for child in sorted(path.iterdir()):
            if child.suffix.lower() not in {".yaml", ".yml", ".txt"}:
                continue
            if child.name.lower().startswith("readme"):
                continue
            catalog.register(load_camera_geometry(child))
        return catalog


def load_camera_geometry(path: str | Path, *, camera_id: str | None = None) -> CameraGeometry:
    """Load one camera from YAML or an official UCMCTrack ``cam_para`` text file."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"camera geometry file not found: {file_path}")
    selected_id = camera_id or file_path.stem
    if file_path.suffix.lower() in {".yaml", ".yml"}:
        with file_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{file_path} must contain a mapping")
        return geometry_from_mapping(raw, default_camera_id=selected_id)
    return _geometry_from_cam_para(file_path, selected_id)


def geometry_from_mapping(values: Mapping[str, Any], *, default_camera_id: str) -> CameraGeometry:
    camera_id = str(values.get("camera_id") or values.get("id") or default_camera_id).strip()
    if not camera_id:
        raise ValueError("camera geometry must include a non-empty camera_id")
    mapper = _mapper_from_mapping(values)
    return CameraGeometry(
        camera_id,
        mapper,
        _optional_positive(values.get("wx")),
        _optional_positive(values.get("wy")),
        _optional_positive(values.get("vmax")),
        _optional_positive(values.get("assignment_threshold")),
    )


def geometry_from_calibration(
    calibration: CalibrationConfig,
    frame_size: tuple[int, int],
    *,
    camera_id: str,
) -> CameraGeometry:
    """Build UCMCTrack geometry from the project's existing homography config."""

    projector = ImageToGroundProjector.from_calibration(calibration, frame_size)
    if not projector.available or projector._matrix is None:
        raise ValueError(f"calibration for {camera_id!r} does not produce a homography")
    mapper = GroundPlaneMapper.from_homography(projector._matrix, unit=projector.unit or "metres")
    return CameraGeometry(camera_id, mapper)


def _mapper_from_mapping(values: Mapping[str, Any]) -> GroundPlaneMapper:
    if values.get("uncalibrated") is True:
        return GroundPlaneMapper.uncalibrated()
    cam_para = values.get("cam_para_file") or values.get("camera_parameter_file")
    if cam_para:
        return load_camera_geometry(cam_para).mapper
    if values.get("homography") is not None:
        unit = str(values.get("unit") or values.get("ground_unit") or "metres")
        return GroundPlaneMapper.from_homography(np.asarray(values["homography"], dtype=np.float64), unit=unit)
    if values.get("calibration") is not None:
        frame = values.get("frame_size") or values.get("image_size")
        if not isinstance(frame, (list, tuple)) or len(frame) != 2:
            raise ValueError("calibration mappings require frame_size: [width, height]")
        calibration = CalibrationConfig.from_mapping(values["calibration"])
        projector = ImageToGroundProjector.from_calibration(
            calibration,
            (int(frame[0]), int(frame[1])),
        )
        if not projector.available or projector._matrix is None:
            raise ValueError("calibration correspondences did not produce a homography")
        return GroundPlaneMapper.from_homography(
            projector._matrix,
            unit=projector.unit or "metres",
        )
    intrinsic = values.get("intrinsics") or values.get("intrinsic")
    extrinsic = values.get("extrinsics") or values.get("extrinsic")
    if intrinsic is not None and extrinsic is not None:
        return _mapper_from_krt(intrinsic, extrinsic, values)
    if intrinsic is None and extrinsic is None:
        return GroundPlaneMapper.uncalibrated()
    raise ValueError(
        "camera geometry needs intrinsics+extrinsics, homography, calibration, or uncalibrated: true"
    )


def _mapper_from_krt(
    intrinsic: Mapping[str, Any] | list[Any],
    extrinsic: Mapping[str, Any],
    values: Mapping[str, Any],
) -> GroundPlaneMapper:
    if isinstance(intrinsic, Mapping):
        fx = float(intrinsic["fx"])
        fy = float(intrinsic.get("fy", fx))
        cx = float(intrinsic["cx"])
        cy = float(intrinsic["cy"])
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    else:
        k = np.asarray(intrinsic, dtype=np.float64)
    if not isinstance(extrinsic, Mapping):
        raise ValueError("extrinsics must be a mapping with rotation and translation")
    rotation = np.asarray(extrinsic["rotation"], dtype=np.float64)
    if extrinsic.get("translation_mm") is not None:
        translation = np.asarray(extrinsic["translation_mm"], dtype=np.float64).reshape(3) / 1000.0
    elif extrinsic.get("translation") is not None:
        translation = np.asarray(extrinsic["translation"], dtype=np.float64).reshape(3)
        if str(extrinsic.get("translation_unit", "metres")).lower() in {"mm", "millimetres", "millimeters"}:
            translation = translation / 1000.0
    else:
        raise ValueError("extrinsics must include translation or translation_mm")
    return GroundPlaneMapper.from_intrinsics_extrinsics(
        k,
        rotation,
        translation,
        ground_z=float(values.get("ground_z", 0.0)),
        unit=str(values.get("unit") or values.get("ground_unit") or "metres"),
    )


def _geometry_from_cam_para(path: Path, camera_id: str) -> CameraGeometry:
    rotation, translation, intrinsic = _parse_cam_para(path)
    mapper = GroundPlaneMapper.from_intrinsics_extrinsics(intrinsic, rotation, translation)
    return CameraGeometry(camera_id, mapper)


def _parse_cam_para(
    path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rotation = np.zeros((3, 3), dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    intrinsic = np.zeros((3, 3), dtype=np.float64)
    found = {"rotation": False, "translation": False, "intrinsic": False}
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    index = 0
    while index < len(lines):
        label = lines[index].strip().replace(" ", "")
        if label in {"RotationMatrices", "RotationMatrix"}:
            index += 1
            for row in range(3):
                rotation[row] = np.fromstring(lines[index], sep=" ")
                index += 1
            found["rotation"] = True
            continue
        if label in {"TranslationVectors", "TranslationVector"}:
            index += 1
            translation = np.fromstring(lines[index], sep=" ").reshape(-1)[:3] / 1000.0
            index += 1
            found["translation"] = True
            continue
        if label in {"IntrinsicMatrix", "IntrinsicMatrices"}:
            index += 1
            for row in range(3):
                intrinsic[row] = np.fromstring(lines[index], sep=" ")
                index += 1
            found["intrinsic"] = True
            continue
        index += 1
    if not all(found.values()):
        missing = ", ".join(name for name, ok in found.items() if not ok)
        raise ValueError(f"{path} is missing camera sections: {missing}")
    return rotation, translation, intrinsic


def _as_intrinsic_3x4(intrinsic: NDArray[np.floating]) -> NDArray[np.float64]:
    matrix = np.asarray(intrinsic, dtype=np.float64)
    if matrix.shape == (3, 4):
        return matrix
    if matrix.shape == (3, 3):
        padded = np.zeros((3, 4), dtype=np.float64)
        padded[:, :3] = matrix
        return padded
    raise ValueError("intrinsic matrix must be 3x3 or 3x4")


def _uv_error(width: float, height: float) -> tuple[float, float]:
    """Official getUVError: 5% of box height, clipped."""

    _ = width
    return float(np.clip(0.05 * height, 2.0, 13.0)), float(np.clip(0.05 * height, 2.0, 10.0))


def _optional_positive(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("camera geometry numeric overrides must be positive")
    return result
