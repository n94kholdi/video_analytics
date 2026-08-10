"""Timestamp-based smoothed movement speed for shared person tracks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
from typing import Sequence

from app.core.models import TrackObservation, TrajectoryPoint
from app.geometry.calibration import ImageToGroundProjector
from app.geometry.config import CameraConfig, SpeedConfig


_METRE_UNITS = {"m", "metre", "metres", "meter", "meters"}


@dataclass(frozen=True, slots=True)
class CameraSpeedConfig:
    """Resolved speed settings and optional projection for one camera."""

    camera_id: str
    settings: SpeedConfig
    projector: ImageToGroundProjector

    @classmethod
    def for_image(
        cls,
        camera_id: str,
        frame_size: tuple[int, int],
        settings: SpeedConfig | None = None,
    ) -> "CameraSpeedConfig":
        """Create pixel-only speed settings when no camera YAML is supplied."""

        if not camera_id.strip():
            raise ValueError("camera_id must be non-empty")
        return cls(
            camera_id,
            settings or SpeedConfig(),
            ImageToGroundProjector.from_calibration(None, frame_size),
        )

    @classmethod
    def from_camera_config(
        cls, config: CameraConfig, frame_size: tuple[int, int]
    ) -> "CameraSpeedConfig":
        return cls(
            config.camera_id,
            config.speed,
            ImageToGroundProjector.from_calibration(config.calibration, frame_size),
        )


@dataclass(frozen=True, slots=True)
class SpeedTrackStatus:
    """Latest speed state for one observed track."""

    track_id: int
    speed_pixels_per_second: float | None
    speed_metres_per_second: float | None
    unavailable_reason: str | None
    rejected_jump: bool = False


@dataclass(frozen=True, slots=True)
class SpeedSnapshot:
    """Per-frame camera speed metrics with explicit calibration warnings."""

    camera_id: str
    timestamp: float
    tracks: tuple[SpeedTrackStatus, ...]
    average_speed_pixels_per_second: float | None
    average_speed_metres_per_second: float | None
    calibration_warning: str | None


@dataclass(frozen=True, slots=True)
class SpeedResult:
    """Observations enriched with speed values plus aggregate metrics."""

    observations: tuple[TrackObservation, ...]
    snapshot: SpeedSnapshot


@dataclass(frozen=True, slots=True)
class CalibrationDistanceCheck:
    """Comparison of a projected image pair with a surveyed ground distance."""

    projected_distance_metres: float
    known_distance_metres: float
    absolute_error_metres: float
    relative_error: float


@dataclass(frozen=True, slots=True)
class _Sample:
    timestamp: float
    image: tuple[float, float]
    ground: tuple[float, float] | None


@dataclass(slots=True)
class _TrackHistory:
    samples: deque[_Sample]


class SpeedEstimator:
    """Estimate speed from accepted, smoothed, irregularly timed samples."""

    def __init__(self, cameras: Sequence[CameraSpeedConfig]) -> None:
        self._cameras = {camera.camera_id: camera for camera in cameras}
        if len(self._cameras) != len(cameras):
            raise ValueError("camera IDs must be unique")
        self._histories: dict[str, dict[int, _TrackHistory]] = {}
        self._timestamps: dict[str, float] = {}
        self.reset()

    def reset(self, camera_id: str | None = None) -> None:
        """Clear accepted samples and timestamp state."""

        if camera_id is not None and camera_id not in self._cameras:
            raise KeyError(f"unknown camera: {camera_id}")
        targets = self._cameras if camera_id is None else (camera_id,)
        for target in targets:
            self._histories[target] = {}
            self._timestamps[target] = 0.0

    def update(
        self,
        camera_id: str,
        observations: Sequence[TrackObservation],
        *,
        timestamp: float | None = None,
    ) -> SpeedResult:
        """Enrich current observations without assuming a fixed frame rate."""

        if camera_id not in self._cameras:
            raise KeyError(f"unknown camera: {camera_id}")
        if any(item.camera_id != camera_id for item in observations):
            raise ValueError("all observations must match the updated camera")
        event_time = timestamp if timestamp is not None else max(
            (item.timestamp for item in observations),
            default=self._timestamps[camera_id],
        )
        if not math.isfinite(event_time):
            raise ValueError("timestamp must be finite")
        if event_time < self._timestamps[camera_id]:
            raise ValueError("timestamps must be monotonic per camera")
        self._timestamps[camera_id] = event_time

        enriched: list[TrackObservation] = []
        statuses: list[SpeedTrackStatus] = []
        for observation in observations:
            item, status = self._estimate(camera_id, observation)
            enriched.append(item)
            statuses.append(status)

        cutoff = event_time - self._cameras[camera_id].settings.window_seconds
        observed_ids = {item.track_id for item in observations}
        for track_id, history in tuple(self._histories[camera_id].items()):
            if (
                track_id not in observed_ids
                and history.samples
                and history.samples[-1].timestamp < cutoff
            ):
                del self._histories[camera_id][track_id]

        pixel_values = [
            item.speed_pixels_per_second
            for item in statuses
            if item.speed_pixels_per_second is not None
        ]
        metre_values = [
            item.speed_metres_per_second
            for item in statuses
            if item.speed_metres_per_second is not None
        ]
        camera = self._cameras[camera_id]
        warning = _calibration_warning(camera.projector)
        return SpeedResult(
            tuple(enriched),
            SpeedSnapshot(
                camera_id,
                event_time,
                tuple(statuses),
                _mean(pixel_values),
                _mean(metre_values),
                warning,
            ),
        )

    def _estimate(
        self, camera_id: str, observation: TrackObservation
    ) -> tuple[TrackObservation, SpeedTrackStatus]:
        camera = self._cameras[camera_id]
        history = self._histories[camera_id].setdefault(
            observation.track_id, _TrackHistory(deque())
        )
        trajectory = observation.trajectory or (
            TrajectoryPoint(
                observation.timestamp,
                observation.frame_index,
                observation.foot_point,
                observation.foot_point,
            ),
        )
        rejected = False
        rejection_reason: str | None = None
        for point in trajectory:
            if history.samples and point.timestamp <= history.samples[-1].timestamp:
                continue
            accepted, reason = _accepted_sample(point, history, camera)
            if accepted is None:
                if point.timestamp == observation.timestamp:
                    rejected = True
                    rejection_reason = reason
                continue
            history.samples.append(accepted)

        _trim(history.samples, observation.timestamp - camera.settings.window_seconds)
        if rejected:
            return _with_speed(
                observation, history, camera, None, None, rejection_reason
            ), SpeedTrackStatus(
                observation.track_id, None, None, rejection_reason, True
            )
        if len(history.samples) < 2:
            reason = "insufficient timestamped trajectory history"
            item = _with_speed(observation, history, camera, None, None, reason)
            return item, SpeedTrackStatus(observation.track_id, None, None, reason)

        elapsed = history.samples[-1].timestamp - history.samples[0].timestamp
        if elapsed <= 0:
            reason = "insufficient positive timestamp span"
            item = _with_speed(observation, history, camera, None, None, reason)
            return item, SpeedTrackStatus(observation.track_id, None, None, reason)

        pixel_distance = _path_distance(history.samples, ground=False)
        pixel_speed = (
            0.0
            if pixel_distance < camera.settings.minimum_displacement_pixels
            else pixel_distance / elapsed
        )
        ground_available = all(sample.ground is not None for sample in history.samples)
        metre_speed: float | None = None
        reason = _calibration_warning(camera.projector)
        if ground_available and reason is None:
            metre_speed = _path_distance(history.samples, ground=True) / elapsed
            if pixel_speed == 0.0:
                metre_speed = 0.0
        item = _with_speed(
            observation, history, camera, pixel_speed, metre_speed, reason
        )
        return item, SpeedTrackStatus(
            observation.track_id, pixel_speed, metre_speed, reason
        )


def queue_progress_speed(
    velocity: tuple[float, float] | None,
    position: tuple[float, float] | None,
    service_point: tuple[float, float] | None,
) -> float | None:
    """Return signed velocity toward a service point; sideways motion is zero."""

    if velocity is None or position is None or service_point is None:
        return None
    offset = (service_point[0] - position[0], service_point[1] - position[1])
    distance = math.hypot(*offset)
    if distance <= 1e-12:
        return 0.0
    return (velocity[0] * offset[0] + velocity[1] * offset[1]) / distance


def validate_known_ground_distance(
    projector: ImageToGroundProjector,
    first_image_point: tuple[float, float],
    second_image_point: tuple[float, float],
    known_distance_metres: float,
) -> CalibrationDistanceCheck:
    """Validate calibration against an independently known ground distance."""

    if not math.isfinite(known_distance_metres) or known_distance_metres <= 0:
        raise ValueError("known_distance_metres must be finite and positive")
    if _calibration_warning(projector) is not None:
        raise ValueError(_calibration_warning(projector))
    first = projector.project(first_image_point)
    second = projector.project(second_image_point)
    if not first.available or not second.available or first.point is None or second.point is None:
        raise ValueError(first.reason or second.reason or "calibration projection unavailable")
    measured = math.dist(first.point, second.point)
    error = abs(measured - known_distance_metres)
    return CalibrationDistanceCheck(
        measured, known_distance_metres, error, error / known_distance_metres
    )


def _accepted_sample(
    point: TrajectoryPoint,
    history: _TrackHistory,
    camera: CameraSpeedConfig,
) -> tuple[_Sample | None, str | None]:
    ground_result = camera.projector.project(point.smoothed_position)
    ground = ground_result.point if ground_result.available else None
    sample = _Sample(point.timestamp, point.smoothed_position, ground)
    if not history.samples:
        return sample, None
    previous = history.samples[-1]
    elapsed = sample.timestamp - previous.timestamp
    if elapsed <= 0:
        return None, "non-increasing trajectory timestamp rejected"
    pixel_speed = math.dist(previous.image, sample.image) / elapsed
    if pixel_speed > camera.settings.maximum_speed_pixels_per_second:
        return None, "unrealistic pixel jump rejected"
    if (
        _calibration_warning(camera.projector) is None
        and previous.ground is not None
        and sample.ground is not None
    ):
        ground_speed = math.dist(previous.ground, sample.ground) / elapsed
        if ground_speed > camera.settings.maximum_speed_metres_per_second:
            return None, "unrealistic calibrated ground jump rejected"
    return sample, None


def _with_speed(
    observation: TrackObservation,
    history: _TrackHistory,
    camera: CameraSpeedConfig,
    pixel_speed: float | None,
    metre_speed: float | None,
    reason: str | None,
) -> TrackObservation:
    latest = history.samples[-1] if history.samples else None
    image_velocity = _velocity(history.samples, ground=False)
    ground_velocity = _velocity(history.samples, ground=True)
    # ``smoothed_velocity`` remains image-space and therefore pixels/second.
    return replace(
        observation,
        ground_position=latest.ground if latest is not None else None,
        smoothed_velocity=image_velocity,
        speed_pixels_per_second=pixel_speed,
        speed_metres_per_second=metre_speed,
        speed_unavailable_reason=reason,
        ground_smoothed_velocity=ground_velocity,
    )


def _velocity(
    samples: deque[_Sample], *, ground: bool
) -> tuple[float, float] | None:
    if len(samples) < 2:
        return None
    first, last = samples[0], samples[-1]
    elapsed = last.timestamp - first.timestamp
    start = first.ground if ground else first.image
    end = last.ground if ground else last.image
    if elapsed <= 0 or start is None or end is None:
        return None
    return ((end[0] - start[0]) / elapsed, (end[1] - start[1]) / elapsed)


def _path_distance(samples: deque[_Sample], *, ground: bool) -> float:
    points = [sample.ground if ground else sample.image for sample in samples]
    if any(point is None for point in points):
        return 0.0
    return sum(
        math.dist(first, second)
        for first, second in zip(points, points[1:])
        if first is not None and second is not None
    )


def _trim(samples: deque[_Sample], cutoff: float) -> None:
    while samples and samples[0].timestamp < cutoff:
        samples.popleft()


def _calibration_warning(projector: ImageToGroundProjector) -> str | None:
    if not projector.available:
        return projector.unavailable_reason or "camera calibration is unavailable"
    if projector.unit is None or projector.unit.strip().lower() not in _METRE_UNITS:
        return "calibration ground unit is not metres; physical speed unavailable"
    return None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None
