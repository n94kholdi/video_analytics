"""Validated YAML camera and analytics configuration models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import yaml

from app.geometry.primitives import denormalize_point, validate_polygon


class CameraConfigError(ValueError):
    """Raised when camera geometry configuration is invalid."""


@dataclass(frozen=True, slots=True)
class NormalizedPoint:
    """A resolution-independent image point in the closed unit square."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise CameraConfigError("normalized point coordinates must be finite")
        if not 0.0 <= self.x <= 1.0 or not 0.0 <= self.y <= 1.0:
            raise CameraConfigError(
                "normalized point coordinates must be between 0 and 1"
            )

    @classmethod
    def from_value(cls, value: Any, field: str) -> "NormalizedPoint":
        point = _point(value, field)
        try:
            return cls(*point)
        except CameraConfigError as exc:
            raise CameraConfigError(f"{field}: {exc}") from exc

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def to_pixels(self, frame_size: tuple[int, int]) -> tuple[float, float]:
        return denormalize_point(self.as_tuple(), frame_size)


@dataclass(frozen=True, slots=True)
class PolygonZone:
    """Named normalized polygon used by a later analytics module."""

    zone_id: str
    points: tuple[NormalizedPoint, ...]

    def __post_init__(self) -> None:
        if not self.zone_id.strip():
            raise CameraConfigError("polygon zone_id must be non-empty")
        try:
            validate_polygon(tuple(point.as_tuple() for point in self.points))
        except ValueError as exc:
            raise CameraConfigError(f"polygon {self.zone_id!r}: {exc}") from exc

    def pixel_points(self, frame_size: tuple[int, int]) -> tuple[tuple[float, float], ...]:
        return tuple(point.to_pixels(frame_size) for point in self.points)

    @classmethod
    def from_mapping(cls, value: Any, field: str) -> "PolygonZone":
        mapping = _mapping_value(value, field)
        zone_id = _string(mapping.get("id"), f"{field}.id")
        points = _point_list(mapping.get("points"), f"{field}.points")
        return cls(zone_id, points)


@dataclass(frozen=True, slots=True)
class ActiveSchedule:
    """Optional weekly wall-clock schedule for a restricted zone."""

    start_minute: int
    end_minute: int
    weekdays: tuple[int, ...] = tuple(range(7))
    timezone: str = "UTC"

    def __post_init__(self) -> None:
        if not 0 <= self.start_minute < 24 * 60:
            raise CameraConfigError("active schedule start must be a valid HH:MM time")
        if not 0 <= self.end_minute < 24 * 60:
            raise CameraConfigError("active schedule end must be a valid HH:MM time")
        if self.start_minute == self.end_minute:
            raise CameraConfigError("active schedule start and end must differ")
        if not self.weekdays or len(set(self.weekdays)) != len(self.weekdays):
            raise CameraConfigError(
                "active schedule weekdays must be non-empty and unique"
            )
        if any(isinstance(day, bool) or not 0 <= day <= 6 for day in self.weekdays):
            raise CameraConfigError("active schedule weekdays must be integers from 0 to 6")
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise CameraConfigError(
                f"active schedule timezone is unknown: {self.timezone}"
            ) from exc

    @classmethod
    def from_mapping(cls, value: Any, field: str) -> "ActiveSchedule":
        mapping = _mapping_value(value, field)
        weekdays_value = mapping.get("weekdays", list(range(7)))
        if not isinstance(weekdays_value, list):
            raise CameraConfigError(f"{field}.weekdays must be a list")
        weekdays = tuple(
            _weekday(day, f"{field}.weekdays[{index}]")
            for index, day in enumerate(weekdays_value)
        )
        return cls(
            _clock_minute(mapping.get("start"), f"{field}.start"),
            _clock_minute(mapping.get("end"), f"{field}.end"),
            weekdays,
            _string(mapping.get("timezone", "UTC"), f"{field}.timezone"),
        )

    def is_active(self, timestamp: float) -> bool:
        """Evaluate a Unix timestamp, including schedules spanning midnight."""

        local = datetime.fromtimestamp(timestamp, ZoneInfo(self.timezone))
        minute = local.hour * 60 + local.minute
        if self.start_minute <= self.end_minute:
            return (
                local.weekday() in self.weekdays
                and self.start_minute <= minute < self.end_minute
            )
        if minute >= self.start_minute:
            return local.weekday() in self.weekdays
        previous_day = (local.weekday() - 1) % 7
        return previous_day in self.weekdays and minute < self.end_minute


@dataclass(frozen=True, slots=True)
class RestrictedZone:
    """Restricted polygon and temporal intrusion thresholds."""

    zone_id: str
    points: tuple[NormalizedPoint, ...]
    entry_dwell_seconds: float = 1.0
    exit_grace_seconds: float = 1.0
    alert_cooldown_seconds: float = 30.0
    active_schedule: ActiveSchedule | None = None

    def __post_init__(self) -> None:
        PolygonZone(self.zone_id, self.points)
        for value, field_name in (
            (self.entry_dwell_seconds, "entry_dwell_seconds"),
            (self.exit_grace_seconds, "exit_grace_seconds"),
            (self.alert_cooldown_seconds, "alert_cooldown_seconds"),
        ):
            if not math.isfinite(value) or value < 0:
                raise CameraConfigError(
                    f"restricted zone {field_name} must be non-negative"
                )

    def pixel_points(self, frame_size: tuple[int, int]) -> tuple[tuple[float, float], ...]:
        return tuple(point.to_pixels(frame_size) for point in self.points)

    @classmethod
    def from_mapping(cls, value: Any, field: str) -> "RestrictedZone":
        mapping = _mapping_value(value, field)
        schedule = mapping.get("active_schedule")
        return cls(
            _string(mapping.get("id"), f"{field}.id"),
            _point_list(mapping.get("points"), f"{field}.points"),
            _number(
                mapping.get("entry_dwell_seconds", 1.0),
                f"{field}.entry_dwell_seconds",
            ),
            _number(
                mapping.get("exit_grace_seconds", 1.0),
                f"{field}.exit_grace_seconds",
            ),
            _number(
                mapping.get("alert_cooldown_seconds", 30.0),
                f"{field}.alert_cooldown_seconds",
            ),
            ActiveSchedule.from_mapping(schedule, f"{field}.active_schedule")
            if schedule is not None
            else None,
        )


@dataclass(frozen=True, slots=True)
class CountingLine:
    """Named directed finite line with normalized endpoints."""

    line_id: str
    start: NormalizedPoint
    end: NormalizedPoint
    positive_label: str = "entry"
    negative_label: str = "exit"
    hysteresis: float = 0.01

    def __post_init__(self) -> None:
        if not self.line_id.strip():
            raise CameraConfigError("counting line id must be non-empty")
        if self.start == self.end:
            raise CameraConfigError(f"counting line {self.line_id!r} endpoints differ")
        if not self.positive_label.strip() or not self.negative_label.strip():
            raise CameraConfigError("counting line direction labels must be non-empty")
        if not math.isfinite(self.hysteresis) or not 0.0 <= self.hysteresis <= 1.0:
            raise CameraConfigError(
                "counting line hysteresis must be between 0 and 1"
            )

    def pixel_points(
        self, frame_size: tuple[int, int]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self.start.to_pixels(frame_size), self.end.to_pixels(frame_size))

    @classmethod
    def from_mapping(cls, value: Any, field: str) -> "CountingLine":
        mapping = _mapping_value(value, field)
        return cls(
            _string(mapping.get("id"), f"{field}.id"),
            NormalizedPoint.from_value(mapping.get("start"), f"{field}.start"),
            NormalizedPoint.from_value(mapping.get("end"), f"{field}.end"),
            _string(mapping.get("positive_label", "entry"), f"{field}.positive_label"),
            _string(mapping.get("negative_label", "exit"), f"{field}.negative_label"),
            _number(mapping.get("hysteresis", 0.01), f"{field}.hysteresis"),
        )


@dataclass(frozen=True, slots=True)
class ServicePoint:
    """Manually configured queue service destination."""

    point: NormalizedPoint
    label: str = "service"

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise CameraConfigError("service point label must be non-empty")


@dataclass(frozen=True, slots=True)
class QueueRegion:
    """Configured queue polygon, destination, and overflow threshold."""

    queue_id: str
    polygon: tuple[NormalizedPoint, ...]
    service_point: ServicePoint
    overflow_threshold: int

    def __post_init__(self) -> None:
        if not self.queue_id.strip():
            raise CameraConfigError("queue id must be non-empty")
        if isinstance(self.overflow_threshold, bool) or self.overflow_threshold <= 0:
            raise CameraConfigError("queue overflow_threshold must be positive")
        try:
            validate_polygon(tuple(point.as_tuple() for point in self.polygon))
        except ValueError as exc:
            raise CameraConfigError(f"queue {self.queue_id!r}: {exc}") from exc

    @classmethod
    def from_mapping(cls, value: Any, field: str) -> "QueueRegion":
        mapping = _mapping_value(value, field)
        service = _mapping_value(mapping.get("service_point"), f"{field}.service_point")
        return cls(
            _string(mapping.get("id"), f"{field}.id"),
            _point_list(mapping.get("polygon"), f"{field}.polygon"),
            ServicePoint(
                NormalizedPoint.from_value(
                    service.get("point"), f"{field}.service_point.point"
                ),
                _string(
                    service.get("label", "service"),
                    f"{field}.service_point.label",
                ),
            ),
            _positive_int(mapping.get("overflow_threshold"), f"{field}.overflow_threshold"),
        )


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Image-to-ground correspondences for an optional homography."""

    image_points: tuple[NormalizedPoint, ...]
    ground_points: tuple[tuple[float, float], ...]
    ground_unit: str = "metres"

    def __post_init__(self) -> None:
        if len(self.image_points) != len(self.ground_points):
            raise CameraConfigError(
                "calibration image_points and ground_points lengths must match"
            )
        if len(self.image_points) < 4:
            raise CameraConfigError("calibration requires at least four correspondences")
        if not self.ground_unit.strip():
            raise CameraConfigError("calibration ground_unit must be non-empty")
        if not all(all(math.isfinite(v) for v in point) for point in self.ground_points):
            raise CameraConfigError("calibration ground points must be finite")
        if len(set(point.as_tuple() for point in self.image_points)) != len(
            self.image_points
        ) or len(set(self.ground_points)) != len(self.ground_points):
            raise CameraConfigError("calibration points must not contain duplicates")
        design = _homography_design_matrix(
            tuple(point.as_tuple() for point in self.image_points), self.ground_points
        )
        if np.linalg.matrix_rank(design, tol=1e-10) < 8:
            raise CameraConfigError(
                "calibration correspondences are degenerate; use non-collinear points"
            )

    @classmethod
    def from_mapping(cls, value: Any, field: str = "calibration") -> "CalibrationConfig":
        mapping = _mapping_value(value, field)
        image_points = _point_list(mapping.get("image_points"), f"{field}.image_points")
        raw_ground = mapping.get("ground_points")
        if not isinstance(raw_ground, list):
            raise CameraConfigError(f"{field}.ground_points must be a list")
        ground_points = tuple(
            _point(point, f"{field}.ground_points[{index}]")
            for index, point in enumerate(raw_ground)
        )
        return cls(
            image_points,
            ground_points,
            _string(mapping.get("ground_unit", "metres"), f"{field}.ground_unit"),
        )


@dataclass(frozen=True, slots=True)
class AnalyticsConfig:
    """Enabled module names and their configured geometry."""

    enabled: tuple[str, ...]
    occupancy_zones: tuple[PolygonZone, ...] = ()
    restricted_zones: tuple[RestrictedZone, ...] = ()
    counting_lines: tuple[CountingLine, ...] = ()
    queues: tuple[QueueRegion, ...] = ()

    def __post_init__(self) -> None:
        allowed = {"occupancy", "line_counting", "restricted_area", "heatmap", "queue", "speed"}
        unknown = set(self.enabled) - allowed
        if unknown:
            raise CameraConfigError(
                f"unknown analytics modules: {', '.join(sorted(unknown))}"
            )
        if len(set(self.enabled)) != len(self.enabled):
            raise CameraConfigError("analytics.enabled must not contain duplicates")
        identifiers = [zone.zone_id for zone in self.occupancy_zones]
        identifiers += [zone.zone_id for zone in self.restricted_zones]
        identifiers += [line.line_id for line in self.counting_lines]
        identifiers += [queue.queue_id for queue in self.queues]
        if len(set(identifiers)) != len(identifiers):
            raise CameraConfigError("zone, line, and queue IDs must be unique per camera")

    @classmethod
    def from_mapping(cls, value: Any) -> "AnalyticsConfig":
        mapping = _mapping_value(value, "analytics")
        raw_enabled = mapping.get("enabled", [])
        if not isinstance(raw_enabled, list):
            raise CameraConfigError("analytics.enabled must be a list")
        enabled = tuple(
            _string(item, f"analytics.enabled[{index}]")
            for index, item in enumerate(raw_enabled)
        )
        return cls(
            enabled,
            _models(mapping.get("occupancy_zones", []), "analytics.occupancy_zones", PolygonZone),
            _models(mapping.get("restricted_zones", []), "analytics.restricted_zones", RestrictedZone),
            _models(mapping.get("counting_lines", []), "analytics.counting_lines", CountingLine),
            _models(mapping.get("queues", []), "analytics.queues", QueueRegion),
        )


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    confidence_threshold: float = 0.4
    iou_threshold: float = 0.7

    def __post_init__(self) -> None:
        _validate_threshold(self.confidence_threshold, "detector.confidence_threshold")
        _validate_threshold(self.iou_threshold, "detector.iou_threshold")


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    activation_threshold: float = 0.4
    lost_track_buffer: int = 30
    match_threshold: float = 0.3

    def __post_init__(self) -> None:
        _validate_threshold(self.activation_threshold, "tracker.activation_threshold")
        _validate_threshold(self.match_threshold, "tracker.match_threshold")
        _positive_int(self.lost_track_buffer, "tracker.lost_track_buffer")


@dataclass(frozen=True, slots=True)
class HeatmapConfig:
    region: tuple[NormalizedPoint, ...] | None = None
    grid_size: tuple[int, int] = (64, 36)

    def __post_init__(self) -> None:
        if self.region is not None:
            try:
                validate_polygon(tuple(point.as_tuple() for point in self.region))
            except ValueError as exc:
                raise CameraConfigError(f"heatmap.region: {exc}") from exc
        if len(self.grid_size) != 2 or any(
            isinstance(value, bool) or value <= 0 for value in self.grid_size
        ):
            raise CameraConfigError("heatmap.grid_size values must be positive integers")


@dataclass(frozen=True, slots=True)
class OutputConfig:
    annotated_video: str | None = None
    events_jsonl: str | None = None


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    enabled: bool = True
    draw_geometry: bool = True


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Complete configuration for one independent camera."""

    camera_id: str
    name: str
    source: str | int
    analytics: AnalyticsConfig
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    calibration: CalibrationConfig | None = None
    outputs: OutputConfig = field(default_factory=OutputConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self) -> None:
        if not self.camera_id.strip() or not self.name.strip():
            raise CameraConfigError("camera id and name must be non-empty")
        if not isinstance(self.source, int) and not str(self.source).strip():
            raise CameraConfigError("camera source must be a path, URI, or device index")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CameraConfig":
        camera = _mapping_value(values.get("camera"), "camera")
        detector = _mapping_value(values.get("detector", {}), "detector")
        tracker = _mapping_value(values.get("tracker", {}), "tracker")
        heatmap = _mapping_value(values.get("heatmap", {}), "heatmap")
        outputs = _mapping_value(values.get("outputs", {}), "outputs")
        visualization = _mapping_value(values.get("visualization", {}), "visualization")
        raw_source = camera.get("source")
        if not isinstance(raw_source, (str, int)) or isinstance(raw_source, bool):
            raise CameraConfigError("camera.source must be a string or integer")
        grid = heatmap.get("grid_size", [64, 36])
        if not isinstance(grid, list) or len(grid) != 2:
            raise CameraConfigError("heatmap.grid_size must contain width and height")
        region_value = heatmap.get("region")
        return cls(
            _string(camera.get("id"), "camera.id"),
            _string(camera.get("name"), "camera.name"),
            raw_source,
            AnalyticsConfig.from_mapping(values.get("analytics")),
            DetectorConfig(
                _number(detector.get("confidence_threshold", 0.4), "detector.confidence_threshold"),
                _number(detector.get("iou_threshold", 0.7), "detector.iou_threshold"),
            ),
            TrackerConfig(
                _number(tracker.get("activation_threshold", 0.4), "tracker.activation_threshold"),
                _positive_int(tracker.get("lost_track_buffer", 30), "tracker.lost_track_buffer"),
                _number(tracker.get("match_threshold", 0.3), "tracker.match_threshold"),
            ),
            HeatmapConfig(
                _point_list(region_value, "heatmap.region") if region_value is not None else None,
                (
                    _positive_int(grid[0], "heatmap.grid_size[0]"),
                    _positive_int(grid[1], "heatmap.grid_size[1]"),
                ),
            ),
            CalibrationConfig.from_mapping(values["calibration"])
            if values.get("calibration") is not None
            else None,
            OutputConfig(
                _optional_string(outputs.get("annotated_video"), "outputs.annotated_video"),
                _optional_string(outputs.get("events_jsonl"), "outputs.events_jsonl"),
            ),
            VisualizationConfig(
                _boolean(visualization.get("enabled", True), "visualization.enabled"),
                _boolean(visualization.get("draw_geometry", True), "visualization.draw_geometry"),
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the stable public YAML shape rather than dataclass field names."""

        result: dict[str, Any] = {
            "camera": {"id": self.camera_id, "name": self.name, "source": self.source},
            "detector": asdict(self.detector),
            "tracker": asdict(self.tracker),
            "analytics": {
                "enabled": list(self.analytics.enabled),
                "occupancy_zones": [_zone_mapping(item) for item in self.analytics.occupancy_zones],
                "restricted_zones": [_restricted_zone_mapping(item) for item in self.analytics.restricted_zones],
                "counting_lines": [_line_mapping(item) for item in self.analytics.counting_lines],
                "queues": [_queue_mapping(item) for item in self.analytics.queues],
            },
            "heatmap": {
                "region": _points_mapping(self.heatmap.region) if self.heatmap.region else None,
                "grid_size": list(self.heatmap.grid_size),
            },
            "outputs": asdict(self.outputs),
            "visualization": asdict(self.visualization),
        }
        if self.calibration is not None:
            result["calibration"] = {
                "image_points": _points_mapping(self.calibration.image_points),
                "ground_points": [list(point) for point in self.calibration.ground_points],
                "ground_unit": self.calibration.ground_unit,
            }
        return result


def load_camera_config(path: str | Path) -> CameraConfig:
    """Load and validate one camera YAML file."""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        raise CameraConfigError(f"camera configuration not found: {config_path}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise CameraConfigError(f"could not read camera configuration: {config_path}") from exc
    if not isinstance(values, Mapping):
        raise CameraConfigError("camera configuration root must be a mapping")
    return CameraConfig.from_mapping(values)


def dump_camera_config(config: CameraConfig, path: str | Path) -> None:
    """Serialize a validated camera configuration as readable YAML."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config.to_mapping(), stream, sort_keys=False)
    except OSError as exc:
        raise CameraConfigError(f"could not write camera configuration: {output_path}") from exc


def _homography_design_matrix(source: Sequence[tuple[float, float]], destination: Sequence[tuple[float, float]]) -> np.ndarray:
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(source, destination):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    return np.asarray(rows, dtype=np.float64)


def _models(values: Any, field: str, model: type[Any]) -> tuple[Any, ...]:
    if not isinstance(values, list):
        raise CameraConfigError(f"{field} must be a list")
    return tuple(model.from_mapping(value, f"{field}[{index}]") for index, value in enumerate(values))


def _point_list(value: Any, field: str) -> tuple[NormalizedPoint, ...]:
    if not isinstance(value, list):
        raise CameraConfigError(f"{field} must be a list")
    return tuple(NormalizedPoint.from_value(point, f"{field}[{index}]") for index, point in enumerate(value))


def _point(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CameraConfigError(f"{field} must contain exactly two coordinates")
    return (_number(value[0], f"{field}[0]"), _number(value[1], f"{field}[1]"))


def _mapping_value(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CameraConfigError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CameraConfigError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, field: str) -> str | None:
    return None if value is None else _string(value, field)


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise CameraConfigError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CameraConfigError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise CameraConfigError(f"{field} must be a finite number")
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CameraConfigError(f"{field} must be a positive integer")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise CameraConfigError(f"{field} must be a boolean")
    return value


def _clock_minute(value: Any, field: str) -> int:
    if not isinstance(value, str):
        raise CameraConfigError(f"{field} must use HH:MM")
    parts = value.split(":")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise CameraConfigError(f"{field} must use HH:MM")
    hour, minute = (int(part) for part in parts)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise CameraConfigError(f"{field} must use HH:MM")
    return hour * 60 + minute


def _weekday(value: Any, field: str) -> int:
    names = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    if isinstance(value, str) and value.lower() in names:
        return names[value.lower()]
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 6:
        return value
    raise CameraConfigError(f"{field} must be a weekday name or integer from 0 to 6")


def _validate_threshold(value: float, field: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise CameraConfigError(f"{field} must be between 0 and 1")


def _points_mapping(points: Sequence[NormalizedPoint] | None) -> list[list[float]]:
    return [[point.x, point.y] for point in points or ()]


def _zone_mapping(zone: PolygonZone) -> dict[str, Any]:
    return {"id": zone.zone_id, "points": _points_mapping(zone.points)}


def _restricted_zone_mapping(zone: RestrictedZone) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": zone.zone_id,
        "points": _points_mapping(zone.points),
        "entry_dwell_seconds": zone.entry_dwell_seconds,
        "exit_grace_seconds": zone.exit_grace_seconds,
        "alert_cooldown_seconds": zone.alert_cooldown_seconds,
    }
    if zone.active_schedule is not None:
        schedule = zone.active_schedule
        result["active_schedule"] = {
            "start": f"{schedule.start_minute // 60:02d}:{schedule.start_minute % 60:02d}",
            "end": f"{schedule.end_minute // 60:02d}:{schedule.end_minute % 60:02d}",
            "weekdays": list(schedule.weekdays),
            "timezone": schedule.timezone,
        }
    return result


def _line_mapping(line: CountingLine) -> dict[str, Any]:
    return {
        "id": line.line_id,
        "start": list(line.start.as_tuple()),
        "end": list(line.end.as_tuple()),
        "positive_label": line.positive_label,
        "negative_label": line.negative_label,
        "hysteresis": line.hysteresis,
    }


def _queue_mapping(queue: QueueRegion) -> dict[str, Any]:
    return {
        "id": queue.queue_id,
        "polygon": _points_mapping(queue.polygon),
        "service_point": {
            "point": list(queue.service_point.point.as_tuple()),
            "label": queue.service_point.label,
        },
        "overflow_threshold": queue.overflow_threshold,
    }
