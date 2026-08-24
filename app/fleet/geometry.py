"""Build validated camera YAML-shaped configs for fleet analytics."""

from __future__ import annotations

from typing import Any

from app.fleet.catalog import FleetCamera, MappedZone
from app.fleet.settings import FleetSettings
from app.geometry.config import CameraConfig


def _points(zone: MappedZone) -> list[list[float]]:
    return [[point[0], point[1]] for point in zone.points]


def _centroid(zone: MappedZone) -> list[float]:
    xs = [point[0] for point in zone.points]
    ys = [point[1] for point in zone.points]
    return [sum(xs) / len(xs), sum(ys) / len(ys)]


def _queue_mapping(zone: MappedZone) -> dict[str, Any]:
    return {
        "id": zone.zone_id,
        "polygon": _points(zone),
        "service_point": {"point": _centroid(zone), "label": "service"},
        "overflow_threshold": 10,
        "minimum_dwell_seconds": 1.0,
        "maximum_speed_pixels_per_second": 80.0,
        "gap_tolerance_seconds": 2.0,
    }


def _restricted_mapping(zone: MappedZone) -> dict[str, Any]:
    return {
        "id": zone.zone_id,
        "points": _points(zone),
        "entry_dwell_seconds": 1.0,
        "exit_grace_seconds": 0.5,
        "alert_cooldown_seconds": 30.0,
    }


def camera_config_mapping(camera: FleetCamera, settings: FleetSettings) -> dict[str, Any]:
    enabled = ["occupancy", "heatmap"]
    analytics: dict[str, Any] = {
        "enabled": enabled,
        "occupancy_zones": [
            {
                "id": "frame",
                "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            }
        ],
    }
    if camera.queues:
        enabled.extend(["queue", "speed"])
        analytics["queues"] = [_queue_mapping(zone) for zone in camera.queues]
    if camera.restricted_zones:
        enabled.append("restricted_area")
        analytics["restricted_zones"] = [_restricted_mapping(zone) for zone in camera.restricted_zones]
    gap = max(4.0, settings.interval_seconds * 2.0)
    lost_track_buffer = max(3, round(8.0 * settings.fps))
    return {
        "camera": {"id": camera.camera_id, "name": camera.name, "source": camera.stream_url},
        "analytics": analytics,
        "tracker": {"lost_track_buffer": lost_track_buffer},
        "speed": {"window_seconds": max(4.0, settings.interval_seconds * 2.0)},
        "heatmap": {"max_sample_gap_seconds": gap, "grid_size": [64, 36]},
    }


def build_camera_config(camera: FleetCamera, settings: FleetSettings) -> CameraConfig:
    return CameraConfig.from_mapping(camera_config_mapping(camera, settings))
