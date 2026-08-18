"""Environment-backed settings for the location-accumulation fleet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import os


# Dashboard accumulation is intentionally cheaper than on-demand live jobs.
# One processed frame every two seconds (0.5 FPS) is the required operating point.
FLEET_FPS = 0.5
FLEET_INTERVAL_SECONDS = 1.0 / FLEET_FPS


def _flag(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _positive_float(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _positive_int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


@dataclass(frozen=True, slots=True)
class FleetSettings:
    """Runtime configuration for continuous location analytics."""

    enabled: bool
    fps: float
    interval_seconds: float
    processing_width: int
    refresh_seconds: float
    max_cameras: int
    reconnect_seconds: float
    database_url: str | None
    mediamtx_rtsp_url: str | None
    expected_samples_per_minute: int
    spatial_publish_seconds: float

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> "FleetSettings":
        environment = os.environ if environ is None else environ
        fps = min(FLEET_FPS, _positive_float(environment.get("VIDEO_ANALYTICS_FLEET_FPS"), FLEET_FPS))
        interval = 1.0 / fps
        return cls(
            enabled=_flag(environment.get("VIDEO_ANALYTICS_FLEET_ENABLED"), False),
            fps=fps,
            interval_seconds=interval,
            processing_width=_positive_int(
                environment.get("VIDEO_ANALYTICS_PROCESSING_WIDTH"), 1280
            ),
            refresh_seconds=_positive_float(
                environment.get("VIDEO_ANALYTICS_FLEET_REFRESH_SECONDS"), 30.0
            ),
            max_cameras=_positive_int(
                environment.get("VIDEO_ANALYTICS_FLEET_MAX_CAMERAS"), 32
            ),
            reconnect_seconds=_positive_float(
                environment.get("VIDEO_ANALYTICS_FLEET_RECONNECT_SECONDS"), 5.0
            ),
            database_url=(
                environment.get("ANALYTICS_DATABASE_URL")
                or environment.get("DATABASE_URL")
                or None
            ),
            mediamtx_rtsp_url=environment.get("MEDIAMTX_RTSP_URL") or None,
            expected_samples_per_minute=max(1, round(fps * 60.0)),
            spatial_publish_seconds=_positive_float(
                environment.get("VIDEO_ANALYTICS_SPATIAL_PUBLISH_SECONDS"), 30.0
            ),
        )
