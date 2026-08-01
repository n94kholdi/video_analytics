"""Camera YAML model serialization and loading tests."""

from pathlib import Path

import pytest

from app.geometry.config import (
    CameraConfig,
    CameraConfigError,
    dump_camera_config,
    load_camera_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_example_camera_configuration_loads() -> None:
    config = load_camera_config(PROJECT_ROOT / "configs/cameras/example_lobby.yaml")

    assert config.camera_id == "lobby_east"
    assert config.analytics.counting_lines[0].line_id == "main_entrance"
    assert config.analytics.queues[0].service_point.label == "reception desk"
    assert config.calibration is not None


def test_camera_configuration_round_trip(tmp_path: Path) -> None:
    original = load_camera_config(PROJECT_ROOT / "configs/cameras/example_lobby.yaml")
    output = tmp_path / "nested" / "camera.yaml"

    dump_camera_config(original, output)
    loaded = load_camera_config(output)

    assert loaded == original


def test_invalid_polygon_has_contextual_configuration_error() -> None:
    mapping = {
        "camera": {"id": "cam", "name": "Camera", "source": 0},
        "analytics": {
            "enabled": ["occupancy"],
            "occupancy_zones": [
                {"id": "bad", "points": [[0, 0], [1, 1], [0, 1], [1, 0]]}
            ],
        },
    }

    with pytest.raises(CameraConfigError, match="polygon 'bad'.*self-intersect"):
        CameraConfig.from_mapping(mapping)


def test_out_of_range_normalized_coordinate_is_rejected() -> None:
    mapping = {
        "camera": {"id": "cam", "name": "Camera", "source": "rtsp://example"},
        "analytics": {
            "enabled": ["line_counting"],
            "counting_lines": [{"id": "line", "start": [-0.1, 0], "end": [1, 1]}],
        },
    }

    with pytest.raises(CameraConfigError, match="between 0 and 1"):
        CameraConfig.from_mapping(mapping)


def test_out_of_range_line_hysteresis_is_rejected() -> None:
    mapping = {
        "camera": {"id": "cam", "name": "Camera", "source": 0},
        "analytics": {
            "enabled": ["line_counting"],
            "counting_lines": [
                {
                    "id": "line",
                    "start": [0, 0.5],
                    "end": [1, 0.5],
                    "hysteresis": 1.1,
                }
            ],
        },
    }

    with pytest.raises(CameraConfigError, match="hysteresis must be between"):
        CameraConfig.from_mapping(mapping)


def test_restricted_zone_thresholds_and_schedule_are_parsed() -> None:
    mapping = {
        "camera": {"id": "cam", "name": "Camera", "source": 0},
        "analytics": {
            "enabled": ["restricted_area"],
            "restricted_zones": [
                {
                    "id": "secure",
                    "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "entry_dwell_seconds": 2.5,
                    "exit_grace_seconds": 0.75,
                    "alert_cooldown_seconds": 60,
                    "active_schedule": {
                        "start": "22:00",
                        "end": "06:00",
                        "weekdays": ["monday", "tuesday"],
                        "timezone": "UTC",
                    },
                }
            ],
        },
    }

    zone = CameraConfig.from_mapping(mapping).analytics.restricted_zones[0]

    assert zone.entry_dwell_seconds == 2.5
    assert zone.exit_grace_seconds == 0.75
    assert zone.alert_cooldown_seconds == 60
    assert zone.active_schedule is not None
    assert zone.active_schedule.weekdays == (0, 1)


def test_negative_restricted_zone_threshold_is_rejected() -> None:
    mapping = {
        "camera": {"id": "cam", "name": "Camera", "source": 0},
        "analytics": {
            "enabled": ["restricted_area"],
            "restricted_zones": [
                {
                    "id": "secure",
                    "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "entry_dwell_seconds": -1,
                }
            ],
        },
    }

    with pytest.raises(CameraConfigError, match="entry_dwell_seconds.*non-negative"):
        CameraConfig.from_mapping(mapping)
