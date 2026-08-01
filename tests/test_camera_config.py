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
