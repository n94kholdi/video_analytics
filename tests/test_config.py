"""Configuration loading and validation tests."""

from pathlib import Path

import pytest

from app.core.config import AppSettings, ConfigError, PROJECT_ROOT, load_settings


def valid_mapping() -> dict[str, object]:
    return {
        "app": {
            "name": "test-analytics",
            "environment": "test",
            "log_level": "INFO",
        },
        "paths": {
            "output_dir": "outputs/test",
            "database_path": "outputs/test.sqlite3",
        },
        "onnx": {
            "providers": ["CPUExecutionProvider"],
            "detector_model": None,
        },
        "detector": {
            "confidence_threshold": 0.4,
            "iou_threshold": 0.7,
        },
        "tracker": {
            "activation_threshold": 0.4,
            "lost_track_buffer": 30,
            "match_threshold": 0.3,
            "history_size": 90,
        },
    }


def test_default_configuration_loads() -> None:
    settings = load_settings(environ={})

    assert settings.name == "video-analytics-mvp"
    assert settings.environment == "development"
    assert settings.onnx_providers == ("CPUExecutionProvider",)
    assert settings.detector_model == (
        PROJECT_ROOT.parent.parent
        / "All_weights"
        / "Weights_final"
        / "HumanDetection_light_input_640.onnx"
    )
    assert settings.reid_model == (
        PROJECT_ROOT
        / "All_weights"
        / "Weights_final"
        / "Tracking_osnet_x0_25_msmt17.onnx"
    )
    assert settings.detector_confidence_threshold == 0.4
    assert settings.detector_iou_threshold == 0.7
    assert settings.tracker_activation_threshold == 0.4
    assert settings.tracker_lost_track_buffer == 30
    assert settings.tracker_match_threshold == 0.3
    assert settings.tracker_history_size == 90
    assert settings.tracker_type == "bytetrack"
    assert settings.tracker_bbd_threshold == 16.0
    assert settings.tracker_inertia == 0.2
    assert settings.tracker_w_association_emb == 0.75
    assert settings.tracker_delta_t_seconds == 2.0
    assert settings.tracker_use_cmc is False
    assert settings.output_dir == PROJECT_ROOT / "outputs"


def test_relative_paths_resolve_from_project_root() -> None:
    settings = AppSettings.from_mapping(valid_mapping(), base_dir=PROJECT_ROOT)

    assert settings.output_dir == PROJECT_ROOT / "outputs" / "test"
    assert settings.database_path == PROJECT_ROOT / "outputs" / "test.sqlite3"


@pytest.mark.parametrize("tracker_name", ["stabletrack", "deepocsort"])
def test_environment_overrides_are_validated(tracker_name: str) -> None:
    settings = load_settings(
        environ={
            "VIDEO_ANALYTICS_LOG_LEVEL": "debug",
            "VIDEO_ANALYTICS_ONNX_PROVIDERS": (
                "CUDAExecutionProvider, CPUExecutionProvider"
            ),
            "VIDEO_ANALYTICS_CONFIDENCE_THRESHOLD": "0.55",
            "VIDEO_ANALYTICS_IOU_THRESHOLD": "0.60",
            "VIDEO_ANALYTICS_TRACKER": tracker_name,
        }
    )

    assert settings.log_level == "DEBUG"
    assert settings.onnx_providers == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    assert settings.detector_confidence_threshold == 0.55
    assert settings.detector_iou_threshold == 0.60
    assert settings.tracker_type == tracker_name


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("app", "name", "", "app.name"),
        ("app", "environment", "staging", "app.environment"),
        ("app", "log_level", "TRACE", "app.log_level"),
        ("paths", "database_path", "outputs/data.txt", "database_path"),
        ("onnx", "providers", [], "onnx.providers"),
        (
            "detector",
            "confidence_threshold",
            1.1,
            "detector.confidence_threshold",
        ),
        ("detector", "iou_threshold", -0.1, "detector.iou_threshold"),
        ("tracker", "activation_threshold", 1.1, "tracker.activation_threshold"),
        ("tracker", "lost_track_buffer", 0, "tracker.lost_track_buffer"),
        ("tracker", "match_threshold", -0.1, "tracker.match_threshold"),
        ("tracker", "history_size", 0, "tracker.history_size"),
        ("tracker", "type", "unknown", "tracker.type"),
    ],
)
def test_invalid_configuration_is_rejected(
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    config = valid_mapping()
    nested = config[section]
    assert isinstance(nested, dict)
    nested[field] = value

    with pytest.raises(ConfigError, match=message):
        AppSettings.from_mapping(config, base_dir=Path("/tmp/project"))


def test_missing_configuration_file_has_useful_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(ConfigError, match="configuration file not found"):
        load_settings(missing_path, environ={})
