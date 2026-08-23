"""Small, dependency-light application configuration loader."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_CAMERA_CONFIG_PATH = PROJECT_ROOT / "configs" / "cameras" / "example_lobby.yaml"
_ALLOWED_ENVIRONMENTS = frozenset({"development", "test", "production"})
_ALLOWED_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_ALLOWED_TRACKERS = frozenset({"bytetrack", "stabletrack", "deepocsort", "botsort"})


class ConfigError(ValueError):
    """Raised when application configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Validated settings required to bootstrap the application."""

    name: str
    environment: str
    log_level: str
    output_dir: Path
    database_path: Path
    onnx_providers: tuple[str, ...]
    detector_confidence_threshold: float
    detector_iou_threshold: float
    tracker_activation_threshold: float
    tracker_lost_track_buffer: int
    tracker_match_threshold: float
    tracker_history_size: int
    tracker_type: str = "bytetrack"
    tracker_bbd_threshold: float = 16.0
    tracker_stable_iou_threshold: float = 0.4
    tracker_reid_high_threshold: float = 0.65
    tracker_reid_low_threshold: float = 0.3
    tracker_max_age_seconds: float | None = None
    tracker_use_visual_tracking: bool = True
    tracker_inertia: float = 0.2
    tracker_w_association_emb: float = 0.75
    tracker_alpha_fixed_emb: float = 0.95
    tracker_aw_param: float = 0.5
    tracker_delta_t_seconds: float = 2.0
    tracker_use_cmc: bool = False
    tracker_proximity_thresh: float = 1.0
    tracker_track_low_threshold: float = 0.1
    tracker_new_track_threshold: float | None = None
    tracker_embedding_alpha: float = 0.9
    detector_model: Path | None = None
    reid_model: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        base_dir: Path = PROJECT_ROOT,
    ) -> "AppSettings":
        """Create settings from the YAML-shaped mapping used by the MVP."""

        app = _mapping(values, "app")
        paths = _mapping(values, "paths")
        onnx = _mapping(values, "onnx")
        detector = _mapping(values, "detector")
        tracker = _mapping(values, "tracker")

        name = _non_empty_string(app.get("name"), "app.name")
        environment = _non_empty_string(
            app.get("environment"), "app.environment"
        ).lower()
        if environment not in _ALLOWED_ENVIRONMENTS:
            allowed = ", ".join(sorted(_ALLOWED_ENVIRONMENTS))
            raise ConfigError(f"app.environment must be one of: {allowed}")

        log_level = _non_empty_string(app.get("log_level"), "app.log_level").upper()
        if log_level not in _ALLOWED_LOG_LEVELS:
            allowed = ", ".join(sorted(_ALLOWED_LOG_LEVELS))
            raise ConfigError(f"app.log_level must be one of: {allowed}")

        output_dir = _resolve_path(
            _non_empty_string(paths.get("output_dir"), "paths.output_dir"),
            base_dir,
        )
        database_path = _resolve_path(
            _non_empty_string(paths.get("database_path"), "paths.database_path"),
            base_dir,
        )
        if database_path.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
            raise ConfigError(
                "paths.database_path must end with .db, .sqlite, or .sqlite3"
            )

        providers_value = onnx.get("providers")
        if not isinstance(providers_value, list) or not providers_value:
            raise ConfigError("onnx.providers must be a non-empty list")
        providers = tuple(
            _non_empty_string(value, f"onnx.providers[{index}]")
            for index, value in enumerate(providers_value)
        )

        detector_value = onnx.get("detector_model")
        detector_model = None
        if detector_value is not None:
            detector_model = _resolve_path(
                _non_empty_string(detector_value, "onnx.detector_model"),
                base_dir,
            )
        reid_value = onnx.get("reid_model")
        reid_model = None
        if reid_value is not None:
            reid_model = _resolve_path(
                _non_empty_string(reid_value, "onnx.reid_model"),
                base_dir,
            )

        confidence_threshold = _threshold(
            detector.get("confidence_threshold"),
            "detector.confidence_threshold",
        )
        iou_threshold = _threshold(
            detector.get("iou_threshold"),
            "detector.iou_threshold",
        )
        tracker_activation_threshold = _threshold(
            tracker.get("activation_threshold"),
            "tracker.activation_threshold",
        )
        tracker_lost_track_buffer = _positive_int(
            tracker.get("lost_track_buffer"),
            "tracker.lost_track_buffer",
        )
        tracker_match_threshold = _threshold(
            tracker.get("match_threshold"),
            "tracker.match_threshold",
        )
        tracker_history_size = _positive_int(
            tracker.get("history_size"),
            "tracker.history_size",
        )
        tracker_type = str(tracker.get("type") or "bytetrack").strip().lower()
        if tracker_type not in _ALLOWED_TRACKERS:
            allowed = ", ".join(sorted(_ALLOWED_TRACKERS))
            raise ConfigError(f"tracker.type must be one of: {allowed}")
        tracker_bbd_threshold = _positive_float(
            tracker.get("bbd_threshold", 16.0),
            "tracker.bbd_threshold",
        )
        tracker_stable_iou_threshold = _threshold(
            tracker.get("iou_threshold", 0.4),
            "tracker.iou_threshold",
        )
        tracker_reid_high_threshold = _threshold(
            tracker.get("reid_high_threshold", 0.65),
            "tracker.reid_high_threshold",
        )
        tracker_reid_low_threshold = _threshold(
            tracker.get("reid_low_threshold", 0.3),
            "tracker.reid_low_threshold",
        )
        max_age_value = tracker.get("max_age_seconds")
        tracker_max_age_seconds = (
            None
            if max_age_value is None
            else _positive_float(max_age_value, "tracker.max_age_seconds")
        )
        tracker_use_visual_tracking = _boolean(
            tracker.get("use_visual_tracking", True),
            "tracker.use_visual_tracking",
        )
        tracker_inertia = _threshold(tracker.get("inertia", 0.2), "tracker.inertia")
        tracker_w_association_emb = _positive_float(
            tracker.get("w_association_emb", 0.75),
            "tracker.w_association_emb",
        )
        tracker_alpha_fixed_emb = _threshold(
            tracker.get("alpha_fixed_emb", 0.95),
            "tracker.alpha_fixed_emb",
        )
        tracker_aw_param = _threshold(tracker.get("aw_param", 0.5), "tracker.aw_param")
        tracker_delta_t_seconds = _positive_float(
            tracker.get("delta_t_seconds", 2.0),
            "tracker.delta_t_seconds",
        )
        tracker_use_cmc = _boolean(tracker.get("use_cmc", False), "tracker.use_cmc")
        tracker_proximity_thresh = _threshold(
            tracker.get("proximity_thresh", 1.0),
            "tracker.proximity_thresh",
        )
        tracker_track_low_threshold = _threshold(
            tracker.get("track_low_threshold", 0.1),
            "tracker.track_low_threshold",
        )
        new_track_value = tracker.get("new_track_threshold")
        tracker_new_track_threshold = (
            None
            if new_track_value is None
            else _threshold(new_track_value, "tracker.new_track_threshold")
        )
        tracker_embedding_alpha = _threshold(
            tracker.get("embedding_alpha", 0.9),
            "tracker.embedding_alpha",
        )

        return cls(
            name=name,
            environment=environment,
            log_level=log_level,
            output_dir=output_dir,
            database_path=database_path,
            onnx_providers=providers,
            detector_confidence_threshold=confidence_threshold,
            detector_iou_threshold=iou_threshold,
            tracker_activation_threshold=tracker_activation_threshold,
            tracker_lost_track_buffer=tracker_lost_track_buffer,
            tracker_match_threshold=tracker_match_threshold,
            tracker_history_size=tracker_history_size,
            tracker_type=tracker_type,
            tracker_bbd_threshold=tracker_bbd_threshold,
            tracker_stable_iou_threshold=tracker_stable_iou_threshold,
            tracker_reid_high_threshold=tracker_reid_high_threshold,
            tracker_reid_low_threshold=tracker_reid_low_threshold,
            tracker_max_age_seconds=tracker_max_age_seconds,
            tracker_use_visual_tracking=tracker_use_visual_tracking,
            tracker_inertia=tracker_inertia,
            tracker_w_association_emb=tracker_w_association_emb,
            tracker_alpha_fixed_emb=tracker_alpha_fixed_emb,
            tracker_aw_param=tracker_aw_param,
            tracker_delta_t_seconds=tracker_delta_t_seconds,
            tracker_use_cmc=tracker_use_cmc,
            tracker_proximity_thresh=tracker_proximity_thresh,
            tracker_track_low_threshold=tracker_track_low_threshold,
            tracker_new_track_threshold=tracker_new_track_threshold,
            tracker_embedding_alpha=tracker_embedding_alpha,
            detector_model=detector_model,
            reid_model=reid_model,
        )


def load_settings(
    path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> AppSettings:
    """Load YAML settings and apply supported environment overrides."""

    environment = os.environ if environ is None else environ
    selected_path = path or environment.get("VIDEO_ANALYTICS_CONFIG")
    config_path = (
        _resolve_path(str(selected_path), PROJECT_ROOT)
        if selected_path
        else DEFAULT_CONFIG_PATH
    )

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            raw_values = yaml.safe_load(config_file)
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {config_path}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not read configuration: {config_path}") from exc

    if not isinstance(raw_values, Mapping):
        raise ConfigError("configuration root must be a mapping")

    settings = AppSettings.from_mapping(raw_values, base_dir=PROJECT_ROOT)
    return _apply_environment_overrides(settings, environment)


def _apply_environment_overrides(
    settings: AppSettings,
    environment: Mapping[str, str],
) -> AppSettings:
    updates: dict[str, Any] = {}

    if value := environment.get("VIDEO_ANALYTICS_LOG_LEVEL"):
        level = value.strip().upper()
        if level not in _ALLOWED_LOG_LEVELS:
            allowed = ", ".join(sorted(_ALLOWED_LOG_LEVELS))
            raise ConfigError(
                f"VIDEO_ANALYTICS_LOG_LEVEL must be one of: {allowed}"
            )
        updates["log_level"] = level

    path_overrides = {
        "VIDEO_ANALYTICS_OUTPUT_DIR": "output_dir",
        "VIDEO_ANALYTICS_DATABASE_PATH": "database_path",
        "VIDEO_ANALYTICS_DETECTOR_MODEL": "detector_model",
        "VIDEO_ANALYTICS_REID_MODEL": "reid_model",
    }
    for variable, field_name in path_overrides.items():
        if value := environment.get(variable):
            updates[field_name] = _resolve_path(value.strip(), PROJECT_ROOT)

    if database_path := updates.get("database_path"):
        if database_path.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
            raise ConfigError(
                "VIDEO_ANALYTICS_DATABASE_PATH must end with "
                ".db, .sqlite, or .sqlite3"
            )

    if value := environment.get("VIDEO_ANALYTICS_ONNX_PROVIDERS"):
        providers = tuple(item.strip() for item in value.split(",") if item.strip())
        if not providers:
            raise ConfigError(
                "VIDEO_ANALYTICS_ONNX_PROVIDERS must contain at least one provider"
            )
        updates["onnx_providers"] = providers

    threshold_overrides = {
        "VIDEO_ANALYTICS_CONFIDENCE_THRESHOLD": (
            "detector_confidence_threshold",
            "confidence threshold",
        ),
        "VIDEO_ANALYTICS_IOU_THRESHOLD": (
            "detector_iou_threshold",
            "IoU threshold",
        ),
    }
    for variable, (field_name, label) in threshold_overrides.items():
        if value := environment.get(variable):
            updates[field_name] = _threshold(value, variable, label=label)

    if value := environment.get("VIDEO_ANALYTICS_TRACKER"):
        tracker_type = value.strip().lower()
        if tracker_type not in _ALLOWED_TRACKERS:
            allowed = ", ".join(sorted(_ALLOWED_TRACKERS))
            raise ConfigError(f"VIDEO_ANALYTICS_TRACKER must be one of: {allowed}")
        updates["tracker_type"] = tracker_type

    return replace(settings, **updates) if updates else settings


def _mapping(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty string")
    return value.strip()


def _threshold(value: Any, field_name: str, *, label: str = "threshold") -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a number between 0 and 1") from exc
    if not 0.0 <= result <= 1.0:
        raise ConfigError(f"{field_name} {label} must be between 0 and 1")
    return result


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field_name} must be a positive integer")
    return value


def _positive_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a positive number") from exc
    if not result > 0:
        raise ConfigError(f"{field_name} must be a positive number")
    return result


def _boolean(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0", "yes", "no"}:
        return value.strip().lower() in {"true", "1", "yes"}
    raise ConfigError(f"{field_name} must be a boolean")


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()
