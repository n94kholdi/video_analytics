"""Configuration-based tracker construction.

New trackers register here. Dashboard and CLIs consume `available_trackers()`
so additional adapters do not require UI rebuilds beyond fetching this catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Callable, Mapping

from app.core.config import AppSettings
from app.tracking.base import BaseTracker
from app.tracking.botsort_adapter import BoTSortAdapter
from app.tracking.bytetrack import ByteTrackAdapter
from app.tracking.deepocsort_adapter import DeepOCSortAdapter
from app.tracking.stabletrack_adapter import StableTrackAdapter


@dataclass(frozen=True, slots=True)
class TrackerSpec:
    type: str
    label: str
    description: str
    factory: Callable[..., BaseTracker]
    requires_reid: bool = False


TRACKER_REGISTRY: dict[str, TrackerSpec] = {
    "bytetrack": TrackerSpec(
        "bytetrack",
        "ByteTrack",
        "Baseline motion tracker. Existing production behavior.",
        ByteTrackAdapter,
    ),
    "stabletrack": TrackerSpec(
        "stabletrack",
        "StableTrack",
        "Low-frequency association (BBD + optional ReID) for 0.5 FPS processing.",
        StableTrackAdapter,
        requires_reid=False,
    ),
    "deepocsort": TrackerSpec(
        "deepocsort",
        "Deep OC-SORT",
        "Observation-centric motion plus adaptive appearance association.",
        DeepOCSortAdapter,
        requires_reid=False,
    ),
    "botsort": TrackerSpec(
        "botsort",
        "BoT-SORT",
        "ByteTrack associations plus camera-motion compensation and optional ReID fusion.",
        BoTSortAdapter,
        requires_reid=False,
    ),
}


def available_trackers() -> tuple[TrackerSpec, ...]:
    return tuple(TRACKER_REGISTRY.values())


def available_tracker_types() -> tuple[str, ...]:
    return tuple(TRACKER_REGISTRY)


def public_tracker_catalog() -> list[dict[str, object]]:
    return [
        {
            "type": item.type,
            "label": item.label,
            "description": item.description,
            "requires_reid": item.requires_reid,
        }
        for item in available_trackers()
    ]


def create_tracker(
    tracker_type: str | None = None,
    *,
    settings: AppSettings | None = None,
    **kwargs: Any,
) -> BaseTracker:
    """Build a tracker from YAML/settings plus explicit overrides."""

    selected = (tracker_type or (settings.tracker_type if settings is not None else None) or "bytetrack")
    selected = selected.strip().lower()
    spec = TRACKER_REGISTRY.get(selected)
    if spec is None:
        known = ", ".join(available_tracker_types())
        raise ValueError(f"unknown tracker type {selected!r}; expected one of: {known}")
    parameters = _default_kwargs(settings)
    parameters.update({key: value for key, value in kwargs.items() if value is not None})
    return spec.factory(**_accepted_kwargs(spec.factory, parameters))


def _default_kwargs(settings: AppSettings | None) -> dict[str, Any]:
    if settings is None:
        return {}
    reid_model: Path | None = None
    return {
        "activation_threshold": settings.tracker_activation_threshold,
        "lost_track_buffer": settings.tracker_lost_track_buffer,
        "match_threshold": settings.tracker_match_threshold,
        "history_size": settings.tracker_history_size,
        "reid_providers": settings.onnx_providers,
        "reid_model": reid_model,
        "bbd_threshold": settings.tracker_bbd_threshold,
        "iou_threshold": settings.tracker_stable_iou_threshold,
        "reid_similarity_threshold": settings.tracker_reid_high_threshold,
        "reid_low_threshold": settings.tracker_reid_low_threshold,
        "max_age_seconds": settings.tracker_max_age_seconds,
        "use_visual_tracking": settings.tracker_use_visual_tracking,
        "inertia": settings.tracker_inertia,
        "w_association_emb": settings.tracker_w_association_emb,
        "alpha_fixed_emb": settings.tracker_alpha_fixed_emb,
        "aw_param": settings.tracker_aw_param,
        "delta_t_seconds": settings.tracker_delta_t_seconds,
        "use_cmc": settings.tracker_use_cmc,
        "proximity_thresh": settings.tracker_proximity_thresh,
        "track_low_threshold": settings.tracker_track_low_threshold,
        "new_track_threshold": settings.tracker_new_track_threshold,
        "embedding_alpha": settings.tracker_embedding_alpha,
    }


def _accepted_kwargs(factory: Callable[..., BaseTracker], values: Mapping[str, Any]) -> dict[str, Any]:
    parameters = signature(factory).parameters
    if any(item.kind is Parameter.VAR_KEYWORD for item in parameters.values()):
        return dict(values)
    allowed = {name for name in parameters if name != "self"}
    return {name: value for name, value in values.items() if name in allowed}


def register_tracker(spec: TrackerSpec) -> None:
    """Tests and future adapters register additional tracker types here."""

    TRACKER_REGISTRY[spec.type] = spec
