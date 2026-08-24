"""Compact per-frame metrics for the management minute publisher."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from app.analytics.heatmap import CrowdedRegion, HeatmapSnapshot
from app.analytics.queue import QueueStatus
from app.analytics.restricted_area import RestrictedAreaSnapshot
from app.core.models import Event


def crowded_regions(regions: Sequence[CrowdedRegion]) -> list[dict[str, object]]:
    return [
        {
            "region_id": region.region_id,
            "row": region.row,
            "column": region.column,
            "normalized_bounds": list(region.normalized_bounds),
            "average_occupancy": region.average_occupancy,
        }
        for region in regions
    ]


def management_spatial_layers(occupancy: np.ndarray, dwell_seconds: np.ndarray) -> dict[str, list[dict[str, float]]]:
    occupancy_points = _grid_points(occupancy)
    dwell_points = _grid_points(dwell_seconds / 60.0)
    maximum = max((point["value"] for point in occupancy_points), default=1.0) or 1.0
    congestion = [
        {**point, "value": point["value"] * 100 / maximum, "intensity": point["value"] / maximum}
        for point in occupancy_points
    ]
    return {
        "occupancy": occupancy_points,
        "dwell": dwell_points,
        "traffic": occupancy_points,
        "congestion": congestion,
    }


def _grid_points(values: np.ndarray) -> list[dict[str, float]]:
    rows, columns = values.shape
    points: list[dict[str, float]] = []
    maximum = float(np.max(values)) if values.size else 0.0
    for row in range(3):
        y1, y2 = round(row * rows / 3), round((row + 1) * rows / 3)
        for column in range(4):
            x1, x2 = round(column * columns / 4), round((column + 1) * columns / 4)
            value = float(np.mean(values[y1:y2, x1:x2])) if y2 > y1 and x2 > x1 else 0.0
            points.append(
                {
                    "x": (column + 0.5) * 25,
                    "y": (row + 0.5) * 100 / 3,
                    "value": value,
                    "intensity": value / maximum if maximum > 0 else 0.0,
                }
            )
    return points


def live_metrics(
    *,
    current_people: int,
    unique_people: int,
    active_tracks: int,
    entries: int,
    exits: int,
    occupancy: Mapping[str, int],
    restricted: RestrictedAreaSnapshot | None,
    restricted_violations: int,
    queue_statuses: Sequence[QueueStatus],
    crowded: Sequence[CrowdedRegion],
    ground: HeatmapSnapshot | None,
    include_spatial_layers: bool,
    processing_fps: float,
    frame_count: int,
    active_tracker: str | None = None,
) -> dict[str, object]:
    queue_lengths = [item.raw_count for item in queue_statuses]
    queue_waits = [
        item.approximate_current_waiting_seconds
        for item in queue_statuses
        if item.approximate_current_waiting_seconds is not None
    ]
    queue_speeds = [
        item.average_speed_pixels_per_second
        for item in queue_statuses
        if item.average_speed_pixels_per_second is not None
    ]
    return {
        "current_people": current_people,
        "total_unique_people": unique_people,
        "active_tracks": active_tracks,
        "entry_count": entries,
        "exit_count": exits,
        "zone_occupancy": dict(occupancy),
        "restricted_occupancy": restricted.current_tracks if restricted is not None else None,
        "restricted_entries": restricted.cumulative_entries if restricted is not None else None,
        "restricted_exits": restricted.cumulative_exits if restricted is not None else None,
        "restricted_violations": restricted_violations,
        "queue_length": sum(queue_lengths) if queue_lengths else None,
        "queue_wait_seconds": sum(queue_waits) / len(queue_waits) if queue_waits else None,
        "queue_speed": sum(queue_speeds) / len(queue_speeds) if queue_speeds else None,
        "queue_details": {
            item.queue_id: {
                "people": item.raw_count,
                "current_wait_seconds": item.approximate_current_waiting_seconds,
                "completed_wait_count": item.completed_wait_count,
                "last_completed_wait_seconds": item.last_completed_waiting_seconds,
                "overflow": item.overflow,
                "average_speed_pixels_per_second": item.average_speed_pixels_per_second,
                "average_speed_metres_per_second": item.average_speed_metres_per_second,
            }
            for item in queue_statuses
        }
        if queue_statuses
        else None,
        "processing_fps": processing_fps,
        "frame_count": frame_count,
        "active_tracker": active_tracker,
        "top_crowded_regions": crowded_regions(crowded) if crowded else None,
        "management_spatial_layers": (
            management_spatial_layers(ground.occupancy, ground.dwell_seconds)
            if include_spatial_layers and ground is not None
            else None
        ),
    }


def collect_events(*groups: Sequence[Event] | None) -> list[Event]:
    events: list[Event] = []
    for group in groups:
        if group:
            events.extend(group)
    return events
