"""Automatic near-vertical grouping of confirmed person tracks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from app.core.models import TrackObservation


@dataclass(frozen=True, slots=True)
class VerticalQueueConfig:
    """Image-space thresholds for automatic vertical-row grouping."""

    camera_id: str
    frame_size: tuple[int, int]
    maximum_center_distance_fraction: float = 0.08
    minimum_people: int = 2
    row_match_distance_fraction: float = 0.12

    def __post_init__(self) -> None:
        width, height = self.frame_size
        if not self.camera_id.strip():
            raise ValueError("camera_id must be non-empty")
        if width <= 0 or height <= 0:
            raise ValueError("frame width and height must be positive")
        for value, name in (
            (self.maximum_center_distance_fraction, "maximum_center_distance_fraction"),
            (self.row_match_distance_fraction, "row_match_distance_fraction"),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if (
            isinstance(self.minimum_people, bool)
            or not isinstance(self.minimum_people, int)
            or self.minimum_people <= 0
        ):
            raise ValueError("minimum_people must be a positive integer")


@dataclass(frozen=True, slots=True)
class VerticalQueueRow:
    row_id: int
    center_x: float
    top_y: float
    bottom_y: float
    track_ids: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.track_ids)


@dataclass(frozen=True, slots=True)
class VerticalQueueSnapshot:
    camera_id: str
    timestamp: float
    rows: tuple[VerticalQueueRow, ...]

    def row_for_track(self, track_id: int) -> VerticalQueueRow | None:
        return next((row for row in self.rows if track_id in row.track_ids), None)


class VerticalQueueAnalyzer:
    """Cluster people by bbox-center X and retain stable row identities.

    This is automatic geometric grouping, not evidence that people are truly
    queueing. Membership is recalculated on every frame.
    """

    def __init__(self, configs: Sequence[VerticalQueueConfig]) -> None:
        self._configs = {config.camera_id: config for config in configs}
        if len(self._configs) != len(configs):
            raise ValueError("camera IDs must be unique")
        self._row_centers: dict[str, dict[int, float]] = {}
        self._next_row_ids: dict[str, int] = {}
        self._snapshots: dict[str, VerticalQueueSnapshot] = {}
        self.reset()

    def reset(self, camera_id: str | None = None) -> None:
        """Clear stable row IDs globally or for one camera."""

        if camera_id is not None and camera_id not in self._configs:
            raise KeyError(f"unknown camera: {camera_id}")
        targets = self._configs if camera_id is None else (camera_id,)
        for target in targets:
            self._row_centers[target] = {}
            self._next_row_ids[target] = 1
            self._snapshots[target] = VerticalQueueSnapshot(target, 0.0, ())

    def update(
        self,
        camera_id: str,
        observations: Sequence[TrackObservation],
        *,
        timestamp: float | None = None,
    ) -> VerticalQueueSnapshot:
        """Recompute vertical rows from confirmed observations in one frame."""

        if camera_id not in self._configs:
            raise KeyError(f"unknown camera: {camera_id}")
        if any(item.camera_id != camera_id for item in observations):
            raise ValueError("all observations must match the updated camera")
        previous_timestamp = self._snapshots[camera_id].timestamp
        event_time = (
            timestamp
            if timestamp is not None
            else max(
                (item.timestamp for item in observations),
                default=previous_timestamp,
            )
        )
        if not math.isfinite(event_time):
            raise ValueError("timestamp must be finite")
        if event_time < previous_timestamp:
            raise ValueError("timestamps must be monotonic per camera")

        config = self._configs[camera_id]
        confirmed = sorted(
            (item for item in observations if item.confirmed),
            key=lambda item: (_center_x(item), item.track_id),
        )
        maximum_distance = (
            config.maximum_center_distance_fraction * config.frame_size[0]
        )
        clusters: list[list[TrackObservation]] = []
        for observation in confirmed:
            if not clusters:
                clusters.append([observation])
                continue
            cluster_center = sum(_center_x(item) for item in clusters[-1]) / len(
                clusters[-1]
            )
            if abs(_center_x(observation) - cluster_center) <= maximum_distance:
                clusters[-1].append(observation)
            else:
                clusters.append([observation])
        clusters = [
            cluster for cluster in clusters if len(cluster) >= config.minimum_people
        ]

        centers = [
            sum(_center_x(item) for item in cluster) / len(cluster)
            for cluster in clusters
        ]
        row_ids = self._match_row_ids(camera_id, centers)
        rows = tuple(
            sorted(
                (
                    VerticalQueueRow(
                        row_id,
                        center,
                        min(item.xyxy[1] for item in cluster),
                        max(item.xyxy[3] for item in cluster),
                        tuple(sorted(item.track_id for item in cluster)),
                    )
                    for row_id, center, cluster in zip(row_ids, centers, clusters)
                ),
                key=lambda row: row.center_x,
            )
        )
        snapshot = VerticalQueueSnapshot(camera_id, event_time, rows)
        self._snapshots[camera_id] = snapshot
        self._row_centers[camera_id] = {
            row.row_id: row.center_x for row in snapshot.rows
        }
        return snapshot

    def snapshot(self, camera_id: str) -> VerticalQueueSnapshot:
        if camera_id not in self._configs:
            raise KeyError(f"unknown camera: {camera_id}")
        return self._snapshots[camera_id]

    def _match_row_ids(self, camera_id: str, centers: Sequence[float]) -> list[int]:
        config = self._configs[camera_id]
        maximum_distance = config.row_match_distance_fraction * config.frame_size[0]
        previous = self._row_centers[camera_id]
        assignments: dict[int, int] = {}
        used_rows: set[int] = set()
        for _distance, center_index, row_id in sorted(
            (
                (abs(center - old_center), center_index, row_id)
                for center_index, center in enumerate(centers)
                for row_id, old_center in previous.items()
                if abs(center - old_center) <= maximum_distance
            )
        ):
            if center_index in assignments or row_id in used_rows:
                continue
            assignments[center_index] = row_id
            used_rows.add(row_id)
        result: list[int] = []
        for index in range(len(centers)):
            row_id = assignments.get(index)
            if row_id is None:
                row_id = self._next_row_ids[camera_id]
                self._next_row_ids[camera_id] += 1
            result.append(row_id)
        return result


def _center_x(observation: TrackObservation) -> float:
    return (observation.xyxy[0] + observation.xyxy[2]) / 2.0
