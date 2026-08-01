"""Current polygon occupancy and directional finite-line people counting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
from uuid import uuid4

from app.core.models import Event, TrackObservation
from app.geometry.config import CameraConfig, CountingLine, PolygonZone
from app.geometry.primitives import (
    LineCrossingDirection,
    detect_line_crossing,
    point_in_polygon,
    signed_line_side,
)

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class CameraCountingConfig:
    """Resolved counting geometry and image size for one camera."""

    camera_id: str
    frame_size: tuple[int, int]
    occupancy_zones: tuple[PolygonZone, ...] = ()
    counting_lines: tuple[CountingLine, ...] = ()

    def __post_init__(self) -> None:
        width, height = self.frame_size
        if not self.camera_id.strip():
            raise ValueError("camera_id must be non-empty")
        if width <= 0 or height <= 0:
            raise ValueError("frame width and height must be positive")
        identifiers = [zone.zone_id for zone in self.occupancy_zones]
        identifiers.extend(line.line_id for line in self.counting_lines)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("counting geometry IDs must be unique per camera")

    @classmethod
    def from_camera_config(
        cls, config: CameraConfig, frame_size: tuple[int, int]
    ) -> "CameraCountingConfig":
        """Select only Phase 5 geometry from a validated camera config."""

        enabled = set(config.analytics.enabled)
        return cls(
            config.camera_id,
            frame_size,
            config.analytics.occupancy_zones if "occupancy" in enabled else (),
            config.analytics.counting_lines if "line_counting" in enabled else (),
        )


@dataclass(frozen=True, slots=True)
class OccupancyCount:
    zone_id: str
    current: int


@dataclass(frozen=True, slots=True)
class LineCount:
    line_id: str
    entries: int
    exits: int


@dataclass(frozen=True, slots=True)
class CountingSnapshot:
    """Current and cumulative counting values for one camera."""

    camera_id: str
    timestamp: float
    occupancy: tuple[OccupancyCount, ...]
    lines: tuple[LineCount, ...]

    @property
    def current_occupancy(self) -> int:
        """Sum independent zone occupancies; overlapping zones count separately."""

        return sum(item.current for item in self.occupancy)

    @property
    def cumulative_entries(self) -> int:
        return sum(item.entries for item in self.lines)

    @property
    def cumulative_exits(self) -> int:
        return sum(item.exits for item in self.lines)

    def occupancy_for(self, zone_id: str) -> int:
        return next(item.current for item in self.occupancy if item.zone_id == zone_id)

    def line_for(self, line_id: str) -> LineCount:
        return next(item for item in self.lines if item.line_id == line_id)


@dataclass(frozen=True, slots=True)
class CountingResult:
    snapshot: CountingSnapshot
    events: tuple[Event, ...]


@dataclass(slots=True)
class _TrackLineState:
    stable_side: int | None = None
    stable_point: Point | None = None


class PeopleCounter:
    """Stateful, explicitly resettable counter for independent cameras."""

    def __init__(self, cameras: Sequence[CameraCountingConfig]) -> None:
        self._cameras = {camera.camera_id: camera for camera in cameras}
        if len(self._cameras) != len(cameras):
            raise ValueError("camera IDs must be unique")
        self._line_totals: dict[str, dict[str, list[int]]] = {}
        self._line_states: dict[str, dict[tuple[str, int], _TrackLineState]] = {}
        self._occupancy: dict[str, dict[str, int]] = {}
        self._timestamps: dict[str, float] = {}
        self.reset()

    def reset(self, camera_id: str | None = None) -> None:
        """Clear cumulative and per-track state globally or for one camera."""

        if camera_id is not None and camera_id not in self._cameras:
            raise KeyError(f"unknown camera: {camera_id}")
        targets = self._cameras if camera_id is None else (camera_id,)
        for target in targets:
            config = self._cameras[target]
            self._line_totals[target] = {
                line.line_id: [0, 0] for line in config.counting_lines
            }
            self._line_states[target] = {}
            self._occupancy[target] = {
                zone.zone_id: 0 for zone in config.occupancy_zones
            }
            self._timestamps[target] = 0.0

    def update(
        self,
        camera_id: str,
        observations: Sequence[TrackObservation],
        *,
        timestamp: float | None = None,
    ) -> CountingResult:
        """Consume one camera frame and return its latest metrics and events."""

        if camera_id not in self._cameras:
            raise KeyError(f"unknown camera: {camera_id}")
        if any(item.camera_id != camera_id for item in observations):
            raise ValueError("all observations must match the updated camera")
        if timestamp is not None and not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")

        # One track may only contribute once per frame. Tracker output is expected
        # to be unique, while this also makes accidental duplicates harmless.
        confirmed = {item.track_id: item for item in observations if item.confirmed}
        event_time = (
            timestamp
            if timestamp is not None
            else max((item.timestamp for item in observations), default=self._timestamps[camera_id])
        )
        self._timestamps[camera_id] = event_time
        config = self._cameras[camera_id]
        self._update_occupancy(camera_id, config, tuple(confirmed.values()))

        active_ids = set(confirmed)
        states = self._line_states[camera_id]
        for key in tuple(states):
            if key[1] not in active_ids:
                del states[key]

        events: list[Event] = []
        for line in config.counting_lines:
            start, end = line.pixel_points(config.frame_size)
            hysteresis_pixels = line.hysteresis * math.hypot(
                config.frame_size[0] - 1, config.frame_size[1] - 1
            )
            for observation in confirmed.values():
                event = self._update_line(
                    camera_id,
                    line,
                    start,
                    end,
                    hysteresis_pixels,
                    observation,
                )
                if event is not None:
                    events.append(event)
        return CountingResult(self.snapshot(camera_id), tuple(events))

    def snapshot(self, camera_id: str) -> CountingSnapshot:
        """Return metrics without advancing state."""

        if camera_id not in self._cameras:
            raise KeyError(f"unknown camera: {camera_id}")
        config = self._cameras[camera_id]
        return CountingSnapshot(
            camera_id,
            self._timestamps[camera_id],
            tuple(
                OccupancyCount(zone.zone_id, self._occupancy[camera_id][zone.zone_id])
                for zone in config.occupancy_zones
            ),
            tuple(
                LineCount(line.line_id, *self._line_totals[camera_id][line.line_id])
                for line in config.counting_lines
            ),
        )

    def _update_occupancy(
        self,
        camera_id: str,
        config: CameraCountingConfig,
        observations: Sequence[TrackObservation],
    ) -> None:
        for zone in config.occupancy_zones:
            polygon = zone.pixel_points(config.frame_size)
            self._occupancy[camera_id][zone.zone_id] = sum(
                point_in_polygon(item.foot_point, polygon) for item in observations
            )

    def _update_line(
        self,
        camera_id: str,
        line: CountingLine,
        start: Point,
        end: Point,
        hysteresis_pixels: float,
        observation: TrackObservation,
    ) -> Event | None:
        key = (line.line_id, observation.track_id)
        state = self._line_states[camera_id].setdefault(key, _TrackLineState())
        point = observation.foot_point
        length = math.dist(start, end)
        distance = signed_line_side(point, start, end) / length
        side = 1 if distance > hysteresis_pixels else -1 if distance < -hysteresis_pixels else 0
        if side == 0:
            return None
        if state.stable_side is None:
            state.stable_side = side
            state.stable_point = point
            return None
        if side == state.stable_side:
            state.stable_point = point
            return None

        crossing = detect_line_crossing(state.stable_point or point, point, start, end)
        state.stable_side = side
        state.stable_point = point
        if not crossing.crossed or crossing.direction is None:
            return None

        is_entry = crossing.direction is LineCrossingDirection.NEGATIVE_TO_POSITIVE
        totals = self._line_totals[camera_id][line.line_id]
        totals[0 if is_entry else 1] += 1
        direction = line.positive_label if is_entry else line.negative_label
        return Event(
            event_id=str(uuid4()),
            event_type="line_crossed",
            camera_id=camera_id,
            timestamp=observation.timestamp,
            track_id=observation.track_id,
            line_id=line.line_id,
            payload={
                "direction": direction,
                "side_transition": crossing.direction.value,
            },
        )
