"""Foot-point restricted-area detection with dwell, grace, and cooldown."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence
from uuid import uuid4

from app.core.models import Event, TrackObservation
from app.geometry.config import CameraConfig, RestrictedZone
from app.geometry.primitives import point_in_polygon
from app.storage import EventSink


class IntrusionState(str, Enum):
    """Externally visible restricted-area lifecycle states."""

    OUTSIDE = "outside"
    ENTERED = "entered"
    CONFIRMED = "confirmed"
    EXITED = "exited"


@dataclass(frozen=True, slots=True)
class CameraRestrictedAreaConfig:
    """Resolved restricted polygons and image size for one camera."""

    camera_id: str
    frame_size: tuple[int, int]
    zones: tuple[RestrictedZone, ...] = ()

    def __post_init__(self) -> None:
        width, height = self.frame_size
        if not self.camera_id.strip():
            raise ValueError("camera_id must be non-empty")
        if width <= 0 or height <= 0:
            raise ValueError("frame width and height must be positive")
        zone_ids = [zone.zone_id for zone in self.zones]
        if len(zone_ids) != len(set(zone_ids)):
            raise ValueError("restricted zone IDs must be unique per camera")

    @classmethod
    def from_camera_config(
        cls, config: CameraConfig, frame_size: tuple[int, int]
    ) -> "CameraRestrictedAreaConfig":
        """Select only enabled Phase 6 geometry from a camera config."""

        zones = (
            config.analytics.restricted_zones
            if "restricted_area" in config.analytics.enabled
            else ()
        )
        return cls(config.camera_id, frame_size, zones)


@dataclass(frozen=True, slots=True)
class IntrusionTrackStatus:
    zone_id: str
    track_id: int
    state: IntrusionState
    inside: bool
    entered_at: float
    confirmed_at: float | None


@dataclass(frozen=True, slots=True)
class RestrictedZoneStatus:
    zone_id: str
    active: bool
    current_tracks: int
    entered_tracks: int
    confirmed_tracks: int
    cumulative_entries: int
    cumulative_exits: int


@dataclass(frozen=True, slots=True)
class RestrictedAreaSnapshot:
    camera_id: str
    timestamp: float
    zones: tuple[RestrictedZoneStatus, ...]
    tracks: tuple[IntrusionTrackStatus, ...]

    def zone_for(self, zone_id: str) -> RestrictedZoneStatus:
        return next(item for item in self.zones if item.zone_id == zone_id)

    def state_for(self, zone_id: str, track_id: int) -> IntrusionState | None:
        if zone_id not in {zone.zone_id for zone in self.zones}:
            return None
        return next(
            (
                item.state
                for item in self.tracks
                if item.zone_id == zone_id and item.track_id == track_id
            ),
            IntrusionState.OUTSIDE,
        )


@dataclass(frozen=True, slots=True)
class RestrictedAreaResult:
    snapshot: RestrictedAreaSnapshot
    events: tuple[Event, ...]


@dataclass(slots=True)
class _IntrusionTrackState:
    status: IntrusionState
    entered_at: float
    inside: bool = True
    outside_since: float | None = None
    confirmed_at: float | None = None
    confirmation_event_emitted: bool = False


class RestrictedAreaDetector:
    """Stateful restricted-zone analytics for independent cameras."""

    def __init__(
        self,
        cameras: Sequence[CameraRestrictedAreaConfig],
        *,
        event_sink: EventSink | None = None,
    ) -> None:
        self._cameras = {camera.camera_id: camera for camera in cameras}
        if len(self._cameras) != len(cameras):
            raise ValueError("camera IDs must be unique")
        self._event_sink = event_sink
        self._states: dict[str, dict[tuple[str, int], _IntrusionTrackState]] = {}
        self._last_alerts: dict[str, dict[tuple[str, int], float]] = {}
        self._zone_totals: dict[str, dict[str, list[int]]] = {}
        self._timestamps: dict[str, float] = {}
        self.reset()

    def reset(self, camera_id: str | None = None) -> None:
        """Clear lifecycle and cooldown state globally or for one camera."""

        if camera_id is not None and camera_id not in self._cameras:
            raise KeyError(f"unknown camera: {camera_id}")
        targets = self._cameras if camera_id is None else (camera_id,)
        for target in targets:
            self._states[target] = {}
            self._last_alerts[target] = {}
            self._zone_totals[target] = {
                zone.zone_id: [0, 0] for zone in self._cameras[target].zones
            }
            self._timestamps[target] = 0.0

    def update(
        self,
        camera_id: str,
        observations: Sequence[TrackObservation],
        *,
        timestamp: float | None = None,
    ) -> RestrictedAreaResult:
        """Consume one camera frame and persist the events it produces."""

        if camera_id not in self._cameras:
            raise KeyError(f"unknown camera: {camera_id}")
        if any(item.camera_id != camera_id for item in observations):
            raise ValueError("all observations must match the updated camera")
        if timestamp is not None and not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        event_time = (
            timestamp
            if timestamp is not None
            else max(
                (item.timestamp for item in observations),
                default=self._timestamps[camera_id],
            )
        )
        if not math.isfinite(event_time):
            raise ValueError("observation timestamps must be finite")
        if event_time < self._timestamps[camera_id]:
            raise ValueError("timestamps must be monotonic per camera")
        self._timestamps[camera_id] = event_time

        confirmed = {item.track_id: item for item in observations if item.confirmed}
        events: list[Event] = []
        config = self._cameras[camera_id]
        states = self._states[camera_id]
        for zone in config.zones:
            active = (
                zone.active_schedule is None
                or zone.active_schedule.is_active(event_time)
            )
            zone_keys = tuple(key for key in states if key[0] == zone.zone_id)
            if not active:
                for key in zone_keys:
                    del states[key]
                continue

            polygon = zone.pixel_points(config.frame_size)
            inside_ids = {
                track_id
                for track_id, observation in confirmed.items()
                if point_in_polygon(observation.foot_point, polygon)
            }
            for track_id in sorted(inside_ids):
                events.extend(
                    self._observe_inside(camera_id, zone, track_id, event_time)
                )
            for key in tuple(key for key in states if key[0] == zone.zone_id):
                if key[1] not in inside_ids:
                    event = self._observe_outside(
                        camera_id, zone, key, event_time
                    )
                    if event is not None:
                        events.append(event)

        event_batch = tuple(events)
        if self._event_sink is not None:
            self._event_sink.persist(event_batch)
        return RestrictedAreaResult(self.snapshot(camera_id), event_batch)

    def snapshot(self, camera_id: str) -> RestrictedAreaSnapshot:
        """Return zone and track status without advancing state."""

        if camera_id not in self._cameras:
            raise KeyError(f"unknown camera: {camera_id}")
        timestamp = self._timestamps[camera_id]
        config = self._cameras[camera_id]
        states = self._states[camera_id]
        tracks = tuple(
            IntrusionTrackStatus(
                zone_id,
                track_id,
                state.status,
                state.inside,
                state.entered_at,
                state.confirmed_at,
            )
            for (zone_id, track_id), state in sorted(states.items())
        )
        zones = tuple(
            RestrictedZoneStatus(
                zone.zone_id,
                zone.active_schedule is None or zone.active_schedule.is_active(timestamp),
                sum(
                    state.status in {IntrusionState.ENTERED, IntrusionState.CONFIRMED}
                    for (zone_id, _), state in states.items()
                    if zone_id == zone.zone_id
                ),
                sum(
                    state.status is IntrusionState.ENTERED
                    for (zone_id, _), state in states.items()
                    if zone_id == zone.zone_id
                ),
                sum(
                    state.status is IntrusionState.CONFIRMED
                    for (zone_id, _), state in states.items()
                    if zone_id == zone.zone_id
                ),
                *self._zone_totals[camera_id][zone.zone_id],
            )
            for zone in config.zones
        )
        return RestrictedAreaSnapshot(camera_id, timestamp, zones, tracks)

    def _observe_inside(
        self,
        camera_id: str,
        zone: RestrictedZone,
        track_id: int,
        timestamp: float,
    ) -> list[Event]:
        key = (zone.zone_id, track_id)
        state = self._states[camera_id].get(key)
        events: list[Event] = []
        if state is None or state.status is IntrusionState.EXITED:
            state = _IntrusionTrackState(IntrusionState.ENTERED, timestamp)
            self._states[camera_id][key] = state
            self._zone_totals[camera_id][zone.zone_id][0] += 1
            events.append(
                self._event(
                    "restricted_area_entered", camera_id, zone.zone_id, track_id, timestamp
                )
            )
        else:
            state.inside = True
            state.outside_since = None

        if (
            state.status is IntrusionState.ENTERED
            and _elapsed_at_least(
                timestamp, state.entered_at, zone.entry_dwell_seconds
            )
        ):
            state.status = IntrusionState.CONFIRMED
            state.confirmed_at = timestamp

        if state.status is IntrusionState.CONFIRMED and not state.confirmation_event_emitted:
            last_alert = self._last_alerts[camera_id].get(key)
            if last_alert is None or _elapsed_at_least(
                timestamp, last_alert, zone.alert_cooldown_seconds
            ):
                state.confirmation_event_emitted = True
                self._last_alerts[camera_id][key] = timestamp
                events.append(
                    self._event(
                        "restricted_area_confirmed",
                        camera_id,
                        zone.zone_id,
                        track_id,
                        timestamp,
                        {"dwell_seconds": timestamp - state.entered_at},
                    )
                )
        return events

    def _observe_outside(
        self,
        camera_id: str,
        zone: RestrictedZone,
        key: tuple[str, int],
        timestamp: float,
    ) -> Event | None:
        state = self._states[camera_id][key]
        if state.status is IntrusionState.EXITED:
            return None
        if state.outside_since is None:
            state.outside_since = timestamp
        state.inside = False
        if not _elapsed_at_least(
            timestamp, state.outside_since, zone.exit_grace_seconds
        ):
            return None
        was_confirmed = state.status is IntrusionState.CONFIRMED
        state.status = IntrusionState.EXITED
        self._zone_totals[camera_id][zone.zone_id][1] += 1
        return self._event(
            "restricted_area_exited",
            camera_id,
            zone.zone_id,
            key[1],
            timestamp,
            {
                "confirmed": was_confirmed,
                "duration_seconds": timestamp - state.entered_at,
            },
        )

    @staticmethod
    def _event(
        event_type: str,
        camera_id: str,
        zone_id: str,
        track_id: int,
        timestamp: float,
        payload: dict[str, object] | None = None,
    ) -> Event:
        return Event(
            event_id=str(uuid4()),
            event_type=event_type,
            camera_id=camera_id,
            timestamp=timestamp,
            track_id=track_id,
            zone_id=zone_id,
            payload=payload,
        )


def _elapsed_at_least(timestamp: float, since: float, threshold: float) -> bool:
    """Compare source times without surprising boundary failures from float noise."""

    return timestamp - since >= threshold - 1e-9
