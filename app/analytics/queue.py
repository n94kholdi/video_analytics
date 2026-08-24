"""Configured, heuristic queue membership and waiting-time analytics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence
from uuid import uuid4

from app.core.models import Event, TrackObservation
from app.analytics.speed import queue_progress_speed
from app.geometry.calibration import ImageToGroundProjector
from app.geometry.config import CameraConfig, QueueRegion
from app.geometry.primitives import bbox_polygon_coverage_ratio, point_in_polygon
from app.storage import EventSink

MINIMUM_BBOX_COVERAGE_RATIO = 0.75
"""Fraction of a person's bbox area that must overlap a queue to count as inside.

Foot-point containment alone is too easy to satisfy at a region's edge; this
also requires most of the person's bbox to actually be inside the polygon.
"""


class QueueTrackState(str, Enum):
    """Visible lifecycle states for tracks being evaluated as queue members."""

    CANDIDATE = "candidate"
    MEMBER = "member"


@dataclass(frozen=True, slots=True)
class CameraQueueConfig:
    """Resolved configured queues and image size for one camera."""

    camera_id: str
    frame_size: tuple[int, int]
    queues: tuple[QueueRegion, ...] = ()
    projector: ImageToGroundProjector | None = None

    def __post_init__(self) -> None:
        width, height = self.frame_size
        if not self.camera_id.strip():
            raise ValueError("camera_id must be non-empty")
        if width <= 0 or height <= 0:
            raise ValueError("frame width and height must be positive")
        queue_ids = [queue.queue_id for queue in self.queues]
        if len(queue_ids) != len(set(queue_ids)):
            raise ValueError("queue IDs must be unique per camera")

    @classmethod
    def from_camera_config(
        cls, config: CameraConfig, frame_size: tuple[int, int]
    ) -> "CameraQueueConfig":
        """Select only manually configured, enabled queues."""

        queues = config.analytics.queues if "queue" in config.analytics.enabled else ()
        return cls(
            config.camera_id,
            frame_size,
            queues,
            ImageToGroundProjector.from_calibration(config.calibration, frame_size),
        )


@dataclass(frozen=True, slots=True)
class QueueTrackStatus:
    queue_id: str
    track_id: int
    state: QueueTrackState
    observed_this_frame: bool
    candidate_since: float
    joined_at: float | None
    waiting_seconds: float
    smoothed_speed_pixels_per_second: float | None
    speed_metres_per_second: float | None
    progress_speed_pixels_per_second: float | None
    progress_speed_metres_per_second: float | None


@dataclass(frozen=True, slots=True)
class QueueStatus:
    queue_id: str
    raw_count: int
    smoothed_count: float
    present: bool
    overflow: bool
    approximate_current_waiting_seconds: float | None
    completed_wait_count: int
    average_completed_waiting_seconds: float | None
    last_completed_waiting_seconds: float | None
    average_speed_pixels_per_second: float | None
    average_speed_metres_per_second: float | None
    average_progress_speed_pixels_per_second: float | None
    average_progress_speed_metres_per_second: float | None
    physical_speed_unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    camera_id: str
    timestamp: float
    queues: tuple[QueueStatus, ...]
    tracks: tuple[QueueTrackStatus, ...]

    def queue_for(self, queue_id: str) -> QueueStatus:
        return next(item for item in self.queues if item.queue_id == queue_id)

    def state_for(self, queue_id: str, track_id: int) -> QueueTrackState | None:
        return next(
            (
                item.state
                for item in self.tracks
                if item.queue_id == queue_id and item.track_id == track_id
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class QueueResult:
    snapshot: QueueSnapshot
    events: tuple[Event, ...]


@dataclass(slots=True)
class _TrackState:
    status: QueueTrackState
    candidate_since: float
    last_seen: float
    last_point: tuple[float, float]
    speed: float | None
    speed_metres: float | None
    progress_speed: float | None
    progress_speed_metres: float | None
    joined_at: float | None = None
    observed_this_frame: bool = True


@dataclass(slots=True)
class _QueueTotals:
    smoothed_count: float = 0.0
    overflow: bool = False
    completed_count: int = 0
    completed_total_seconds: float = 0.0
    last_completed_seconds: float | None = None


class QueueAnalyzer:
    """Stateful analytics for independent manually configured queues.

    Membership is an estimate, not a classification guarantee. It combines
    confirmed foot-point containment, dwell time, smoothed image speed, and a
    short missing-observation tolerance.
    """

    def __init__(
        self,
        cameras: Sequence[CameraQueueConfig],
        *,
        event_sink: EventSink | None = None,
    ) -> None:
        self._cameras = {camera.camera_id: camera for camera in cameras}
        if len(self._cameras) != len(cameras):
            raise ValueError("camera IDs must be unique")
        self._event_sink = event_sink
        self._states: dict[str, dict[tuple[str, int], _TrackState]] = {}
        self._totals: dict[str, dict[str, _QueueTotals]] = {}
        self._timestamps: dict[str, float] = {}
        self.reset()

    def reset(self, camera_id: str | None = None) -> None:
        """Clear queue lifecycle, smoothing, overflow, and wait totals."""

        if camera_id is not None and camera_id not in self._cameras:
            raise KeyError(f"unknown camera: {camera_id}")
        targets = self._cameras if camera_id is None else (camera_id,)
        for target in targets:
            self._states[target] = {}
            self._totals[target] = {
                queue.queue_id: _QueueTotals()
                for queue in self._cameras[target].queues
            }
            self._timestamps[target] = 0.0

    def update(
        self,
        camera_id: str,
        observations: Sequence[TrackObservation],
        *,
        timestamp: float | None = None,
    ) -> QueueResult:
        """Consume confirmed shared trajectories for one camera frame."""

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
        config = self._cameras[camera_id]
        states = self._states[camera_id]
        events: list[Event] = []
        for state in states.values():
            state.observed_this_frame = False

        for queue in config.queues:
            polygon = tuple(
                point.to_pixels(config.frame_size) for point in queue.polygon
            )
            seen_ids: set[int] = set()
            for track_id, observation in sorted(confirmed.items()):
                seen_ids.add(track_id)
                key = (queue.queue_id, track_id)
                speed = _smoothed_speed(observation)
                progress, progress_metres = self._progress_speeds(
                    config, queue, observation
                )
                within_region = point_in_polygon(
                    observation.foot_point, polygon
                ) and (
                    bbox_polygon_coverage_ratio(observation.xyxy, polygon)
                    >= MINIMUM_BBOX_COVERAGE_RATIO - 1e-9
                )
                qualifies = within_region and (
                    speed is None
                    or speed <= queue.maximum_speed_pixels_per_second + 1e-9
                )
                if qualifies:
                    joined = self._observe_qualifying(
                        camera_id,
                        queue,
                        observation,
                        event_time,
                        speed,
                        progress,
                        progress_metres,
                    )
                    if joined is not None:
                        events.append(joined)
                elif key in states:
                    left = self._leave(
                        camera_id,
                        queue,
                        key,
                        event_time,
                        observation.foot_point,
                        reason=(
                            "outside_polygon" if not within_region else "speed_exceeded"
                        ),
                    )
                    if left is not None:
                        events.append(left)

            for key in tuple(key for key in states if key[0] == queue.queue_id):
                if key[1] in seen_ids:
                    continue
                state = states[key]
                if (
                    event_time - state.last_seen
                    > queue.gap_tolerance_seconds + 1e-9
                ):
                    left = self._leave(
                        camera_id,
                        queue,
                        key,
                        event_time,
                        state.last_point,
                        reason="tracking_gap",
                    )
                    if left is not None:
                        events.append(left)

            raw_count = sum(
                state.status is QueueTrackState.MEMBER
                for (queue_id, _), state in states.items()
                if queue_id == queue.queue_id
            )
            total = self._totals[camera_id][queue.queue_id]
            total.smoothed_count = (
                queue.count_smoothing_alpha * raw_count
                + (1.0 - queue.count_smoothing_alpha) * total.smoothed_count
            )
            overflowing = raw_count >= queue.overflow_threshold
            if overflowing != total.overflow:
                total.overflow = overflowing
                events.append(
                    self._event(
                        "queue_overflow_started"
                        if overflowing
                        else "queue_overflow_ended",
                        camera_id,
                        queue.queue_id,
                        event_time,
                        payload={
                            "raw_count": raw_count,
                            "overflow_threshold": queue.overflow_threshold,
                        },
                    )
                )

        event_batch = tuple(events)
        if self._event_sink is not None:
            self._event_sink.persist(event_batch)
        return QueueResult(self.snapshot(camera_id), event_batch)

    def snapshot(self, camera_id: str) -> QueueSnapshot:
        """Return current heuristic queue state without advancing it."""

        if camera_id not in self._cameras:
            raise KeyError(f"unknown camera: {camera_id}")
        timestamp = self._timestamps[camera_id]
        states = self._states[camera_id]
        tracks = tuple(
            QueueTrackStatus(
                queue_id,
                track_id,
                state.status,
                state.observed_this_frame,
                state.candidate_since,
                state.joined_at,
                max(0.0, timestamp - state.candidate_since),
                state.speed,
                state.speed_metres,
                state.progress_speed,
                state.progress_speed_metres,
            )
            for (queue_id, track_id), state in sorted(states.items())
        )
        queues: list[QueueStatus] = []
        for queue in self._cameras[camera_id].queues:
            members = [
                state
                for (queue_id, _), state in states.items()
                if queue_id == queue.queue_id
                and state.status is QueueTrackState.MEMBER
            ]
            raw_count = len(members)
            total = self._totals[camera_id][queue.queue_id]
            current_wait = (
                sum(timestamp - state.candidate_since for state in members) / raw_count
                if raw_count
                else None
            )
            pixel_speeds = [state.speed for state in members if state.speed is not None]
            metre_speeds = [
                state.speed_metres for state in members if state.speed_metres is not None
            ]
            pixel_progress = [
                state.progress_speed
                for state in members
                if state.progress_speed is not None
            ]
            metre_progress = [
                state.progress_speed_metres
                for state in members
                if state.progress_speed_metres is not None
            ]
            projector = self._cameras[camera_id].projector
            physical_reason = None
            if not metre_speeds:
                physical_reason = (
                    projector.unavailable_reason
                    if projector is not None and not projector.available
                    else "physical queue speed unavailable"
                )
            queues.append(
                QueueStatus(
                    queue.queue_id,
                    raw_count,
                    total.smoothed_count,
                    raw_count > 0,
                    total.overflow,
                    current_wait,
                    total.completed_count,
                    total.completed_total_seconds / total.completed_count
                    if total.completed_count
                    else None,
                    total.last_completed_seconds,
                    _mean(pixel_speeds),
                    _mean(metre_speeds),
                    _mean(pixel_progress),
                    _mean(metre_progress),
                    physical_reason,
                )
            )
        return QueueSnapshot(camera_id, timestamp, tuple(queues), tracks)

    def _observe_qualifying(
        self,
        camera_id: str,
        queue: QueueRegion,
        observation: TrackObservation,
        timestamp: float,
        speed: float | None,
        progress_speed: float | None,
        progress_speed_metres: float | None,
    ) -> Event | None:
        key = (queue.queue_id, observation.track_id)
        state = self._states[camera_id].get(key)
        if state is None:
            state = _TrackState(
                QueueTrackState.CANDIDATE,
                timestamp,
                timestamp,
                observation.foot_point,
                speed,
                observation.speed_metres_per_second,
                progress_speed,
                progress_speed_metres,
            )
            self._states[camera_id][key] = state
        else:
            state.last_seen = timestamp
            state.last_point = observation.foot_point
            state.speed = speed
            state.speed_metres = observation.speed_metres_per_second
            state.progress_speed = progress_speed
            state.progress_speed_metres = progress_speed_metres
            state.observed_this_frame = True
        if (
            state.status is QueueTrackState.CANDIDATE
            and timestamp - state.candidate_since
            >= queue.minimum_dwell_seconds - 1e-9
        ):
            state.status = QueueTrackState.MEMBER
            state.joined_at = timestamp
            return self._event(
                "queue_joined",
                camera_id,
                queue.queue_id,
                timestamp,
                track_id=observation.track_id,
                payload={
                    "qualifying_dwell_seconds": timestamp - state.candidate_since,
                    "smoothed_speed_pixels_per_second": speed,
                    "speed_metres_per_second": observation.speed_metres_per_second,
                    "queue_progress_pixels_per_second": progress_speed,
                    "queue_progress_metres_per_second": progress_speed_metres,
                    "membership_is_heuristic": True,
                },
            )
        return None

    def _leave(
        self,
        camera_id: str,
        queue: QueueRegion,
        key: tuple[str, int],
        timestamp: float,
        exit_point: tuple[float, float],
        *,
        reason: str,
    ) -> Event | None:
        state = self._states[camera_id].pop(key)
        if state.status is QueueTrackState.CANDIDATE:
            return None
        waiting_seconds = max(0.0, timestamp - state.candidate_since)
        completed = _near_service(
            exit_point,
            queue.service_point.point.as_tuple(),
            self._cameras[camera_id].frame_size,
            queue.service_completion_radius,
        )
        total = self._totals[camera_id][queue.queue_id]
        if completed:
            total.completed_count += 1
            total.completed_total_seconds += waiting_seconds
            total.last_completed_seconds = waiting_seconds
        return self._event(
            "queue_left",
            camera_id,
            queue.queue_id,
            timestamp,
            track_id=key[1],
            payload={
                "waiting_seconds": waiting_seconds,
                "completed_near_service": completed,
                "leave_reason": reason,
            },
        )

    @staticmethod
    def _event(
        event_type: str,
        camera_id: str,
        queue_id: str,
        timestamp: float,
        *,
        track_id: int | None = None,
        payload: dict[str, object] | None = None,
    ) -> Event:
        return Event(
            event_id=str(uuid4()),
            event_type=event_type,
            camera_id=camera_id,
            timestamp=timestamp,
            track_id=track_id,
            zone_id=queue_id,
            payload=payload,
        )

    @staticmethod
    def _progress_speeds(
        config: CameraQueueConfig,
        queue: QueueRegion,
        observation: TrackObservation,
    ) -> tuple[float | None, float | None]:
        service_image = queue.service_point.point.to_pixels(config.frame_size)
        pixel = queue_progress_speed(
            observation.smoothed_velocity,
            observation.foot_point,
            service_image,
        )
        service_ground = None
        if config.projector is not None:
            projected = config.projector.project(service_image)
            service_ground = projected.point if projected.available else None
        physical = queue_progress_speed(
            observation.ground_smoothed_velocity,
            observation.ground_position,
            service_ground,
        )
        return pixel, physical


def _smoothed_speed(observation: TrackObservation) -> float | None:
    if observation.speed_pixels_per_second is not None:
        return observation.speed_pixels_per_second
    if observation.smoothed_velocity is not None:
        return math.hypot(*observation.smoothed_velocity)
    if len(observation.trajectory) < 2:
        return None
    previous, current = observation.trajectory[-2:]
    elapsed = current.timestamp - previous.timestamp
    if elapsed <= 0:
        return None
    return math.dist(previous.smoothed_position, current.smoothed_position) / elapsed


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _near_service(
    image_point: tuple[float, float],
    normalized_service: tuple[float, float],
    frame_size: tuple[int, int],
    radius: float,
) -> bool:
    width, height = frame_size
    normalized = (
        image_point[0] / max(width - 1, 1),
        image_point[1] / max(height - 1, 1),
    )
    return math.dist(normalized, normalized_service) <= radius + 1e-9
