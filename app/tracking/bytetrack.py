"""ByteTrack adapter and bounded trajectory state management."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
import math
from time import perf_counter
from typing import Any
import warnings

import numpy as np
import supervision as sv
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from trackers import ByteTrackTracker

from app.core.models import Detection, TrackObservation, TrajectoryPoint
from app.tracking.base import TrackingResult


def foot_point(xyxy: Sequence[float]) -> tuple[float, float]:
    """Return the bottom-center image point for an ``xyxy`` box."""

    x1, _y1, x2, y2 = xyxy
    return ((float(x1) + float(x2)) / 2.0, float(y2))


def detections_to_supervision(detections: Sequence[Detection]) -> sv.Detections:
    """Convert shared person detections to ByteTrack's input representation."""

    people = [
        detection
        for detection in detections
        if detection.class_id == 0
        and (detection.class_name is None or detection.class_name.lower() == "person")
    ]
    if not people:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.asarray([item.xyxy for item in people], dtype=np.float32),
        confidence=np.asarray([item.confidence for item in people], dtype=np.float32),
        class_id=np.zeros(len(people), dtype=int),
    )


class ByteTrackAdapter:
    """Adapt maintained ByteTrack output to shared timestamped observations."""

    def __init__(
        self,
        *,
        activation_threshold: float = 0.4,
        lost_track_buffer: int = 30,
        match_threshold: float = 0.3,
        history_size: int = 90,
        frame_rate: float = 30.0,
        confirmation_frames: int = 2,
        smoothing_alpha: float = 0.35,
    ) -> None:
        _unit_interval(activation_threshold, "activation_threshold")
        _unit_interval(match_threshold, "match_threshold")
        _unit_interval(smoothing_alpha, "smoothing_alpha", lower_open=True)
        if lost_track_buffer <= 0:
            raise ValueError("lost_track_buffer must be positive")
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        if confirmation_frames <= 0:
            raise ValueError("confirmation_frames must be positive")
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("frame_rate must be finite and positive")

        self.history_size = history_size
        self.smoothing_alpha = smoothing_alpha
        # Trackers 2.5 emits deprecation-library warnings during construction;
        # they are external packaging warnings, not tracker behavior warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker = ByteTrackTracker(
                lost_track_buffer=lost_track_buffer,
                frame_rate=frame_rate,
                track_activation_threshold=activation_threshold,
                minimum_consecutive_frames=confirmation_frames,
                minimum_iou_threshold=match_threshold,
                high_conf_det_threshold=activation_threshold,
            )
        self._histories: dict[int, deque[TrajectoryPoint]] = {}
        self._tracklet_ids: dict[int, int] = {}
        self._next_track_id = 1
        self._last_frame_index: int | None = None

    def update(
        self,
        detections: Sequence[Detection],
        *,
        camera_id: str,
        timestamp: float,
        frame_index: int,
    ) -> TrackingResult:
        """Advance ByteTrack and retain only observations seen in this frame."""

        self._validate_frame(camera_id, timestamp, frame_index)
        started = perf_counter()
        tracker_input = detections_to_supervision(detections)
        tracked = self._tracker.update(tracker_input)

        active_tracklets = [
            tracklet
            for tracklet in self._tracker.tracks
            if tracklet.time_since_update == 0
        ]
        row_matches = _match_rows_to_tracklets(tracked, active_tracklets)
        observations: list[TrackObservation] = []
        for row_index, tracklet in row_matches:
            identity = id(tracklet)
            track_id = self._tracklet_ids.get(identity)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracklet_ids[identity] = track_id

            box = tuple(float(value) for value in tracklet.get_state_bbox())
            raw_point = foot_point(box)
            history = self._histories.setdefault(
                track_id, deque(maxlen=self.history_size)
            )
            smoothed = _smooth(raw_point, history, self.smoothing_alpha)
            history.append(
                TrajectoryPoint(
                    timestamp=float(timestamp),
                    frame_index=frame_index,
                    position=raw_point,
                    smoothed_position=smoothed,
                )
            )
            confidence = (
                float(tracked.confidence[row_index])
                if tracked.confidence is not None
                else 1.0
            )
            observations.append(
                TrackObservation(
                    camera_id=camera_id,
                    track_id=track_id,
                    timestamp=float(timestamp),
                    frame_index=frame_index,
                    xyxy=box,
                    foot_point=raw_point,
                    detection_confidence=confidence,
                    confirmed=tracklet.tracker_id != -1,
                    trajectory=tuple(history),
                )
            )

        alive_identities = {id(tracklet) for tracklet in self._tracker.tracks}
        expired = tuple(
            sorted(
                track_id
                for identity, track_id in self._tracklet_ids.items()
                if identity not in alive_identities
            )
        )
        for identity, track_id in tuple(self._tracklet_ids.items()):
            if identity not in alive_identities:
                del self._tracklet_ids[identity]
                self._histories.pop(track_id, None)

        self._last_frame_index = frame_index
        return TrackingResult(
            observations=tuple(sorted(observations, key=lambda item: item.track_id)),
            expired_track_ids=expired,
            tracking_ms=(perf_counter() - started) * 1000.0,
        )

    def reset(self) -> None:
        """Clear backend tracks, trajectories, IDs, and frame ordering state."""

        self._tracker.reset()
        self._histories.clear()
        self._tracklet_ids.clear()
        self._next_track_id = 1
        self._last_frame_index = None

    @property
    def retained_track_count(self) -> int:
        """Return the number of active or buffered track histories."""

        return len(self._histories)

    def _validate_frame(
        self, camera_id: str, timestamp: float, frame_index: int
    ) -> None:
        if not camera_id.strip():
            raise ValueError("camera_id must be non-empty")
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("frame_index must increase on every update")


def _match_rows_to_tracklets(
    tracked: sv.Detections, active_tracklets: Sequence[Any]
) -> list[tuple[int, Any]]:
    """Associate ByteTrack output rows with its live tracklet objects."""

    if len(tracked) == 0 or not active_tracklets:
        return []
    backend_ids = (
        tracked.tracker_id
        if tracked.tracker_id is not None
        else np.full(len(tracked), -1, dtype=int)
    )
    matches: list[tuple[int, Any]] = []
    matched_rows: set[int] = set()
    matched_tracklets: set[int] = set()

    # Confirmed ByteTrack IDs are authoritative and avoid an ambiguous second
    # IoU association when people overlap heavily.
    by_backend_id = {
        int(tracklet.tracker_id): (index, tracklet)
        for index, tracklet in enumerate(active_tracklets)
        if tracklet.tracker_id != -1
    }
    for row_index, backend_id in enumerate(backend_ids):
        match = by_backend_id.get(int(backend_id))
        if match is None:
            continue
        tracklet_index, tracklet = match
        matches.append((row_index, tracklet))
        matched_rows.add(row_index)
        matched_tracklets.add(tracklet_index)

    # New tracks have backend ID -1 until confirmed. Match only those rows to
    # unconfirmed active tracklets; unmatched low-confidence rows are ignored.
    candidates: list[tuple[float, int, int]] = []
    for row_index, detection_box in enumerate(tracked.xyxy):
        if row_index in matched_rows or int(backend_ids[row_index]) != -1:
            continue
        for tracklet_index, tracklet in enumerate(active_tracklets):
            if tracklet_index in matched_tracklets or tracklet.tracker_id != -1:
                continue
            candidates.append(
                (
                    _intersection_over_union(
                        detection_box, tracklet.get_state_bbox()
                    ),
                    row_index,
                    tracklet_index,
                )
            )
    for overlap, row_index, tracklet_index in sorted(candidates, reverse=True):
        if overlap <= 0.0:
            break
        if row_index in matched_rows or tracklet_index in matched_tracklets:
            continue
        matched_rows.add(row_index)
        matched_tracklets.add(tracklet_index)
        matches.append((row_index, active_tracklets[tracklet_index]))
    return matches


def _intersection_over_union(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _smooth(
    point: tuple[float, float],
    history: deque[TrajectoryPoint],
    alpha: float,
) -> tuple[float, float]:
    if not history:
        return point
    previous = history[-1].smoothed_position
    return (
        alpha * point[0] + (1.0 - alpha) * previous[0],
        alpha * point[1] + (1.0 - alpha) * previous[1],
    )


def _unit_interval(value: float, name: str, *, lower_open: bool = False) -> None:
    valid_lower = value > 0.0 if lower_open else value >= 0.0
    if not math.isfinite(value) or not valid_lower or value > 1.0:
        bracket = "(0, 1]" if lower_open else "[0, 1]"
        raise ValueError(f"{name} must be in {bracket}")
