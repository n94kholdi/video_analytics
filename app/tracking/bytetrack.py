"""ByteTrack adapter and bounded trajectory state management."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
import math
from pathlib import Path
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
from app.tracking.reid import OsNetReIdentifier, ReIdGallery


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
        frame_size: tuple[int, int] | None = None,
        reid_model: str | Path | None = None,
        reid_providers: Sequence[str] = ("CPUExecutionProvider",),
        reid_similarity_threshold: float = 0.75,
        reid_max_age_frames: int = 300,
        reid_interval: int = 5,
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
        if frame_size is not None and (frame_size[0] <= 0 or frame_size[1] <= 0):
            raise ValueError("frame_size must contain positive width and height")
        if reid_interval <= 0:
            raise ValueError("reid_interval must be positive")

        self.history_size = history_size
        self.smoothing_alpha = smoothing_alpha
        self.frame_size = frame_size
        self.reid_interval = reid_interval
        self._reidentifier = (
            OsNetReIdentifier(reid_model, providers=reid_providers)
            if reid_model is not None
            else None
        )
        self._reid_gallery = (
            ReIdGallery(
                similarity_threshold=reid_similarity_threshold,
                max_age_frames=reid_max_age_frames,
            )
            if self._reidentifier is not None
            else None
        )
        # Trackers 2.5 emits deprecation-library warnings during construction;
        # they are external packaging warnings, not tracker behavior warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            try:
                self._tracker = ByteTrackTracker(
                    lost_track_buffer=lost_track_buffer,
                    frame_rate=frame_rate,
                    track_activation_threshold=activation_threshold,
                    minimum_consecutive_frames=confirmation_frames,
                    minimum_iou_threshold=match_threshold,
                    high_conf_det_threshold=activation_threshold,
                )
            except Exception as exc:
                raise RuntimeError(
                    "ByteTrack initialization failed; verify trackers==2.5.x and tracker "
                    f"thresholds (activation={activation_threshold}, match={match_threshold}, "
                    f"buffer={lost_track_buffer}, fps={frame_rate}): {exc}"
                ) from exc
        self._histories: dict[int, deque[TrajectoryPoint]] = {}
        self._tracklet_ids: dict[int, int] = {}
        self._confirmed_track_ids: set[int] = set()
        self._next_track_id = 1
        self._last_frame_index: int | None = None

    def update(
        self,
        detections: Sequence[Detection],
        *,
        camera_id: str,
        timestamp: float,
        frame_index: int,
        frame: np.ndarray | None = None,
    ) -> TrackingResult:
        """Advance ByteTrack and retain only observations seen in this frame."""

        self._validate_frame(camera_id, timestamp, frame_index)
        self._validate_detections(detections, frame_index)
        if self._reidentifier is not None and frame is None:
            raise ValueError("frame is required when OSNet ReID is enabled")
        started = perf_counter()
        tracker_input = detections_to_supervision(detections)
        try:
            tracked = self._tracker.update(tracker_input)
        except Exception as exc:
            raise RuntimeError(
                "ByteTrack update failed at frame "
                f"{frame_index} with {len(tracker_input)} person detections "
                f"(bbox format=xyxy, frame_size={self.frame_size}): {exc}"
            ) from exc

        active_tracklets = [
            tracklet
            for tracklet in self._tracker.tracks
            if tracklet.time_since_update == 0
        ]
        row_matches = _match_rows_to_tracklets(tracked, active_tracklets)
        observations: list[TrackObservation] = []
        alive_identities = {id(tracklet) for tracklet in self._tracker.tracks}
        occupied_track_ids = {
            track_id
            for identity, track_id in self._tracklet_ids.items()
            if identity in alive_identities
        }
        for row_index, tracklet in row_matches:
            identity = id(tracklet)
            track_id = self._tracklet_ids.get(identity)
            is_confirmed = tracklet.tracker_id != -1
            embedding = None
            should_embed = self._reidentifier is not None and (
                track_id is None
                or (is_confirmed and track_id not in self._confirmed_track_ids)
                or frame_index % self.reid_interval == 0
            )
            if should_embed:
                assert frame is not None
                embedding = self._reidentifier.embed(frame, tracklet.get_state_bbox())
            if track_id is None:
                track_id = (
                    self._reid_gallery.match(
                        embedding,
                        frame_index,
                        excluded_track_ids=occupied_track_ids,
                    )
                    if embedding is not None and self._reid_gallery is not None
                    else None
                )
                if track_id is None:
                    track_id = self._next_track_id
                    self._next_track_id += 1
                self._tracklet_ids[identity] = track_id
            elif (
                is_confirmed
                and track_id not in self._confirmed_track_ids
                and embedding is not None
                and self._reid_gallery is not None
            ):
                matched_id = self._reid_gallery.match(
                    embedding,
                    frame_index,
                    excluded_track_ids=occupied_track_ids,
                )
                if matched_id is not None and matched_id != track_id:
                    provisional_id = track_id
                    track_id = matched_id
                    self._tracklet_ids[identity] = track_id
                    occupied_track_ids.discard(provisional_id)
                    provisional_history = self._histories.pop(provisional_id, None)
                    if provisional_history is not None:
                        self._histories[track_id] = provisional_history
            occupied_track_ids.add(track_id)
            if is_confirmed and embedding is not None and self._reid_gallery is not None:
                self._reid_gallery.update(track_id, embedding, frame_index)
                self._confirmed_track_ids.add(track_id)

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

        alive_public_ids = {
            track_id
            for identity, track_id in self._tracklet_ids.items()
            if identity in alive_identities
        }
        expired = tuple(
            sorted(
                track_id
                for identity, track_id in self._tracklet_ids.items()
                if identity not in alive_identities and track_id not in alive_public_ids
            )
        )
        for identity, track_id in tuple(self._tracklet_ids.items()):
            if identity not in alive_identities:
                del self._tracklet_ids[identity]
                if track_id not in alive_public_ids:
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
        self._confirmed_track_ids.clear()
        self._next_track_id = 1
        self._last_frame_index = None
        if self._reid_gallery is not None:
            self._reid_gallery.clear()

    @property
    def reid_enabled(self) -> bool:
        return self._reidentifier is not None

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

    def _validate_detections(
        self, detections: Sequence[Detection], frame_index: int
    ) -> None:
        if self.frame_size is None:
            return
        width, height = self.frame_size
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.xyxy
            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"invalid zero-area xyxy detection {index} at frame {frame_index}: {detection.xyxy}"
                )
            tolerance = 1.0
            if x1 < -tolerance or y1 < -tolerance or x2 > width + tolerance or y2 > height + tolerance:
                raise ValueError(
                    f"xyxy detection {index} is outside {width}x{height} frame at frame "
                    f"{frame_index}: {detection.xyxy}"
                )


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
