"""Paper-faithful StableTrack core (arXiv:2511.20418).

Adapted for processed-only streams: when no intermediate frame is supplied,
association runs in the current frame using real elapsed time ``Δτ`` inside
Bbox-Based Distance instead of assuming consecutive 30 FPS frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from app.tracking.third_party.stabletrack.kalman import StableKalmanFilter
from app.tracking.third_party.stabletrack.matching import (
    INF,
    bbox_based_distance,
    cosine_similarity,
    iou_distance,
    linear_assignment,
    xyxy_to_xywh,
)
from app.tracking.third_party.stabletrack.visual import camshift_predict, displacement

Embedding = NDArray[np.float32]
EmbedFn = Callable[[NDArray[np.uint8], Sequence[float]], Embedding | None]


@dataclass(slots=True)
class StableTrackConfig:
    """Paper hyperparameters plus 0.5 FPS operating-point knobs."""

    activation_threshold: float = 0.4
    bbd_threshold: float = 16.0
    iou_threshold: float = 0.4
    reid_high_threshold: float = 0.65
    reid_low_threshold: float = 0.3
    bbd_alpha: float = 0.025
    bbd_beta: float = 0.25
    bbd_scale: float = 1.0
    max_age_seconds: float = 8.0
    confirmation_hits: int = 1
    ema: float = 0.9
    use_visual_tracking: bool = True


@dataclass(slots=True)
class TrackState:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    hits: int
    time_since_update: float
    last_timestamp: float
    embedding: Embedding | None = None
    kalman: StableKalmanFilter = field(default_factory=StableKalmanFilter)
    confirmed: bool = False


@dataclass(frozen=True, slots=True)
class TrackOutput:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    confirmed: bool
    embedding: Embedding | None = None


class StableTrack:
    """Two-stage BBD + ReID matcher with timestamp-aware Kalman prediction."""

    def __init__(self, config: StableTrackConfig | None = None) -> None:
        self.config = config or StableTrackConfig()
        self.tracks: list[TrackState] = []
        self._next_id = 1
        self._last_timestamp: float | None = None
        self._last_frame: NDArray[np.uint8] | None = None
        self.last_reid_ms = 0.0

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        self._last_timestamp = None
        self._last_frame = None
        self.last_reid_ms = 0.0

    def update(
        self,
        *,
        boxes: NDArray[np.floating],
        scores: NDArray[np.floating],
        class_ids: NDArray[np.integer] | None,
        timestamp: float,
        frame: NDArray[np.uint8] | None = None,
        intermediate_frame: NDArray[np.uint8] | None = None,
        embed: EmbedFn | None = None,
    ) -> list[TrackOutput]:
        from time import perf_counter

        dt = 0.0 if self._last_timestamp is None else max(timestamp - self._last_timestamp, 1e-6)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4) if len(boxes) else np.empty((0, 4), dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1) if len(scores) else np.empty((0,), dtype=np.float32)
        if class_ids is None:
            class_ids = np.zeros(len(boxes), dtype=int)
        else:
            class_ids = np.asarray(class_ids).reshape(-1)

        embeddings: list[Embedding | None] = [None] * len(boxes)
        reid_started = perf_counter()
        if embed is not None and frame is not None:
            for index, box in enumerate(boxes):
                embeddings[index] = embed(frame, box)
        self.last_reid_ms = (perf_counter() - reid_started) * 1000.0 if embed is not None else 0.0

        predicted, velocities = self._predict_tracks(dt, frame, intermediate_frame)
        detection_boxes = self._warp_detections(boxes, frame, intermediate_frame, dt)

        high_mask = scores >= self.config.activation_threshold
        high_indices = np.flatnonzero(high_mask)
        low_indices = np.flatnonzero(~high_mask)

        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_dets = list(range(len(boxes)))
        matches: list[tuple[int, int]] = []

        if len(high_indices):
            first = self._associate(
                unmatched_tracks,
                list(high_indices),
                predicted,
                detection_boxes,
                embeddings,
                dt,
                stage="bbd",
            )
            matches.extend(first)
            unmatched_tracks = [index for index in unmatched_tracks if index not in {row for row, _col in first}]
            unmatched_dets = [index for index in unmatched_dets if index not in {col for _row, col in first}]

        remaining_high = [index for index in unmatched_dets if index in set(high_indices)]
        remaining_low = [index for index in unmatched_dets if index in set(low_indices)]
        second_candidates = remaining_high + remaining_low
        if unmatched_tracks and second_candidates:
            second = self._associate(
                unmatched_tracks,
                second_candidates,
                predicted,
                detection_boxes,
                embeddings,
                dt,
                stage="iou",
            )
            matches.extend(second)
            unmatched_tracks = [index for index in unmatched_tracks if index not in {row for row, _col in second}]
            unmatched_dets = [index for index in unmatched_dets if index not in {col for _row, col in second}]

        for track_index, det_index in matches:
            velocity = velocities.get(track_index)
            self._update_track(
                self.tracks[track_index],
                boxes[det_index],
                float(scores[det_index]),
                embeddings[det_index],
                timestamp,
                velocity,
            )

        for track_index in unmatched_tracks:
            track = self.tracks[track_index]
            track.time_since_update += dt
            track.xyxy = predicted[track_index] if track_index < len(predicted) else track.xyxy

        for det_index in unmatched_dets:
            if float(scores[det_index]) < self.config.activation_threshold:
                continue
            self._create_track(
                boxes[det_index],
                float(scores[det_index]),
                int(class_ids[det_index]) if det_index < len(class_ids) else 0,
                embeddings[det_index],
                timestamp,
            )

        self.tracks = [
            track for track in self.tracks if track.time_since_update <= self.config.max_age_seconds
        ]
        self._last_timestamp = timestamp
        self._last_frame = None if frame is None else np.ascontiguousarray(frame)

        outputs: list[TrackOutput] = []
        for track in self.tracks:
            if track.time_since_update > 1e-6:
                continue
            outputs.append(
                TrackOutput(
                    track_id=track.track_id,
                    xyxy=track.xyxy,
                    confidence=track.confidence,
                    class_id=track.class_id,
                    confirmed=track.confirmed,
                    embedding=track.embedding,
                )
            )
        return outputs

    def _predict_tracks(
        self,
        dt: float,
        frame: NDArray[np.uint8] | None,
        intermediate_frame: NDArray[np.uint8] | None,
        ) -> tuple[list[tuple[float, float, float, float]], dict[int, tuple[float, float]]]:
        predicted: list[tuple[float, float, float, float]] = []
        velocities: dict[int, tuple[float, float]] = {}
        half_dt = max(dt / 2.0, 1e-6)
        target = intermediate_frame if intermediate_frame is not None else frame
        source = self._last_frame
        for index, track in enumerate(self.tracks):
            visual_box = None
            if self.config.use_visual_tracking and source is not None and target is not None:
                visual_box = camshift_predict(source, target, track.xyxy)
            track.kalman.predict(half_dt if intermediate_frame is not None else dt)
            if visual_box is not None:
                visual_dt = half_dt if intermediate_frame is not None else dt
                velocities[index] = displacement(track.xyxy, visual_box, visual_dt)
                predicted.append(visual_box)
            else:
                predicted.append(track.kalman.to_xyxy())
        return predicted, velocities

    def _warp_detections(
        self,
        boxes: NDArray[np.floating],
        frame: NDArray[np.uint8] | None,
        intermediate_frame: NDArray[np.uint8] | None,
        dt: float,
    ) -> list[tuple[float, float, float, float]]:
        if (
            not self.config.use_visual_tracking
            or intermediate_frame is None
            or frame is None
            or not len(boxes)
        ):
            return [tuple(float(value) for value in box) for box in boxes]
        warped: list[tuple[float, float, float, float]] = []
        _ = dt
        for box in boxes:
            predicted = camshift_predict(frame, intermediate_frame, box)
            warped.append(tuple(float(value) for value in (predicted if predicted is not None else box)))
        return warped

    def _associate(
        self,
        track_indices: Sequence[int],
        det_indices: Sequence[int],
        predicted: Sequence[tuple[float, float, float, float]],
        detection_boxes: Sequence[tuple[float, float, float, float]],
        embeddings: Sequence[Embedding | None],
        dt: float,
        *,
        stage: str,
    ) -> list[tuple[int, int]]:
        if not track_indices or not det_indices:
            return []
        cost = np.full((len(track_indices), len(det_indices)), INF, dtype=np.float64)
        appearance_available = any(item is not None for item in embeddings) or any(
            self.tracks[index].embedding is not None for index in track_indices
        )
        for row, track_index in enumerate(track_indices):
            track = self.tracks[track_index]
            pred = predicted[track_index]
            delta_tau = max(track.time_since_update + dt, dt)
            for col, det_index in enumerate(det_indices):
                det_box = detection_boxes[det_index]
                similarity = cosine_similarity(track.embedding, embeddings[det_index])
                if stage == "bbd":
                    distance = bbox_based_distance(
                        pred,
                        det_box,
                        delta_tau,
                        alpha=self.config.bbd_alpha,
                        beta=self.config.bbd_beta,
                        scale=self.config.bbd_scale,
                    )
                    spatial_ok = distance < self.config.bbd_threshold
                    appearance_ok = (
                        (not appearance_available)
                        or similarity >= self.config.reid_high_threshold
                    )
                    if spatial_ok and appearance_ok:
                        cost[row, col] = (1.0 - similarity) if appearance_available else distance
                else:
                    distance = iou_distance(pred, det_box)
                    spatial_ok = distance < self.config.iou_threshold
                    appearance_ok = (
                        (not appearance_available)
                        or similarity >= self.config.reid_low_threshold
                    )
                    if spatial_ok and appearance_ok:
                        cost[row, col] = (1.0 - similarity) if appearance_available else distance
        assigned = linear_assignment(cost)
        return [(track_indices[row], det_indices[col]) for row, col in assigned]

    def _update_track(
        self,
        track: TrackState,
        box: NDArray[np.floating],
        confidence: float,
        embedding: Embedding | None,
        timestamp: float,
        velocity: tuple[float, float] | None,
    ) -> None:
        xyxy = tuple(float(value) for value in box)
        track.kalman.update(np.asarray(xyxy_to_xywh(xyxy)), velocity=velocity)
        track.xyxy = xyxy
        track.confidence = confidence
        track.hits += 1
        track.time_since_update = 0.0
        track.last_timestamp = timestamp
        track.confirmed = track.hits >= self.config.confirmation_hits
        if embedding is not None:
            if track.embedding is None:
                track.embedding = embedding
            else:
                track.embedding = (
                    self.config.ema * track.embedding + (1.0 - self.config.ema) * embedding
                )

    def _create_track(
        self,
        box: NDArray[np.floating],
        confidence: float,
        class_id: int,
        embedding: Embedding | None,
        timestamp: float,
    ) -> None:
        xyxy = tuple(float(value) for value in box)
        kalman = StableKalmanFilter()
        kalman.initiate(np.asarray(xyxy_to_xywh(xyxy)))
        track = TrackState(
            track_id=self._next_id,
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            hits=1,
            time_since_update=0.0,
            last_timestamp=timestamp,
            embedding=embedding,
            kalman=kalman,
            confirmed=self.config.confirmation_hits <= 1,
        )
        self._next_id += 1
        self.tracks.append(track)
