"""Deep OC-SORT core (arXiv:2302.11813), detector-agnostic and timestamp-aware.

Isolated from application adapters. Association, Dynamic Appearance, Adaptive
Weighting, OCR, and optional CMC follow GerardMaggiolino/Deep-OC-SORT
``trackers/ocsort_embedding``. Kalman prediction uses elapsed seconds instead of
a unit frame step so 0.5 FPS processing remains well-defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from app.tracking.third_party.deepocsort.association import associate, cosine_cost, iou_batch
from app.tracking.third_party.deepocsort.cmc import SparseFlowCMC, apply_affine_to_xyxy
from app.tracking.third_party.deepocsort.kalman import DeepOCSortKalman, xyxy_to_xywh
from app.tracking.third_party.stabletrack.matching import INF, linear_assignment

Embedding = NDArray[np.float32]
EmbedFn = Callable[[NDArray[np.uint8], Sequence[float]], Embedding | None]


@dataclass(slots=True)
class DeepOCSortConfig:
    """Paper hyperparameters plus 0.5 FPS operating-point knobs."""

    activation_threshold: float = 0.4
    iou_threshold: float = 0.3
    max_age_seconds: float = 8.0
    confirmation_hits: int = 1
    delta_t_seconds: float = 2.0
    inertia: float = 0.2
    w_association_emb: float = 0.75
    alpha_fixed_emb: float = 0.95
    aw_param: float = 0.5
    aw_off: bool = False
    use_cmc: bool = False
    reid_recovery_threshold: float = 0.65


@dataclass(slots=True)
class TrackState:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    hits: int
    hit_streak: int
    time_since_update: float
    last_timestamp: float
    last_observation: tuple[float, float, float, float, float]
    observations: list[tuple[float, tuple[float, float, float, float, float]]] = field(default_factory=list)
    velocity: tuple[float, float] | None = None
    embedding: Embedding | None = None
    kalman: DeepOCSortKalman = field(default_factory=DeepOCSortKalman)
    frozen: bool = False
    confirmed: bool = False


@dataclass(frozen=True, slots=True)
class TrackOutput:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    confirmed: bool
    embedding: Embedding | None = None


def _speed_direction(
    first: Sequence[float],
    second: Sequence[float],
) -> tuple[float, float]:
    cx1 = (float(first[0]) + float(first[2])) / 2.0
    cy1 = (float(first[1]) + float(first[3])) / 2.0
    cx2 = (float(second[0]) + float(second[2])) / 2.0
    cy2 = (float(second[1]) + float(second[3])) / 2.0
    dy, dx = cy2 - cy1, cx2 - cx1
    norm = float(np.sqrt(dy * dy + dx * dx)) + 1e-6
    return (dy / norm, dx / norm)


def _placeholder_observation() -> tuple[float, float, float, float, float]:
    return (-1.0, -1.0, -1.0, -1.0, -1.0)


def _k_previous_obs(
    observations: Sequence[tuple[float, tuple[float, float, float, float, float]]],
    last_timestamp: float,
    delta_t_seconds: float,
) -> tuple[float, float, float, float, float]:
    if not observations:
        return _placeholder_observation()
    target = last_timestamp - max(float(delta_t_seconds), 1e-6)
    chosen = observations[0][1]
    best = abs(observations[0][0] - target)
    for stamp, box in observations:
        distance = abs(stamp - target)
        if distance <= best:
            best = distance
            chosen = box
    return chosen


class DeepOCSort:
    """OC-SORT + Dynamic Appearance + Adaptive Weighting, with real timestamps."""

    def __init__(self, config: DeepOCSortConfig | None = None) -> None:
        self.config = config or DeepOCSortConfig()
        self.tracks: list[TrackState] = []
        self._next_id = 1
        self._last_timestamp: float | None = None
        self._cmc = SparseFlowCMC() if self.config.use_cmc else None
        self.last_reid_ms = 0.0

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        self._last_timestamp = None
        self.last_reid_ms = 0.0
        if self._cmc is not None:
            self._cmc.reset()

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
        _ = intermediate_frame
        dt = 0.0 if self._last_timestamp is None else max(timestamp - self._last_timestamp, 1e-6)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4) if len(boxes) else np.empty((0, 4), dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1) if len(scores) else np.empty((0,), dtype=np.float32)
        if class_ids is None:
            class_ids = np.zeros(len(boxes), dtype=int)
        else:
            class_ids = np.asarray(class_ids).reshape(-1)

        if self._cmc is not None and frame is not None:
            transform = self._cmc.compute_affine(frame, boxes)
            for track in self.tracks:
                self._apply_cmc(track, transform)

        embeddings: list[Embedding | None] = [None] * len(boxes)
        reid_started = perf_counter()
        if embed is not None and frame is not None:
            for index, box in enumerate(boxes):
                embeddings[index] = embed(frame, box)
        self.last_reid_ms = (perf_counter() - reid_started) * 1000.0 if embed is not None else 0.0

        high_mask = scores >= self.config.activation_threshold
        high_indices = np.flatnonzero(high_mask)
        high_boxes = boxes[high_indices] if len(high_indices) else np.empty((0, 4), dtype=np.float32)
        high_scores = scores[high_indices] if len(high_indices) else np.empty((0,), dtype=np.float32)
        high_class_ids = class_ids[high_indices] if len(high_indices) else np.empty((0,), dtype=int)
        high_embeddings = [embeddings[index] for index in high_indices]
        dets = (
            np.concatenate([high_boxes, high_scores[:, None]], axis=1)
            if len(high_boxes)
            else np.empty((0, 5), dtype=np.float32)
        )

        predicted: list[tuple[float, float, float, float]] = []
        alive: list[TrackState] = []
        for track in self.tracks:
            track.kalman.predict(dt, freeze_scale_velocity=track.frozen)
            box = track.kalman.to_xyxy()
            if not np.isfinite(box).all():
                continue
            track.xyxy = box
            predicted.append(box)
            alive.append(track)
        self.tracks = alive

        trks = (
            np.asarray([[*box, 0.0] for box in predicted], dtype=np.float64)
            if predicted
            else np.empty((0, 5), dtype=np.float64)
        )
        velocities = np.asarray(
            [track.velocity if track.velocity is not None else (0.0, 0.0) for track in self.tracks],
            dtype=np.float64,
        ).reshape(-1, 2)
        last_boxes = np.asarray([track.last_observation for track in self.tracks], dtype=np.float64).reshape(-1, 5)
        previous = np.asarray(
            [
                _k_previous_obs(track.observations, track.last_timestamp, self.config.delta_t_seconds)
                for track in self.tracks
            ],
            dtype=np.float64,
        ).reshape(-1, 5)
        track_embeddings = [track.embedding for track in self.tracks]
        appearance = cosine_cost(high_embeddings, track_embeddings)

        matched, unmatched_dets, unmatched_trks = associate(
            dets,
            trks,
            self.config.iou_threshold,
            velocities if len(self.tracks) else np.empty((0, 2), dtype=np.float64),
            previous if len(self.tracks) else np.empty((0, 5), dtype=np.float64),
            self.config.inertia,
            appearance,
            self.config.w_association_emb,
            aw_off=self.config.aw_off,
            aw_param=self.config.aw_param,
        )
        alphas = self._dynamic_appearance_alphas(high_scores)
        for det_index, track_index in matched:
            self._update_track(
                self.tracks[int(track_index)],
                dets[int(det_index)],
                high_embeddings[int(det_index)],
                timestamp,
                alpha=float(alphas[int(det_index)]),
            )

        if len(unmatched_dets) and len(unmatched_trks):
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = iou_batch(left_dets, left_trks)
            left_appearance = cosine_cost(
                [high_embeddings[int(index)] for index in unmatched_dets],
                [track_embeddings[int(index)] for index in unmatched_trks],
            )
            if left_appearance is None:
                left_appearance = np.zeros_like(iou_left)
            if iou_left.size and float(iou_left.max(initial=0.0)) > self.config.iou_threshold:
                pairs = linear_assignment(-(iou_left + left_appearance))
                used_dets: set[int] = set()
                used_trks: set[int] = set()
                for row, col in pairs:
                    if iou_left[row, col] < self.config.iou_threshold:
                        continue
                    det_index = int(unmatched_dets[row])
                    track_index = int(unmatched_trks[col])
                    self._update_track(
                        self.tracks[track_index],
                        dets[det_index],
                        high_embeddings[det_index],
                        timestamp,
                        alpha=float(alphas[det_index]),
                    )
                    used_dets.add(det_index)
                    used_trks.add(track_index)
                unmatched_dets = np.asarray(
                    [index for index in unmatched_dets if int(index) not in used_dets],
                    dtype=np.int64,
                )
                unmatched_trks = np.asarray(
                    [index for index in unmatched_trks if int(index) not in used_trks],
                    dtype=np.int64,
                )

        if appearance is not None and len(unmatched_dets) and len(unmatched_trks):
            leftover = cosine_cost(
                [high_embeddings[int(index)] for index in unmatched_dets],
                [self.tracks[int(index)].embedding for index in unmatched_trks],
            )
            if leftover is not None and leftover.size:
                cost = 1.0 - leftover
                cost[leftover < self.config.reid_recovery_threshold] = INF
                used_dets = set()
                used_trks = set()
                for row, col in linear_assignment(cost):
                    if leftover[row, col] < self.config.reid_recovery_threshold:
                        continue
                    det_index = int(unmatched_dets[row])
                    track_index = int(unmatched_trks[col])
                    self._update_track(
                        self.tracks[track_index],
                        dets[det_index],
                        high_embeddings[det_index],
                        timestamp,
                        alpha=float(alphas[det_index]),
                    )
                    used_dets.add(det_index)
                    used_trks.add(track_index)
                unmatched_dets = np.asarray(
                    [index for index in unmatched_dets if int(index) not in used_dets],
                    dtype=np.int64,
                )
                unmatched_trks = np.asarray(
                    [index for index in unmatched_trks if int(index) not in used_trks],
                    dtype=np.int64,
                )

        for track_index in unmatched_trks:
            track = self.tracks[int(track_index)]
            track.time_since_update += dt
            track.hit_streak = 0
            track.frozen = True

        for det_index in unmatched_dets:
            self._create_track(
                dets[int(det_index)],
                int(high_class_ids[int(det_index)]) if int(det_index) < len(high_class_ids) else 0,
                high_embeddings[int(det_index)],
                timestamp,
            )

        self.tracks = [
            track for track in self.tracks if track.time_since_update <= self.config.max_age_seconds
        ]
        self._last_timestamp = timestamp
        outputs: list[TrackOutput] = []
        for track in self.tracks:
            if track.time_since_update > 1e-6:
                continue
            box = track.last_observation[:4]
            outputs.append(
                TrackOutput(
                    track_id=track.track_id,
                    xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=track.confidence,
                    class_id=track.class_id,
                    confirmed=track.confirmed,
                    embedding=track.embedding,
                )
            )
        return outputs

    def _dynamic_appearance_alphas(self, scores: NDArray[np.floating]) -> NDArray[np.float64]:
        threshold = min(max(self.config.activation_threshold, 0.0), 0.999)
        trust = np.clip((np.asarray(scores, dtype=np.float64) - threshold) / (1.0 - threshold), 0.0, 1.0)
        fixed = self.config.alpha_fixed_emb
        return fixed + (1.0 - fixed) * (1.0 - trust)

    def _update_track(
        self,
        track: TrackState,
        detection: NDArray[np.floating],
        embedding: Embedding | None,
        timestamp: float,
        *,
        alpha: float,
    ) -> None:
        bbox = tuple(float(value) for value in detection[:5])
        xyxy = bbox[:4]
        if track.last_observation[0] >= 0:
            previous = _k_previous_obs(track.observations, track.last_timestamp, self.config.delta_t_seconds)
            if previous[0] < 0:
                previous = track.last_observation
            track.velocity = _speed_direction(previous, xyxy)
        track.last_observation = bbox
        track.observations.append((timestamp, bbox))
        track.xyxy = xyxy
        track.confidence = float(detection[4]) if detection.shape[0] > 4 else track.confidence
        track.hits += 1
        track.hit_streak += 1
        track.time_since_update = 0.0
        track.last_timestamp = timestamp
        track.frozen = False
        track.confirmed = track.hits >= self.config.confirmation_hits
        track.kalman.update(xyxy_to_xywh(xyxy))
        if embedding is not None:
            if track.embedding is None:
                track.embedding = np.asarray(embedding, dtype=np.float32)
            else:
                mixed = alpha * track.embedding + (1.0 - alpha) * np.asarray(embedding, dtype=np.float32)
                norm = float(np.linalg.norm(mixed))
                track.embedding = mixed / norm if norm > 1e-12 else track.embedding

    def _create_track(
        self,
        detection: NDArray[np.floating],
        class_id: int,
        embedding: Embedding | None,
        timestamp: float,
    ) -> None:
        bbox = tuple(float(value) for value in detection[:5])
        xyxy = bbox[:4]
        kalman = DeepOCSortKalman()
        kalman.initiate(xyxy_to_xywh(xyxy))
        track = TrackState(
            track_id=self._next_id,
            xyxy=xyxy,
            confidence=float(detection[4]) if detection.shape[0] > 4 else 1.0,
            class_id=class_id,
            hits=1,
            hit_streak=1,
            time_since_update=0.0,
            last_timestamp=timestamp,
            last_observation=bbox,
            observations=[(timestamp, bbox)],
            embedding=None if embedding is None else np.asarray(embedding, dtype=np.float32),
            kalman=kalman,
            confirmed=self.config.confirmation_hits <= 1,
        )
        self._next_id += 1
        self.tracks.append(track)

    def _apply_cmc(self, track: TrackState, transform: NDArray[np.floating]) -> None:
        if track.last_observation[0] >= 0:
            warped = apply_affine_to_xyxy(track.last_observation[:4], transform)
            track.last_observation = (*warped, track.last_observation[4])
        updated: list[tuple[float, tuple[float, float, float, float, float]]] = []
        for stamp, box in track.observations:
            warped = apply_affine_to_xyxy(box[:4], transform)
            updated.append((stamp, (*warped, box[4])))
        track.observations = updated
        matrix = np.asarray(transform, dtype=np.float64).reshape(2, 3)
        track.kalman.apply_affine(matrix[:, :2], matrix[:, 2])
        track.xyxy = track.kalman.to_xyxy()
