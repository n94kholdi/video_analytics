"""BoT-SORT core (arXiv:2206.14651), detector-agnostic and timestamp-aware.

Isolated from application adapters. Two-stage ByteTrack association, width-height
Kalman, IoU–ReID min-fusion, and optional GMC follow NirAharon/BoT-SORT
``tracker/bot_sort.py``. Kalman prediction uses elapsed seconds instead of a
unit frame step so 0.5 FPS processing remains well-defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from app.tracking.third_party.botsort.association import associate, associate_center
from app.tracking.third_party.botsort.kalman import BoTSortKalman, xyxy_to_xywh
from app.tracking.third_party.deepocsort.cmc import SparseFlowCMC, apply_affine_to_xyxy

Embedding = NDArray[np.float32]
EmbedFn = Callable[[NDArray[np.uint8], Sequence[float]], Embedding | None]


@dataclass(slots=True)
class BoTSortConfig:
    """Paper hyperparameters plus 0.5 FPS operating-point knobs."""

    activation_threshold: float = 0.4
    track_low_threshold: float = 0.1
    new_track_threshold: float = 0.4
    iou_threshold: float = 0.3
    second_match_threshold: float = 0.5
    unconfirmed_match_threshold: float = 0.7
    max_age_seconds: float = 8.0
    confirmation_hits: int = 1
    proximity_thresh: float = 1.0
    appearance_thresh: float = 0.25
    embedding_alpha: float = 0.9
    use_cmc: bool = False
    sparse_dt_seconds: float = 0.4
    center_match_threshold: float = 4.0


@dataclass(slots=True)
class TrackState:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    hits: int
    time_since_update: float
    last_timestamp: float
    last_observation: tuple[float, float, float, float]
    embedding: Embedding | None = None
    kalman: BoTSortKalman = field(default_factory=BoTSortKalman)
    confirmed: bool = False


@dataclass(frozen=True, slots=True)
class TrackOutput:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    confirmed: bool
    embedding: Embedding | None = None


class BoTSort:
    """ByteTrack associations + BoT-SORT Kalman, GMC, and optional ReID fusion."""

    def __init__(self, config: BoTSortConfig | None = None) -> None:
        self.config = config or BoTSortConfig()
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

        high_mask = scores >= self.config.activation_threshold
        low_mask = (scores >= self.config.track_low_threshold) & ~high_mask
        high_indices = np.flatnonzero(high_mask)
        low_indices = np.flatnonzero(low_mask)

        embeddings: list[Embedding | None] = [None] * len(boxes)
        reid_started = perf_counter()
        if embed is not None and frame is not None:
            for index in high_indices:
                embeddings[int(index)] = embed(frame, boxes[int(index)])
        self.last_reid_ms = (perf_counter() - reid_started) * 1000.0 if embed is not None else 0.0

        for track in self.tracks:
            track.kalman.predict(dt, freeze_scale_velocity=track.time_since_update > 1e-6)
            box = track.kalman.to_xyxy()
            if np.isfinite(box).all():
                track.xyxy = box

        if self._cmc is not None and frame is not None:
            transform = self._cmc.compute_affine(frame, boxes)
            for track in self.tracks:
                self._apply_cmc(track, transform)

        self.tracks = [track for track in self.tracks if np.isfinite(track.xyxy).all()]
        sparse = dt >= self.config.sparse_dt_seconds

        high_boxes, high_scores, high_class_ids, high_embeddings = self._slice(
            boxes, scores, class_ids, embeddings, high_indices
        )
        low_boxes, low_scores, _low_class_ids, _low_embeddings = self._slice(
            boxes, scores, class_ids, embeddings, low_indices
        )

        confirmed_indices = [index for index, track in enumerate(self.tracks) if track.confirmed]
        unconfirmed_indices = [index for index, track in enumerate(self.tracks) if not track.confirmed]
        first_thresh = max(1.0 - self.config.iou_threshold, 1e-6)
        matched_high, unmatched_high, unmatched_confirmed_local = associate(
            high_boxes,
            self._association_boxes(confirmed_indices, sparse=sparse),
            match_thresh=first_thresh,
            det_scores=high_scores,
            det_embeddings=high_embeddings if embed is not None else None,
            track_embeddings=[self.tracks[index].embedding for index in confirmed_indices]
            if embed is not None
            else None,
            proximity_thresh=self.config.proximity_thresh,
            appearance_thresh=self.config.appearance_thresh,
            fuse_det_score=not sparse,
            alternate_boxes=self._observation_boxes(confirmed_indices) if not sparse else None,
        )
        used_tracks: set[int] = set()
        for det_index, local_index in matched_high:
            track_index = confirmed_indices[local_index]
            self._update_track(
                self.tracks[track_index],
                high_boxes[det_index],
                float(high_scores[det_index]),
                high_embeddings[det_index],
                timestamp,
            )
            used_tracks.add(track_index)

        leftover_confirmed = [
            confirmed_indices[local]
            for local in unmatched_confirmed_local
            if confirmed_indices[local] not in used_tracks
        ]
        leftover_high = list(unmatched_high)
        leftover_high_boxes = high_boxes[leftover_high] if leftover_high else np.empty((0, 4), dtype=np.float32)
        leftover_high_scores = high_scores[leftover_high] if leftover_high else np.empty((0,), dtype=np.float32)
        leftover_high_embeddings = [high_embeddings[index] for index in leftover_high]
        recovered, unmatched_high_local, _unmatched_confirmed = associate_center(
            leftover_high_boxes,
            self._observation_boxes(leftover_confirmed),
            delta_tau=dt,
            match_thresh=self.config.center_match_threshold,
        )
        for local_det, local_track in recovered:
            det_index = leftover_high[local_det]
            track_index = leftover_confirmed[local_track]
            self._update_track(
                self.tracks[track_index],
                high_boxes[det_index],
                float(high_scores[det_index]),
                leftover_high_embeddings[local_det],
                timestamp,
            )
            used_tracks.add(track_index)
        unmatched_high = [leftover_high[index] for index in unmatched_high_local]

        leftover_tracked = [
            track_index
            for track_index in leftover_confirmed
            if track_index not in used_tracks and self.tracks[track_index].time_since_update <= 1e-6
        ]
        second_matched, _unmatched_low, _unmatched_tracked = associate(
            low_boxes,
            self._association_boxes(leftover_tracked, sparse=sparse),
            match_thresh=self.config.second_match_threshold,
            det_scores=low_scores,
            fuse_det_score=False,
        )
        for det_index, local_index in second_matched:
            track_index = leftover_tracked[local_index]
            self._update_track(
                self.tracks[track_index],
                low_boxes[det_index],
                float(low_scores[det_index]),
                None,
                timestamp,
            )
            used_tracks.add(track_index)

        leftover_unconfirmed = [index for index in unconfirmed_indices if index not in used_tracks]
        leftover_high = list(unmatched_high)
        leftover_high_boxes = high_boxes[leftover_high] if leftover_high else np.empty((0, 4), dtype=np.float32)
        leftover_high_scores = high_scores[leftover_high] if leftover_high else np.empty((0,), dtype=np.float32)
        third_matched, unmatched_high_local, _unmatched_unconfirmed = associate(
            leftover_high_boxes,
            self._association_boxes(leftover_unconfirmed, sparse=sparse),
            match_thresh=self.config.unconfirmed_match_threshold,
            det_scores=leftover_high_scores,
            fuse_det_score=not sparse,
        )
        for local_det, local_track in third_matched:
            det_index = leftover_high[local_det]
            track_index = leftover_unconfirmed[local_track]
            self._update_track(
                self.tracks[track_index],
                high_boxes[det_index],
                float(high_scores[det_index]),
                high_embeddings[det_index],
                timestamp,
            )
            used_tracks.add(track_index)
        unmatched_high = [leftover_high[index] for index in unmatched_high_local]

        for track_index, track in enumerate(self.tracks):
            if track_index in used_tracks:
                continue
            track.time_since_update += dt

        for det_index in unmatched_high:
            if float(high_scores[det_index]) < self.config.new_track_threshold:
                continue
            self._create_track(
                high_boxes[det_index],
                float(high_scores[det_index]),
                int(high_class_ids[det_index]) if det_index < len(high_class_ids) else 0,
                high_embeddings[det_index],
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

    def _slice(
        self,
        boxes: NDArray[np.floating],
        scores: NDArray[np.floating],
        class_ids: NDArray[np.integer],
        embeddings: Sequence[Embedding | None],
        indices: NDArray[np.integer],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int_], list[Embedding | None]]:
        if len(indices) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=int),
                [],
            )
        return (
            boxes[indices],
            scores[indices],
            class_ids[indices],
            [embeddings[int(index)] for index in indices],
        )

    def _track_boxes(self, indices: Sequence[int] | None = None) -> NDArray[np.float32]:
        selected = self.tracks if indices is None else [self.tracks[index] for index in indices]
        if not selected:
            return np.empty((0, 4), dtype=np.float32)
        return np.asarray([track.xyxy for track in selected], dtype=np.float32)

    def _observation_boxes(self, indices: Sequence[int] | None = None) -> NDArray[np.float32]:
        selected = self.tracks if indices is None else [self.tracks[index] for index in indices]
        if not selected:
            return np.empty((0, 4), dtype=np.float32)
        return np.asarray([track.last_observation for track in selected], dtype=np.float32)

    def _association_boxes(self, indices: Sequence[int], *, sparse: bool) -> NDArray[np.float32]:
        return self._observation_boxes(indices) if sparse else self._track_boxes(indices)

    def _update_track(
        self,
        track: TrackState,
        xyxy: NDArray[np.floating],
        confidence: float,
        embedding: Embedding | None,
        timestamp: float,
    ) -> None:
        box = tuple(float(value) for value in xyxy[:4])
        track.xyxy = box
        track.last_observation = box
        track.confidence = confidence
        track.hits += 1
        track.time_since_update = 0.0
        track.last_timestamp = timestamp
        track.confirmed = track.hits >= self.config.confirmation_hits
        track.kalman.update(xyxy_to_xywh(box))
        if embedding is None:
            return
        current = np.asarray(embedding, dtype=np.float32)
        if track.embedding is None:
            track.embedding = current
            return
        mixed = self.config.embedding_alpha * track.embedding + (1.0 - self.config.embedding_alpha) * current
        norm = float(np.linalg.norm(mixed))
        track.embedding = mixed / norm if norm > 1e-12 else track.embedding

    def _create_track(
        self,
        xyxy: NDArray[np.floating],
        confidence: float,
        class_id: int,
        embedding: Embedding | None,
        timestamp: float,
    ) -> None:
        box = tuple(float(value) for value in xyxy[:4])
        kalman = BoTSortKalman()
        kalman.initiate(xyxy_to_xywh(box))
        self.tracks.append(
            TrackState(
                track_id=self._next_id,
                xyxy=box,
                confidence=confidence,
                class_id=class_id,
                hits=1,
                time_since_update=0.0,
                last_timestamp=timestamp,
                last_observation=box,
                embedding=None if embedding is None else np.asarray(embedding, dtype=np.float32),
                kalman=kalman,
                confirmed=self.config.confirmation_hits <= 1,
            )
        )
        self._next_id += 1

    def _apply_cmc(self, track: TrackState, transform: NDArray[np.floating]) -> None:
        matrix = np.asarray(transform, dtype=np.float64).reshape(2, 3)
        track.kalman.apply_affine(matrix[:, :2], matrix[:, 2])
        track.xyxy = apply_affine_to_xyxy(track.xyxy, transform)
        track.last_observation = apply_affine_to_xyxy(track.last_observation, transform)
