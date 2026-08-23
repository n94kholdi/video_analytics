"""UCMCTrack core (arXiv:2312.08952), detector-agnostic and timestamp-aware.

Isolated from application adapters. High/low BYTE-style association with mapped
Mahalanobis distance follows corfyi/UCMCTrack ``tracker/ucmc.py``. Kalman
prediction uses elapsed seconds instead of a unit frame step so 0.5 FPS
processing remains well-defined. Ground-plane mapping is injected; this module
never loads camera files.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from app.tracking.calibration.camera import GroundPlaneMapper, MappedMeasurement
from app.tracking.third_party.stabletrack.matching import intersection_over_union, linear_assignment
from app.tracking.third_party.ucmctrack.kalman import GroundKalman


class TrackStatus(str, Enum):
    tentative = "tentative"
    confirmed = "confirmed"
    coasted = "coasted"


@dataclass(slots=True)
class UCMCTrackConfig:
    """Paper hyperparameters plus 0.5 FPS operating-point knobs."""

    activation_threshold: float = 0.4
    track_low_threshold: float = 0.1
    assignment_threshold: float = 15.0
    second_assignment_threshold: float | None = None
    max_age_seconds: float = 2.0
    confirmation_hits: int = 1
    wx: float = 5.0
    wy: float = 5.0
    vmax: float = 100.0
    iou_keep_threshold: float = 0.25
    iou_cost_weight: float = 2.0


@dataclass(slots=True)
class TrackState:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    hits: int
    time_since_update: float
    last_timestamp: float
    kalman: GroundKalman
    last_xy: NDArray[np.float64]
    confirmed: bool = False
    status: TrackStatus = TrackStatus.tentative


@dataclass(frozen=True, slots=True)
class TrackOutput:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    confirmed: bool


@dataclass(frozen=True, slots=True)
class _MappedDetection:
    index: int
    xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    measurement: MappedMeasurement


class UCMCTrack:
    """BYTE-style association on the mapped ground plane (or image plane)."""

    def __init__(self, config: UCMCTrackConfig | None = None) -> None:
        self.config = config or UCMCTrackConfig()
        self.tracks: list[TrackState] = []
        self._next_id = 1
        self._last_timestamp: float | None = None
        self.last_reid_ms = 0.0

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        self._last_timestamp = None
        self.last_reid_ms = 0.0

    def update(
        self,
        *,
        boxes: NDArray[np.floating],
        scores: NDArray[np.floating],
        class_ids: NDArray[np.integer] | None,
        timestamp: float,
        mapper: GroundPlaneMapper,
        wx: float | None = None,
        wy: float | None = None,
        vmax: float | None = None,
        assignment_threshold: float | None = None,
    ) -> list[TrackOutput]:
        dt = 0.0 if self._last_timestamp is None else max(timestamp - self._last_timestamp, 1e-6)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4) if len(boxes) else np.empty((0, 4), dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1) if len(scores) else np.empty((0,), dtype=np.float32)
        if class_ids is None:
            class_ids = np.zeros(len(boxes), dtype=int)
        else:
            class_ids = np.asarray(class_ids).reshape(-1)

        process_x = float(self.config.wx if wx is None else wx)
        process_y = float(self.config.wy if wy is None else wy)
        speed = float(self.config.vmax if vmax is None else vmax)
        high_gate = float(self.config.assignment_threshold if assignment_threshold is None else assignment_threshold)
        low_gate = float(
            self.config.second_assignment_threshold
            if self.config.second_assignment_threshold is not None
            else high_gate
        )

        detections = self._map_detections(boxes, scores, class_ids, mapper)
        for track in self.tracks:
            if dt > 0:
                track.kalman.predict(dt)
                track.time_since_update += dt

        high = [item for item in detections if item.score >= self.config.activation_threshold]
        low = [
            item
            for item in detections
            if self.config.track_low_threshold <= item.score < self.config.activation_threshold
        ]
        confirmed = [track for track in self.tracks if track.status in {TrackStatus.confirmed, TrackStatus.coasted}]
        tentative = [track for track in self.tracks if track.status is TrackStatus.tentative]

        unmatched_high, unmatched_confirmed = self._associate(
            high, confirmed, high_gate, timestamp, speed, mapper.calibrated
        )
        unmatched_low, unmatched_confirmed = self._associate(
            low, unmatched_confirmed, low_gate, timestamp, speed, mapper.calibrated
        )
        _ = unmatched_low
        unmatched_high, unmatched_tentative = self._associate(
            unmatched_high, tentative, high_gate, timestamp, speed, mapper.calibrated
        )

        for detection in unmatched_high:
            self._start_track(detection, timestamp, process_x, process_y, speed)

        for track in unmatched_confirmed + unmatched_tentative:
            if track.status is TrackStatus.confirmed:
                track.status = TrackStatus.coasted
            track.confirmed = track.status is TrackStatus.confirmed and track.hits >= self.config.confirmation_hits

        self._prune(timestamp)
        self._last_timestamp = timestamp
        self.last_reid_ms = 0.0
        return [
            TrackOutput(track.track_id, track.xyxy, track.confidence, track.class_id, track.confirmed)
            for track in self.tracks
            if track.time_since_update <= 1e-9 and track.confirmed
        ]

    def _map_detections(
        self,
        boxes: NDArray[np.float32],
        scores: NDArray[np.float32],
        class_ids: NDArray[np.integer],
        mapper: GroundPlaneMapper,
    ) -> list[_MappedDetection]:
        mapped: list[_MappedDetection] = []
        for index, box in enumerate(boxes):
            xyxy = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            measurement = mapper.map_box(xyxy)
            if measurement is None:
                continue
            mapped.append(
                _MappedDetection(
                    index,
                    xyxy,
                    float(scores[index]),
                    int(class_ids[index]) if index < len(class_ids) else 0,
                    measurement,
                )
            )
        return mapped

    def _associate(
        self,
        detections: list[_MappedDetection],
        tracks: list[TrackState],
        gate: float,
        timestamp: float,
        vmax: float,
        calibrated: bool,
    ) -> tuple[list[_MappedDetection], list[TrackState]]:
        if not detections or not tracks:
            return detections, tracks
        cost = np.full((len(detections), len(tracks)), 1e5, dtype=np.float64)
        allowed = np.zeros((len(detections), len(tracks)), dtype=bool)
        keep_iou = float(self.config.iou_keep_threshold)
        iou_weight = float(self.config.iou_cost_weight)
        for row, detection in enumerate(detections):
            det_xy = np.asarray(detection.measurement.xy, dtype=np.float64).reshape(2)
            sigma = float(np.sqrt(max(np.trace(detection.measurement.covariance), 1e-9)))
            scale = max(sigma, 1e-3)
            for col, track in enumerate(tracks):
                iou = intersection_over_union(track.xyxy, detection.xyxy)
                predicted = np.asarray(track.kalman.position(), dtype=np.float64).reshape(2)
                last_xy = np.asarray(track.last_xy, dtype=np.float64).reshape(2)
                dist_obs = float(np.linalg.norm(det_xy - last_xy))
                dist_pred = float(np.linalg.norm(det_xy - predicted))
                age = max(timestamp - track.last_timestamp, 1e-6)
                coasted = track.status is TrackStatus.coasted or track.time_since_update > 1e-6
                chi2, mmd = track.kalman.association_scores(
                    det_xy,
                    detection.measurement.covariance,
                )
                budget = self._motion_budget(vmax, age, sigma, track.xyxy, calibrated=calibrated)
                dist = dist_obs if coasted else min(dist_obs, dist_pred)
                if not self._association_allowed(
                    iou=iou,
                    dist=dist,
                    budget=budget,
                    chi2=chi2,
                    gate=gate,
                    keep_iou=keep_iou,
                    coasted=coasted,
                    calibrated=calibrated,
                ):
                    continue
                allowed[row, col] = True
                if calibrated:
                    # Paper ranking: mapped Mahalanobis (MMD).
                    cost[row, col] = mmd
                else:
                    # Pixel-space MMD is poorly scaled at 0.5 FPS.
                    cost[row, col] = dist / scale + iou_weight * (1.0 - iou) + 0.1 * min(chi2, 50.0)
        matches = [
            (row, col)
            for row, col in linear_assignment(cost)
            if allowed[row, col]
        ]
        used_dets = {row for row, _col in matches}
        used_tracks = {col for _row, col in matches}
        for row, col in matches:
            self._update_track(tracks[col], detections[row], timestamp, vmax)
        leftover_dets = [item for index, item in enumerate(detections) if index not in used_dets]
        leftover_tracks = [item for index, item in enumerate(tracks) if index not in used_tracks]
        return leftover_dets, leftover_tracks

    @staticmethod
    def _motion_budget(
        vmax: float,
        age: float,
        sigma: float,
        xyxy: tuple[float, float, float, float],
        *,
        calibrated: bool,
    ) -> float:
        mapped_budget = float(vmax) * age + 3.0 * sigma
        if calibrated:
            return mapped_budget
        box_height = max(float(xyxy[3] - xyxy[1]), 1.0)
        return min(float(vmax) * age, 1.5 * box_height) + 3.0 * sigma

    @staticmethod
    def _association_allowed(
        *,
        iou: float,
        dist: float,
        budget: float,
        chi2: float,
        gate: float,
        keep_iou: float,
        coasted: bool,
        calibrated: bool,
    ) -> bool:
        if iou >= keep_iou or dist <= budget:
            return True
        if calibrated and not coasted and chi2 <= gate:
            return True
        return False

    def _start_track(
        self,
        detection: _MappedDetection,
        timestamp: float,
        wx: float,
        wy: float,
        vmax: float,
    ) -> None:
        hits = 1
        confirmed = hits >= self.config.confirmation_hits
        self.tracks.append(
            TrackState(
                track_id=self._next_id,
                xyxy=detection.xyxy,
                confidence=detection.score,
                class_id=detection.class_id,
                hits=hits,
                time_since_update=0.0,
                last_timestamp=timestamp,
                kalman=GroundKalman(
                    detection.measurement.xy,
                    detection.measurement.covariance,
                    wx=wx,
                    wy=wy,
                    vmax=vmax,
                ),
                last_xy=np.asarray(detection.measurement.xy, dtype=np.float64).reshape(2).copy(),
                confirmed=confirmed,
                status=TrackStatus.confirmed if confirmed else TrackStatus.tentative,
            )
        )
        self._next_id += 1

    def _update_track(
        self,
        track: TrackState,
        detection: _MappedDetection,
        timestamp: float,
        vmax: float,
    ) -> None:
        iou = intersection_over_union(track.xyxy, detection.xyxy)
        track.kalman.update(detection.measurement.xy, detection.measurement.covariance)
        track.kalman.limit_speed(vmax)
        if iou >= self.config.iou_keep_threshold:
            track.kalman.dampen_velocity(0.25)
        track.xyxy = detection.xyxy
        track.confidence = detection.score
        track.class_id = detection.class_id
        track.hits += 1
        track.time_since_update = 0.0
        track.last_timestamp = timestamp
        track.last_xy = np.asarray(detection.measurement.xy, dtype=np.float64).reshape(2).copy()
        if track.hits >= self.config.confirmation_hits:
            track.confirmed = True
            track.status = TrackStatus.confirmed
        elif track.status is TrackStatus.coasted:
            track.status = TrackStatus.confirmed

    def _prune(self, timestamp: float) -> None:
        tentative_limit = max(self.config.max_age_seconds / 2.0, 2.0)
        kept: list[TrackState] = []
        for track in self.tracks:
            age = timestamp - track.last_timestamp
            if track.status is TrackStatus.tentative and age > tentative_limit:
                continue
            if age > self.config.max_age_seconds:
                continue
            kept.append(track)
        self.tracks = kept
