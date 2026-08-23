"""UCMCTrack adapter: official algorithm → shared TrackObservation outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from app.core.models import Detection
from app.tracking.base import BaseTracker, TrackingResult
from app.tracking.calibration.camera import CameraGeometry, CameraGeometryCatalog
from app.tracking.conversion import person_detections
from app.tracking.third_party.ucmctrack import UCMCTrack, UCMCTrackConfig
from app.tracking.trajectories import TrajectoryBook


class UCMCTrackAdapter(BaseTracker):
    """Timestamp-aware UCMCTrack wrapper with optional per-camera geometry."""

    name = "ucmctrack"

    def __init__(
        self,
        *,
        activation_threshold: float = 0.4,
        lost_track_buffer: int = 4,
        match_threshold: float = 0.3,
        history_size: int = 90,
        frame_rate: float = 0.5,
        confirmation_frames: int = 1,
        smoothing_alpha: float = 0.35,
        frame_size: tuple[int, int] | None = None,
        reid_model: str | Path | None = None,
        reid_providers: Sequence[str] = ("CPUExecutionProvider",),
        reid_similarity_threshold: float = 0.65,
        reid_max_age_frames: int = 4,
        reid_interval: int = 1,
        iou_threshold: float | None = None,
        reid_low_threshold: float = 0.3,
        max_age_seconds: float | None = None,
        recovery_seconds: float | None = None,
        track_low_threshold: float = 0.1,
        new_track_threshold: float | None = None,
        wx: float = 5.0,
        wy: float = 5.0,
        vmax: float = 100.0,
        assignment_threshold: float | None = None,
        camera_geometry_dir: str | Path | None = None,
        camera_catalog: CameraGeometryCatalog | None = None,
        camera_geometries: Mapping[str, CameraGeometry] | None = None,
        bbd_threshold: float = 16.0,
        use_visual_tracking: bool = True,
        inertia: float = 0.2,
        w_association_emb: float = 0.75,
        alpha_fixed_emb: float = 0.95,
        aw_param: float = 0.5,
        delta_t_seconds: float | None = None,
        use_cmc: bool = False,
        proximity_thresh: float = 1.0,
        embedding_alpha: float = 0.9,
    ) -> None:
        _ = (
            reid_model,
            reid_providers,
            reid_similarity_threshold,
            reid_max_age_frames,
            reid_interval,
            reid_low_threshold,
            new_track_threshold,
            iou_threshold,
            match_threshold,
            bbd_threshold,
            use_visual_tracking,
            inertia,
            w_association_emb,
            alpha_fixed_emb,
            aw_param,
            delta_t_seconds,
            use_cmc,
            proximity_thresh,
            embedding_alpha,
        )
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("frame_rate must be finite and positive")
        if frame_size is not None and (frame_size[0] <= 0 or frame_size[1] <= 0):
            raise ValueError("frame_size must contain positive width and height")
        frame_gap = 1.0 / frame_rate
        default_recovery = max(1.0, frame_gap)
        if recovery_seconds is not None:
            age_seconds = float(recovery_seconds)
        elif max_age_seconds is not None:
            age_seconds = min(float(max_age_seconds), default_recovery)
        else:
            age_seconds = default_recovery
        _ = lost_track_buffer
        gate = 15.0 if assignment_threshold is None else float(assignment_threshold)
        self.frame_size = frame_size
        self._wx = float(wx)
        self._wy = float(wy)
        self._vmax = float(vmax)
        self._assignment_threshold = gate
        self._catalog = _build_catalog(camera_catalog, camera_geometries, camera_geometry_dir)
        self._book = TrajectoryBook(history_size=history_size, smoothing_alpha=smoothing_alpha)
        self._backend = UCMCTrack(
            UCMCTrackConfig(
                activation_threshold=activation_threshold,
                track_low_threshold=track_low_threshold,
                assignment_threshold=gate,
                max_age_seconds=age_seconds,
                confirmation_hits=max(1, confirmation_frames),
                wx=self._wx,
                wy=self._wy,
                vmax=self._vmax,
            )
        )
        self._last_frame_index: int | None = None
        self._last_alive: set[int] = set()

    def update(
        self,
        detections: Sequence[Detection],
        *,
        camera_id: str,
        timestamp: float,
        frame_index: int,
        frame: np.ndarray | None = None,
        intermediate_frame: np.ndarray | None = None,
    ) -> TrackingResult:
        _ = (frame, intermediate_frame)
        self._validate_frame(camera_id, timestamp, frame_index)
        self._validate_detections(detections, frame_index)
        started = perf_counter()
        people = person_detections(detections)
        boxes = np.asarray([item.xyxy for item in people], dtype=np.float32) if people else np.empty((0, 4), dtype=np.float32)
        scores = (
            np.asarray([item.confidence for item in people], dtype=np.float32)
            if people
            else np.empty((0,), dtype=np.float32)
        )
        class_ids = np.asarray([item.class_id for item in people], dtype=int) if people else np.empty((0,), dtype=int)
        mapper = self._catalog.mapper_for(camera_id)
        wx, wy, vmax, gate = self._catalog.parameters_for(
            camera_id,
            wx=self._wx,
            wy=self._wy,
            vmax=self._vmax,
            assignment_threshold=self._assignment_threshold,
        )
        outputs = self._backend.update(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            timestamp=float(timestamp),
            mapper=mapper,
            wx=wx,
            wy=wy,
            vmax=vmax,
            assignment_threshold=gate,
        )
        observations = [
            self._book.observe(
                camera_id=camera_id,
                track_id=item.track_id,
                timestamp=timestamp,
                frame_index=frame_index,
                xyxy=item.xyxy,
                confidence=item.confidence,
                confirmed=item.confirmed,
                class_id=item.class_id,
            )
            for item in outputs
        ]
        alive = {item.track_id for item in self._backend.tracks}
        expired = tuple(sorted(self._last_alive - alive))
        self._book.prune(alive)
        self._last_alive = alive
        self._last_frame_index = frame_index
        return TrackingResult(
            observations=tuple(sorted(observations, key=lambda item: item.track_id)),
            expired_track_ids=expired,
            tracking_ms=(perf_counter() - started) * 1000.0,
            reid_ms=self._backend.last_reid_ms,
            tracker_name=self.name,
        )

    def reset(self) -> None:
        self._backend.reset()
        self._book.clear()
        self._last_frame_index = None
        self._last_alive = set()

    @property
    def reid_enabled(self) -> bool:
        return False

    @property
    def retained_track_count(self) -> int:
        return len(self._book)

    def _validate_frame(self, camera_id: str, timestamp: float, frame_index: int) -> None:
        if not camera_id.strip():
            raise ValueError("camera_id must be non-empty")
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("frame_index must increase on every update")

    def _validate_detections(self, detections: Sequence[Detection], frame_index: int) -> None:
        if self.frame_size is None:
            return
        width, height = self.frame_size
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.xyxy
            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"invalid zero-area xyxy detection {index} at frame {frame_index}: {detection.xyxy}"
                )
            if x1 < -1.0 or y1 < -1.0 or x2 > width + 1.0 or y2 > height + 1.0:
                raise ValueError(
                    f"xyxy detection {index} is outside {width}x{height} frame at frame "
                    f"{frame_index}: {detection.xyxy}"
                )


def _build_catalog(
    catalog: CameraGeometryCatalog | None,
    geometries: Mapping[str, CameraGeometry] | None,
    directory: str | Path | None,
) -> CameraGeometryCatalog:
    if catalog is not None:
        return catalog
    if geometries is not None:
        return CameraGeometryCatalog(geometries)
    if directory is not None:
        return CameraGeometryCatalog.from_directory(directory)
    return CameraGeometryCatalog()
