"""StableTrack adapter: paper backend → shared TrackObservation outputs."""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from app.core.models import Detection
from app.tracking.base import BaseTracker, TrackingResult
from app.tracking.conversion import person_detections
from app.tracking.reid import OsNetReIdentifier
from app.tracking.third_party.stabletrack import StableTrack, StableTrackConfig
from app.tracking.trajectories import TrajectoryBook


class StableTrackAdapter(BaseTracker):
    """Timestamp-aware StableTrack wrapper with normalized observations."""

    name = "stabletrack"

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
        bbd_threshold: float = 16.0,
        iou_threshold: float = 0.4,
        reid_low_threshold: float = 0.3,
        max_age_seconds: float | None = None,
        use_visual_tracking: bool = True,
    ) -> None:
        _ = (match_threshold, reid_max_age_frames, reid_interval)
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("frame_rate must be finite and positive")
        if frame_size is not None and (frame_size[0] <= 0 or frame_size[1] <= 0):
            raise ValueError("frame_size must contain positive width and height")
        age_seconds = (
            float(max_age_seconds)
            if max_age_seconds is not None
            else max(lost_track_buffer, 1) / frame_rate
        )
        self.frame_size = frame_size
        self._book = TrajectoryBook(history_size=history_size, smoothing_alpha=smoothing_alpha)
        self._reidentifier = (
            OsNetReIdentifier(reid_model, providers=reid_providers)
            if reid_model is not None
            else None
        )
        self._backend = StableTrack(
            StableTrackConfig(
                activation_threshold=activation_threshold,
                bbd_threshold=bbd_threshold,
                iou_threshold=iou_threshold,
                reid_high_threshold=reid_similarity_threshold,
                reid_low_threshold=reid_low_threshold,
                max_age_seconds=age_seconds,
                confirmation_hits=max(1, confirmation_frames),
                use_visual_tracking=use_visual_tracking,
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
        self._validate_frame(camera_id, timestamp, frame_index)
        self._validate_detections(detections, frame_index)
        if self._reidentifier is not None and frame is None:
            raise ValueError("frame is required when OSNet ReID is enabled")
        started = perf_counter()
        people = person_detections(detections)
        boxes = np.asarray([item.xyxy for item in people], dtype=np.float32) if people else np.empty((0, 4), dtype=np.float32)
        scores = (
            np.asarray([item.confidence for item in people], dtype=np.float32)
            if people
            else np.empty((0,), dtype=np.float32)
        )
        class_ids = np.asarray([item.class_id for item in people], dtype=int) if people else np.empty((0,), dtype=int)
        embed = self._reidentifier.embed if self._reidentifier is not None else None
        outputs = self._backend.update(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            timestamp=float(timestamp),
            frame=frame,
            intermediate_frame=intermediate_frame,
            embed=embed,
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
        return self._reidentifier is not None

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
