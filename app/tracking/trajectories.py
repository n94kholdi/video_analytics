"""Bounded, tracker-independent trajectory history and foot-point smoothing."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from app.core.models import TrackObservation, TrajectoryPoint


def foot_point(xyxy: Sequence[float]) -> tuple[float, float]:
    """Return the bottom-center image point for an ``xyxy`` box."""

    x1, _y1, x2, y2 = xyxy
    return ((float(x1) + float(x2)) / 2.0, float(y2))


def smooth_point(
    point: tuple[float, float],
    history: Sequence[TrajectoryPoint],
    alpha: float,
) -> tuple[float, float]:
    if not history:
        return point
    previous = history[-1].smoothed_position
    return (
        alpha * point[0] + (1.0 - alpha) * previous[0],
        alpha * point[1] + (1.0 - alpha) * previous[1],
    )


class TrajectoryBook:
    """Retain EMA-smoothed foot-point histories keyed by public track ID."""

    def __init__(self, *, history_size: int, smoothing_alpha: float) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        self.history_size = history_size
        self.smoothing_alpha = smoothing_alpha
        self._histories: dict[int, deque[TrajectoryPoint]] = {}

    def __len__(self) -> int:
        return len(self._histories)

    def observe(
        self,
        *,
        camera_id: str,
        track_id: int,
        timestamp: float,
        frame_index: int,
        xyxy: tuple[float, float, float, float],
        confidence: float,
        confirmed: bool,
        class_id: int = 0,
    ) -> TrackObservation:
        raw_point = foot_point(xyxy)
        history = self._histories.setdefault(track_id, deque(maxlen=self.history_size))
        smoothed = smooth_point(raw_point, history, self.smoothing_alpha)
        history.append(
            TrajectoryPoint(
                timestamp=float(timestamp),
                frame_index=frame_index,
                position=raw_point,
                smoothed_position=smoothed,
            )
        )
        return TrackObservation(
            camera_id=camera_id,
            track_id=track_id,
            timestamp=float(timestamp),
            frame_index=frame_index,
            xyxy=xyxy,
            foot_point=raw_point,
            detection_confidence=confidence,
            confirmed=confirmed,
            trajectory=tuple(history),
            class_id=class_id,
        )

    def prune(self, alive_track_ids: set[int]) -> tuple[int, ...]:
        expired = tuple(sorted(track_id for track_id in self._histories if track_id not in alive_track_ids))
        for track_id in expired:
            del self._histories[track_id]
        return expired

    def pop(self, track_id: int) -> deque[TrajectoryPoint] | None:
        return self._histories.pop(track_id, None)

    def transfer(self, source_id: int, destination_id: int) -> None:
        history = self._histories.pop(source_id, None)
        if history is not None:
            self._histories[destination_id] = history

    def clear(self) -> None:
        self._histories.clear()
