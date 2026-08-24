"""OpenCV annotation for shared track observations."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from app.core.models import TrackObservation, TrajectoryPoint

TRAIL_MAX_SECONDS = 4.0


def annotate_tracks(
    frame: NDArray[np.uint8],
    observations: Sequence[TrackObservation],
    *,
    tracking_ms: float | None = None,
    show_trajectories: bool = True,
    current_people: int | None = None,
    total_unique_people: int | None = None,
    tracker_name: str | None = None,
    max_trail_seconds: float = TRAIL_MAX_SECONDS,
) -> NDArray[np.uint8]:
    """Draw track metadata and optionally draw recent trajectory trails."""

    annotated = frame.copy()
    height, width = annotated.shape[:2]
    scale = max(1.0, min(width / 1280.0, height / 720.0))
    box_thickness = max(3, int(round(3 * scale)))
    trail_thickness = max(3, int(round(3 * scale)))
    font_scale = 0.65 * scale
    text_thickness = max(2, int(round(2 * scale)))
    for observation in observations:
        color = _track_color(observation.track_id) if observation.confirmed else (0, 140, 255)
        points = _recent_trail_points(observation, max_trail_seconds)
        if show_trajectories and len(points) > 1:
            cv2.polylines(annotated, [points], False, color, trail_thickness, cv2.LINE_AA)
        x1, y1, x2, y2 = (int(round(value)) for value in observation.xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 0), box_thickness + 2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)
        cv2.circle(
            annotated,
            tuple(int(round(value)) for value in observation.foot_point),
            max(5, int(round(5 * scale))),
            color,
            -1,
            cv2.LINE_AA,
        )
        state = "confirmed" if observation.confirmed else "unconfirmed"
        label = f"ID {observation.track_id} {state}"
        if observation.speed_pixels_per_second is not None:
            label += f" {observation.speed_pixels_per_second:.1f} px/s"
        if observation.speed_metres_per_second is not None:
            label += f" {observation.speed_metres_per_second:.2f} m/s"
        _draw_label(
            annotated,
            label,
            (x1, max(int(round(22 * scale)), y1 - int(round(10 * scale)))),
            color,
            font_scale,
            text_thickness,
        )
    if tracking_ms is not None:
        prefix = f"{tracker_name} " if tracker_name else ""
        _draw_label(
            annotated,
            f"{prefix}tracking {tracking_ms:.1f} ms",
            (10, int(round(28 * scale))),
            (255, 255, 255),
            0.7 * scale,
            text_thickness,
        )
    count_rows = (
        *(() if current_people is None else (f"people now: {current_people}",)),
        *(
            ()
            if total_unique_people is None
            else (f"unique people total: {total_unique_people}",)
        ),
    )
    for index, text in enumerate(count_rows):
        _draw_label(
            annotated,
            text,
            (10, int(round(56 * scale)) + index * int(round(28 * scale))),
            (255, 255, 255),
            0.7 * scale,
            text_thickness,
        )
    return annotated


def _recent_trail_points(
    observation: TrackObservation,
    max_trail_seconds: float,
) -> NDArray[np.int32]:
    cutoff = float(observation.timestamp) - max(0.0, float(max_trail_seconds))
    recent: list[TrajectoryPoint] = [
        sample for sample in observation.trajectory if sample.timestamp >= cutoff
    ]
    if not recent:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray([sample.smoothed_position for sample in recent], dtype=np.int32)


def _draw_label(
    image: NDArray[np.uint8],
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, origin, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, font, font_scale, color, thickness, cv2.LINE_AA)


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Saturated BGR colors so boxes stay readable on bright market footage."""

    hue = int((track_id * 37) % 180)
    hsv = np.uint8([[[hue, 230, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])
