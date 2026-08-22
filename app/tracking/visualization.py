"""OpenCV annotation for shared track observations."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from app.core.models import TrackObservation


def annotate_tracks(
    frame: NDArray[np.uint8],
    observations: Sequence[TrackObservation],
    *,
    tracking_ms: float | None = None,
    show_trajectories: bool = True,
    current_people: int | None = None,
    total_unique_people: int | None = None,
    tracker_name: str | None = None,
) -> NDArray[np.uint8]:
    """Draw track metadata and optionally draw smoothed trajectory trails."""

    annotated = frame.copy()
    for observation in observations:
        color = _track_color(observation.track_id) if observation.confirmed else (0, 180, 255)
        points = np.asarray(
            [sample.smoothed_position for sample in observation.trajectory],
            dtype=np.int32,
        )
        if show_trajectories and len(points) > 1:
            cv2.polylines(annotated, [points], False, color, 2, cv2.LINE_AA)
        x1, y1, x2, y2 = (int(round(value)) for value in observation.xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(
            annotated,
            tuple(int(round(value)) for value in observation.foot_point),
            4,
            color,
            -1,
            cv2.LINE_AA,
        )
        state = "confirmed" if observation.confirmed else "unconfirmed"
        label = f"person #{observation.track_id} {state}"
        if observation.speed_pixels_per_second is not None:
            label += f" {observation.speed_pixels_per_second:.1f} px/s"
        if observation.speed_metres_per_second is not None:
            label += f" {observation.speed_metres_per_second:.2f} m/s"
        cv2.putText(
            annotated,
            label,
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    if tracking_ms is not None:
        prefix = f"{tracker_name} " if tracker_name else ""
        cv2.putText(
            annotated,
            f"{prefix}tracking {tracking_ms:.1f} ms",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
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
        cv2.putText(
            annotated,
            text,
            (10, 50 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def _track_color(track_id: int) -> tuple[int, int, int]:
    return (
        64 + (track_id * 53) % 192,
        64 + (track_id * 97) % 192,
        64 + (track_id * 151) % 192,
    )
