"""OpenCV overlays for automatic vertical queue rows."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from app.analytics.vertical_queue import VerticalQueueSnapshot
from app.core.models import TrackObservation


_ROW_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 160, 0),
    (0, 200, 255),
    (180, 80, 255),
    (80, 220, 80),
    (255, 100, 180),
    (220, 220, 60),
    (60, 140, 255),
    (200, 120, 80),
)


def vertical_row_color(row_id: int) -> tuple[int, int, int]:
    """Return the stable BGR color assigned to one vertical row ID."""

    return _ROW_COLORS[(row_id - 1) % len(_ROW_COLORS)]


def hotter_row_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Shift a queue color toward a brighter red/yellow line accent."""

    blue, green, red = color
    return (
        max(0, blue - 70),
        min(255, green + 55),
        min(255, red + 110),
    )


def annotate_vertical_queues(
    frame: NDArray[np.uint8],
    snapshot: VerticalQueueSnapshot,
    observations: Sequence[TrackObservation] = (),
    *,
    copy: bool = True,
) -> NDArray[np.uint8]:
    """Draw vertical row lines, same-color members, and one count summary."""

    annotated = frame.copy() if copy else frame
    rows_by_track = {
        track_id: row
        for row in snapshot.rows
        for track_id in row.track_ids
    }
    for row in snapshot.rows:
        color = vertical_row_color(row.row_id)
        hot_color = hotter_row_color(color)
        x = int(round(row.center_x))
        top = 0
        bottom = annotated.shape[0] - 1
        cv2.line(
            annotated, (x, top), (x, bottom), hot_color, 11, cv2.LINE_AA
        )
        cv2.line(annotated, (x, top), (x, bottom), color, 5, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"row {row.row_id}: {row.count}",
            (max(0, x + 6), max(18, top + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            hot_color,
            3,
            cv2.LINE_AA,
        )

    for observation in observations:
        row = rows_by_track.get(observation.track_id)
        if row is None:
            continue
        color = vertical_row_color(row.row_id)
        x1, y1, x2, y2 = (int(round(value)) for value in observation.xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            annotated,
            f"person {observation.track_id} row {row.row_id}",
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    if snapshot.rows:
        summary = "Queues | " + " | ".join(
            f"row {row.row_id}: {row.count}" for row in snapshot.rows
        )
    else:
        summary = "Queues | no vertical rows"
    height, width = annotated.shape[:2]
    scale = max(0.65, min(width / 1280.0, height / 720.0))
    line_height = max(26, int(round(34 * scale)))
    cv2.rectangle(
        annotated,
        (0, max(0, height - line_height)),
        (width - 1, height - 1),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        annotated,
        summary,
        (10, height - max(7, int(round(9 * scale)))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65 * scale,
        (255, 255, 255),
        max(1, int(round(2 * scale))),
        cv2.LINE_AA,
    )
    return annotated
