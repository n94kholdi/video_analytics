"""OpenCV counter overlays for annotated video frames."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from app.analytics.counting import CountingSnapshot


def annotate_people_counts(
    frame: NDArray[np.uint8],
    snapshot: CountingSnapshot,
    *,
    confirmed_humans: int | None = None,
    copy: bool = True,
) -> NDArray[np.uint8]:
    """Add visible-track, zone occupancy, and line totals to a frame."""

    annotated = frame.copy() if copy else frame
    rows = [
        *(
            (f"confirmed humans: {confirmed_humans}",)
            if confirmed_humans is not None
            else ()
        ),
        *(f"occupancy {item.zone_id}: {item.current}" for item in snapshot.occupancy),
        *(
            f"line {item.line_id}: entries {item.entries} exits {item.exits}"
            for item in snapshot.lines
        ),
    ]
    if not rows:
        rows.append("people counting: no configured geometry")
    height, width = annotated.shape[:2]
    scale = max(1.0, min(width / 1280.0, height / 720.0))
    margin = int(round(6 * scale))
    line_height = int(round(24 * scale))
    panel_height = margin * 2 + line_height * len(rows)
    panel_width = min(width - margin, int(round(430 * scale)))
    cv2.rectangle(
        annotated,
        (margin, margin),
        (panel_width, panel_height),
        (20, 20, 20),
        -1,
    )
    for index, text in enumerate(rows):
        cv2.putText(
            annotated,
            text,
            (
                int(round(14 * scale)),
                int(round(26 * scale)) + index * line_height,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * scale,
            (255, 255, 255),
            max(1, int(round(scale))),
            cv2.LINE_AA,
        )
    return annotated
