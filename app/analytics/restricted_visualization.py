"""OpenCV restricted-zone status and intrusion overlays."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from app.analytics.restricted_area import (
    CameraRestrictedAreaConfig,
    IntrusionState,
    RestrictedAreaSnapshot,
)
from app.core.models import TrackObservation


def annotate_restricted_areas(
    frame: NDArray[np.uint8],
    config: CameraRestrictedAreaConfig,
    snapshot: RestrictedAreaSnapshot,
    observations: Sequence[TrackObservation] = (),
    *,
    copy: bool = True,
) -> NDArray[np.uint8]:
    """Draw named zone state plus boxes and foot points for intruding tracks."""

    if snapshot.camera_id != config.camera_id:
        raise ValueError("snapshot and restricted-area config cameras must match")
    annotated = frame.copy() if copy else frame
    overlay = annotated.copy()
    statuses = {item.zone_id: item for item in snapshot.zones}
    for zone in config.zones:
        status = statuses[zone.zone_id]
        points = np.rint(zone.pixel_points(config.frame_size)).astype(np.int32)
        if not status.active:
            color = (128, 128, 128)
            label = f"{zone.zone_id}: inactive"
        elif status.confirmed_tracks:
            color = (0, 0, 255)
            label = (
                f"{zone.zone_id}: current {status.current_tracks} "
                f"entries {status.cumulative_entries} exits {status.cumulative_exits}"
            )
        elif status.entered_tracks:
            color = (0, 200, 255)
            label = (
                f"{zone.zone_id}: current {status.current_tracks} "
                f"entries {status.cumulative_entries} exits {status.cumulative_exits}"
            )
        else:
            color = (0, 180, 0)
            label = (
                f"{zone.zone_id}: current 0 entries {status.cumulative_entries} "
                f"exits {status.cumulative_exits}"
            )
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(annotated, [points], True, color, 2, cv2.LINE_AA)
        anchor = tuple(points[np.argmin(points[:, 1])].tolist())
        cv2.putText(
            annotated,
            label,
            (anchor[0], max(16, anchor[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.14, annotated, 0.86, 0, annotated)

    track_states: dict[int, IntrusionState] = {}
    for item in snapshot.tracks:
        if item.state is IntrusionState.EXITED:
            continue
        previous = track_states.get(item.track_id)
        if previous is None or item.state is IntrusionState.CONFIRMED:
            track_states[item.track_id] = item.state
    for observation in observations:
        state = track_states.get(observation.track_id)
        if state is None:
            continue
        color = (0, 0, 255) if state is IntrusionState.CONFIRMED else (0, 200, 255)
        x1, y1, x2, y2 = (int(round(value)) for value in observation.xyxy)
        foot = tuple(int(round(value)) for value in observation.foot_point)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, foot, 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"intrusion {observation.track_id}: {state.value}",
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated
