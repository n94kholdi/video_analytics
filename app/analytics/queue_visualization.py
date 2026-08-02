"""OpenCV overlays for configured heuristic queue analytics."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from app.analytics.queue import CameraQueueConfig, QueueSnapshot, QueueTrackState
from app.core.models import TrackObservation


_QUEUE_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 160, 0),
    (0, 200, 255),
    (180, 80, 255),
    (80, 220, 80),
    (255, 100, 180),
    (220, 220, 60),
    (60, 140, 255),
    (200, 120, 80),
)


def annotate_queues(
    frame: NDArray[np.uint8],
    config: CameraQueueConfig,
    snapshot: QueueSnapshot,
    observations: Sequence[TrackObservation] = (),
    *,
    copy: bool = True,
) -> NDArray[np.uint8]:
    """Draw queue polygons, service points, metrics, and per-track status."""

    if snapshot.camera_id != config.camera_id:
        raise ValueError("snapshot and queue config cameras must match")
    annotated = frame.copy() if copy else frame
    overlay = annotated.copy()
    statuses = {item.queue_id: item for item in snapshot.queues}
    queue_colors = {
        queue.queue_id: _QUEUE_COLORS[index % len(_QUEUE_COLORS)]
        for index, queue in enumerate(config.queues)
    }
    for queue in config.queues:
        status = statuses[queue.queue_id]
        color = queue_colors[queue.queue_id]
        points = np.rint(
            [point.to_pixels(config.frame_size) for point in queue.polygon]
        ).astype(np.int32)
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(annotated, [points], True, color, 2, cv2.LINE_AA)
        service = tuple(
            int(round(value))
            for value in queue.service_point.point.to_pixels(config.frame_size)
        )
        cv2.drawMarker(
            annotated, service, (255, 255, 255), cv2.MARKER_CROSS, 16, 2
        )
        anchor = tuple(points[np.argmin(points[:, 1])].tolist())
        cv2.putText(
            annotated,
            queue.queue_id,
            (anchor[0], max(16, anchor[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.10, annotated, 0.90, 0, annotated)

    by_track: dict[int, list[tuple[str, QueueTrackState]]] = {}
    for track in snapshot.tracks:
        by_track.setdefault(track.track_id, []).append((track.queue_id, track.state))
    queue_order = {queue.queue_id: index for index, queue in enumerate(config.queues)}
    for observation in observations:
        memberships = by_track.get(observation.track_id)
        if not memberships:
            continue
        selected_queue, selected_state = min(
            memberships,
            key=lambda item: (
                item[1] is not QueueTrackState.MEMBER,
                queue_order[item[0]],
            ),
        )
        color = queue_colors[selected_queue]
        x1, y1, x2, y2 = (int(round(value)) for value in observation.xyxy)
        thickness = 3 if selected_state is QueueTrackState.MEMBER else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        labels = ",".join(
            f"{queue_id}:{state.value}" for queue_id, state in memberships
        )
        cv2.putText(
            annotated,
            f"queue {observation.track_id} {labels}",
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )

    if config.queues:
        summary = "Queues | " + " | ".join(
            f"{status.queue_id}: {status.raw_count}"
            + (" OVERFLOW" if status.overflow else "")
            for status in snapshot.queues
        )
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
