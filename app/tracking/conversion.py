"""Shared detection conversion used by every tracker adapter."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import supervision as sv

from app.core.models import Detection


def is_person_detection(detection: Detection) -> bool:
    """Return whether a detection is labelled as a person."""

    if detection.class_id != 0:
        return False
    return detection.class_name is None or detection.class_name.lower() == "person"


def person_detections(detections: Sequence[Detection]) -> tuple[Detection, ...]:
    """Keep only person detections for tracker input."""

    return tuple(item for item in detections if is_person_detection(item))


def detections_to_xyxy(detections: Sequence[Detection]) -> np.ndarray:
    """Return an ``(N, 4)`` float32 array of person boxes, or empty."""

    people = person_detections(detections)
    if not people:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray([item.xyxy for item in people], dtype=np.float32)


def detections_to_supervision(detections: Sequence[Detection]) -> sv.Detections:
    """Convert shared person detections to Supervision's tracker input."""

    people = person_detections(detections)
    if not people:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.asarray([item.xyxy for item in people], dtype=np.float32),
        confidence=np.asarray([item.confidence for item in people], dtype=np.float32),
        class_id=np.zeros(len(people), dtype=int),
    )
