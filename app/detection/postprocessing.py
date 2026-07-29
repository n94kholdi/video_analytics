"""Output parsing and non-maximum suppression for the light person model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.core.models import Detection
from app.detection.preprocessing import LetterboxTransform


class DetectorOutputError(ValueError):
    """Raised when detector output does not match the supported semantics."""


def parse_light_output(
    raw_output: NDArray[np.floating],
    transform: LetterboxTransform,
    *,
    confidence_threshold: float,
    iou_threshold: float,
) -> tuple[Detection, ...]:
    """Parse one-class YOLO output shaped ``[1, 5, N]`` or ``[1, N, 5]``."""

    _validate_threshold(confidence_threshold, "confidence_threshold")
    _validate_threshold(iou_threshold, "iou_threshold")

    output = np.asarray(raw_output)
    if output.ndim != 3 or output.shape[0] != 1:
        raise DetectorOutputError(
            "light detector output must have shape [1, 5, N] or [1, N, 5]; "
            f"received {output.shape}"
        )

    predictions = output[0]
    if predictions.shape[0] == 5:
        predictions = predictions.T
    elif predictions.shape[1] != 5:
        raise DetectorOutputError(
            "light detector output must have exactly five values per candidate; "
            f"received {output.shape}"
        )

    if predictions.shape[0] == 0:
        return ()

    predictions = np.asarray(predictions, dtype=np.float32)
    finite = np.all(np.isfinite(predictions), axis=1)
    confident = predictions[:, 4] >= confidence_threshold
    predictions = predictions[finite & confident]
    if predictions.shape[0] == 0:
        return ()

    boxes = xywh_to_xyxy(predictions[:, :4])
    scores = predictions[:, 4]
    keep = non_max_suppression(boxes, scores, iou_threshold)
    restored = transform.restore_boxes(boxes[keep])
    kept_scores = scores[keep]

    valid_size = (restored[:, 2] > restored[:, 0]) & (
        restored[:, 3] > restored[:, 1]
    )
    detections = []
    for box, score in zip(restored[valid_size], kept_scores[valid_size]):
        detections.append(
            Detection(
                xyxy=tuple(float(value) for value in box),
                confidence=float(score),
                class_id=0,
                class_name="person",
            )
        )
    return tuple(detections)


def xywh_to_xyxy(boxes: NDArray[np.floating]) -> NDArray[np.float32]:
    """Convert center-based ``xywh`` boxes to corner-based ``xyxy`` boxes."""

    source = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    converted = np.empty_like(source)
    converted[:, 0] = source[:, 0] - source[:, 2] / 2.0
    converted[:, 1] = source[:, 1] - source[:, 3] / 2.0
    converted[:, 2] = source[:, 0] + source[:, 2] / 2.0
    converted[:, 3] = source[:, 1] + source[:, 3] / 2.0
    return converted


def non_max_suppression(
    boxes: NDArray[np.floating],
    scores: NDArray[np.floating],
    iou_threshold: float,
) -> NDArray[np.int64]:
    """Return score-ordered indices after class-agnostic NMS."""

    _validate_threshold(iou_threshold, "iou_threshold")
    boxes_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores_array = np.asarray(scores, dtype=np.float32).reshape(-1)
    if boxes_array.shape[0] != scores_array.shape[0]:
        raise ValueError("boxes and scores must contain the same number of items")
    if boxes_array.shape[0] == 0:
        return np.empty(0, dtype=np.int64)

    x1, y1, x2, y2 = boxes_array.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores_array)[::-1]
    kept: list[int] = []

    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break

        remaining = order[1:]
        intersection_width = np.maximum(
            0.0,
            np.minimum(x2[current], x2[remaining])
            - np.maximum(x1[current], x1[remaining]),
        )
        intersection_height = np.maximum(
            0.0,
            np.minimum(y2[current], y2[remaining])
            - np.maximum(y1[current], y1[remaining]),
        )
        intersection = intersection_width * intersection_height
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = remaining[iou <= iou_threshold]

    return np.asarray(kept, dtype=np.int64)


def _validate_threshold(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")

