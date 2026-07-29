"""Drawing helpers kept outside detector inference."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from app.detection.base import DetectionResult


def annotate_frame(
    frame: NDArray[np.uint8],
    result: DetectionResult,
    *,
    copy: bool = True,
) -> NDArray[np.uint8]:
    """Draw detection boxes, confidence labels, and detector timing."""

    annotated = frame.copy() if copy else frame
    for detection in result.detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"{detection.class_name or detection.class_id} {detection.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )

    timing_label = (
        f"pre {result.timings.preprocessing_ms:.1f} ms | "
        f"infer {result.timings.inference_ms:.1f} ms | "
        f"post {result.timings.postprocessing_ms:.1f} ms"
    )
    cv2.putText(
        annotated,
        timing_label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 230),
        2,
        cv2.LINE_AA,
    )
    return annotated

