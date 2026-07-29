"""Detector interfaces and result metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from app.core.models import Detection


@dataclass(frozen=True, slots=True)
class DetectionTimings:
    """Wall-clock time spent in each detector stage, in milliseconds."""

    preprocessing_ms: float
    inference_ms: float
    postprocessing_ms: float

    @property
    def total_ms(self) -> float:
        """Return total detector time excluding visualization and I/O."""

        return self.preprocessing_ms + self.inference_ms + self.postprocessing_ms


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Detections and timings produced for one frame."""

    detections: tuple[Detection, ...]
    timings: DetectionTimings


class PersonDetector(Protocol):
    """Interface implemented by person detectors."""

    def detect(self, frame: NDArray[np.uint8]) -> DetectionResult:
        """Detect people in one OpenCV BGR frame."""

