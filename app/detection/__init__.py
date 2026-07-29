"""ONNX person detection interfaces."""

from app.detection.base import DetectionResult, DetectionTimings, PersonDetector
from app.detection.onnx_detector import OnnxPersonDetector

__all__ = [
    "DetectionResult",
    "DetectionTimings",
    "OnnxPersonDetector",
    "PersonDetector",
]
