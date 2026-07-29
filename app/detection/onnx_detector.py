"""ONNX Runtime implementation of the light person detector."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort

from app.detection.base import DetectionResult, DetectionTimings
from app.detection.postprocessing import parse_light_output
from app.detection.preprocessing import preprocess_bgr


class ModelValidationError(ValueError):
    """Raised when an ONNX model does not match the supported light contract."""


class DetectorInferenceError(RuntimeError):
    """Raised when ONNX Runtime cannot execute the detector."""


class OnnxPersonDetector:
    """One-class 640x640 ONNX person detector with CPU provider fallback."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.7,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"detector model file does not exist: {self.model_path}"
            )
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.providers = select_providers(providers, ort.get_available_providers())

        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=list(self.providers),
            )
        except Exception as exc:
            raise ModelValidationError(
                f"could not load detector model {self.model_path}: {exc}"
            ) from exc

        input_meta = self.session.get_inputs()
        output_meta = self.session.get_outputs()
        self.input_name, self.output_name, self.input_size = validate_model_contract(
            input_meta,
            output_meta,
        )

    def detect(self, frame: NDArray[np.uint8]) -> DetectionResult:
        """Detect people in one OpenCV BGR frame."""

        started = perf_counter()
        preprocessed = preprocess_bgr(frame, self.input_size)
        after_preprocessing = perf_counter()

        try:
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: preprocessed.tensor},
            )
        except Exception as exc:
            raise DetectorInferenceError(f"ONNX detector inference failed: {exc}") from exc
        after_inference = perf_counter()

        detections = parse_light_output(
            outputs[0],
            preprocessed.transform,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
        )
        finished = perf_counter()

        timings = DetectionTimings(
            preprocessing_ms=(after_preprocessing - started) * 1000.0,
            inference_ms=(after_inference - after_preprocessing) * 1000.0,
            postprocessing_ms=(finished - after_inference) * 1000.0,
        )
        return DetectionResult(detections=detections, timings=timings)


def select_providers(
    requested: Sequence[str],
    available: Sequence[str],
) -> tuple[str, ...]:
    """Select available providers in request order and always add CPU fallback."""

    requested_clean = tuple(provider.strip() for provider in requested if provider.strip())
    if not requested_clean:
        raise ValueError("at least one ONNX Runtime provider must be requested")

    available_set = set(available)
    selected = [provider for provider in requested_clean if provider in available_set]
    if (
        "CPUExecutionProvider" in available_set
        and "CPUExecutionProvider" not in selected
    ):
        selected.append("CPUExecutionProvider")
    if not selected:
        raise ModelValidationError(
            "none of the requested ONNX Runtime providers are available; "
            f"requested={list(requested_clean)}, available={list(available)}"
        )
    return tuple(dict.fromkeys(selected))


def validate_model_contract(
    inputs: Sequence[Any],
    outputs: Sequence[Any],
) -> tuple[str, str, tuple[int, int]]:
    """Validate the verified light-model input and output metadata."""

    if len(inputs) != 1:
        raise ModelValidationError(
            f"light detector must expose exactly one input; received {len(inputs)}"
        )
    if not outputs:
        raise ModelValidationError("light detector must expose at least one output")

    input_meta = inputs[0]
    input_shape = tuple(input_meta.shape)
    if getattr(input_meta, "type", None) != "tensor(float)":
        raise ModelValidationError(
            "light detector input must be tensor(float); "
            f"received {getattr(input_meta, 'type', None)!r}"
        )
    if len(input_shape) != 4:
        raise ModelValidationError(
            f"light detector input must be rank-4 NCHW; received {input_shape}"
        )
    batch, channels, height, width = input_shape
    if not _valid_batch_dimension(batch):
        raise ModelValidationError(
            "light detector batch dimension must be dynamic or fixed at 1; "
            f"received {batch!r}"
        )
    if channels != 3 or height != 640 or width != 640:
        raise ModelValidationError(
            "light detector input must be [batch, 3, 640, 640]; "
            f"received {input_shape}"
        )

    output_meta = outputs[0]
    output_shape = tuple(output_meta.shape)
    if getattr(output_meta, "type", None) != "tensor(float)":
        raise ModelValidationError(
            "light detector output must be tensor(float); "
            f"received {getattr(output_meta, 'type', None)!r}"
        )
    if len(output_shape) != 3 or not _valid_batch_dimension(output_shape[0]):
        raise ModelValidationError(
            "light detector output must be rank 3 with a dynamic or unit batch; "
            f"received {output_shape}"
        )
    if tuple(output_shape[1:]) not in {(5, 8400), (8400, 5)}:
        raise ModelValidationError(
            "light detector output must be [batch, 5, 8400] or "
            f"[batch, 8400, 5]; received {output_shape}"
        )

    return input_meta.name, output_meta.name, (int(height), int(width))


def _valid_batch_dimension(value: Any) -> bool:
    return value is None or isinstance(value, str) or (
        isinstance(value, Integral) and not isinstance(value, bool) and int(value) == 1
    )

