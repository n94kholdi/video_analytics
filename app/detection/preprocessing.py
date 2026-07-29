"""Image preprocessing for fixed-size ONNX detectors."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    """Geometry needed to restore boxes from model to source coordinates."""

    original_height: int
    original_width: int
    input_height: int
    input_width: int
    resized_height: int
    resized_width: int
    pad_top: int
    pad_left: int

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.original_width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.original_height

    def restore_boxes(
        self,
        boxes: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Restore model-space ``xyxy`` boxes and clip them to the source frame."""

        restored = np.asarray(boxes, dtype=np.float32).copy().reshape(-1, 4)
        if restored.size == 0:
            return restored

        restored[:, [0, 2]] = (
            restored[:, [0, 2]] - float(self.pad_left)
        ) / self.scale_x
        restored[:, [1, 3]] = (
            restored[:, [1, 3]] - float(self.pad_top)
        ) / self.scale_y
        restored[:, [0, 2]] = np.clip(
            restored[:, [0, 2]], 0.0, float(self.original_width)
        )
        restored[:, [1, 3]] = np.clip(
            restored[:, [1, 3]], 0.0, float(self.original_height)
        )
        return restored


@dataclass(frozen=True, slots=True)
class PreprocessedFrame:
    """Model tensor and the transform used to create it."""

    tensor: NDArray[np.float32]
    transform: LetterboxTransform


def letterbox(
    frame: NDArray[np.uint8],
    input_size: tuple[int, int] = (640, 640),
    *,
    padding_color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[NDArray[np.uint8], LetterboxTransform]:
    """Resize a BGR frame without distortion and pad it to ``input_size``."""

    if (
        not isinstance(frame, np.ndarray)
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.shape[0] <= 0
        or frame.shape[1] <= 0
    ):
        raise ValueError("frame must be a non-empty HxWx3 BGR NumPy array")

    input_height, input_width = input_size
    if input_height <= 0 or input_width <= 0:
        raise ValueError("input_size dimensions must be positive")

    original_height, original_width = frame.shape[:2]
    scale = min(input_width / original_width, input_height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))

    interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    horizontal_padding = input_width - resized_width
    vertical_padding = input_height - resized_height
    pad_left = horizontal_padding // 2
    pad_right = horizontal_padding - pad_left
    pad_top = vertical_padding // 2
    pad_bottom = vertical_padding - pad_top
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=padding_color,
    )

    transform = LetterboxTransform(
        original_height=original_height,
        original_width=original_width,
        input_height=input_height,
        input_width=input_width,
        resized_height=resized_height,
        resized_width=resized_width,
        pad_top=pad_top,
        pad_left=pad_left,
    )
    return padded, transform


def preprocess_bgr(
    frame: NDArray[np.uint8],
    input_size: tuple[int, int] = (640, 640),
) -> PreprocessedFrame:
    """Letterbox BGR input and produce normalized RGB NCHW float32 data."""

    padded, transform = letterbox(frame, input_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(
        rgb.transpose(2, 0, 1)[np.newaxis],
        dtype=np.float32,
    )
    tensor /= 255.0
    return PreprocessedFrame(tensor=tensor, transform=transform)

