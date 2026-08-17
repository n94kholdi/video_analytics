"""Grab a still JPEG from a recorded file or a live RTSP camera."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import base64
import re

import cv2
import numpy as np

from app.api.live import processing_frame_size

SNAPSHOT_WIDTH = 1280
STREAM_OPEN_TIMEOUT_SECONDS = 8.0
STREAM_WARMUP_READS = 5


class FrameCaptureError(Exception):
    """Raised when a still cannot be decoded from a camera or file."""

    def __init__(self, message: str, *, status_code: int = 503) -> None:
        super().__init__(message)
        self.status_code = status_code


def require_rtsp_url(value: str) -> str:
    candidate = value.strip()
    if not re.match(r"^rtsps?://[^/]+/.+", candidate, flags=re.IGNORECASE):
        raise ValueError("stream_url must be an RTSP URL with a stream path")
    return candidate


def encode_jpeg_data_url(frame: np.ndarray, *, quality: int = 88) -> dict[str, object]:
    """Return a dashboard-ready JPEG data URL plus frame geometry."""

    if frame is None or frame.size == 0:
        raise FrameCaptureError("could not encode an empty frame", status_code=500)
    height, width = frame.shape[:2]
    target_width, target_height = processing_frame_size(width, height, SNAPSHOT_WIDTH)
    if (target_width, target_height) != (width, height):
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        width, height = target_width, target_height
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise FrameCaptureError("failed to encode frame", status_code=500)
    return {
        "data_url": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
    }


def grab_stream_frame(
    stream_url: str,
    *,
    timeout_seconds: float = STREAM_OPEN_TIMEOUT_SECONDS,
    warmup_reads: int = STREAM_WARMUP_READS,
) -> np.ndarray:
    """Read one usable frame from RTSP using the same OpenCV path as live jobs."""

    def _read() -> np.ndarray:
        capture = cv2.VideoCapture(stream_url)
        try:
            if not capture.isOpened():
                raise FrameCaptureError("could not open camera stream")
            last: np.ndarray | None = None
            for _ in range(max(1, warmup_reads)):
                ok, frame = capture.read()
                if ok and frame is not None and frame.size > 0:
                    last = frame
            if last is None:
                raise FrameCaptureError("could not read a frame from the camera")
            return last
        finally:
            capture.release()

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stream-frame")
    future = pool.submit(_read)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError as exc:
        raise FrameCaptureError(
            "timed out waiting for a camera frame",
            status_code=504,
        ) from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
