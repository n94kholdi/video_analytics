"""Grab a still JPEG from a recorded file or a live RTSP camera."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from collections.abc import Iterator
import base64
import re
import time

import cv2
import numpy as np

from app.api.live import processing_frame_size

SNAPSHOT_WIDTH = 1280
PREVIEW_WIDTH = 960
STREAM_OPEN_TIMEOUT_SECONDS = 8.0
STREAM_WARMUP_READS = 5
PREVIEW_FPS = 8.0
PREVIEW_JPEG_QUALITY = 70


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


def encode_jpeg(
    frame: np.ndarray,
    *,
    quality: int = 88,
    max_width: int = SNAPSHOT_WIDTH,
) -> tuple[bytes, int, int]:
    """JPEG-encode a frame, shrinking wide sources to a dashboard-friendly size."""

    if frame is None or frame.size == 0:
        raise FrameCaptureError("could not encode an empty frame", status_code=500)
    height, width = frame.shape[:2]
    target_width, target_height = processing_frame_size(width, height, max_width)
    if (target_width, target_height) != (width, height):
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        width, height = target_width, target_height
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise FrameCaptureError("failed to encode frame", status_code=500)
    return encoded.tobytes(), width, height


def encode_jpeg_data_url(frame: np.ndarray, *, quality: int = 88) -> dict[str, object]:
    """Return a dashboard-ready JPEG data URL plus frame geometry."""

    payload, width, height = encode_jpeg(frame, quality=quality)
    return {
        "data_url": "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii"),
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
    }


def _multipart_jpeg(image: bytes) -> bytes:
    return (
        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
        + str(len(image)).encode("ascii")
        + b"\r\n\r\n"
        + image
        + b"\r\n"
    )


def iter_mjpeg_preview(
    stream_url: str,
    *,
    fps: float = PREVIEW_FPS,
    jpeg_quality: int = PREVIEW_JPEG_QUALITY,
    timeout_seconds: float = STREAM_OPEN_TIMEOUT_SECONDS,
) -> Iterator[bytes]:
    """Yield an MJPEG live view from the same RTSP source used by analytics jobs."""

    capture = _open_capture(stream_url, timeout_seconds=timeout_seconds)
    interval = 1.0 / max(1.0, fps)
    consecutive_failures = 0
    try:
        while True:
            started = time.monotonic()
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    return
                time.sleep(interval)
                continue
            consecutive_failures = 0
            payload, _, _ = encode_jpeg(frame, quality=jpeg_quality, max_width=PREVIEW_WIDTH)
            yield _multipart_jpeg(payload)
            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        capture.release()


def _open_capture(stream_url: str, *, timeout_seconds: float) -> cv2.VideoCapture:
    def _open() -> cv2.VideoCapture:
        capture = cv2.VideoCapture(stream_url)
        if not capture.isOpened():
            capture.release()
            raise FrameCaptureError("could not open camera stream")
        return capture

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preview-open")
    future = pool.submit(_open)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError as exc:
        raise FrameCaptureError(
            "timed out waiting for a camera frame",
            status_code=504,
        ) from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


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
