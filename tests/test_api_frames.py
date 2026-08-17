from __future__ import annotations

import base64
import time
from unittest.mock import MagicMock

import cv2
import numpy as np
from app.api.frames import FrameCaptureError, encode_jpeg_data_url, grab_stream_frame, require_rtsp_url


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray | None], *, opened: bool = True) -> None:
        self._frames = list(frames)
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._frames:
            return False, None
        frame = self._frames.pop(0)
        return frame is not None, frame

    def release(self) -> None:
        self.released = True


def test_require_rtsp_url_rejects_non_rtsp_sources() -> None:
    assert require_rtsp_url(" rtsp://mediamtx:8554/mobile-1 ") == "rtsp://mediamtx:8554/mobile-1"
    try:
        require_rtsp_url("http://example.test/stream")
    except ValueError as exc:
        assert "RTSP" in str(exc)
    else:
        raise AssertionError("non-RTSP URLs must be rejected")


def test_encode_jpeg_data_url_resizes_wide_frames() -> None:
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    payload = encode_jpeg_data_url(frame)

    assert payload["width"] == 1280
    assert payload["height"] == 720
    assert payload["aspect_ratio"] == 1280 / 720
    assert str(payload["data_url"]).startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(str(payload["data_url"]).split(",", 1)[1])
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[1] == 1280


def test_grab_stream_frame_returns_last_usable_warmup_frame(monkeypatch) -> None:
    first = np.full((40, 80, 3), 10, dtype=np.uint8)
    last = np.full((40, 80, 3), 200, dtype=np.uint8)
    capture = _FakeCapture([first, None, last])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _url: capture)

    frame = grab_stream_frame("rtsp://mediamtx:8554/cam", warmup_reads=3)

    assert np.array_equal(frame, last)
    assert capture.released


def test_grab_stream_frame_fails_when_stream_cannot_open(monkeypatch) -> None:
    capture = _FakeCapture([], opened=False)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _url: capture)

    try:
        grab_stream_frame("rtsp://mediamtx:8554/offline")
    except FrameCaptureError as exc:
        assert exc.status_code == 503
        assert "could not open" in str(exc)
    else:
        raise AssertionError("closed streams must raise FrameCaptureError")
    assert capture.released


def test_grab_stream_frame_times_out(monkeypatch) -> None:
    def hanging_capture(_url: str) -> MagicMock:
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.side_effect = lambda: time.sleep(2)
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", hanging_capture)

    try:
        grab_stream_frame("rtsp://mediamtx:8554/slow", timeout_seconds=0.05, warmup_reads=1)
    except FrameCaptureError as exc:
        assert exc.status_code == 504
    else:
        raise AssertionError("a hung capture must time out")
