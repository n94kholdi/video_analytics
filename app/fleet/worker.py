"""Keep one RTSP camera open and process a single frame every two seconds."""

from __future__ import annotations

import logging
import os
from threading import Event, Lock, Thread
from time import monotonic
from typing import Callable

import cv2

from app.fleet.catalog import FleetCamera
from app.fleet.pipeline import CameraPipeline, PersonDetector
from app.fleet.settings import FleetSettings

logger = logging.getLogger(__name__)

CaptureFactory = Callable[[str], cv2.VideoCapture]
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def open_capture(stream_url: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(stream_url)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class CameraWorker:
    """Background thread that samples one connected camera at the fleet FPS."""

    def __init__(
        self,
        camera: FleetCamera,
        settings: FleetSettings,
        detector: PersonDetector,
        detector_lock: Lock,
        *,
        capture_factory: CaptureFactory = open_capture,
        pipeline: CameraPipeline | None = None,
    ) -> None:
        self.camera = camera
        self.settings = settings
        self.status = "starting"
        self.last_error: str | None = None
        self.processed_frames = 0
        self.last_metrics: dict[str, object] | None = None
        self._last_processed = 0.0
        self._detector = detector
        self._detector_lock = detector_lock
        self._capture_factory = capture_factory
        self._pipeline = pipeline
        self._stop = Event()
        self._thread: Thread | None = None

    @property
    def signature(self) -> str:
        return self.camera.signature

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(
            target=self._run,
            name=f"fleet-{self.camera.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float = 8.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._pipeline is not None:
            self._pipeline.close()

    def snapshot(self) -> dict[str, object]:
        return {
            "cameraId": self.camera.camera_id,
            "name": self.camera.name,
            "marketId": self.camera.market_id,
            "fieldId": self.camera.field_id,
            "status": self.status,
            "processedFrames": self.processed_frames,
            "currentPeople": self.last_metrics.get("current_people") if self.last_metrics else None,
            "queueLength": self.last_metrics.get("queue_length") if self.last_metrics else None,
            "lastError": self.last_error,
        }

    def _run(self) -> None:
        pipeline = self._pipeline or CameraPipeline(
            self.camera, self.settings, self._detector, self._detector_lock
        )
        self._pipeline = pipeline
        capture: cv2.VideoCapture | None = None
        try:
            while not self._stop.is_set():
                if capture is None:
                    capture = self._open()
                    if capture is None:
                        self.status = "reconnect"
                        self._stop.wait(self.settings.reconnect_seconds)
                        continue
                remaining = self.settings.interval_seconds - (monotonic() - self._last_processed)
                if remaining > 0:
                    capture.grab()
                    self._stop.wait(min(0.05, remaining))
                    continue
                ok, frame = capture.read()
                if not ok or frame is None or getattr(frame, "size", 0) == 0:
                    self.last_error = "lost camera stream"
                    capture.release()
                    capture = None
                    self.status = "reconnect"
                    continue
                try:
                    self.last_metrics = pipeline.process(frame)
                    self.processed_frames = pipeline.processed_frames
                    self._last_processed = monotonic()
                    self.status = "running"
                    self.last_error = None
                except Exception as exc:  # noqa: BLE001 - keep the worker alive
                    self.last_error = str(exc)
                    logger.exception("fleet camera %s failed a frame", self.camera.camera_id)
        finally:
            if capture is not None:
                capture.release()
            pipeline.close()
            self.status = "stopped"

    def _open(self) -> cv2.VideoCapture | None:
        try:
            capture = self._capture_factory(self.camera.stream_url)
            if not capture.isOpened():
                capture.release()
                self.last_error = "could not open camera stream"
                return None
            self.status = "connected"
            return capture
        except Exception as exc:  # noqa: BLE001 - reconnect on the next loop
            self.last_error = str(exc)
            return None
