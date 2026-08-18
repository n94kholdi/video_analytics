"""Discover connected cameras and keep one 0.5 FPS worker per camera."""

from __future__ import annotations

import logging
import os
from threading import Event, Lock, Thread
from time import time
from typing import Callable

from app.core.config import ConfigError, load_settings
from app.detection.onnx_detector import OnnxPersonDetector
from app.fleet.catalog import FleetCamera, load_fleet_cameras
from app.fleet.pipeline import PersonDetector
from app.fleet.settings import FleetSettings
from app.fleet.worker import CameraWorker

logger = logging.getLogger(__name__)

CameraLoader = Callable[[FleetSettings], tuple[FleetCamera, ...]]
WorkerFactory = Callable[[FleetCamera, FleetSettings, PersonDetector, Lock], CameraWorker]


class FleetSupervisor:
    """Reconcile camera catalog membership without touching on-demand jobs."""

    def __init__(
        self,
        settings: FleetSettings,
        *,
        load_cameras: CameraLoader = load_fleet_cameras,
        detector: PersonDetector | None = None,
        worker_factory: WorkerFactory | None = None,
    ) -> None:
        self.settings = settings
        self._load_cameras = load_cameras
        self._detector = detector
        self._worker_factory = worker_factory or (
            lambda camera, fleet_settings, person_detector, lock: CameraWorker(
                camera, fleet_settings, person_detector, lock
            )
        )
        self._detector_lock = Lock()
        self._workers: dict[str, CameraWorker] = {}
        self._lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None
        self.last_refresh: float | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        if not self.settings.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        os.environ.setdefault(
            "ANALYTICS_EXPECTED_SAMPLES_PER_MINUTE",
            str(self.settings.expected_samples_per_minute),
        )
        try:
            self._ensure_detector()
        except Exception as exc:  # noqa: BLE001 - leave fleet idle instead of crashing the API
            self.last_error = str(exc)
            logger.exception("fleet detector could not start")
            return
        self._stop.clear()
        self.reconcile()
        self._thread = Thread(target=self._loop, name="fleet-supervisor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.stop()

    def reconcile(self) -> None:
        try:
            cameras = self._load_cameras(self.settings)
            self.last_error = None
        except Exception as exc:  # noqa: BLE001 - keep existing workers
            self.last_error = str(exc)
            logger.warning("fleet camera catalog refresh failed: %s", exc)
            return
        desired = {camera.camera_id: camera for camera in cameras}
        with self._lock:
            current_ids = set(self._workers)
            for camera_id in current_ids - set(desired):
                self._workers.pop(camera_id).stop()
            for camera_id, camera in desired.items():
                existing = self._workers.get(camera_id)
                if existing is not None and existing.signature == camera.signature:
                    continue
                if existing is not None:
                    existing.stop()
                if self._detector is None:
                    continue
                worker = self._worker_factory(camera, self.settings, self._detector, self._detector_lock)
                self._workers[camera_id] = worker
                worker.start()
            self.last_refresh = time()

    def status(self) -> dict[str, object]:
        with self._lock:
            workers = [worker.snapshot() for worker in self._workers.values()]
        running = sum(1 for item in workers if item["status"] in {"running", "connected", "starting"})
        return {
            "enabled": self.settings.enabled,
            "fps": self.settings.fps,
            "intervalSeconds": self.settings.interval_seconds,
            "maxCameras": self.settings.max_cameras,
            "cameras": len(workers),
            "running": running,
            "lastRefresh": self.last_refresh,
            "lastError": self.last_error,
            "workers": workers,
        }

    def _loop(self) -> None:
        while not self._stop.wait(self.settings.refresh_seconds):
            self.reconcile()

    def _ensure_detector(self) -> None:
        if self._detector is not None:
            return
        settings = load_settings()
        if settings.detector_model is None:
            raise ConfigError("fleet analytics requires VIDEO_ANALYTICS_DETECTOR_MODEL")
        self._detector = OnnxPersonDetector(
            settings.detector_model,
            confidence_threshold=settings.detector_confidence_threshold,
            iou_threshold=settings.detector_iou_threshold,
            providers=settings.onnx_providers,
        )


fleet_supervisor = FleetSupervisor(FleetSettings.from_environ())
