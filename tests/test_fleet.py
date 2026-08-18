from threading import Lock

import numpy as np
import pytest

from app.fleet.catalog import cameras_from_rows, parse_polygon, rewrite_stream_url
from app.fleet.geometry import build_camera_config, camera_config_mapping
from app.fleet.pipeline import CameraPipeline, EmptyDetector
from app.fleet.sampler import SampleInterval
from app.fleet.settings import FLEET_FPS, FLEET_INTERVAL_SECONDS, FleetSettings
from app.fleet.supervisor import FleetSupervisor
from app.fleet.catalog import FleetCamera, MappedZone
from app.management.publisher import MinutePublisher


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class FakeWorker:
    def __init__(self, camera: FleetCamera) -> None:
        self.camera = camera
        self.started = False
        self.stopped = False

    @property
    def signature(self) -> str:
        return self.camera.signature

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float = 3.0) -> None:
        self.stopped = True

    def is_alive(self) -> bool:
        return self.started and not self.stopped

    def snapshot(self) -> dict[str, object]:
        return {
            "cameraId": self.camera.camera_id,
            "name": self.camera.name,
            "marketId": self.camera.market_id,
            "fieldId": self.camera.field_id,
            "status": "running" if self.started and not self.stopped else "stopped",
            "processedFrames": 0,
            "lastError": None,
        }


def _camera(**overrides: object) -> FleetCamera:
    values = {
        "camera_id": "cam-1",
        "name": "Entrance",
        "stream_url": "rtsp://mediamtx:8554/bazar1-a",
        "field_id": "field-1",
        "market_id": "market-1",
        "booth_id": None,
        "queues": (),
        "restricted_zones": (),
        "signature": "sig-1",
    }
    values.update(overrides)
    return FleetCamera(**values)  # type: ignore[arg-type]


def test_fleet_fps_is_one_frame_every_two_seconds() -> None:
    assert FLEET_FPS == 0.5
    assert FLEET_INTERVAL_SECONDS == 2.0
    settings = FleetSettings.from_environ(
        {"VIDEO_ANALYTICS_FLEET_ENABLED": "true", "VIDEO_ANALYTICS_FLEET_FPS": "5"}
    )
    assert settings.fps == 0.5
    assert settings.interval_seconds == 2.0
    assert settings.expected_samples_per_minute == 30
    assert settings.spatial_publish_seconds == 30.0


def test_sample_interval_emits_once_per_two_seconds_and_does_not_catch_up() -> None:
    clock = FakeClock(0.0)
    gate = SampleInterval(2.0, clock=clock)
    assert gate.due() is True
    assert gate.due() is False
    clock.value = 1.99
    assert gate.due() is False
    clock.value = 2.0
    assert gate.due() is True
    clock.value = 7.5
    assert gate.due() is True
    assert gate.due() is False


def test_localhost_mediamtx_url_is_rewritten_for_docker() -> None:
    rewritten = rewrite_stream_url("rtsp://localhost:8554/cam-a", "rtsp://mediamtx:8554")
    assert rewritten == "rtsp://mediamtx:8554/cam-a"
    assert rewrite_stream_url("rtsp://10.1.2.3:554/axis", "rtsp://mediamtx:8554") == "rtsp://10.1.2.3:554/axis"


def test_catalog_groups_cameras_and_queue_polygons_by_market() -> None:
    settings = FleetSettings.from_environ({"MEDIAMTX_RTSP_URL": "rtsp://mediamtx:8554"})
    cameras = cameras_from_rows(
        [
            {
                "id": "cam-a",
                "name": "Bazar 1 A",
                "stream_url": "rtsp://localhost:8554/bazar1-a",
                "field_id": "field-1",
                "market_id": "market-bazar-1",
                "booth_id": None,
            },
            {"id": "cam-offline", "name": "Dead", "stream_url": None, "field_id": None, "market_id": None, "booth_id": None},
        ],
        [
            {
                "camera_id": "cam-a",
                "region_id": "region-queue-123456",
                "name": "Checkout",
                "type": "QUEUE",
                "polygon": [{"x": 0.1, "y": 0.2}, {"x": 0.4, "y": 0.2}, {"x": 0.4, "y": 0.8}, {"x": 0.1, "y": 0.8}],
            }
        ],
        settings,
    )
    assert len(cameras) == 1
    camera = cameras[0]
    assert camera.stream_url == "rtsp://mediamtx:8554/bazar1-a"
    assert camera.market_id == "market-bazar-1"
    assert camera.queues[0].zone_id.endswith("123456")
    assert parse_polygon([[0.0, 0.0], [1.0, 0.0]]) is None


def test_fleet_camera_config_enables_queue_heatmap_and_wide_sample_gap() -> None:
    settings = FleetSettings.from_environ({})
    camera = _camera(
        queues=(
            MappedZone(
                "queue-line-abc",
                "line",
                ((0.2, 0.2), (0.8, 0.2), (0.8, 0.9), (0.2, 0.9)),
            ),
        )
    )
    mapping = camera_config_mapping(camera, settings)
    config = build_camera_config(camera, settings)
    assert mapping["heatmap"]["max_sample_gap_seconds"] >= 4.0
    assert "heatmap" in config.analytics.enabled
    assert "queue" in config.analytics.enabled
    assert config.heatmap.max_sample_gap_seconds >= 4.0
    assert config.tracker.lost_track_buffer == 4


def test_pipeline_processes_one_empty_frame_without_writing_job_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VIDEO_ANALYTICS_INGEST_URL", raising=False)
    monkeypatch.setenv("ANALYTICS_OUTBOX_DIR", str(tmp_path))
    settings = FleetSettings.from_environ({})
    pipeline = CameraPipeline(
        _camera(),
        settings,
        EmptyDetector(),
        Lock(),
        publisher=MinutePublisher("cam-1", "Entrance"),
    )
    metrics = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8), timestamp=1.0)
    pipeline.close()
    assert metrics["frame_count"] == 1
    assert metrics["current_people"] == 0
    assert metrics["processing_fps"] is not None
    assert metrics["management_spatial_layers"] is not None
    assert list(tmp_path.iterdir()) == []


def test_supervisor_starts_and_replaces_cameras_without_touching_on_demand_jobs() -> None:
    first = _camera(signature="one")
    updated = _camera(signature="two")
    removed_market = _camera(camera_id="cam-2", name="Other", signature="other")
    catalog = {"items": (first, removed_market)}
    created: list[FakeWorker] = []

    def load(_settings: FleetSettings) -> tuple[FleetCamera, ...]:
        return catalog["items"]

    def factory(camera, settings, detector, lock) -> FakeWorker:
        worker = FakeWorker(camera)
        created.append(worker)
        return worker

    supervisor = FleetSupervisor(
        FleetSettings.from_environ({"VIDEO_ANALYTICS_FLEET_ENABLED": "true"}),
        load_cameras=load,
        detector=EmptyDetector(),
        worker_factory=factory,
    )
    supervisor.reconcile()
    assert {worker.camera.camera_id for worker in created} == {"cam-1", "cam-2"}
    assert all(worker.started for worker in created)

    catalog["items"] = (updated,)
    supervisor.reconcile()
    assert created[0].stopped is True
    assert created[1].stopped is True
    assert created[-1].camera.signature == "two"
    status = supervisor.status()
    assert status["fps"] == 0.5
    assert status["cameras"] == 1
    assert status["intervalSeconds"] == 2.0
