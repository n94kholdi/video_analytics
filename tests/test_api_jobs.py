from pathlib import Path
import json

import app.api.jobs as jobs_module
import numpy as np
from app.api.jobs import JobManager, _last_json_object
from app.api.live import LiveReporter, processing_frame_size
from app.api.presets import APPLICATIONS, get_application


def test_application_catalog_has_unique_ids() -> None:
    identifiers = [item.application_id for item in APPLICATIONS]
    assert len(identifiers) == len(set(identifiers))
    assert {"detection", "people_counting", "heatmap", "vertical_queue"} <= set(
        identifiers
    )
    assert all(item.metrics for item in APPLICATIONS)
    assert all(
        definition.display in {"card", "chart", "status", "counter", "table"}
        for item in APPLICATIONS
        for definition in item.metrics
    )


def test_configured_applications_declare_camera_config_requirement() -> None:
    assert get_application("restricted_area").requires_camera_config
    assert get_application("configured_queue").requires_camera_config
    assert get_application("full_analytics").requires_camera_config
    assert not get_application("people_counting").requires_camera_config


def test_restricted_area_exposes_lifecycle_counters() -> None:
    keys = {item.key for item in get_application("restricted_area").metrics}

    assert {
        "restricted_occupancy",
        "restricted_entries",
        "restricted_exits",
        "restricted_violations",
    } <= keys
    assert "entry_count" not in keys
    assert "exit_count" not in keys


def test_heatmap_applications_expose_top_crowded_regions_metric() -> None:
    for application_id in ("heatmap", "full_analytics"):
        metrics = get_application(application_id).metrics
        definition = next(
            item for item in metrics if item.key == "top_crowded_regions"
        )
        assert definition.value_type == "table"
        assert definition.display == "table"


def test_job_command_uses_existing_analytics_cli(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, python_executable="python-test")
    job_dir = tmp_path / "job-1"
    job_dir.mkdir()
    source = job_dir / "input.mp4"
    source.touch()
    record = manager.register(
        job_id="job-1",
        application_id="vertical_queue",
        original_filename="shop.mp4",
        camera_id="test-camera",
        input_video=source,
        camera_config=None,
        max_frames=25,
    )

    command, expected = manager._build_command(record, get_application("vertical_queue"))

    assert command[:3] == ["python-test", "-m", "app.analytics.cli"]
    assert "--enable-queue" in command
    assert command[command.index("--queue-mode") + 1] == "vertical"
    assert command[command.index("--max-frames") + 1] == "25"
    assert command[command.index("--processing-width") + 1] == "1280"
    assert command[command.index("--frame-stride") + 1] == "5"
    assert expected["counts_csv"] == job_dir / "counts.csv"


def test_detection_command_receives_supported_max_frames(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)
    job_dir = tmp_path / "job-2"
    job_dir.mkdir()
    source = job_dir / "input.mp4"
    source.touch()
    record = manager.register(
        job_id="job-2",
        application_id="detection",
        original_filename="shop.mp4",
        camera_id="test-camera",
        input_video=source,
        camera_config=None,
        max_frames=10,
    )

    command, _ = manager._build_command(record, get_application("detection"))

    assert command[command.index("--max-frames") + 1] == "10"
    assert "--camera-id" not in command


def test_stream_job_passes_rtsp_url_to_existing_pipeline(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, python_executable="python-test")
    job_dir = tmp_path / "live-job"
    job_dir.mkdir()
    source = "rtsp://mediamtx:8554/mobile-1"
    record = manager.register(
        job_id="live-job",
        application_id="people_counting",
        original_filename="live:phone",
        camera_id="phone",
        input_video=source,
        camera_config=None,
        max_frames=100,
        source_type="rtsp",
    )

    command, _ = manager._build_command(record, get_application("people_counting"))
    configuration = json.loads((job_dir / "configuration.json").read_text())

    assert command[3] == source
    assert configuration["source_type"] == "rtsp"
    assert configuration["input_video"] == "<live-rtsp-stream>"


def test_tracking_command_does_not_receive_analytics_camera_config(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)
    job_dir = tmp_path / "job-3"
    job_dir.mkdir()
    source = job_dir / "input.mp4"
    source.touch()
    camera_config = job_dir / "camera.yaml"
    camera_config.touch()
    record = manager.register(
        job_id="job-3",
        application_id="tracking",
        original_filename="shop.mp4",
        camera_id="test-camera",
        input_video=source,
        camera_config=camera_config,
        max_frames=10,
    )

    command, _ = manager._build_command(record, get_application("tracking"))

    assert "--camera-config" not in command
    assert command[command.index("--camera-id") + 1] == "test-camera"
    assert command[command.index("--max-frames") + 1] == "10"


def test_admin_selected_reid_is_persisted_and_added_to_tracking_command(
    tmp_path: Path,
) -> None:
    manager = JobManager(tmp_path)
    job_dir = tmp_path / "job-reid"
    job_dir.mkdir()
    source = job_dir / "input.mp4"
    source.touch()
    record = manager.register(
        job_id="job-reid",
        application_id="people_counting",
        original_filename="shop.mp4",
        camera_id="test-camera",
        input_video=source,
        camera_config=None,
        max_frames=None,
        enable_reid=True,
    )

    command, _ = manager._build_command(record, get_application("people_counting"))
    configuration = json.loads((job_dir / "configuration.json").read_text())

    assert "--enable-reid" in command
    assert record.enable_reid is True
    assert manager.public_dict(record)["enable_reid"] is True
    assert configuration["enable_reid"] is True


def test_heatmap_occupancy_video_gets_stable_browser_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    manager = JobManager(tmp_path)
    job_dir = tmp_path / "job-4"
    heatmap_dir = job_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True)
    source = job_dir / "input.mp4"
    source.touch()
    raw_video = job_dir / "annotated_raw.mp4"
    raw_video.touch()
    occupancy_video = heatmap_dir / "camera_image_occupancy.mp4"
    occupancy_video.touch()
    log_path = job_dir / "process.log"
    log_path.touch()
    record = manager.register(
        job_id="job-4",
        application_id="heatmap",
        original_filename="shop.mp4",
        camera_id="test-camera",
        input_video=source,
        camera_config=None,
        max_frames=None,
    )

    def fake_browser_video(_source: Path, destination: Path) -> bool:
        destination.touch()
        return True

    monkeypatch.setattr(jobs_module, "_make_browser_video", fake_browser_video)
    artifacts = manager._collect_artifacts(
        record, {"annotated_video_raw": raw_video}, log_path
    )

    assert Path(artifacts["heatmap_video"]).name == "heatmap_occupancy.mp4"
    assert Path(artifacts["heatmap_video"]).parent == job_dir


def test_last_json_object_ignores_leading_output() -> None:
    assert _last_json_object('progress\n{"frames": 12, "ok": true}\n') == {
        "frames": 12,
        "ok": True,
    }


def test_live_reporter_persists_sampled_preview_metrics_and_events(tmp_path: Path) -> None:
    reporter = LiveReporter(
        tmp_path,
        "job-live",
        total_frames=10,
        metric_interval=10,
        preview_interval=10,
    )
    reporter.publish(
        1,
        {"current_people": 2, "processing_fps": 12.5},
        frame=np.zeros((720, 1280, 3), dtype=np.uint8),
        force=True,
    )

    assert (tmp_path / "preview.jpg").is_file()
    assert (tmp_path / "metrics.jsonl").is_file()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert {event["type"] for event in events} == {
        "preview_updated",
        "metrics_updated",
        "progress_updated",
    }
    assert events[-1]["job_id"] == "job-live"
    assert events[-1]["progress"] == 20.0
    assert events[-1]["preview_reference"] == "/api/v1/jobs/job-live/preview"


def test_cancel_request_is_persisted_for_running_process(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)
    job_dir = tmp_path / "job-cancel"
    job_dir.mkdir()
    source = job_dir / "input.mp4"
    source.touch()
    record = manager.register(
        job_id="job-cancel",
        application_id="tracking",
        original_filename="shop.mp4",
        camera_id="camera",
        input_video=source,
        camera_config=None,
        max_frames=None,
    )

    cancelled = manager.cancel(record.id)

    assert cancelled.status == "cancelling"
    assert (job_dir / "cancel.requested").is_file()


def test_dashboard_processing_size_preserves_aspect_ratio_and_avoids_upscale() -> None:
    assert processing_frame_size(3840, 2160, 1280) == (1280, 720)
    assert processing_frame_size(640, 480, 1280) == (640, 480)
