from pathlib import Path

import app.api.jobs as jobs_module
from app.api.jobs import JobManager, _last_json_object
from app.api.presets import APPLICATIONS, get_application


def test_application_catalog_has_unique_ids() -> None:
    identifiers = [item.application_id for item in APPLICATIONS]
    assert len(identifiers) == len(set(identifiers))
    assert {"detection", "people_counting", "heatmap", "vertical_queue"} <= set(
        identifiers
    )


def test_configured_applications_declare_camera_config_requirement() -> None:
    assert get_application("restricted_area").requires_camera_config
    assert get_application("configured_queue").requires_camera_config
    assert get_application("full_analytics").requires_camera_config
    assert not get_application("people_counting").requires_camera_config


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
