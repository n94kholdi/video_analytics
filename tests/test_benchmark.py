"""Cached-detection benchmark pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.models import Detection
from app.tracking.benchmark.cache import CachedDetection, DetectionCache
from app.tracking.benchmark.metrics import GroundTruthBox, evaluate_tracks
from app.tracking.benchmark.runner import BenchmarkRunner
from app.tracking.factory import create_tracker


def person(x: float) -> Detection:
    return Detection((x, 10.0, x + 20.0, 50.0), 0.9)


def test_detection_cache_round_trip(tmp_path: Path) -> None:
    cache = DetectionCache(tmp_path / "dets.jsonl")
    frames = (
        CachedDetection(0, 0.0, 0, (person(0.0),)),
        CachedDetection(1, 2.0, 60, (person(4.0),)),
    )
    cache.write(frames)
    loaded = cache.read()

    assert [item.timestamp for item in loaded] == [0.0, 2.0]
    assert loaded[0].detections[0].xyxy == person(0.0).xyxy


def test_perfect_tracks_score_high_clear_and_hota() -> None:
    ground_truth = [
        GroundTruthBox(0, 1, (0.0, 10.0, 20.0, 50.0)),
        GroundTruthBox(1, 1, (4.0, 10.0, 24.0, 50.0)),
        GroundTruthBox(2, 1, (8.0, 10.0, 28.0, 50.0)),
    ]
    hypotheses = [
        (0, 7, (0.0, 10.0, 20.0, 50.0)),
        (1, 7, (4.0, 10.0, 24.0, 50.0)),
        (2, 7, (8.0, 10.0, 28.0, 50.0)),
    ]
    metrics = evaluate_tracks(ground_truth, hypotheses)

    assert metrics.mota == pytest.approx(1.0)
    assert metrics.idf1 == pytest.approx(1.0)
    assert metrics.id_switches == 0
    assert metrics.fragmentation == 0
    assert metrics.hota > 0.9


def test_benchmark_runner_compares_trackers_on_cached_detections(tmp_path: Path) -> None:
    frames = [
        CachedDetection(index, index * 2.0, index, (person(float(index * 3)),))
        for index in range(4)
    ]
    ground_truth = [
        GroundTruthBox(item.frame_index, 1, item.detections[0].xyxy) for item in frames
    ]
    runner = BenchmarkRunner()
    reports = []
    for tracker_type in ("bytetrack", "stabletrack", "deepocsort", "botsort"):
        report, observations = runner.run(
            frames,
            tracker_type=tracker_type,
            frame_rate=0.5,
            ground_truth=ground_truth,
        )
        reports.append(report)
        path = tmp_path / f"{tracker_type}.json"
        runner.write_report(report, path)
        assert path.is_file()
        assert path.with_suffix(".csv").is_file()
        assert report.tracker == tracker_type
        assert report.frames == 4
        assert report.effective_fps > 0
        assert observations
        assert report.metrics is not None
        assert report.metrics.id_switches == 0

    assert {item.tracker for item in reports} == {"bytetrack", "stabletrack", "deepocsort", "botsort"}


def test_same_cached_detections_are_reused_by_factory_trackers() -> None:
    detections = (person(0.0),)
    first = create_tracker("bytetrack", frame_rate=0.5, confirmation_frames=1)
    second = create_tracker("stabletrack", frame_rate=0.5, confirmation_frames=1)
    third = create_tracker("deepocsort", frame_rate=0.5, confirmation_frames=1)
    fourth = create_tracker("botsort", frame_rate=0.5, confirmation_frames=1)
    left = first.update(detections, camera_id="cam", timestamp=0.0, frame_index=0)
    right = second.update(detections, camera_id="cam", timestamp=0.0, frame_index=0)
    deep = third.update(detections, camera_id="cam", timestamp=0.0, frame_index=0)
    bot = fourth.update(detections, camera_id="cam", timestamp=0.0, frame_index=0)

    assert left.observations[0].xyxy == detections[0].xyxy
    assert right.observations[0].xyxy == detections[0].xyxy
    assert deep.observations[0].xyxy == detections[0].xyxy
    assert bot.observations[0].xyxy == detections[0].xyxy
    assert left.normalized()[0].class_id == right.normalized()[0].class_id == deep.normalized()[0].class_id == bot.normalized()[0].class_id == 0
