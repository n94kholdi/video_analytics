"""Phase 9 synthetic timestamped speed and queue-progress tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.analytics.queue import CameraQueueConfig, QueueAnalyzer
from app.analytics.speed import (
    CameraSpeedConfig,
    SpeedEstimator,
    validate_known_ground_distance,
)
from app.core.models import TrackObservation, TrajectoryPoint
from app.geometry.calibration import ImageToGroundProjector
from app.geometry.config import (
    CalibrationConfig,
    NormalizedPoint,
    QueueRegion,
    ServicePoint,
    SpeedConfig,
)


def _calibration() -> CalibrationConfig:
    return CalibrationConfig(
        (
            NormalizedPoint(0, 0),
            NormalizedPoint(1, 0),
            NormalizedPoint(1, 1),
            NormalizedPoint(0, 1),
        ),
        ((0, 0), (10, 0), (10, 10), (0, 10)),
        "metres",
    )


def _estimator(
    *, calibrated: bool = False, settings: SpeedConfig | None = None
) -> SpeedEstimator:
    projector = ImageToGroundProjector.from_calibration(
        _calibration() if calibrated else None, (101, 101)
    )
    return SpeedEstimator(
        (CameraSpeedConfig("cam", settings or SpeedConfig(), projector),)
    )


def _observation(
    points: list[tuple[float, float, float]], *, track_id: int = 1
) -> TrackObservation:
    trajectory = tuple(
        TrajectoryPoint(timestamp, index, (x, y), (x, y))
        for index, (timestamp, x, y) in enumerate(points)
    )
    timestamp, x, y = points[-1]
    return TrackObservation(
        "cam",
        track_id,
        timestamp,
        len(points) - 1,
        (x - 2, y - 10, x + 2, y),
        (x, y),
        0.9,
        True,
        trajectory,
    )


def test_constant_synthetic_velocity_and_pixel_label_state() -> None:
    result = _estimator().update(
        "cam", [_observation([(0, 10, 20), (0.5, 15, 20), (1, 20, 20)])]
    )

    track = result.observations[0]
    assert track.speed_pixels_per_second == pytest.approx(10)
    assert track.smoothed_velocity == pytest.approx((10, 0))
    assert track.speed_metres_per_second is None
    assert result.snapshot.average_speed_pixels_per_second == pytest.approx(10)


def test_stationary_track_reports_zero_after_sufficient_history() -> None:
    result = _estimator().update(
        "cam", [_observation([(0, 25, 25), (0.4, 25, 25), (1, 25, 25)])]
    )

    assert result.observations[0].speed_pixels_per_second == 0


def test_irregular_timestamps_and_skipped_frames_use_elapsed_time() -> None:
    result = _estimator(settings=SpeedConfig(window_seconds=3)).update(
        "cam", [_observation([(0, 0, 10), (0.2, 2, 10), (1.7, 17, 10)])]
    )

    assert result.observations[0].speed_pixels_per_second == pytest.approx(10)


def test_calibrated_physical_speed_is_metres_per_second() -> None:
    result = _estimator(calibrated=True).update(
        "cam", [_observation([(0, 10, 50), (1, 20, 50)])]
    )

    assert result.observations[0].speed_pixels_per_second == pytest.approx(10)
    assert result.observations[0].speed_metres_per_second == pytest.approx(1)
    assert result.snapshot.calibration_warning is None


def test_missing_calibration_reports_physical_speed_unavailable() -> None:
    result = _estimator().update(
        "cam", [_observation([(0, 10, 50), (1, 20, 50)])]
    )

    assert result.observations[0].speed_metres_per_second is None
    assert "not configured" in result.observations[0].speed_unavailable_reason
    assert "not configured" in result.snapshot.calibration_warning


def test_unrealistic_jump_is_rejected() -> None:
    estimator = _estimator(
        settings=SpeedConfig(maximum_speed_pixels_per_second=50)
    )
    estimator.update("cam", [_observation([(0, 10, 20)])])
    result = estimator.update("cam", [_observation([(1, 100, 20)])])

    assert result.observations[0].speed_pixels_per_second is None
    assert result.snapshot.tracks[0].rejected_jump
    assert result.observations[0].speed_unavailable_reason == (
        "unrealistic pixel jump rejected"
    )


def test_insufficient_history_does_not_report_speed() -> None:
    result = _estimator().update("cam", [_observation([(0, 10, 20)])])

    assert result.observations[0].speed_pixels_per_second is None
    assert "insufficient" in result.observations[0].speed_unavailable_reason


def test_queue_progress_distinguishes_toward_service_from_sideways_motion() -> None:
    queue = QueueRegion(
        "desk",
        (
            NormalizedPoint(0, 0),
            NormalizedPoint(1, 0),
            NormalizedPoint(1, 1),
            NormalizedPoint(0, 1),
        ),
        ServicePoint(NormalizedPoint(1, 0.5)),
        overflow_threshold=5,
        minimum_dwell_seconds=0,
        maximum_speed_pixels_per_second=100,
    )
    analyzer = QueueAnalyzer((CameraQueueConfig("cam", (101, 101), (queue,)),))
    toward = replace(
        _observation([(0, 40, 50), (1, 50, 50)], track_id=1),
        speed_pixels_per_second=10,
        smoothed_velocity=(10, 0),
    )
    sideways = replace(
        _observation([(0, 40, 40), (1, 40, 50)], track_id=2),
        speed_pixels_per_second=10,
        smoothed_velocity=(0, 10),
    )

    snapshot = analyzer.update("cam", [toward, sideways], timestamp=1).snapshot
    states = {item.track_id: item for item in snapshot.tracks}
    assert states[1].progress_speed_pixels_per_second == pytest.approx(10)
    assert states[2].progress_speed_pixels_per_second == pytest.approx(0)
    assert snapshot.queue_for("desk").average_progress_speed_pixels_per_second == pytest.approx(5)


def test_known_ground_distance_validates_small_calibration_example() -> None:
    projector = ImageToGroundProjector.from_calibration(_calibration(), (101, 101))

    check = validate_known_ground_distance(projector, (10, 50), (60, 50), 5)

    assert check.projected_distance_metres == pytest.approx(5)
    assert check.absolute_error_metres == pytest.approx(0)
    assert check.relative_error == pytest.approx(0)
