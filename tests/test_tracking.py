"""ByteTrack conversion, lifecycle, trajectory, and timestamp tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.models import Detection
from app.tracking.bytetrack import (
    ByteTrackAdapter,
    detections_to_supervision,
    foot_point,
)


def person(x: float, *, confidence: float = 0.9) -> Detection:
    return Detection((x, 10.0, x + 20.0, 50.0), confidence)


def test_detection_to_tracker_conversion_filters_non_people() -> None:
    detections = [
        person(0.0, confidence=0.8),
        Detection((1.0, 2.0, 3.0, 4.0), 0.7, class_id=2, class_name="car"),
    ]

    converted = detections_to_supervision(detections)

    np.testing.assert_allclose(converted.xyxy, [[0.0, 10.0, 20.0, 50.0]])
    np.testing.assert_allclose(converted.confidence, [0.8])
    np.testing.assert_array_equal(converted.class_id, [0])


def test_stable_synthetic_track_progression_and_confirmation() -> None:
    tracker = ByteTrackAdapter(activation_threshold=0.4, match_threshold=0.2)

    first = tracker.update([person(0.0)], camera_id="cam", timestamp=1.0, frame_index=0)
    second = tracker.update([person(2.0)], camera_id="cam", timestamp=1.1, frame_index=1)
    third = tracker.update([person(4.0)], camera_id="cam", timestamp=1.2, frame_index=2)

    assert [len(result.observations) for result in (first, second, third)] == [1, 1, 1]
    ids = [result.observations[0].track_id for result in (first, second, third)]
    assert ids == [ids[0]] * 3
    assert first.observations[0].confirmed is False
    assert second.observations[0].confirmed is True


def test_empty_frames_are_accepted_without_fabricated_observations() -> None:
    tracker = ByteTrackAdapter(lost_track_buffer=3)
    tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    tracker.update([person(1.0)], camera_id="cam", timestamp=0.1, frame_index=1)

    result = tracker.update([], camera_id="cam", timestamp=0.2, frame_index=2)

    assert result.observations == ()
    assert result.expired_track_ids == ()
    assert tracker.retained_track_count == 1


def test_track_expiration_removes_trajectory_state() -> None:
    tracker = ByteTrackAdapter(lost_track_buffer=2, frame_rate=30.0)
    created = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    track_id = created.observations[0].track_id
    tracker.update([person(1.0)], camera_id="cam", timestamp=0.1, frame_index=1)

    tracker.update([], camera_id="cam", timestamp=0.2, frame_index=2)
    expired = tracker.update([], camera_id="cam", timestamp=0.3, frame_index=3)

    assert expired.expired_track_ids == (track_id,)
    assert tracker.retained_track_count == 0


def test_trajectory_history_is_bounded_and_smoothed() -> None:
    tracker = ByteTrackAdapter(history_size=3, smoothing_alpha=0.5, match_threshold=0.1)
    latest = None
    for frame_index in range(5):
        latest = tracker.update(
            [person(float(frame_index))],
            camera_id="cam",
            timestamp=frame_index / 10,
            frame_index=frame_index,
        )

    assert latest is not None
    trajectory = latest.observations[0].trajectory
    assert len(trajectory) == 3
    assert [point.frame_index for point in trajectory] == [2, 3, 4]
    assert trajectory[-1].smoothed_position[0] < trajectory[-1].position[0]


def test_foot_point_is_bottom_center() -> None:
    assert foot_point((10.0, 20.0, 30.0, 80.0)) == (20.0, 80.0)


def test_source_timestamp_and_frame_index_are_preserved() -> None:
    tracker = ByteTrackAdapter()

    result = tracker.update(
        [person(0.0)], camera_id="entrance", timestamp=12.345, frame_index=17
    )

    observation = result.observations[0]
    assert observation.camera_id == "entrance"
    assert observation.timestamp == pytest.approx(12.345)
    assert observation.frame_index == 17
    assert observation.trajectory[-1].timestamp == pytest.approx(12.345)
    assert observation.trajectory[-1].frame_index == 17


def test_reset_allows_frame_indices_to_restart() -> None:
    tracker = ByteTrackAdapter()
    tracker.update([person(0.0)], camera_id="cam", timestamp=1.0, frame_index=10)

    tracker.reset()
    result = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)

    assert result.observations[0].track_id == 1
