"""Tracker factory, switching, reset, empty input, and 0.5 FPS tests."""

from __future__ import annotations

import pytest

from app.core.models import Detection
from app.tracking.base import BaseTracker, NormalizedTrack
from app.tracking.factory import available_tracker_types, create_tracker, public_tracker_catalog


def person(x: float, *, confidence: float = 0.9) -> Detection:
    return Detection((x, 10.0, x + 20.0, 50.0), confidence)


def test_factory_initializes_registered_trackers() -> None:
    catalog = {item["type"] for item in public_tracker_catalog()}
    assert catalog == {"bytetrack", "stabletrack", "deepocsort"}
    for tracker_type in available_tracker_types():
        tracker = create_tracker(tracker_type, frame_rate=0.5)
        assert isinstance(tracker, BaseTracker)
        assert tracker.name == tracker_type


def test_factory_rejects_unknown_tracker_type() -> None:
    with pytest.raises(ValueError, match="unknown tracker type"):
        create_tracker("not-a-tracker")


def test_switching_trackers_does_not_share_state() -> None:
    first = create_tracker("bytetrack", frame_rate=0.5, confirmation_frames=1)
    second = create_tracker("stabletrack", frame_rate=0.5, confirmation_frames=1)
    first.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    second.update([person(80.0)], camera_id="cam", timestamp=0.0, frame_index=0)

    assert first.name != second.name
    assert first.retained_track_count == 1
    assert second.retained_track_count == 1
    first_next = first.update([person(1.0)], camera_id="cam", timestamp=2.0, frame_index=1)
    second_next = second.update([person(81.0)], camera_id="cam", timestamp=2.0, frame_index=1)
    assert first_next.observations and second_next.observations
    assert first_next.tracker_name == "bytetrack"
    assert second_next.tracker_name == "stabletrack"


@pytest.mark.parametrize("tracker_type", ["bytetrack", "stabletrack", "deepocsort"])
def test_reset_clears_ids_and_allows_frame_index_restart(tracker_type: str) -> None:
    tracker = create_tracker(tracker_type, frame_rate=0.5, confirmation_frames=1)
    tracker.update([person(0.0)], camera_id="cam", timestamp=1.0, frame_index=10)
    tracker.reset()
    result = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)

    assert result.observations[0].track_id == 1
    assert tracker.retained_track_count == 1


@pytest.mark.parametrize("tracker_type", ["bytetrack", "stabletrack", "deepocsort"])
def test_empty_detections_do_not_fabricate_observations(tracker_type: str) -> None:
    tracker = create_tracker(tracker_type, frame_rate=0.5, confirmation_frames=1, lost_track_buffer=4)
    tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)

    empty = tracker.update([], camera_id="cam", timestamp=2.0, frame_index=1)

    assert empty.observations == ()
    assert all(hasattr(empty, field) for field in ("observations", "expired_track_ids", "tracking_ms"))


@pytest.mark.parametrize("tracker_type", ["bytetrack", "stabletrack", "deepocsort"])
def test_outputs_are_normalized(tracker_type: str) -> None:
    tracker = create_tracker(tracker_type, frame_rate=0.5, confirmation_frames=1)
    result = tracker.update([person(5.0, confidence=0.82)], camera_id="cam", timestamp=0.0, frame_index=0)
    normalized = result.normalized()

    assert normalized
    item = normalized[0]
    assert isinstance(item, NormalizedTrack)
    assert item.track_id >= 1
    assert item.bbox == result.observations[0].xyxy
    assert item.confidence == pytest.approx(0.82)
    assert item.class_id == 0
    assert result.observations[0].class_id == 0
    assert result.tracker_name == tracker_type


@pytest.mark.parametrize("tracker_type", ["bytetrack", "stabletrack", "deepocsort"])
def test_half_fps_timestamp_gap_keeps_identity(tracker_type: str) -> None:
    tracker = create_tracker(
        tracker_type,
        frame_rate=0.5,
        confirmation_frames=1,
        lost_track_buffer=4,
        max_age_seconds=8.0,
        match_threshold=0.1,
    )
    first = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    second = tracker.update([person(3.0)], camera_id="cam", timestamp=2.0, frame_index=1)
    third = tracker.update([person(6.0)], camera_id="cam", timestamp=4.0, frame_index=2)

    assert [len(result.observations) for result in (first, second, third)] == [1, 1, 1]
    ids = [result.observations[0].track_id for result in (first, second, third)]
    assert ids == [ids[0]] * 3
    assert second.observations[0].timestamp == pytest.approx(2.0)
    assert third.observations[0].timestamp == pytest.approx(4.0)
