"""StableTrack association, timestamp, and adapter-isolation tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.models import Detection
from app.tracking.stabletrack_adapter import StableTrackAdapter
from app.tracking.third_party.stabletrack.matching import bbox_based_distance


def person(x: float, *, y: float = 10.0, confidence: float = 0.9) -> Detection:
    return Detection((x, y, x + 20.0, y + 40.0), confidence)


def test_bbox_based_distance_grows_with_center_offset_not_frame_count() -> None:
    box = (10.0, 10.0, 50.0, 90.0)
    near = bbox_based_distance(box, (12.0, 12.0, 52.0, 92.0), delta_tau=2.0)
    far = bbox_based_distance(box, (80.0, 10.0, 120.0, 90.0), delta_tau=2.0)
    same_far_at_full_fps = bbox_based_distance(box, (80.0, 10.0, 120.0, 90.0), delta_tau=1.0 / 30.0)

    assert near < 2.0
    assert far > near
    assert same_far_at_full_fps > far


def test_stabletrack_reuses_id_across_two_second_gap() -> None:
    tracker = StableTrackAdapter(frame_rate=0.5, confirmation_frames=1, use_visual_tracking=False)
    first = tracker.update([person(0.0)], camera_id="cam", timestamp=10.0, frame_index=0)
    second = tracker.update([person(4.0)], camera_id="cam", timestamp=12.0, frame_index=1)

    assert first.observations[0].track_id == second.observations[0].track_id
    assert second.observations[0].timestamp == pytest.approx(12.0)


def test_stabletrack_recovers_id_when_the_same_person_returns_nearby() -> None:
    tracker = StableTrackAdapter(
        frame_rate=0.5,
        confirmation_frames=1,
        max_age_seconds=8.0,
        use_visual_tracking=False,
    )
    first = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    tracker.update([], camera_id="cam", timestamp=2.0, frame_index=1)
    resumed = tracker.update([person(4.0)], camera_id="cam", timestamp=4.0, frame_index=2)

    assert resumed.observations[0].track_id == first.observations[0].track_id


def test_stabletrack_does_not_give_a_lost_id_to_a_distant_person() -> None:
    tracker = StableTrackAdapter(
        frame_rate=0.5,
        confirmation_frames=1,
        max_age_seconds=8.0,
        use_visual_tracking=False,
    )
    first = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    tracker.update([], camera_id="cam", timestamp=2.0, frame_index=1)
    resumed = tracker.update([person(80.0)], camera_id="cam", timestamp=4.0, frame_index=2)

    assert resumed.observations[0].track_id != first.observations[0].track_id


def test_stabletrack_does_not_recover_after_the_lost_window() -> None:
    tracker = StableTrackAdapter(
        frame_rate=0.5,
        confirmation_frames=1,
        max_age_seconds=8.0,
        lost_recovery_seconds=4.0,
        use_visual_tracking=False,
    )
    first = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    tracker.update([], camera_id="cam", timestamp=2.0, frame_index=1)
    tracker.update([], camera_id="cam", timestamp=4.0, frame_index=2)
    tracker.update([], camera_id="cam", timestamp=6.0, frame_index=3)
    resumed = tracker.update([person(4.0)], camera_id="cam", timestamp=8.0, frame_index=4)

    assert resumed.observations[0].track_id != first.observations[0].track_id


def test_stabletrack_expires_after_max_age_seconds() -> None:
    tracker = StableTrackAdapter(
        frame_rate=0.5,
        confirmation_frames=1,
        max_age_seconds=2.0,
        use_visual_tracking=False,
    )
    created = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    tracker.update([], camera_id="cam", timestamp=2.0, frame_index=1)
    expired = tracker.update([], camera_id="cam", timestamp=4.1, frame_index=2)

    assert created.observations[0].track_id in expired.expired_track_ids
    assert tracker.retained_track_count == 0


def test_stabletrack_requires_frame_when_reid_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeReIdentifier:
        def __init__(self, _model_path: str, *, providers: tuple[str, ...]) -> None:
            assert providers

        def embed(self, _frame: np.ndarray, _xyxy: object) -> np.ndarray:
            return np.asarray((1.0, 0.0), dtype=np.float32)

    import app.tracking.stabletrack_adapter as module

    monkeypatch.setattr(module, "OsNetReIdentifier", FakeReIdentifier)
    tracker = StableTrackAdapter(reid_model="fake.onnx", frame_rate=0.5)

    with pytest.raises(ValueError, match="frame is required"):
        tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
