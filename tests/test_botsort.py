"""BoT-SORT association, timestamp, appearance, and adapter-isolation tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.models import Detection
from app.tracking.botsort_adapter import BoTSortAdapter
from app.tracking.third_party.botsort.association import fuse_iou_reid, fuse_score
from app.tracking.third_party.botsort.kalman import BoTSortKalman, xyxy_to_xywh
from app.tracking.third_party.botsort.tracker import BoTSort, BoTSortConfig


def person(x: float, *, y: float = 10.0, confidence: float = 0.9) -> Detection:
    return Detection((x, y, x + 20.0, y + 40.0), confidence)


def test_kalman_prediction_scales_with_elapsed_seconds_not_frame_count() -> None:
    filter_ = BoTSortKalman()
    filter_.initiate(xyxy_to_xywh((10.0, 10.0, 30.0, 50.0)))
    filter_.mean[4] = 5.0
    one_frame = BoTSortKalman()
    one_frame.initiate(xyxy_to_xywh((10.0, 10.0, 30.0, 50.0)))
    one_frame.mean[4] = 5.0
    filter_.predict(2.0)
    one_frame.predict(1.0 / 30.0)

    assert filter_.mean[0] > one_frame.mean[0]
    assert filter_.to_xyxy()[0] > 10.0


def test_score_fusion_prefers_confident_overlaps() -> None:
    cost = np.array([[0.2]], dtype=np.float64)
    high = fuse_score(cost, np.array([0.9], dtype=np.float64))
    low = fuse_score(cost, np.array([0.4], dtype=np.float64))

    assert high[0, 0] < low[0, 0]
    assert high[0, 0] == pytest.approx(0.1 + 0.9 * 0.2)


def test_reid_min_fusion_recovers_when_iou_is_zero_and_proximity_gate_is_open() -> None:
    iou_dist = np.array([[1.0]], dtype=np.float64)
    appearance = np.array([[1.0]], dtype=np.float64)
    fused = fuse_iou_reid(iou_dist, appearance, proximity_thresh=1.0, appearance_thresh=0.25)
    gated = fuse_iou_reid(iou_dist, appearance, proximity_thresh=0.5, appearance_thresh=0.25)

    assert fused[0, 0] == pytest.approx(0.0)
    assert gated[0, 0] == pytest.approx(1.0)


def test_proximity_gate_uses_raw_iou_not_fused_score() -> None:
    raw_iou = np.array([[0.8]], dtype=np.float64)
    fused_score = fuse_score(raw_iou, np.array([0.2], dtype=np.float64))
    appearance = np.array([[1.0]], dtype=np.float64)
    gated = fuse_iou_reid(
        fused_score,
        appearance,
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        iou_for_gate=raw_iou,
    )

    assert float(fused_score[0, 0]) > 0.8
    assert gated[0, 0] == pytest.approx(float(fused_score[0, 0]))


def test_botsort_reuses_id_across_two_second_gap() -> None:
    tracker = BoTSortAdapter(frame_rate=0.5, confirmation_frames=1)
    first = tracker.update([person(0.0)], camera_id="cam", timestamp=10.0, frame_index=0)
    second = tracker.update([person(4.0)], camera_id="cam", timestamp=12.0, frame_index=1)

    assert first.observations[0].track_id == second.observations[0].track_id
    assert second.observations[0].timestamp == pytest.approx(12.0)


def test_botsort_keeps_id_when_person_walks_beyond_iou() -> None:
    tracker = BoTSortAdapter(
        frame_rate=0.5,
        confirmation_frames=1,
        match_threshold=0.3,
        iou_threshold=0.4,
    )
    first = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    second = tracker.update([person(30.0)], camera_id="cam", timestamp=2.0, frame_index=1)
    third = tracker.update([person(60.0)], camera_id="cam", timestamp=4.0, frame_index=2)

    assert second.observations[0].track_id == first.observations[0].track_id
    assert third.observations[0].track_id == first.observations[0].track_id
    assert tracker.retained_track_count == 1


def test_botsort_recovers_id_after_one_missed_frame_while_walking() -> None:
    tracker = BoTSortAdapter(frame_rate=0.5, confirmation_frames=1, max_age_seconds=8.0)
    created = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    tracker.update([], camera_id="cam", timestamp=2.0, frame_index=1)
    recovered = tracker.update([person(40.0)], camera_id="cam", timestamp=4.0, frame_index=2)

    assert recovered.observations[0].track_id == created.observations[0].track_id


def test_two_separated_people_keep_distinct_ids() -> None:
    tracker = BoTSortAdapter(frame_rate=0.5, confirmation_frames=1)
    first = tracker.update([person(0.0), person(200.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    second = tracker.update([person(25.0), person(225.0)], camera_id="cam", timestamp=2.0, frame_index=1)

    first_ids = {item.track_id for item in first.observations}
    second_ids = {item.track_id for item in second.observations}
    assert first_ids == second_ids
    assert len(second_ids) == 2


def test_botsort_expires_after_max_age_seconds() -> None:
    tracker = BoTSortAdapter(frame_rate=0.5, confirmation_frames=1, max_age_seconds=2.0)
    created = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    tracker.update([], camera_id="cam", timestamp=2.0, frame_index=1)
    expired = tracker.update([], camera_id="cam", timestamp=4.1, frame_index=2)

    assert created.observations[0].track_id in expired.expired_track_ids
    assert tracker.retained_track_count == 0


def test_low_score_second_association_keeps_identity() -> None:
    tracker = BoTSortAdapter(
        frame_rate=0.5,
        confirmation_frames=1,
        activation_threshold=0.4,
        track_low_threshold=0.1,
        max_age_seconds=8.0,
    )
    created = tracker.update([person(0.0, confidence=0.9)], camera_id="cam", timestamp=0.0, frame_index=0)
    recovered = tracker.update([person(2.0, confidence=0.2)], camera_id="cam", timestamp=2.0, frame_index=1)

    assert recovered.observations[0].track_id == created.observations[0].track_id


def test_reid_recovery_reuses_id_when_iou_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeReIdentifier:
        def __init__(self, _model_path: str, *, providers: tuple[str, ...]) -> None:
            assert providers

        def embed(self, _frame: np.ndarray, _xyxy: object) -> np.ndarray:
            return np.asarray((1.0, 0.0), dtype=np.float32)

    import app.tracking.botsort_adapter as module

    monkeypatch.setattr(module, "OsNetReIdentifier", FakeReIdentifier)
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    tracker = BoTSortAdapter(
        reid_model="fake.onnx",
        frame_rate=0.5,
        confirmation_frames=1,
        max_age_seconds=8.0,
        match_threshold=0.9,
        proximity_thresh=1.0,
        reid_similarity_threshold=0.8,
    )
    first = tracker.update(
        [person(0.0)],
        camera_id="cam",
        timestamp=0.0,
        frame_index=0,
        frame=frame,
    )
    tracker.update([], camera_id="cam", timestamp=2.0, frame_index=1, frame=frame)
    recovered = tracker.update(
        [person(80.0)],
        camera_id="cam",
        timestamp=4.0,
        frame_index=2,
        frame=frame,
    )

    assert recovered.observations[0].track_id == first.observations[0].track_id


def test_botsort_requires_frame_when_reid_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeReIdentifier:
        def __init__(self, _model_path: str, *, providers: tuple[str, ...]) -> None:
            assert providers

        def embed(self, _frame: np.ndarray, _xyxy: object) -> np.ndarray:
            return np.asarray((1.0, 0.0), dtype=np.float32)

    import app.tracking.botsort_adapter as module

    monkeypatch.setattr(module, "OsNetReIdentifier", FakeReIdentifier)
    tracker = BoTSortAdapter(reid_model="fake.onnx", frame_rate=0.5)

    with pytest.raises(ValueError, match="frame is required"):
        tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)


def test_backend_does_not_import_application_adapter() -> None:
    backend = BoTSort(BoTSortConfig(activation_threshold=0.4, confirmation_hits=1))
    boxes = np.asarray([[0.0, 10.0, 20.0, 50.0]], dtype=np.float32)
    scores = np.asarray([0.9], dtype=np.float32)
    outputs = backend.update(boxes=boxes, scores=scores, class_ids=None, timestamp=0.0)

    assert outputs[0].track_id == 1
    assert backend.last_reid_ms == 0.0


def test_reid_embeddings_are_extracted_only_for_high_score_boxes() -> None:
    seen: list[tuple[float, float, float, float]] = []

    def embed(_frame: np.ndarray, xyxy: object) -> np.ndarray:
        seen.append(tuple(float(value) for value in xyxy[:4]))
        return np.asarray((1.0, 0.0), dtype=np.float32)

    backend = BoTSort(BoTSortConfig(activation_threshold=0.4, confirmation_hits=1))
    boxes = np.asarray([[0.0, 10.0, 20.0, 50.0], [40.0, 10.0, 60.0, 50.0]], dtype=np.float32)
    scores = np.asarray([0.9, 0.2], dtype=np.float32)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    backend.update(boxes=boxes, scores=scores, class_ids=None, timestamp=0.0, frame=frame, embed=embed)

    assert seen == [(0.0, 10.0, 20.0, 50.0)]
