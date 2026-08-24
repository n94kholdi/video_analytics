"""Deep OC-SORT association, timestamp, appearance, and adapter-isolation tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.models import Detection
from app.tracking.deepocsort_adapter import DeepOCSortAdapter
from app.tracking.third_party.deepocsort.association import compute_aw_max_metric
from app.tracking.third_party.deepocsort.kalman import DeepOCSortKalman, xyxy_to_xywh
from app.tracking.third_party.deepocsort.tracker import DeepOCSort, DeepOCSortConfig


def person(x: float, *, y: float = 10.0, confidence: float = 0.9) -> Detection:
    return Detection((x, y, x + 20.0, y + 40.0), confidence)


def test_kalman_prediction_scales_with_elapsed_seconds_not_frame_count() -> None:
    filter_ = DeepOCSortKalman()
    filter_.initiate(xyxy_to_xywh((10.0, 10.0, 30.0, 50.0)))
    filter_.mean[4] = 5.0
    one_frame = DeepOCSortKalman()
    one_frame.initiate(xyxy_to_xywh((10.0, 10.0, 30.0, 50.0)))
    one_frame.mean[4] = 5.0
    filter_.predict(2.0)
    one_frame.predict(1.0 / 30.0)

    assert filter_.mean[0] > one_frame.mean[0]
    assert filter_.to_xyxy()[0] > 10.0


def test_adaptive_weighting_boosts_unique_appearance_matches() -> None:
    cost = np.array([[0.95, 0.20], [0.21, 0.94]], dtype=np.float64)
    weighted = compute_aw_max_metric(cost, 0.75, bottom=0.5)
    flat = np.array([[0.50, 0.50], [0.50, 0.49]], dtype=np.float64)
    ambiguous = compute_aw_max_metric(flat, 0.75, bottom=0.5)

    assert weighted[0, 0] > ambiguous[0, 0]
    assert weighted[1, 1] > weighted[0, 1]


def test_dynamic_appearance_alpha_rises_when_confidence_falls() -> None:
    backend = DeepOCSort(DeepOCSortConfig(activation_threshold=0.4, alpha_fixed_emb=0.95))
    high = backend._dynamic_appearance_alphas(np.asarray([0.99], dtype=np.float64))
    low = backend._dynamic_appearance_alphas(np.asarray([0.41], dtype=np.float64))

    assert high[0] == pytest.approx(0.95, abs=0.02)
    assert low[0] > high[0]
    assert low[0] <= 1.0


def test_deepocsort_reuses_id_across_two_second_gap() -> None:
    tracker = DeepOCSortAdapter(frame_rate=0.5, confirmation_frames=1)
    first = tracker.update([person(0.0)], camera_id="cam", timestamp=10.0, frame_index=0)
    second = tracker.update([person(4.0)], camera_id="cam", timestamp=12.0, frame_index=1)

    assert first.observations[0].track_id == second.observations[0].track_id
    assert second.observations[0].timestamp == pytest.approx(12.0)


def test_deepocsort_expires_after_max_age_seconds() -> None:
    tracker = DeepOCSortAdapter(frame_rate=0.5, confirmation_frames=1, max_age_seconds=2.0)
    created = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    tracker.update([], camera_id="cam", timestamp=2.0, frame_index=1)
    expired = tracker.update([], camera_id="cam", timestamp=4.1, frame_index=2)

    assert created.observations[0].track_id in expired.expired_track_ids
    assert tracker.retained_track_count == 0


def test_ocr_recovers_lost_track_from_last_observation() -> None:
    tracker = DeepOCSortAdapter(frame_rate=0.5, confirmation_frames=1, max_age_seconds=8.0)
    created = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    tracker.update([], camera_id="cam", timestamp=2.0, frame_index=1)
    recovered = tracker.update([person(2.0)], camera_id="cam", timestamp=4.0, frame_index=2)

    assert recovered.observations[0].track_id == created.observations[0].track_id


def test_reid_recovery_reuses_id_when_iou_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeReIdentifier:
        def __init__(self, _model_path: str, *, providers: tuple[str, ...]) -> None:
            assert providers

        def embed(self, _frame: np.ndarray, _xyxy: object) -> np.ndarray:
            return np.asarray((1.0, 0.0), dtype=np.float32)

    import app.tracking.deepocsort_adapter as module

    monkeypatch.setattr(module, "OsNetReIdentifier", FakeReIdentifier)
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    tracker = DeepOCSortAdapter(
        reid_model="fake.onnx",
        frame_rate=0.5,
        confirmation_frames=1,
        max_age_seconds=8.0,
        match_threshold=0.9,
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


def test_deepocsort_requires_frame_when_reid_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeReIdentifier:
        def __init__(self, _model_path: str, *, providers: tuple[str, ...]) -> None:
            assert providers

        def embed(self, _frame: np.ndarray, _xyxy: object) -> np.ndarray:
            return np.asarray((1.0, 0.0), dtype=np.float32)

    import app.tracking.deepocsort_adapter as module

    monkeypatch.setattr(module, "OsNetReIdentifier", FakeReIdentifier)
    tracker = DeepOCSortAdapter(reid_model="fake.onnx", frame_rate=0.5)

    with pytest.raises(ValueError, match="frame is required"):
        tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
