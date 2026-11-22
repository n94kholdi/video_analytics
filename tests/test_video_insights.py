from __future__ import annotations

import cv2
import numpy as np
from contextlib import nullcontext

from app.insights.service import (
    FRUIT_QUALITY_PROMPT,
    PER_FRUIT_CROP_QUALITY_PROMPT,
    VideoInsightError,
    VideoInsightService,
    _parse_fruit_quality,
    _parse_yes_no_answers,
    build_estimated_per_fruit_scores,
    extract_stream_frames,
    extract_video_frames,
)


class _SeekableCapture:
    def __init__(self, total: int = 12, fps: float = 10.0) -> None:
        self.total = total
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.total)
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.index = int(value)
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        # BGR values make the color conversion observable.
        return True, np.full((4, 6, 3), [self.index, 10, 20], dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def test_extract_video_frames_samples_uniformly_and_converts_to_rgb(monkeypatch) -> None:
    capture = _SeekableCapture(total=12)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: capture)

    frames = extract_video_frames("sample.mp4", num_frames=4)

    assert len(frames) == 4
    assert [int(frame[0, 0, 2]) for frame in frames] == [0, 3, 7, 11]
    assert frames[0][0, 0].tolist() == [20, 10, 0]
    assert capture.released


def test_extract_video_frames_samples_only_requested_time_window(monkeypatch) -> None:
    capture = _SeekableCapture(total=120, fps=10)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: capture)

    frames = extract_video_frames(
        "sample.mp4",
        num_frames=4,
        window_start_seconds=2,
        window_end_seconds=5,
    )

    assert [int(frame[0, 0, 2]) for frame in frames] == [20, 29, 39, 49]


def test_extract_stream_frames_returns_consecutive_frames(monkeypatch) -> None:
    capture = _SeekableCapture()

    def read() -> tuple[bool, np.ndarray]:
        capture.index += 1
        return True, np.full((4, 6, 3), capture.index, dtype=np.uint8)

    capture.read = read  # type: ignore[method-assign]
    monkeypatch.setattr(cv2, "VideoCapture", lambda _url: capture)

    frames = extract_stream_frames(
        "rtsp://mediamtx:8554/camera", num_frames=3, sample_interval_seconds=0
    )

    assert len(frames) == 3
    assert [int(frame[0, 0, 0]) for frame in frames] == [1, 2, 3]
    assert capture.released


def test_interpret_returns_only_normalized_yes_no_answers(monkeypatch) -> None:
    service = VideoInsightService()
    fake_torch = type("FakeTorch", (), {"inference_mode": staticmethod(nullcontext)})
    monkeypatch.setattr(service, "_load", lambda: (object(), object(), fake_torch))
    calls: list[dict[str, object]] = []

    def fake_generate(*_args, **kwargs):
        calls.append(kwargs)
        return ('```json\n{"fighting": "yes", "floor_clean": false}\n```', 5.3)

    monkeypatch.setattr(service, "_generate", fake_generate)
    result = service.interpret([np.zeros((4, 6, 3), dtype=np.uint8)] * 8)

    assert result["answers"] == {"fighting": "Yes", "floor_clean": "No"}
    assert result["inference_seconds"] == 5.3
    assert set(result) == {"answers", "frame_count", "inference_seconds"}
    assert len(calls) == 1
    assert calls[0]["max_new_tokens"] == 40
    assert len(calls[0]["images"]) == 8  # type: ignore[arg-type]


def test_answer_parser_rejects_free_form_model_output() -> None:
    with np.testing.assert_raises(VideoInsightError):
        _parse_yes_no_answers("There does not seem to be a fight.")


def test_interpret_prompt_contains_both_fixed_questions(monkeypatch) -> None:
    service = VideoInsightService()
    fake_torch = type("FakeTorch", (), {"inference_mode": staticmethod(nullcontext)})
    monkeypatch.setattr(service, "_load", lambda: (object(), object(), fake_torch))
    prompts: list[object] = []

    def fake_generate(*args, **_kwargs):
        prompts.append(args[3])
        return ('{"fighting":"No","floor_clean":"Yes"}', 1.0)

    monkeypatch.setattr(service, "_generate", fake_generate)
    service.interpret([np.zeros((4, 6, 3), dtype=np.uint8)] * 8)

    visual_content = prompts[0][0]["content"]  # type: ignore[index]
    assert "Are there persons fighting in the video?" in visual_content[-1]["text"]
    assert "Is the floor of the scene clean?" in visual_content[-1]["text"]


def test_parse_fruit_quality_returns_normalized_contract() -> None:
    result = _parse_fruit_quality(
        '```json\n{"has_fruit":true,"label":"تقریباً تازه",'
        '"freshness_score":78,"distribution":{"fresh":75,"middle":20,"rotten":5},'
        '"fruit_count_estimate":9,"confidence":84,"summary_fa":"بیشتر میوه‌ها تازه هستند."}\n```'
    )

    assert result["label"] == "تقریباً تازه"
    assert result["distribution"] == {"fresh": 75, "middle": 20, "rotten": 5}
    assert result["freshness_score"] == 78


def test_parse_fruit_quality_rejects_invalid_distribution() -> None:
    with np.testing.assert_raises(VideoInsightError):
        _parse_fruit_quality(
            '{"has_fruit":true,"label":"متوسط","freshness_score":50,'
            '"distribution":{"fresh":40,"middle":40,"rotten":40},'
            '"fruit_count_estimate":3,"confidence":70,"summary_fa":"کیفیت متوسط است."}'
        )


def test_interpret_fruit_quality_uses_common_prompt(monkeypatch) -> None:
    service = VideoInsightService()
    fake_torch = type("FakeTorch", (), {"inference_mode": staticmethod(nullcontext)})
    monkeypatch.setattr(service, "_load", lambda: (object(), object(), fake_torch))
    prompts: list[object] = []

    def fake_generate(*args, **_kwargs):
        prompts.append(args[3])
        return (
            '{"has_fruit":true,"label":"تازه","freshness_score":96,'
            '"distribution":{"fresh":100,"middle":0,"rotten":0},'
            '"fruit_count_estimate":4,"confidence":92,"summary_fa":"همه میوه‌ها تازه‌اند."}',
            1.2,
        )

    monkeypatch.setattr(service, "_generate", fake_generate)
    result = service.interpret_fruit_quality([np.zeros((4, 6, 3), dtype=np.uint8)])

    assert result["label"] == "تازه"
    assert result["frame_count"] == 1
    assert result["quality_mode"] == "estimated"
    assert prompts[0][0]["content"][-1]["text"] == FRUIT_QUALITY_PROMPT  # type: ignore[index]


def test_interpret_fruit_quality_with_counting_crops_region_once_and_scales_across_fruits(monkeypatch) -> None:
    service = VideoInsightService()
    fake_torch = type("FakeTorch", (), {"inference_mode": staticmethod(nullcontext)})
    monkeypatch.setattr(service, "_load", lambda: (object(), object(), fake_torch))
    calls: list[dict[str, object]] = []

    def fake_generate(*_args, **kwargs):
        calls.append(kwargs)
        return (
            '{"has_fruit":true,"label":"تازه","freshness_score":85,'
            '"distribution":{"fresh":60,"middle":40,"rotten":0},'
            '"fruit_count_estimate":3,"confidence":88,"summary_fa":"میوه‌های پالت تازه‌اند."}',
            0.4,
        )

    monkeypatch.setattr(service, "_generate", fake_generate)
    counting_result = {
        "contract_version": 1,
        "job_id": "job-123",
        "pallet_polygon": [[10.0, 10.0], [150.0, 10.0], [150.0, 150.0], [10.0, 150.0]],
        "frames": [
            {
                "frame_id": "frame_000000",
                "image_width": 200,
                "image_height": 200,
                "pallet_polygon": [[10.0, 10.0], [150.0, 10.0], [150.0, 150.0], [10.0, 150.0]],
                "fruits": [
                    {"fruit_id": 1, "box": [20.0, 20.0, 60.0, 60.0], "inside_pallet": True},
                    {"fruit_id": 2, "box": [70.0, 70.0, 110.0, 110.0], "inside_pallet": True},
                    {"fruit_id": 3, "box": [160.0, 160.0, 195.0, 195.0], "inside_pallet": False},
                ],
            }
        ],
    }

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    result_all = service.interpret_fruit_quality(
        [frame],
        counting_result=counting_result,
        filter_outside_roi=False,
    )
    # Single Qwen call on the cropped region, never 3 separate per-fruit calls
    assert len(calls) == 1
    assert result_all["quality_mode"] == "estimated"
    assert result_all["counted_fruits_total"] == 3
    assert result_all["evaluated_fruits_count"] == 3
    assert [f["fruit_id"] for f in result_all["per_fruit_quality"]] == [1, 2, 3]
    assert result_all["roi_crop_thumbnail"].startswith("data:image/jpeg;base64,")
    assert result_all["freshness_score"] == 85
    assert "سرشکن" not in result_all["summary_fa"]


def test_interpret_fruit_quality_with_roi_filtering_excludes_outside_fruits(monkeypatch) -> None:
    service = VideoInsightService()
    fake_torch = type("FakeTorch", (), {"inference_mode": staticmethod(nullcontext)})
    monkeypatch.setattr(service, "_load", lambda: (object(), object(), fake_torch))

    def fake_generate(*_args, **_kwargs):
        return (
            '{"has_fruit":true,"label":"تازه","freshness_score":86,'
            '"distribution":{"fresh":100,"middle":0,"rotten":0},'
            '"fruit_count_estimate":2,"confidence":90,"summary_fa":"سالم"}',
            0.3,
        )

    monkeypatch.setattr(service, "_generate", fake_generate)
    counting_result = {
        "pallet_polygon": [[10.0, 10.0], [120.0, 10.0], [120.0, 120.0], [10.0, 120.0]],
        "frames": [
            {
                "image_width": 200,
                "image_height": 200,
                "fruits": [
                    {"fruit_id": 1, "box": [20.0, 20.0, 50.0, 50.0], "inside_pallet": True},
                    {"fruit_id": 2, "box": [60.0, 60.0, 90.0, 90.0], "inside_pallet": True},
                    {"fruit_id": 3, "box": [150.0, 150.0, 190.0, 190.0], "inside_pallet": False},
                ],
            }
        ],
    }
    result = service.interpret_fruit_quality(
        [np.zeros((200, 200, 3), dtype=np.uint8)],
        counting_result=counting_result,
        filter_outside_roi=True,
    )
    assert result["counted_fruits_total"] == 3
    assert result["evaluated_fruits_count"] == 2
    assert [f["fruit_id"] for f in result["per_fruit_quality"]] == [1, 2]
    assert result["freshness_score"] == 86


def test_estimated_per_fruit_scores_scales_fruits_into_three_tiers_within_10_to_15_percent() -> None:
    fruits = [{"fruit_id": i + 1, "box": [10, 10, 30, 30]} for i in range(25)]
    estimated = build_estimated_per_fruit_scores(85, fruits, distribution={"fresh": 90, "middle": 10, "rotten": 0})
    assert len(estimated) == 25
    assert all(item["measurement_mode"] == "estimated" for item in estimated)
    scores = [int(item["freshness_score"]) for item in estimated]
    unique_tiers = sorted(set(scores), reverse=True)
    assert len(unique_tiers) == 3
    assert all(72 <= s <= 97 for s in scores)
    assert abs((sum(scores) / len(scores)) - 85.0) <= 1.0

    # Even with 361 fruits, produces exactly 3 tiers within 10-15% of 85%
    large_lot = [{"fruit_id": i + 1, "box": [10, 10, 30, 30]} for i in range(361)]
    estimated_large = build_estimated_per_fruit_scores(85, large_lot, distribution={"fresh": 90, "middle": 10, "rotten": 0})
    large_scores = [int(item["freshness_score"]) for item in estimated_large]
    assert len(set(large_scores)) == 3
    assert min(large_scores) >= 72
    assert max(large_scores) <= 97
    assert abs((sum(large_scores) / len(large_scores)) - 85.0) <= 1.0
