"""Optimized Qwen2.5-VL inference for recorded videos and RTSP sequences."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any

import cv2
import numpy as np


class VideoInsightError(RuntimeError):
    """A user-facing video interpretation failure."""


FIGHT_QUESTION = "Are there persons fighting in the video?"
FLOOR_CLEAN_QUESTION = "Is the floor of the scene clean?"

FRUIT_QUALITY_PROMPT = """Evaluate the visible freshness of the fruits in these images. The images may be one photo or ordered samples from one video. Judge only visible fruit and summarize the whole scene; when video frames repeat the same fruits, do not treat each appearance as a different fruit.

Use visible evidence such as natural color, discoloration, bruising, mold, soft or collapsed areas, wrinkling, dryness, and decay. Do not claim anything about taste, smell, internal quality, or food safety that cannot be seen. Ignore non-fruit objects. If no fruit is clearly visible, set has_fruit to false and use the label "نامشخص".

Return exactly one JSON object with no prose or markdown, using this schema:
{"has_fruit":true,"label":"تازه","freshness_score":95,"distribution":{"fresh":100,"middle":0,"rotten":0},"fruit_count_estimate":12,"confidence":90,"summary_fa":"توضیح کوتاه فارسی"}

Rules:
- freshness_score, confidence, and distribution values are integers from 0 to 100; distribution must total 100.
- fruit_count_estimate is a non-negative integer estimate, or null when it cannot be estimated reliably.
- summary_fa must be one concise Persian sentence grounded in visible evidence.
- label must be exactly one of: "تازه", "تقریباً تازه", "متوسط", "تقریباً فاسد", "فاسد", "نامشخص".
- If essentially all visible fruits are fresh, use "تازه". If most are fresh but a minority show aging or defects, use "تقریباً تازه". Use "متوسط" for a mixed or mid-quality lot, "تقریباً فاسد" when most show substantial deterioration, and "فاسد" when essentially all show clear decay.
"""

PER_FRUIT_CROP_QUALITY_PROMPT = """Evaluate the visible freshness of the single fruit shown in this cropped image.

Use visible evidence such as natural color, discoloration, bruising, mold, soft or collapsed areas, wrinkling, dryness, and decay. Do not claim anything about taste, smell, internal quality, or food safety that cannot be seen. If no fruit is clearly visible in the crop, set has_fruit to false and use the label "نامشخص".

Return exactly one JSON object with no prose or markdown, using this schema:
{"has_fruit":true,"label":"تازه","freshness_score":92,"confidence":88,"note_fa":"توضیح کوتاه فارسی درباره وضعیت ظاهری این میوه"}

Rules:
- freshness_score and confidence are integers from 0 to 100.
- note_fa must be one short Persian phrase or sentence grounded in visible evidence.
- label must be exactly one of: "تازه", "تقریباً تازه", "متوسط", "تقریباً فاسد", "فاسد", "نامشخص".
"""

FRUIT_QUALITY_LABELS = {
    "تازه",
    "تقریباً تازه",
    "متوسط",
    "تقریباً فاسد",
    "فاسد",
    "نامشخص",
}


def _parse_yes_no_answers(raw_output: str) -> dict[str, str]:
    """Convert the private model response into the public two-answer contract."""

    candidate = raw_output.strip()
    json_object = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if json_object:
        candidate = json_object.group(0)
    try:
        payload = json.loads(candidate)
    except (TypeError, json.JSONDecodeError) as exc:
        raise VideoInsightError("the video answers could not be determined") from exc

    def normalize(key: str) -> str:
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, str) and value.strip().lower() in {"yes", "no"}:
            return value.strip().title()
        raise VideoInsightError("the video answers could not be determined")

    return {"fighting": normalize("fighting"), "floor_clean": normalize("floor_clean")}


def _parse_fruit_quality(raw_output: str) -> dict[str, object]:
    """Validate the model response before exposing it through the API."""

    candidate = raw_output.strip()
    json_object = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if json_object:
        candidate = json_object.group(0)
    try:
        payload = json.loads(candidate)
    except (TypeError, json.JSONDecodeError) as exc:
        raise VideoInsightError("the fruit quality could not be determined") from exc
    if not isinstance(payload, dict):
        raise VideoInsightError("the fruit quality could not be determined")

    has_fruit = payload.get("has_fruit")
    label = payload.get("label")
    summary = payload.get("summary_fa")
    distribution = payload.get("distribution")
    if not isinstance(has_fruit, bool) or label not in FRUIT_QUALITY_LABELS:
        raise VideoInsightError("the fruit quality could not be determined")
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 500:
        raise VideoInsightError("the fruit quality could not be determined")
    if not isinstance(distribution, dict):
        raise VideoInsightError("the fruit quality could not be determined")

    def percentage(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VideoInsightError("the fruit quality could not be determined")
        rounded = round(value)
        if not 0 <= rounded <= 100:
            raise VideoInsightError("the fruit quality could not be determined")
        return rounded

    freshness_score = percentage(payload.get("freshness_score"))
    confidence = percentage(payload.get("confidence"))
    normalized_distribution = {
        key: percentage(distribution.get(key)) for key in ("fresh", "middle", "rotten")
    }
    if sum(normalized_distribution.values()) != 100:
        raise VideoInsightError("the fruit quality percentages must total 100")
    count = payload.get("fruit_count_estimate")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
        raise VideoInsightError("the fruit quality could not be determined")
    if not has_fruit:
        label = "نامشخص"
        count = None

    return {
        "has_fruit": has_fruit,
        "label": label,
        "freshness_score": freshness_score,
        "distribution": normalized_distribution,
        "fruit_count_estimate": count,
        "confidence": confidence,
        "summary_fa": summary.strip(),
    }


def _label_for_score(score: float, has_fruit: bool = True) -> str:
    if not has_fruit:
        return "نامشخص"
    if score >= 85:
        return "تازه"
    if score >= 70:
        return "تقریباً تازه"
    if score >= 50:
        return "متوسط"
    if score >= 30:
        return "تقریباً فاسد"
    return "فاسد"


def _grade_for_score(score: float, has_fruit: bool = True) -> str | None:
    if not has_fruit:
        return None
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def _parse_single_fruit_crop_quality(raw_output: str) -> dict[str, object]:
    """Validate per-fruit crop quality output from Qwen."""

    candidate = raw_output.strip()
    json_object = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if json_object:
        candidate = json_object.group(0)
    try:
        payload = json.loads(candidate)
    except (TypeError, json.JSONDecodeError) as exc:
        raise VideoInsightError("the per-fruit quality could not be determined") from exc
    if not isinstance(payload, dict):
        raise VideoInsightError("the per-fruit quality could not be determined")

    has_fruit = payload.get("has_fruit", True)
    if not isinstance(has_fruit, bool):
        has_fruit = True

    def percentage(value: object, default: int = 80) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        rounded = round(float(value))
        return max(0, min(100, rounded))

    freshness_score = percentage(payload.get("freshness_score"), 80)
    confidence = percentage(payload.get("confidence"), 85)
    label = payload.get("label")
    if label not in FRUIT_QUALITY_LABELS:
        label = _label_for_score(freshness_score, has_fruit)
    note_fa = payload.get("note_fa") or payload.get("summary_fa") or ""
    if not isinstance(note_fa, str) or not note_fa.strip():
        note_fa = f"وضعیت ظاهری میوه: {label}"
    if not has_fruit:
        label = "نامشخص"

    return {
        "has_fruit": has_fruit,
        "label": label,
        "freshness_score": freshness_score,
        "grade": _grade_for_score(freshness_score, has_fruit),
        "confidence": confidence,
        "note_fa": note_fa.strip()[:300],
    }


def _point_inside_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    if len(polygon) < 3:
        return True
    pts = np.asarray(polygon, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0


def _crop_fruit_rgb(
    frame_rgb: np.ndarray,
    box: list[float],
    mask_polygon: list[list[float]] | None = None,
    padding_ratio: float = 0.08,
) -> np.ndarray | None:
    """Extract a padded RGB crop for one detected fruit box."""

    if len(box) != 4:
        return None
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio
    cx1 = max(0, int(np.floor(x1 - pad_x)))
    cy1 = max(0, int(np.floor(y1 - pad_y)))
    cx2 = min(w, int(np.ceil(x2 + pad_x)))
    cy2 = min(h, int(np.ceil(y2 + pad_y)))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return frame_rgb[cy1:cy2, cx1:cx2].copy()


def _distribution_from_scores(scores: list[int | float]) -> dict[str, int]:
    """Compute exact integer percentages (summing to 100) for fresh/middle/rotten."""

    if not scores:
        return {"fresh": 0, "middle": 0, "rotten": 0}
    total = len(scores)
    fresh_count = sum(1 for s in scores if s >= 75)
    middle_count = sum(1 for s in scores if 45 <= s < 75)
    rotten_count = total - fresh_count - middle_count

    raw = [
        ("fresh", fresh_count * 100.0 / total),
        ("middle", middle_count * 100.0 / total),
        ("rotten", rotten_count * 100.0 / total),
    ]
    floored = {k: int(np.floor(v)) for k, v in raw}
    remainder = 100 - sum(floored.values())
    for k, _ in sorted(raw, key=lambda item: item[1] - np.floor(item[1]), reverse=True)[:remainder]:
        floored[k] += 1
    return floored


def build_estimated_per_fruit_scores(
    frame_score: int,
    fruits: list[dict[str, Any]],
) -> list[dict[str, object]]:
    """Build an explicitly labeled estimated per-fruit distribution whose exact average equals frame_score.

    Used only when per-crop evaluation is unavailable or skipped. Every fruit score
    is marked measurement_mode='estimated', and the arithmetic mean across the
    fruits equals frame_score (avoiding artificial mismatched averages).
    """

    if not fruits:
        return []
    n = len(fruits)
    target_sum = round(float(frame_score) * n)
    base_score = target_sum // n
    remainder = target_sum - (base_score * n)

    items: list[dict[str, object]] = []
    for idx, fruit in enumerate(fruits):
        score = max(0, min(100, base_score + (1 if idx < remainder else 0)))
        label = _label_for_score(score, True)
        items.append({
            "fruit_id": int(fruit.get("fruit_id", fruit.get("instance_id", idx + 1))),
            "box": [round(float(v), 2) for v in fruit.get("box", [0, 0, 0, 0])],
            "mask_polygon": fruit.get("mask_polygon"),
            "inside_pallet": bool(fruit.get("inside_pallet", True)),
            "freshness_score": score,
            "label": label,
            "grade": _grade_for_score(score, True),
            "confidence": None,
            "note_fa": "امتیاز تخمینی بر پایه امتیاز کلی فریم",
            "measurement_mode": "estimated",
            "size": fruit.get("size"),
        })
    return items


def extract_image_frame(image_path: str | Path) -> np.ndarray:
    """Decode one still image as RGB for Qwen."""

    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None or not frame.size:
        raise VideoInsightError("could not decode image")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_video_frames(
    video_path: str | Path,
    *,
    num_frames: int = 8,
    window_start_seconds: float | None = None,
    window_end_seconds: float | None = None,
) -> list[np.ndarray]:
    """Return evenly spaced RGB frames from the requested video-time window."""

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VideoInsightError("could not open video")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise VideoInsightError("video does not contain any decodable frames")
        first_index = 0
        last_index = total_frames - 1
        if window_start_seconds is not None or window_end_seconds is not None:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                raise VideoInsightError("video frame rate is unavailable")
            first_index = max(0, int((window_start_seconds or 0.0) * fps))
            requested_end = window_end_seconds if window_end_seconds is not None else total_frames / fps
            last_index = min(total_frames - 1, max(first_index, int(requested_end * fps) - 1))
            if first_index >= total_frames:
                raise VideoInsightError("video time window is outside the recording")
        count = min(num_frames, last_index - first_index + 1)
        indexes = np.linspace(first_index, last_index, num=count, dtype=int)
        frames: list[np.ndarray] = []
        for index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise VideoInsightError("could not decode frames from video")
    return frames


def extract_stream_frames(
    stream_url: str,
    *,
    num_frames: int = 8,
    sample_interval_seconds: float = 0.75,
) -> list[np.ndarray]:
    """Sample consecutive RGB frames from a live RTSP stream."""

    capture = cv2.VideoCapture(stream_url)
    try:
        if not capture.isOpened():
            raise VideoInsightError("could not open camera stream")
        frames: list[np.ndarray] = []
        next_sample = time.monotonic()
        failures = 0
        deadline = next_sample + max(15.0, num_frames * sample_interval_seconds + 10.0)
        while len(frames) < num_frames and time.monotonic() < deadline:
            ok, frame = capture.read()
            if not ok or frame is None or not frame.size:
                failures += 1
                if failures >= 20:
                    break
                continue
            failures = 0
            now = time.monotonic()
            if now < next_sample:
                time.sleep(min(next_sample - now, 0.05))
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            next_sample = now + sample_interval_seconds
    finally:
        capture.release()
    if not frames:
        raise VideoInsightError("could not capture frames from camera stream")
    return frames


class VideoInsightService:
    """Lazily loads one quantized model and serializes GPU generation calls."""

    def __init__(self) -> None:
        self.model_path = os.environ.get(
            "VIDEO_INSIGHT_MODEL_PATH", "/models/Qwen2.5-VL-3B-Instruct"
        )
        self.min_pixels = int(os.environ.get("VIDEO_INSIGHT_MIN_PIXELS", str(256 * 256)))
        self.max_pixels = int(os.environ.get("VIDEO_INSIGHT_MAX_PIXELS", str(512 * 512)))
        self._model: Any = None
        self._processor: Any = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    def preload(self) -> None:
        """Load model weights before the first dashboard request."""

        self._load()

    def unload(self) -> bool:
        """Release Qwen model weights from GPU memory so sequential pipelines can run."""

        import gc

        with self._load_lock, self._inference_lock:
            was_loaded = self._model is not None or self._processor is not None
            self._model = None
            self._processor = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            return was_loaded

    def _load(self) -> tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise VideoInsightError(
                "video insight dependencies are unavailable; install the video-insight extra"
            ) from exc

        if self._model is not None and self._processor is not None:
            return self._model, self._processor, torch
        with self._load_lock:
            if self._model is not None and self._processor is not None:
                return self._model, self._processor, torch
            model_path = Path(self.model_path)
            if not model_path.is_dir():
                raise VideoInsightError(
                    f"Qwen model is missing at {model_path}; run the deployment model downloader"
                )
            processor = AutoProcessor.from_pretrained(
                str(model_path),
                local_files_only=True,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            load_options: dict[str, Any] = {"local_files_only": True}
            if torch.cuda.is_available():
                load_options.update(
                    device_map="auto",
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    ),
                    torch_dtype=torch.float16,
                )
            else:
                load_options.update(device_map="cpu", torch_dtype=torch.float32)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(model_path), **load_options
            )
            model.eval()
            self._processor = processor
            self._model = model
        return self._model, self._processor, torch

    def interpret(self, frames: list[np.ndarray]) -> dict[str, object]:
        """Return only normalized answers; never expose generated model text."""

        if not frames:
            raise VideoInsightError("at least one frame is required")
        model, processor, torch = self._load()
        with self._inference_lock, torch.inference_mode():
            content = [{"type": "image", "image": frame} for frame in frames]
            content.append({
                "type": "text",
                "text": (
                    "These are consecutive frames from a CCTV video, ordered from oldest to newest. "
                    "Answer both questions using only visible evidence. Treat uncertainty as No. "
                    "Return exactly one JSON object with no prose or markdown. Each value must be "
                    'either "Yes" or "No". Example: '
                    '{"fighting":"No","floor_clean":"Yes"}.\n'
                    f"1. {FIGHT_QUESTION}\n2. {FLOOR_CLEAN_QUESTION}"
                ),
            })
            raw_output, inference_seconds = self._generate(
                model,
                processor,
                torch,
                [{"role": "user", "content": content}],
                images=frames,
                max_new_tokens=40,
            )

        return {
            "answers": _parse_yes_no_answers(raw_output),
            "frame_count": len(frames),
            "inference_seconds": round(inference_seconds, 3),
        }

    def interpret_fruit_quality(
        self,
        frames: list[np.ndarray],
        *,
        counting_result: dict[str, Any] | None = None,
        filter_outside_roi: bool = False,
        evaluate_per_fruit: bool = True,
    ) -> dict[str, object]:
        """Score visible fruit freshness, optionally merging per-fruit crop quality with counting JSON."""

        if not frames:
            raise VideoInsightError("at least one frame is required")

        if counting_result and isinstance(counting_result, dict):
            return self._interpret_fruit_quality_with_counting(
                frames,
                counting_result=counting_result,
                filter_outside_roi=filter_outside_roi,
                evaluate_per_fruit=evaluate_per_fruit,
            )

        model, processor, torch = self._load()
        with self._inference_lock, torch.inference_mode():
            content = [{"type": "image", "image": frame} for frame in frames]
            content.append({"type": "text", "text": FRUIT_QUALITY_PROMPT})
            raw_output, inference_seconds = self._generate(
                model,
                processor,
                torch,
                [{"role": "user", "content": content}],
                images=frames,
                max_new_tokens=220,
            )
        parsed = _parse_fruit_quality(raw_output)
        score = int(parsed["freshness_score"])
        has_fruit = bool(parsed["has_fruit"])
        return {
            **parsed,
            "grade": _grade_for_score(score, has_fruit),
            "quality_mode": "estimated",
            "frame_count": len(frames),
            "inference_seconds": round(inference_seconds, 3),
        }

    def _interpret_fruit_quality_with_counting(
        self,
        frames: list[np.ndarray],
        *,
        counting_result: dict[str, Any],
        filter_outside_roi: bool = False,
        evaluate_per_fruit: bool = True,
    ) -> dict[str, object]:
        """Evaluate quality for each detected fruit crop inside the selected area and merge by fruit_id."""

        counting_frames = counting_result.get("frames")
        first_counting_frame = (
            counting_frames[0]
            if isinstance(counting_frames, list) and counting_frames and isinstance(counting_frames[0], dict)
            else counting_result
        )
        raw_fruits = (
            first_counting_frame.get("fruits")
            if isinstance(first_counting_frame.get("fruits"), list)
            else counting_result.get("detections", [])
        )
        if not isinstance(raw_fruits, list):
            raw_fruits = []

        pallet_polygon = (
            first_counting_frame.get("pallet_polygon")
            or counting_result.get("pallet_polygon")
            or []
        )
        image_width = first_counting_frame.get("image_width") or frames[0].shape[1]
        image_height = first_counting_frame.get("image_height") or frames[0].shape[0]
        frame_rgb = frames[0]
        if (
            isinstance(image_width, (int, float))
            and isinstance(image_height, (int, float))
            and int(image_width) > 0
            and int(image_height) > 0
            and (frame_rgb.shape[1], frame_rgb.shape[0]) != (int(image_width), int(image_height))
        ):
            frame_rgb = cv2.resize(
                frame_rgb,
                (int(image_width), int(image_height)),
                interpolation=cv2.INTER_AREA,
            )

        normalized_fruits: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_fruits):
            if not isinstance(item, dict):
                continue
            box = item.get("box")
            if not isinstance(box, list) or len(box) != 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            inside = item.get("inside_pallet")
            if not isinstance(inside, bool):
                inside = _point_inside_polygon(cx, cy, pallet_polygon) if pallet_polygon else True
            normalized_fruits.append({
                **item,
                "fruit_id": int(item.get("fruit_id", item.get("instance_id", idx + 1))),
                "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "inside_pallet": inside,
            })

        total_counted = len(normalized_fruits)
        selected_fruits = (
            [f for f in normalized_fruits if f["inside_pallet"]]
            if filter_outside_roi
            else normalized_fruits
        )

        if not selected_fruits:
            return {
                "has_fruit": False,
                "label": "نامشخص",
                "grade": None,
                "freshness_score": 0,
                "distribution": {"fresh": 0, "middle": 0, "rotten": 0},
                "fruit_count_estimate": 0,
                "counted_fruits_total": total_counted,
                "evaluated_fruits_count": 0,
                "confidence": 100,
                "summary_fa": "هیچ میوه‌ای در ناحیه انتخاب‌شده برای ارزیابی کیفیت یافت نشد.",
                "quality_mode": "measured" if evaluate_per_fruit else "estimated",
                "filter_outside_roi": filter_outside_roi,
                "pallet_polygon": pallet_polygon,
                "image_width": int(image_width),
                "image_height": int(image_height),
                "counting_job_id": counting_result.get("job_id"),
                "per_fruit_quality": [],
                "frame_count": len(frames),
                "inference_seconds": 0.0,
            }

        model, processor, torch = self._load()
        total_inference_seconds = 0.0
        per_fruit_records: list[dict[str, object]] = []

        if evaluate_per_fruit:
            with self._inference_lock, torch.inference_mode():
                for fruit in selected_fruits:
                    crop_rgb = _crop_fruit_rgb(
                        frame_rgb,
                        fruit["box"],
                        fruit.get("mask_polygon"),
                    )
                    if crop_rgb is None:
                        continue
                    content = [
                        {"type": "image", "image": crop_rgb},
                        {"type": "text", "text": PER_FRUIT_CROP_QUALITY_PROMPT},
                    ]
                    raw_output, elapsed = self._generate(
                        model,
                        processor,
                        torch,
                        [{"role": "user", "content": content}],
                        images=[crop_rgb],
                        max_new_tokens=140,
                    )
                    total_inference_seconds += elapsed
                    crop_quality = _parse_single_fruit_crop_quality(raw_output)
                    per_fruit_records.append({
                        "fruit_id": fruit["fruit_id"],
                        "box": fruit["box"],
                        "mask_polygon": fruit.get("mask_polygon"),
                        "inside_pallet": fruit["inside_pallet"],
                        "pallet_overlap": fruit.get("pallet_overlap"),
                        "freshness_score": int(crop_quality["freshness_score"]),
                        "label": crop_quality["label"],
                        "grade": crop_quality["grade"],
                        "confidence": int(crop_quality["confidence"]),
                        "note_fa": crop_quality["note_fa"],
                        "measurement_mode": "measured",
                        "size": fruit.get("size"),
                    })

            scores = [int(item["freshness_score"]) for item in per_fruit_records]
            confidences = [
                int(item["confidence"])
                for item in per_fruit_records
                if isinstance(item.get("confidence"), (int, float))
            ]
            mean_score = round(sum(scores) / len(scores)) if scores else 0
            exact_mean = round(sum(scores) / len(scores), 2) if scores else 0.0
            mean_conf = round(sum(confidences) / len(confidences)) if confidences else 85
            distribution = _distribution_from_scores(scores)
            overall_label = _label_for_score(exact_mean, bool(scores))
            overall_grade = _grade_for_score(exact_mean, bool(scores))
            summary_fa = (
                f"کیفیت {len(per_fruit_records)} میوه شمارش‌شده به‌صورت تک‌به‌تک اندازه‌گیری شد؛ "
                f"میانگین امتیاز تازگی {exact_mean} از ۱۰۰ ({overall_label}) است."
            )
            quality_mode = "measured"
        else:
            with self._inference_lock, torch.inference_mode():
                content = [
                    {"type": "image", "image": frame_rgb},
                    {"type": "text", "text": FRUIT_QUALITY_PROMPT},
                ]
                raw_output, elapsed = self._generate(
                    model,
                    processor,
                    torch,
                    [{"role": "user", "content": content}],
                    images=[frame_rgb],
                    max_new_tokens=220,
                )
                total_inference_seconds += elapsed
            whole_parsed = _parse_fruit_quality(raw_output)
            mean_score = int(whole_parsed["freshness_score"])
            exact_mean = float(mean_score)
            mean_conf = int(whole_parsed["confidence"])
            distribution = dict(whole_parsed["distribution"])  # type: ignore[arg-type]
            overall_label = str(whole_parsed["label"])
            overall_grade = _grade_for_score(mean_score, True)
            per_fruit_records = build_estimated_per_fruit_scores(mean_score, selected_fruits)
            summary_fa = (
                f"{whole_parsed['summary_fa']} (امتیاز تک‌میوه‌ها تخمینی و هم‌تراز با میانگین فریم است)."
            )
            quality_mode = "estimated"

        return {
            "has_fruit": True,
            "label": overall_label,
            "grade": overall_grade,
            "freshness_score": mean_score,
            "freshness_score_mean": exact_mean,
            "distribution": distribution,
            "fruit_count_estimate": len(per_fruit_records),
            "counted_fruits_total": total_counted,
            "evaluated_fruits_count": len(per_fruit_records),
            "confidence": mean_conf,
            "summary_fa": summary_fa,
            "quality_mode": quality_mode,
            "filter_outside_roi": filter_outside_roi,
            "pallet_polygon": pallet_polygon,
            "image_width": int(image_width),
            "image_height": int(image_height),
            "counting_job_id": counting_result.get("job_id"),
            "per_fruit_quality": per_fruit_records,
            "frame_count": len(frames),
            "inference_seconds": round(total_inference_seconds, 3),
        }

    @staticmethod
    def _generate(
        model: Any,
        processor: Any,
        torch: Any,
        messages: list[dict[str, Any]],
        *,
        images: list[np.ndarray] | None = None,
        max_new_tokens: int,
    ) -> tuple[str, float]:
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        processor_options: dict[str, Any] = {
            "text": [prompt],
            "return_tensors": "pt",
        }
        if images is not None:
            processor_options["images"] = images
        inputs = processor(**processor_options).to(next(model.parameters()).device)
        input_token_len = int(inputs.input_ids.shape[1])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = output[:, input_token_len:]
        text = processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        del inputs, output, generated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text, elapsed
