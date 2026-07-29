"""Headless image and recorded-video CLI for Phase 2 detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2

from app.core.config import AppSettings, ConfigError, load_settings
from app.detection.base import DetectionTimings
from app.detection.onnx_detector import OnnxPersonDetector
from app.detection.visualization import annotate_frame


IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 2 ONNX person detector on an image or video."
    )
    parser.add_argument("source", type=Path, help="input image or recorded video")
    parser.add_argument("--config", type=Path, help="application YAML configuration")
    parser.add_argument("--model", type=Path, help="override detector model path")
    parser.add_argument("--output", type=Path, help="annotated image or video path")
    parser.add_argument(
        "--input-type",
        choices=("auto", "image", "video"),
        default="auto",
    )
    parser.add_argument("--confidence", type=float, help="confidence threshold [0,1]")
    parser.add_argument("--iou", type=float, help="NMS IoU threshold [0,1]")
    parser.add_argument(
        "--providers",
        nargs="+",
        help="ONNX Runtime providers in priority order; CPU is added as fallback",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="optional video frame limit for smoke tests",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input source does not exist: {source}")

    model = (args.model or settings.detector_model)
    if model is None:
        raise ConfigError("a detector model is required via config or --model")
    providers = tuple(args.providers or settings.onnx_providers)
    detector = OnnxPersonDetector(
        model,
        confidence_threshold=(
            settings.detector_confidence_threshold
            if args.confidence is None
            else args.confidence
        ),
        iou_threshold=(
            settings.detector_iou_threshold if args.iou is None else args.iou
        ),
        providers=providers,
    )

    input_type = args.input_type
    if input_type == "auto":
        input_type = "image" if source.suffix.lower() in IMAGE_SUFFIXES else "video"
    output = args.output or _default_output(settings, source, input_type)

    if input_type == "image":
        summary = _run_image(detector, source, output)
    else:
        summary = _run_video(
            detector,
            source,
            output,
            max_frames=args.max_frames,
        )
    summary["providers"] = list(detector.providers)
    summary["model"] = str(detector.model_path)
    print(json.dumps(summary, indent=2))


def _run_image(
    detector: OnnxPersonDetector,
    source: Path,
    output: Path,
) -> dict[str, object]:
    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError(f"OpenCV could not decode image: {source}")
    result = detector.detect(frame)
    annotated = annotate_frame(frame, result)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), annotated):
        raise RuntimeError(f"OpenCV could not write image: {output}")
    return {
        "input_type": "image",
        "frames": 1,
        "detections": len(result.detections),
        "average_timings_ms": _timing_summary([result.timings]),
        "output": str(output.resolve()),
    }


def _run_video(
    detector: OnnxPersonDetector,
    source: Path,
    output: Path,
    *,
    max_frames: int | None,
) -> dict[str, object]:
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"video has invalid dimensions: {width}x{height}")
    if fps <= 0:
        fps = 30.0

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not create output video: {output}")

    frames = 0
    detections = 0
    timings = []
    try:
        while max_frames is None or frames < max_frames:
            readable, frame = capture.read()
            if not readable:
                break
            result = detector.detect(frame)
            writer.write(annotate_frame(frame, result))
            frames += 1
            detections += len(result.detections)
            timings.append(result.timings)
    finally:
        capture.release()
        writer.release()

    if frames == 0:
        raise RuntimeError(f"video contained no readable frames: {source}")
    return {
        "input_type": "video",
        "frames": frames,
        "detections": detections,
        "average_timings_ms": _timing_summary(timings),
        "output": str(output.resolve()),
    }


def _timing_summary(timings: Sequence[DetectionTimings]) -> dict[str, float]:
    count = len(timings)
    return {
        "preprocessing": sum(item.preprocessing_ms for item in timings) / count,
        "inference": sum(item.inference_ms for item in timings) / count,
        "postprocessing": sum(item.postprocessing_ms for item in timings) / count,
        "total": sum(item.total_ms for item in timings) / count,
    }


def _default_output(
    settings: AppSettings,
    source: Path,
    input_type: str,
) -> Path:
    suffix = source.suffix if input_type == "image" else ".mp4"
    return settings.output_dir / f"{source.stem}_detected{suffix}"


if __name__ == "__main__":
    main()
