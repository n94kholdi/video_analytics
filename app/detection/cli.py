"""Headless image and recorded-video CLI for Phase 2 detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2

from app.api.live import (
    LiveReporter,
    processed_frame_count,
    processing_frame_size,
    resize_processing_frame,
)
from app.core.config import AppSettings, ConfigError, load_settings
from app.core.video_source import is_network_video_source, resolve_video_source, video_source_stem
from app.detection.base import DetectionTimings
from app.detection.onnx_detector import OnnxPersonDetector
from app.detection.visualization import annotate_frame


IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 2 ONNX person detector on an image or video."
    )
    parser.add_argument("source", help="input image, recorded video, or RTSP URL")
    parser.add_argument("--config", type=Path, help="application YAML configuration")
    parser.add_argument("--model", type=Path, help="override detector model path")
    parser.add_argument("--output", type=Path, help="annotated image or video path")
    parser.add_argument(
        "--input-type",
        choices=("auto", "image", "video"),
        default="auto",
    )
    parser.add_argument("--live-dir", type=Path, help="optional dashboard job directory")
    parser.add_argument("--job-id", help="dashboard job ID (requires --live-dir)")
    parser.add_argument(
        "--processing-width",
        type=int,
        help="downscale wider input frames to this width before processing",
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="process every Nth source frame")
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
    if (args.live_dir is None) != (args.job_id is None):
        raise ValueError("--live-dir and --job-id must be provided together")
    if args.processing_width is not None and args.processing_width < 2:
        raise ValueError("--processing-width must be at least 2")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    settings = load_settings(args.config)
    source = resolve_video_source(args.source)

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
        input_type = (
            "video"
            if is_network_video_source(source)
            else "image" if source.suffix.lower() in IMAGE_SUFFIXES else "video"
        )
    if input_type == "image" and is_network_video_source(source):
        raise ValueError("RTSP sources can only be processed as video")
    output = args.output or _default_output(settings, source, input_type)

    if input_type == "image":
        summary = _run_image(detector, source, output, live_dir=args.live_dir, job_id=args.job_id)
    else:
        summary = _run_video(
            detector,
            source,
            output,
            max_frames=args.max_frames,
            live_dir=args.live_dir,
            job_id=args.job_id,
            processing_width=args.processing_width,
            frame_stride=args.frame_stride,
        )
    summary["providers"] = list(detector.providers)
    summary["model"] = str(detector.model_path)
    print(json.dumps(summary, indent=2))


def _run_image(
    detector: OnnxPersonDetector,
    source: Path,
    output: Path,
    *,
    live_dir: Path | None = None,
    job_id: str | None = None,
) -> dict[str, object]:
    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError(f"OpenCV could not decode image: {source}")
    result = detector.detect(frame)
    annotated = annotate_frame(frame, result)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), annotated):
        raise RuntimeError(f"OpenCV could not write image: {output}")
    reporter = LiveReporter(live_dir, job_id, total_frames=1)
    reporter.publish(0, {"current_people": len(result.detections), "total_detections": len(result.detections), "frame_count": 1, "progress": 100.0, "elapsed_seconds": reporter.elapsed}, frame=annotated, force=True)
    return {
        "input_type": "image",
        "frames": 1,
        "detections": len(result.detections),
        "average_timings_ms": _timing_summary([result.timings]),
        "output": str(output.resolve()),
    }


def _run_video(
    detector: OnnxPersonDetector,
    source: str | Path,
    output: Path,
    *,
    max_frames: int | None,
    live_dir: Path | None = None,
    job_id: str | None = None,
    processing_width: int | None = None,
    frame_stride: int = 1,
) -> dict[str, object]:
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"video has invalid dimensions: {width}x{height}")
    width, height = processing_frame_size(width, height, processing_width)
    if fps <= 0:
        fps = 30.0
    output_fps = fps / frame_stride
    reporter = LiveReporter(
        live_dir,
        job_id,
        total_frames=processed_frame_count(capture, max_frames, frame_stride),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not create output video: {output}")

    frames = 0
    detections = 0
    timings = []
    latest_metrics: dict[str, object] = {}
    source_frames = 0
    try:
        while max_frames is None or frames < max_frames:
            readable, frame = capture.read()
            if not readable:
                break
            source_index = source_frames
            source_frames += 1
            if source_index % frame_stride != 0:
                continue
            frame = resize_processing_frame(frame, (width, height))
            result = detector.detect(frame)
            annotated = annotate_frame(frame, result)
            writer.write(annotated)
            frames += 1
            detections += len(result.detections)
            timings.append(result.timings)
            latest_metrics = {
                    "current_people": len(result.detections),
                    "total_detections": detections,
                    "processing_fps": 1000.0 / max(result.timings.total_ms, 0.001),
                    "frame_count": frames,
                    "progress": min(100.0, frames * 100.0 / reporter.total_frames) if reporter.total_frames else None,
                    "elapsed_seconds": reporter.elapsed,
                }
            reporter.publish(
                frames - 1,
                latest_metrics,
                frame=annotated,
            )
    finally:
        capture.release()
        writer.release()

    if frames == 0:
        raise RuntimeError(f"video contained no readable frames: {source}")
    reporter.publish(frames - 1, latest_metrics, frame=annotated, force=True)
    return {
        "input_type": "video",
        "frames": frames,
        "fps": fps,
        "frame_stride": frame_stride,
        "processing_fps": 1000.0 / _timing_summary(timings)["total"],
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
    source: str | Path,
    input_type: str,
) -> Path:
    suffix = source.suffix if input_type == "image" and isinstance(source, Path) else ".mp4"
    return settings.output_dir / f"{video_source_stem(source)}_detected{suffix}"


if __name__ == "__main__":
    main()
