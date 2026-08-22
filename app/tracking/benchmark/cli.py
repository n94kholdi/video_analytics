"""Cache detections once, then benchmark one or more registered trackers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import cv2

from app.api.live import processing_frame_size, resize_processing_frame
from app.core.config import ConfigError, load_settings
from app.core.video_source import resolve_video_source
from app.detection.onnx_detector import OnnxPersonDetector
from app.tracking.benchmark.cache import CachedDetection, DetectionCache
from app.tracking.benchmark.metrics import GroundTruthBox
from app.tracking.benchmark.runner import BenchmarkRunner
from app.tracking.factory import available_tracker_types, public_tracker_catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark registered person trackers on cached detections.")
    parser.add_argument("source", nargs="?", help="recorded video used to (re)build the detection cache")
    parser.add_argument("--config", type=Path, help="application YAML configuration")
    parser.add_argument("--cache", type=Path, required=True, help="JSONL detection cache path")
    parser.add_argument("--output", type=Path, help="JSON report path (CSV is written beside it)")
    parser.add_argument(
        "--tracker",
        action="append",
        dest="trackers",
        help="tracker type; repeat to compare. Default: every registered tracker",
    )
    parser.add_argument("--gt", type=Path, help="optional MOT or CSV ground-truth path")
    parser.add_argument("--camera-id", default="benchmark")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--processing-width", type=int)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    trackers = args.trackers or list(available_tracker_types())
    unknown = [item for item in trackers if item not in available_tracker_types()]
    if unknown:
        known = ", ".join(available_tracker_types())
        raise ValueError(f"unknown tracker(s) {unknown}; expected one of: {known}")
    cache = DetectionCache(args.cache)
    if args.rebuild_cache or not args.cache.is_file():
        if args.source is None:
            raise ValueError("a video source is required to build the detection cache")
        _build_cache(args, settings, cache)
    frames = cache.read()
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    ground_truth = _load_ground_truth(args.gt) if args.gt is not None else ()
    video_frames = _load_video_frames(args) if args.source is not None else None
    runner = BenchmarkRunner(settings)
    reports = []
    frame_rate = 0.5
    if len(frames) >= 2:
        delta = frames[1].timestamp - frames[0].timestamp
        if delta > 0:
            frame_rate = 1.0 / delta
    for tracker_type in trackers:
        report, _observations = runner.run(
            frames,
            tracker_type=tracker_type,
            camera_id=args.camera_id,
            video_frames=video_frames,
            ground_truth=ground_truth,
            frame_rate=frame_rate,
        )
        reports.append(report.as_dict())
        if args.output is not None:
            target = args.output if len(trackers) == 1 else args.output.with_name(f"{args.output.stem}_{tracker_type}{args.output.suffix}")
            runner.write_report(report, target)
    print(json.dumps({"trackers": public_tracker_catalog(), "results": reports}, indent=2))


def _build_cache(args: argparse.Namespace, settings, cache: DetectionCache) -> None:
    model = settings.detector_model
    if model is None:
        raise ConfigError("a detector model is required via config to build a detection cache")
    detector = OnnxPersonDetector(
        model,
        confidence_threshold=settings.detector_confidence_threshold,
        iou_threshold=settings.detector_iou_threshold,
        providers=settings.onnx_providers,
    )
    source = resolve_video_source(args.source)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width, height = processing_frame_size(width, height, args.processing_width)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    frames: list[CachedDetection] = []
    source_frames = 0
    processed = 0
    try:
        while args.max_frames is None or processed < args.max_frames:
            readable, frame = capture.read()
            if not readable:
                break
            source_index = source_frames
            source_frames += 1
            if source_index % args.frame_stride != 0:
                continue
            frame = resize_processing_frame(frame, (width, height))
            source_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp = source_ms / 1000.0 if math.isfinite(source_ms) and source_ms >= 0 else source_index / fps
            detected = detector.detect(frame)
            frames.append(
                CachedDetection(
                    frame_index=processed,
                    timestamp=timestamp,
                    source_index=source_index,
                    detections=detected.detections,
                )
            )
            processed += 1
    finally:
        capture.release()
    cache.write(frames)


def _load_video_frames(args: argparse.Namespace) -> list:
    source = resolve_video_source(args.source)
    capture = cv2.VideoCapture(str(source))
    frames = []
    source_frames = 0
    try:
        while args.max_frames is None or len(frames) < args.max_frames:
            readable, frame = capture.read()
            if not readable:
                break
            source_index = source_frames
            source_frames += 1
            if source_index % args.frame_stride != 0:
                continue
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width, height = processing_frame_size(width, height, args.processing_width)
            frames.append(resize_processing_frame(frame, (width, height)))
    finally:
        capture.release()
    return frames


def _load_ground_truth(path: Path) -> tuple[GroundTruthBox, ...]:
    rows: list[GroundTruthBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [item.strip() for item in line.replace(" ", ",").split(",") if item.strip()]
        if len(parts) < 6:
            continue
        frame = int(float(parts[0]))
        track_id = int(float(parts[1]))
        x, y, width, height = (float(parts[i]) for i in range(2, 6))
        # MOT uses 1-based frames and xywh; CSV with x2>x1+x is treated as xyxy.
        if width > x and height > y:
            xyxy = (x, y, width, height)
            frame_index = frame
        else:
            xyxy = (x, y, x + width, y + height)
            frame_index = max(frame - 1, 0)
        rows.append(GroundTruthBox(frame_index=frame_index, track_id=track_id, xyxy=xyxy))
    return tuple(rows)


if __name__ == "__main__":
    main()
