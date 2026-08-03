"""Recorded-video detector, ByteTrack, and trajectory annotation runner."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Sequence

import cv2

from app.core.config import ConfigError, load_settings
from app.detection.onnx_detector import OnnxPersonDetector
from app.tracking.bytetrack import ByteTrackAdapter
from app.tracking.visualization import annotate_tracks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 3 person tracking on a video.")
    parser.add_argument("source", type=Path, help="input recorded video")
    parser.add_argument("--config", type=Path, help="application YAML configuration")
    parser.add_argument("--model", type=Path, help="override detector model path")
    parser.add_argument("--output", type=Path, help="annotated MP4 path")
    parser.add_argument("--camera-id", default="camera-1")
    parser.add_argument("--max-frames", type=int)
    trajectory_group = parser.add_mutually_exclusive_group()
    trajectory_group.add_argument(
        "--show-trajectories",
        dest="show_trajectories",
        action="store_true",
        help="show smoothed trajectory trails (default: enabled)",
    )
    trajectory_group.add_argument(
        "--no-trajectories",
        dest="show_trajectories",
        action="store_false",
        help="hide trajectory trails while retaining trajectory history",
    )
    parser.set_defaults(show_trajectories=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    settings = load_settings(args.config)
    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input source does not exist: {source}")
    model = args.model or settings.detector_model
    if model is None:
        raise ConfigError("a detector model is required via config or --model")
    output = args.output or settings.output_dir / f"{source.stem}_tracked.mp4"

    detector = OnnxPersonDetector(
        model,
        confidence_threshold=settings.detector_confidence_threshold,
        iou_threshold=settings.detector_iou_threshold,
        providers=settings.onnx_providers,
    )
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"video has invalid dimensions: {width}x{height}")
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    tracker = ByteTrackAdapter(
        activation_threshold=settings.tracker_activation_threshold,
        lost_track_buffer=settings.tracker_lost_track_buffer,
        match_threshold=settings.tracker_match_threshold,
        history_size=settings.tracker_history_size,
        frame_rate=fps,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not create output video: {output}")

    frames = 0
    observations = 0
    detection_ms = 0.0
    tracking_ms = 0.0
    total_ms = 0.0
    try:
        while args.max_frames is None or frames < args.max_frames:
            readable, frame = capture.read()
            if not readable:
                break
            frame_started = perf_counter()
            source_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp = source_ms / 1000.0 if math.isfinite(source_ms) and source_ms >= 0 else frames / fps
            detected = detector.detect(frame)
            tracked = tracker.update(
                detected.detections,
                camera_id=args.camera_id,
                timestamp=timestamp,
                frame_index=frames,
            )
            writer.write(
                annotate_tracks(
                    frame,
                    tracked.observations,
                    tracking_ms=tracked.tracking_ms,
                    show_trajectories=args.show_trajectories,
                )
            )
            frames += 1
            observations += len(tracked.observations)
            detection_ms += detected.timings.total_ms
            tracking_ms += tracked.tracking_ms
            total_ms += (perf_counter() - frame_started) * 1000.0
    finally:
        capture.release()
        writer.release()
    if frames == 0:
        raise RuntimeError(f"video contained no readable frames: {source}")
    print(
        json.dumps(
            {
                "frames": frames,
                "fps": fps,
                "processing_fps": 1000.0 / (total_ms / frames),
                "track_observations": observations,
                "average_timings_ms": {
                    "detection": detection_ms / frames,
                    "tracking": tracking_ms / frames,
                    "total_frame": total_ms / frames,
                },
                "providers": list(detector.providers),
                "model": str(detector.model_path),
                "output": str(output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
