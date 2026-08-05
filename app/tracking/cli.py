"""Recorded-video detector, ByteTrack, and trajectory annotation runner."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Sequence

import cv2

from app.api.live import (
    LiveReporter,
    processed_frame_count,
    processing_frame_size,
    resize_processing_frame,
)
from app.core.config import ConfigError, load_settings
from app.detection.onnx_detector import OnnxPersonDetector
from app.tracking.bytetrack import ByteTrackAdapter
from app.tracking.visualization import annotate_tracks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 3 person tracking on a video.")
    parser.add_argument("source", type=Path, help="input recorded video")
    parser.add_argument("--config", type=Path, help="application YAML configuration")
    parser.add_argument("--model", type=Path, help="override detector model path")
    parser.add_argument(
        "--enable-reid",
        action="store_true",
        help="enable higher-cost OSNet appearance re-identification",
    )
    parser.add_argument("--reid-model", type=Path, help="override OSNet ReID model path")
    parser.add_argument("--output", type=Path, help="annotated MP4 path")
    parser.add_argument("--camera-id", default="camera-1")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--live-dir", type=Path, help="optional dashboard job directory")
    parser.add_argument("--job-id", help="dashboard job ID (requires --live-dir)")
    parser.add_argument(
        "--processing-width",
        type=int,
        help="downscale wider input frames to this width before processing",
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="process every Nth source frame")
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
    if (args.live_dir is None) != (args.job_id is None):
        raise ValueError("--live-dir and --job-id must be provided together")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.processing_width is not None and args.processing_width < 2:
        raise ValueError("--processing-width must be at least 2")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    settings = load_settings(args.config)
    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input source does not exist: {source}")
    model = args.model or settings.detector_model
    if model is None:
        raise ConfigError("a detector model is required via config or --model")
    reid_model = args.reid_model or settings.reid_model
    if args.enable_reid and reid_model is None:
        raise ConfigError("--enable-reid requires a ReID model via config or --reid-model")
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
    width, height = processing_frame_size(width, height, args.processing_width)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    output_fps = fps / args.frame_stride
    tracker = ByteTrackAdapter(
        activation_threshold=settings.tracker_activation_threshold,
        lost_track_buffer=settings.tracker_lost_track_buffer,
        match_threshold=settings.tracker_match_threshold,
        history_size=settings.tracker_history_size,
        frame_rate=output_fps,
        frame_size=(width, height),
        reid_model=reid_model if args.enable_reid else None,
        reid_providers=settings.onnx_providers,
    )
    reporter = LiveReporter(
        args.live_dir,
        args.job_id,
        total_frames=processed_frame_count(capture, args.max_frames, args.frame_stride),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not create output video: {output}")

    frames = 0
    observations = 0
    detection_ms = 0.0
    tracking_ms = 0.0
    total_ms = 0.0
    unique_track_ids: set[int] = set()
    lost_tracks = 0
    latest_metrics: dict[str, object] = {}
    source_frames = 0
    tracking_stream = (
        (args.live_dir / "tracking.jsonl").open("w", encoding="utf-8")
        if args.live_dir is not None else None
    )
    try:
        while args.max_frames is None or frames < args.max_frames:
            readable, frame = capture.read()
            if not readable:
                break
            source_index = source_frames
            source_frames += 1
            if source_index % args.frame_stride != 0:
                continue
            frame = resize_processing_frame(frame, (width, height))
            frame_started = perf_counter()
            source_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp = source_ms / 1000.0 if math.isfinite(source_ms) and source_ms >= 0 else source_index / fps
            detected = detector.detect(frame)
            tracked = tracker.update(
                detected.detections,
                camera_id=args.camera_id,
                timestamp=timestamp,
                frame_index=frames,
                frame=frame,
            )
            current_track_ids = {
                item.track_id for item in tracked.observations if item.confirmed
            }
            unique_track_ids.update(current_track_ids)
            annotated = annotate_tracks(
                    frame,
                    tracked.observations,
                    tracking_ms=tracked.tracking_ms,
                    show_trajectories=args.show_trajectories,
                    current_people=len(current_track_ids),
                    total_unique_people=len(unique_track_ids),
                )
            writer.write(annotated)
            frames += 1
            observations += len(tracked.observations)
            lost_tracks += len(tracked.expired_track_ids)
            if tracking_stream is not None:
                tracking_stream.write(json.dumps({
                    "frame_index": frames - 1,
                    "timestamp_seconds": timestamp,
                    "tracks": [
                        {
                            "track_id": item.track_id,
                            "xyxy": item.xyxy,
                            "confidence": item.detection_confidence,
                            "confirmed": item.confirmed,
                        }
                        for item in tracked.observations
                    ],
                    "expired_track_ids": tracked.expired_track_ids,
                }, separators=(",", ":")) + "\n")
            detection_ms += detected.timings.total_ms
            tracking_ms += tracked.tracking_ms
            total_ms += (perf_counter() - frame_started) * 1000.0
            latest_metrics = {
                    "current_people": len(current_track_ids),
                    "total_unique_people": len(unique_track_ids),
                    "active_tracks": len(tracked.observations),
                    "lost_tracks": lost_tracks,
                    "processing_fps": 1000.0 / max(total_ms / frames, 0.001),
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
        if tracking_stream is not None:
            tracking_stream.close()
    if frames == 0:
        raise RuntimeError(f"video contained no readable frames: {source}")
    reporter.publish(frames - 1, latest_metrics, frame=annotated, force=True)
    print(
        json.dumps(
            {
                "frames": frames,
                "fps": fps,
                "frame_stride": args.frame_stride,
                "processing_fps": 1000.0 / (total_ms / frames),
                "track_observations": observations,
                "total_unique_people": len(unique_track_ids),
                "reid_enabled": tracker.reid_enabled,
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
