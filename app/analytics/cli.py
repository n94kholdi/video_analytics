"""Recorded-video detector, tracker, and per-frame people-counting runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Sequence

import cv2

from app.analytics.counting import CameraCountingConfig, PeopleCounter
from app.analytics.restricted_area import (
    CameraRestrictedAreaConfig,
    RestrictedAreaDetector,
)
from app.analytics.restricted_visualization import annotate_restricted_areas
from app.analytics.visualization import annotate_people_counts
from app.core.config import ConfigError, load_settings
from app.detection.onnx_detector import OnnxPersonDetector
from app.geometry.config import (
    CameraConfig,
    NormalizedPoint,
    PolygonZone,
    load_camera_config,
)
from app.storage import JsonlEventSink
from app.tracking.bytetrack import ByteTrackAdapter
from app.tracking.visualization import annotate_tracks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count tracked people in every frame of a recorded video."
    )
    parser.add_argument("source", type=Path, help="input recorded video")
    parser.add_argument(
        "--camera-config",
        type=Path,
        help="camera YAML with occupancy polygons; defaults to the whole frame",
    )
    parser.add_argument("--config", type=Path, help="application YAML configuration")
    parser.add_argument("--model", type=Path, help="override detector model path")
    parser.add_argument("--output", type=Path, help="annotated MP4 path")
    parser.add_argument("--counts-csv", type=Path, help="per-frame count CSV path")
    parser.add_argument(
        "--events-jsonl",
        type=Path,
        help="append restricted-area events as JSONL (camera YAML output is the default)",
    )
    parser.add_argument("--camera-id", default="camera-1")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--no-trajectories",
        action="store_true",
        help="hide track trails in the annotated video",
    )
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
    output = args.output or settings.output_dir / f"{source.stem}_counted.mp4"
    counts_csv = args.counts_csv or settings.output_dir / f"{source.stem}_counts.csv"
    if output.expanduser().resolve() == source:
        raise ValueError("output video must differ from the input video")

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

    camera_config = (
        load_camera_config(args.camera_config) if args.camera_config is not None else None
    )
    camera_counting = _counting_config(
        camera_config, args.camera_id, (width, height)
    )
    restricted_config = _restricted_config(
        camera_config, camera_counting.camera_id, (width, height)
    )
    event_path = args.events_jsonl
    if event_path is None and camera_config is not None:
        configured_event_path = camera_config.outputs.events_jsonl
        event_path = Path(configured_event_path) if configured_event_path else None
    detector = OnnxPersonDetector(
        model,
        confidence_threshold=settings.detector_confidence_threshold,
        iou_threshold=settings.detector_iou_threshold,
        providers=settings.onnx_providers,
    )
    tracker = ByteTrackAdapter(
        activation_threshold=settings.tracker_activation_threshold,
        lost_track_buffer=settings.tracker_lost_track_buffer,
        match_threshold=settings.tracker_match_threshold,
        history_size=settings.tracker_history_size,
        frame_rate=fps,
    )
    counter = PeopleCounter((camera_counting,))
    restricted = RestrictedAreaDetector(
        (restricted_config,),
        event_sink=JsonlEventSink(event_path) if event_path is not None else None,
    )
    counter.reset()  # Explicit processing-run boundary.
    restricted.reset()

    output.parent.mkdir(parents=True, exist_ok=True)
    counts_csv.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not create output video: {output}")

    frames = 0
    events = 0
    restricted_events = 0
    maximum_confirmed = 0
    maximum_occupancy = 0
    total_ms = 0.0
    zone_ids = tuple(zone.zone_id for zone in camera_counting.occupancy_zones)
    restricted_zone_ids = tuple(zone.zone_id for zone in restricted_config.zones)
    try:
        with counts_csv.open("w", encoding="utf-8", newline="") as stream:
            csv_writer = csv.writer(stream)
            csv_writer.writerow(
                [
                    "frame_index",
                    "timestamp_seconds",
                    "confirmed_humans",
                    "total_zone_occupancy",
                    *(f"occupancy_{zone_id}" for zone_id in zone_ids),
                    "cumulative_entries",
                    "cumulative_exits",
                    *(f"restricted_current_{zone_id}" for zone_id in restricted_zone_ids),
                    *(f"restricted_entries_{zone_id}" for zone_id in restricted_zone_ids),
                    *(f"restricted_exits_{zone_id}" for zone_id in restricted_zone_ids),
                ]
            )
            while args.max_frames is None or frames < args.max_frames:
                readable, frame = capture.read()
                if not readable:
                    break
                started = perf_counter()
                source_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                timestamp = (
                    source_ms / 1000.0
                    if math.isfinite(source_ms) and source_ms >= 0
                    else frames / fps
                )
                detected = detector.detect(frame)
                tracked = tracker.update(
                    detected.detections,
                    camera_id=camera_counting.camera_id,
                    timestamp=timestamp,
                    frame_index=frames,
                )
                counted = counter.update(
                    camera_counting.camera_id,
                    tracked.observations,
                    timestamp=timestamp,
                )
                intrusion = restricted.update(
                    restricted_config.camera_id,
                    tracked.observations,
                    timestamp=timestamp,
                )
                confirmed_humans = sum(
                    observation.confirmed for observation in tracked.observations
                )
                snapshot = counted.snapshot
                annotated = annotate_tracks(
                    frame,
                    tracked.observations,
                    tracking_ms=tracked.tracking_ms,
                    show_trajectories=not args.no_trajectories,
                )
                counted_frame = annotate_people_counts(
                    annotated,
                    snapshot,
                    confirmed_humans=confirmed_humans,
                    restricted_snapshot=intrusion.snapshot,
                )
                writer.write(
                    annotate_restricted_areas(
                        counted_frame,
                        restricted_config,
                        intrusion.snapshot,
                        tracked.observations,
                        copy=False,
                    )
                )
                csv_writer.writerow(
                    [
                        frames,
                        f"{timestamp:.6f}",
                        confirmed_humans,
                        snapshot.current_occupancy,
                        *(snapshot.occupancy_for(zone_id) for zone_id in zone_ids),
                        snapshot.cumulative_entries,
                        snapshot.cumulative_exits,
                        *(
                            intrusion.snapshot.zone_for(zone_id).current_tracks
                            for zone_id in restricted_zone_ids
                        ),
                        *(
                            intrusion.snapshot.zone_for(zone_id).cumulative_entries
                            for zone_id in restricted_zone_ids
                        ),
                        *(
                            intrusion.snapshot.zone_for(zone_id).cumulative_exits
                            for zone_id in restricted_zone_ids
                        ),
                    ]
                )
                maximum_confirmed = max(maximum_confirmed, confirmed_humans)
                maximum_occupancy = max(
                    maximum_occupancy, snapshot.current_occupancy
                )
                events += len(counted.events)
                restricted_events += len(intrusion.events)
                frames += 1
                total_ms += (perf_counter() - started) * 1000.0
    finally:
        capture.release()
        writer.release()

    if frames == 0:
        raise RuntimeError(f"video contained no readable frames: {source}")
    print(
        json.dumps(
            {
                "frames": frames,
                "maximum_confirmed_humans": maximum_confirmed,
                "maximum_total_zone_occupancy": maximum_occupancy,
                "line_crossed_events": events,
                "restricted_area_events": restricted_events,
                "restricted_events_jsonl": str(event_path.resolve())
                if event_path is not None
                else None,
                "average_total_frame_ms": total_ms / frames,
                "annotated_video": str(output.resolve()),
                "per_frame_counts_csv": str(counts_csv.resolve()),
            },
            indent=2,
        )
    )


def _counting_config(
    camera_config: CameraConfig | None,
    default_camera_id: str,
    frame_size: tuple[int, int],
) -> CameraCountingConfig:
    if camera_config is not None:
        return CameraCountingConfig.from_camera_config(
            camera_config, frame_size
        )
    whole_frame = PolygonZone(
        "frame",
        (
            NormalizedPoint(0.0, 0.0),
            NormalizedPoint(1.0, 0.0),
            NormalizedPoint(1.0, 1.0),
            NormalizedPoint(0.0, 1.0),
        ),
    )
    return CameraCountingConfig(default_camera_id, frame_size, (whole_frame,))


def _restricted_config(
    camera_config: CameraConfig | None,
    default_camera_id: str,
    frame_size: tuple[int, int],
) -> CameraRestrictedAreaConfig:
    if camera_config is not None:
        return CameraRestrictedAreaConfig.from_camera_config(camera_config, frame_size)
    return CameraRestrictedAreaConfig(default_camera_id, frame_size)


if __name__ == "__main__":
    main()
