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

from app.api.live import (
    LiveReporter,
    processed_frame_count,
    processing_frame_size,
    resize_processing_frame,
)
from app.analytics.counting import CameraCountingConfig, PeopleCounter
from app.analytics.heatmap import (
    HeatmapExportPaths,
    HeatmapVideoWriter,
    MovementHeatmaps,
    colorize_heatmap,
    export_heatmap_snapshot,
    overlay_heatmap,
)
from app.analytics.restricted_area import (
    CameraRestrictedAreaConfig,
    RestrictedAreaDetector,
)
from app.analytics.restricted_visualization import annotate_restricted_areas
from app.analytics.queue import CameraQueueConfig, QueueAnalyzer
from app.analytics.queue_visualization import annotate_queues
from app.analytics.speed import CameraSpeedConfig, SpeedEstimator
from app.analytics.vertical_queue import VerticalQueueAnalyzer, VerticalQueueConfig
from app.analytics.vertical_queue_visualization import annotate_vertical_queues
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
        "--enable-heatmap",
        action="store_true",
        help="enable evolving occupancy/dwell videos and final heatmap exports",
    )
    parser.add_argument(
        "--enable-queue",
        action="store_true",
        help="enable queue grouping, CSV metrics, and overlays",
    )
    parser.add_argument(
        "--queue-mode",
        choices=("vertical", "configured"),
        default="vertical",
        help="vertical bbox-center grouping (default) or configured polygons",
    )
    parser.add_argument(
        "--queue-column-distance",
        type=float,
        default=0.08,
        help="maximum horizontal bbox-center distance as a frame-width fraction",
    )
    parser.add_argument(
        "--queue-min-people",
        type=int,
        default=2,
        help="minimum people needed to display an automatic vertical queue",
    )
    parser.add_argument(
        "--enable-restricted-area",
        action="store_true",
        help="enable configured restricted-area state, events, CSV, and overlays",
    )
    parser.add_argument(
        "--heatmap-dir",
        type=Path,
        help="heatmap output directory (requires --enable-heatmap)",
    )
    parser.add_argument(
        "--events-jsonl",
        type=Path,
        help=(
            "append restricted-area and configured-queue events as JSONL "
            "(camera YAML output is the default)"
        ),
    )
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
    parser.add_argument(
        "--no-trajectories",
        action="store_true",
        help="hide track trails in the annotated video",
    )
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
    if args.heatmap_dir is not None and not args.enable_heatmap:
        raise ValueError("--heatmap-dir requires --enable-heatmap")
    if not 0.0 < args.queue_column_distance <= 1.0:
        raise ValueError("--queue-column-distance must be in (0, 1]")
    if args.queue_min_people <= 0:
        raise ValueError("--queue-min-people must be positive")

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
    width, height = processing_frame_size(width, height, args.processing_width)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    output_fps = fps / args.frame_stride

    camera_config = (
        load_camera_config(args.camera_config) if args.camera_config is not None else None
    )
    camera_counting = _counting_config(
        camera_config, args.camera_id, (width, height)
    )
    restricted_config = _restricted_config(
        camera_config, camera_counting.camera_id, (width, height)
    )
    queue_config = _queue_config(
        camera_config, camera_counting.camera_id, (width, height)
    )
    speed_requested = args.enable_queue or (
        camera_config is not None and "speed" in camera_config.analytics.enabled
    )
    speed_config = (
        CameraSpeedConfig.from_camera_config(camera_config, (width, height))
        if camera_config is not None
        else CameraSpeedConfig.for_image(
            camera_counting.camera_id, (width, height)
        )
    )
    speed_estimator = SpeedEstimator((speed_config,)) if speed_requested else None
    if args.enable_restricted_area and not restricted_config.zones:
        raise ValueError(
            "--enable-restricted-area requires at least one enabled restricted zone"
        )
    if (
        args.enable_queue
        and args.queue_mode == "configured"
        and not queue_config.queues
    ):
        raise ValueError(
            "configured queue mode requires at least one enabled configured queue"
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
        frame_rate=output_fps,
        frame_size=(width, height),
    )
    reporter = LiveReporter(
        args.live_dir,
        args.job_id,
        total_frames=processed_frame_count(capture, args.max_frames, args.frame_stride),
    )
    counter = PeopleCounter((camera_counting,))
    restricted = (
        RestrictedAreaDetector(
            (restricted_config,),
            event_sink=JsonlEventSink(event_path) if event_path is not None else None,
        )
        if args.enable_restricted_area
        else None
    )
    configured_queues = (
        QueueAnalyzer(
            (queue_config,),
            event_sink=JsonlEventSink(event_path) if event_path is not None else None,
        )
        if args.enable_queue and args.queue_mode == "configured"
        else None
    )
    vertical_queues = (
        VerticalQueueAnalyzer(
            (
                VerticalQueueConfig(
                    camera_counting.camera_id,
                    (width, height),
                    maximum_center_distance_fraction=args.queue_column_distance,
                    minimum_people=args.queue_min_people,
                ),
            )
        )
        if args.enable_queue and args.queue_mode == "vertical"
        else None
    )
    counter.reset()  # Explicit processing-run boundary.
    if restricted is not None:
        restricted.reset()
    if configured_queues is not None:
        configured_queues.reset()
    if vertical_queues is not None:
        vertical_queues.reset()
    if speed_estimator is not None:
        speed_estimator.reset()
    heatmaps = None
    heatmap_directory = None
    heatmap_settings = camera_config.heatmap if camera_config is not None else None
    if args.enable_heatmap:
        heatmaps = (
            MovementHeatmaps.from_camera_config(camera_config, (width, height))
            if camera_config is not None
            else MovementHeatmaps.for_image(camera_counting.camera_id, (width, height))
        )
        configured_directory = (
            camera_config.outputs.heatmap_directory
            if camera_config is not None
            else None
        )
        heatmap_directory = (
            args.heatmap_dir
            or (Path(configured_directory) if configured_directory else None)
            or settings.output_dir / f"{source.stem}_heatmaps"
        )
    if heatmaps is not None:
        heatmaps.reset()

    output.parent.mkdir(parents=True, exist_ok=True)
    counts_csv.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not create output video: {output}")
    heatmap_video_writer = None
    if heatmaps is not None:
        assert heatmap_directory is not None
        color_map = heatmap_settings.color_map if heatmap_settings else "jet"
        opacity = heatmap_settings.opacity if heatmap_settings else 0.55
        smoothing_sigma = (
            heatmap_settings.smoothing_sigma_cells if heatmap_settings else 1.0
        )
        try:
            heatmap_video_writer = HeatmapVideoWriter(
                heatmap_directory,
                prefix=f"{heatmaps.camera_id}_image",
                fps=output_fps,
                frame_size=(width, height),
                color_map=color_map,
                opacity=opacity,
                smoothing_sigma_cells=smoothing_sigma,
            )
        except Exception:
            capture.release()
            writer.release()
            raise

    frames = 0
    events = 0
    restricted_events = 0
    queue_events = 0
    maximum_confirmed = 0
    maximum_occupancy = 0
    total_ms = 0.0
    reference_frame = None
    unique_track_ids: set[int] = set()
    lost_tracks = 0
    zone_ids = tuple(zone.zone_id for zone in camera_counting.occupancy_zones)
    restricted_zone_ids = (
        tuple(zone.zone_id for zone in restricted_config.zones)
        if restricted is not None
        else ()
    )
    queue_ids = (
        tuple(queue.queue_id for queue in queue_config.queues)
        if configured_queues is not None
        else ()
    )
    queue_headers = (
        tuple(f"queue_raw_{queue_id}" for queue_id in queue_ids)
        + tuple(f"queue_smoothed_{queue_id}" for queue_id in queue_ids)
        + tuple(f"queue_wait_seconds_{queue_id}" for queue_id in queue_ids)
        + tuple(f"queue_overflow_{queue_id}" for queue_id in queue_ids)
        + tuple(f"queue_average_speed_pixels_per_second_{queue_id}" for queue_id in queue_ids)
        + tuple(f"queue_average_speed_metres_per_second_{queue_id}" for queue_id in queue_ids)
        + tuple(f"queue_progress_pixels_per_second_{queue_id}" for queue_id in queue_ids)
        + tuple(f"queue_progress_metres_per_second_{queue_id}" for queue_id in queue_ids)
        if configured_queues is not None
        else (
            (
                "vertical_queue_rows",
                "vertical_queue_people",
                "vertical_queue_counts",
                "vertical_queue_speeds_pixels_per_second",
                "vertical_queue_speeds_metres_per_second",
            )
            if vertical_queues is not None
            else ()
        )
    )
    source_frames = 0
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
                    *(
                        (
                            "average_speed_pixels_per_second",
                            "average_speed_metres_per_second",
                        )
                        if speed_estimator is not None
                        else ()
                    ),
                    *(f"restricted_current_{zone_id}" for zone_id in restricted_zone_ids),
                    *(f"restricted_entries_{zone_id}" for zone_id in restricted_zone_ids),
                    *(f"restricted_exits_{zone_id}" for zone_id in restricted_zone_ids),
                    *queue_headers,
                ]
            )
            while args.max_frames is None or frames < args.max_frames:
                readable, frame = capture.read()
                if not readable:
                    break
                source_index = source_frames
                source_frames += 1
                if source_index % args.frame_stride != 0:
                    continue
                frame = resize_processing_frame(frame, (width, height))
                if reference_frame is None:
                    reference_frame = frame.copy()
                started = perf_counter()
                source_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                timestamp = (
                    source_ms / 1000.0
                    if math.isfinite(source_ms) and source_ms >= 0
                    else source_index / fps
                )
                detected = detector.detect(frame)
                tracked = tracker.update(
                    detected.detections,
                    camera_id=camera_counting.camera_id,
                    timestamp=timestamp,
                    frame_index=frames,
                )
                unique_track_ids.update(item.track_id for item in tracked.observations)
                lost_tracks += len(tracked.expired_track_ids)
                speed_result = (
                    speed_estimator.update(
                        camera_counting.camera_id,
                        tracked.observations,
                        timestamp=timestamp,
                    )
                    if speed_estimator is not None
                    else None
                )
                observations = (
                    speed_result.observations
                    if speed_result is not None
                    else tracked.observations
                )
                counted = counter.update(
                    camera_counting.camera_id,
                    observations,
                    timestamp=timestamp,
                )
                intrusion = (
                    restricted.update(
                        restricted_config.camera_id,
                        observations,
                        timestamp=timestamp,
                    )
                    if restricted is not None
                    else None
                )
                queue_result = (
                    configured_queues.update(
                        queue_config.camera_id,
                        observations,
                        timestamp=timestamp,
                    )
                    if configured_queues is not None
                    else None
                )
                vertical_queue_snapshot = (
                    vertical_queues.update(
                        camera_counting.camera_id,
                        observations,
                        timestamp=timestamp,
                    )
                    if vertical_queues is not None
                    else None
                )
                if heatmaps is not None:
                    heatmap_snapshot = heatmaps.update(
                        observations, timestamp=timestamp
                    )
                    assert heatmap_video_writer is not None
                    heatmap_video_writer.write(
                        heatmap_snapshot.image,
                        frame,
                        counted_points=tuple(
                            observation.foot_point
                            for observation in observations
                            if observation.confirmed
                        ),
                    )
                confirmed_humans = sum(
                    observation.confirmed for observation in observations
                )
                snapshot = counted.snapshot
                annotated = annotate_tracks(
                    frame,
                    observations,
                    tracking_ms=tracked.tracking_ms,
                    show_trajectories=not args.no_trajectories,
                )
                counted_frame = annotate_people_counts(
                    annotated,
                    snapshot,
                    confirmed_humans=confirmed_humans,
                    restricted_snapshot=(
                        intrusion.snapshot if intrusion is not None else None
                    ),
                )
                final_frame = counted_frame
                if intrusion is not None:
                    final_frame = annotate_restricted_areas(
                        final_frame,
                        restricted_config,
                        intrusion.snapshot,
                        observations,
                        copy=False,
                    )
                if queue_result is not None:
                    final_frame = annotate_queues(
                        final_frame,
                        queue_config,
                        queue_result.snapshot,
                        observations,
                        copy=False,
                    )
                if vertical_queue_snapshot is not None:
                    final_frame = annotate_vertical_queues(
                        final_frame,
                        vertical_queue_snapshot,
                        observations,
                        copy=False,
                    )
                writer.write(final_frame)
                queue_csv_values: list[object] = []
                if queue_result is not None:
                    queue_csv_values.extend(
                        queue_result.snapshot.queue_for(queue_id).raw_count
                        for queue_id in queue_ids
                    )
                    queue_csv_values.extend(
                        f"{queue_result.snapshot.queue_for(queue_id).smoothed_count:.6f}"
                        for queue_id in queue_ids
                    )
                    queue_csv_values.extend(
                        queue_result.snapshot.queue_for(
                            queue_id
                        ).approximate_current_waiting_seconds
                        for queue_id in queue_ids
                    )
                    queue_csv_values.extend(
                        queue_result.snapshot.queue_for(queue_id).overflow
                        for queue_id in queue_ids
                    )
                    queue_csv_values.extend(
                        queue_result.snapshot.queue_for(
                            queue_id
                        ).average_speed_pixels_per_second
                        for queue_id in queue_ids
                    )
                    queue_csv_values.extend(
                        queue_result.snapshot.queue_for(
                            queue_id
                        ).average_speed_metres_per_second
                        for queue_id in queue_ids
                    )
                    queue_csv_values.extend(
                        queue_result.snapshot.queue_for(
                            queue_id
                        ).average_progress_speed_pixels_per_second
                        for queue_id in queue_ids
                    )
                    queue_csv_values.extend(
                        queue_result.snapshot.queue_for(
                            queue_id
                        ).average_progress_speed_metres_per_second
                        for queue_id in queue_ids
                    )
                elif vertical_queue_snapshot is not None:
                    queue_csv_values.extend(
                        (
                            len(vertical_queue_snapshot.rows),
                            sum(row.count for row in vertical_queue_snapshot.rows),
                            ";".join(
                                f"row_{row.row_id}:{row.count}"
                                for row in vertical_queue_snapshot.rows
                            ),
                            ";".join(
                                f"row_{row.row_id}:{row.average_speed_pixels_per_second:.6f}"
                                for row in vertical_queue_snapshot.rows
                                if row.average_speed_pixels_per_second is not None
                            ),
                            ";".join(
                                f"row_{row.row_id}:{row.average_speed_metres_per_second:.6f}"
                                for row in vertical_queue_snapshot.rows
                                if row.average_speed_metres_per_second is not None
                            ),
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
                            (
                                speed_result.snapshot.average_speed_pixels_per_second,
                                speed_result.snapshot.average_speed_metres_per_second,
                            )
                            if speed_result is not None
                            else ()
                        ),
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
                        *queue_csv_values,
                    ]
                )
                maximum_confirmed = max(maximum_confirmed, confirmed_humans)
                maximum_occupancy = max(
                    maximum_occupancy, snapshot.current_occupancy
                )
                events += len(counted.events)
                if intrusion is not None:
                    restricted_events += len(intrusion.events)
                if queue_result is not None:
                    queue_events += len(queue_result.events)
                frames += 1
                total_ms += (perf_counter() - started) * 1000.0
                restricted_occupancy = (
                    sum(zone.current_tracks for zone in intrusion.snapshot.zones)
                    if intrusion is not None else None
                )
                queue_statuses = (
                    queue_result.snapshot.queues if queue_result is not None else ()
                )
                vertical_rows = (
                    vertical_queue_snapshot.rows if vertical_queue_snapshot is not None else ()
                )
                queue_lengths = [item.raw_count for item in queue_statuses] or [
                    item.count for item in vertical_rows
                ]
                queue_waits = [
                    item.approximate_current_waiting_seconds
                    for item in queue_statuses
                    if item.approximate_current_waiting_seconds is not None
                ]
                queue_speeds = [
                    item.average_speed_pixels_per_second
                    for item in queue_statuses
                    if item.average_speed_pixels_per_second is not None
                ] or [
                    item.average_speed_pixels_per_second
                    for item in vertical_rows
                    if item.average_speed_pixels_per_second is not None
                ]
                live_metrics = {
                    "current_people": confirmed_humans,
                    "total_unique_people": len(unique_track_ids),
                    "active_tracks": len(observations),
                    "lost_tracks": lost_tracks,
                    "entry_count": snapshot.cumulative_entries,
                    "exit_count": snapshot.cumulative_exits,
                    "zone_occupancy": {
                        item.zone_id: item.current for item in snapshot.occupancy
                    },
                    "restricted_occupancy": restricted_occupancy,
                    "restricted_violations": restricted_events,
                    "queue_length": sum(queue_lengths) if queue_lengths else None,
                    "queue_wait_seconds": sum(queue_waits) / len(queue_waits) if queue_waits else None,
                    "queue_speed": sum(queue_speeds) / len(queue_speeds) if queue_speeds else None,
                    "average_person_speed": (
                        speed_result.snapshot.average_speed_pixels_per_second
                        if speed_result is not None else None
                    ),
                    "processing_fps": 1000.0 / max(total_ms / frames, 0.001),
                    "frame_count": frames,
                    "progress": min(100.0, frames * 100.0 / reporter.total_frames) if reporter.total_frames else None,
                    "elapsed_seconds": reporter.elapsed,
                }
                preview_frame = final_frame
                if heatmaps is not None:
                    preview_frame = overlay_heatmap(
                        final_frame,
                        colorize_heatmap(
                            heatmap_snapshot.image.occupancy,
                            color_map=(heatmap_settings.color_map if heatmap_settings else "jet"),
                            output_size=(width, height),
                            smoothing_sigma_cells=(
                                heatmap_settings.smoothing_sigma_cells if heatmap_settings else 1.0
                            ),
                        ),
                        opacity=(heatmap_settings.opacity if heatmap_settings else 0.55),
                    )
                reporter.publish(frames - 1, live_metrics, frame=preview_frame)
    finally:
        capture.release()
        writer.release()
        if heatmap_video_writer is not None:
            heatmap_video_writer.close()

    if frames == 0:
        raise RuntimeError(f"video contained no readable frames: {source}")
    reporter.publish(frames - 1, live_metrics, frame=preview_frame, force=True)
    heatmap_exports: dict[str, object] | None = None
    if heatmaps is not None:
        assert heatmap_directory is not None
        assert heatmap_video_writer is not None
        color_map = heatmap_settings.color_map if heatmap_settings else "jet"
        opacity = heatmap_settings.opacity if heatmap_settings else 0.55
        smoothing_sigma = (
            heatmap_settings.smoothing_sigma_cells if heatmap_settings else 1.0
        )
        snapshot = heatmaps.snapshot()
        image_exports = export_heatmap_snapshot(
            snapshot.image,
            heatmap_directory,
            prefix=f"{heatmaps.camera_id}_image",
            color_map=color_map,
            opacity=opacity,
            smoothing_sigma_cells=smoothing_sigma,
            output_size=(width, height),
            reference_frame=reference_frame,
        )
        heatmap_exports = {
            "image": _export_paths(image_exports),
            "videos": {
                "occupancy": str(heatmap_video_writer.paths.occupancy.resolve()),
                "dwell": str(heatmap_video_writer.paths.dwell.resolve()),
            },
            "ground": None,
            "ground_unavailable_reason": snapshot.ground_unavailable_reason,
        }
        if snapshot.ground is not None:
            ground_exports = export_heatmap_snapshot(
                snapshot.ground,
                heatmap_directory,
                prefix=f"{heatmaps.camera_id}_ground",
                color_map=color_map,
                opacity=opacity,
                smoothing_sigma_cells=smoothing_sigma,
            )
            heatmap_exports["ground"] = _export_paths(ground_exports)
    print(
        json.dumps(
            {
                "frames": frames,
                "fps": fps,
                "frame_stride": args.frame_stride,
                "processing_fps": 1000.0 / (total_ms / frames),
                "maximum_confirmed_humans": maximum_confirmed,
                "maximum_total_zone_occupancy": maximum_occupancy,
                "line_crossed_events": events,
                "restricted_area_enabled": restricted is not None,
                "restricted_area_events": (
                    restricted_events if restricted is not None else None
                ),
                "queue_analytics_enabled": args.enable_queue,
                "queue_mode": args.queue_mode if args.enable_queue else None,
                "queue_events": (
                    queue_events if configured_queues is not None else None
                ),
                "speed_analytics_enabled": speed_estimator is not None,
                "physical_speed_warning": (
                    speed_result.snapshot.calibration_warning
                    if speed_estimator is not None and speed_result is not None
                    else None
                ),
                "restricted_events_jsonl": str(event_path.resolve())
                if event_path is not None
                else None,
                "average_total_frame_ms": total_ms / frames,
                "annotated_video": str(output.resolve()),
                "per_frame_counts_csv": str(counts_csv.resolve()),
                "movement_heatmaps": heatmap_exports,
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


def _queue_config(
    camera_config: CameraConfig | None,
    default_camera_id: str,
    frame_size: tuple[int, int],
) -> CameraQueueConfig:
    if camera_config is not None:
        return CameraQueueConfig.from_camera_config(camera_config, frame_size)
    return CameraQueueConfig(default_camera_id, frame_size)


def _export_paths(paths: HeatmapExportPaths) -> dict[str, str | None]:
    """Convert a heatmap export result to the CLI's JSON-safe path mapping."""

    return {
        name: str(value.resolve()) if value is not None else None
        for name, value in (
            ("occupancy_grid", paths.occupancy_grid),
            ("dwell_grid", paths.dwell_grid),
            ("occupancy_image", paths.occupancy_image),
            ("dwell_image", paths.dwell_image),
            ("occupancy_overlay", paths.occupancy_overlay),
            ("dwell_overlay", paths.dwell_overlay),
        )
    }


if __name__ == "__main__":
    main()
