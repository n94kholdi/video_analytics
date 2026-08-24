"""Run selected trackers against a frozen detection cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from app.core.config import AppSettings
from app.core.models import TrackObservation
from app.tracking.benchmark.cache import CachedDetection
from app.tracking.benchmark.metrics import GroundTruthBox, MotMetrics, evaluate_tracks
from app.tracking.benchmark.resources import ResourceSampler
from app.tracking.factory import create_tracker


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    tracker: str
    frames: int
    tracker_latency_ms: float
    reid_latency_ms: float
    effective_fps: float
    cpu_load: float | None
    cpu_time_seconds: float
    gpu_memory_mb: float | None
    rss_mb: float | None
    metrics: MotMetrics | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.metrics is not None:
            payload["metrics"] = self.metrics.as_dict()
        return payload


class BenchmarkRunner:
    """video → cached detections → selected tracker → evaluation."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings
        self._resources = ResourceSampler()

    def run(
        self,
        frames: Sequence[CachedDetection],
        *,
        tracker_type: str,
        camera_id: str = "benchmark",
        video_frames: Sequence[np.ndarray] | None = None,
        ground_truth: Sequence[GroundTruthBox] = (),
        frame_rate: float = 0.5,
        reid_model: Path | None = None,
    ) -> tuple[BenchmarkReport, tuple[TrackObservation, ...]]:
        tracker = create_tracker(
            tracker_type,
            settings=self.settings,
            frame_rate=frame_rate,
            reid_model=reid_model,
        )
        tracker.reset()
        started = perf_counter()
        tracking_ms = 0.0
        reid_ms = 0.0
        observations: list[TrackObservation] = []
        before = self._resources.snapshot()
        for item in frames:
            frame = None
            if video_frames is not None and item.frame_index < len(video_frames):
                frame = video_frames[item.frame_index]
            result = tracker.update(
                item.detections,
                camera_id=camera_id,
                timestamp=item.timestamp,
                frame_index=item.frame_index,
                frame=frame,
            )
            tracking_ms += result.tracking_ms
            reid_ms += result.reid_ms
            observations.extend(result.observations)
        elapsed = max(perf_counter() - started, 1e-9)
        after = self._resources.snapshot()
        metrics = evaluate_tracks(ground_truth, observations) if ground_truth else None
        report = BenchmarkReport(
            tracker=tracker.name,
            frames=len(frames),
            tracker_latency_ms=tracking_ms / max(len(frames), 1),
            reid_latency_ms=reid_ms / max(len(frames), 1),
            effective_fps=len(frames) / elapsed,
            cpu_load=after.cpu_percent,
            cpu_time_seconds=after.cpu_time_seconds - before.cpu_time_seconds,
            gpu_memory_mb=after.gpu_memory_mb,
            rss_mb=after.rss_mb,
            metrics=metrics,
        )
        return report, tuple(observations)

    def write_report(self, report: BenchmarkReport, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        csv_path = path.with_suffix(".csv")
        payload = report.as_dict()
        metrics = payload.pop("metrics") or {}
        extra = payload.pop("extra") or {}
        row = {**payload, **metrics, **extra}
        csv_path.write_text(
            ",".join(str(key) for key in row) + "\n" + ",".join(_csv_cell(value) for value in row.values()) + "\n",
            encoding="utf-8",
        )


def _csv_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if "," in text:
        return f'"{text}"'
    return text
