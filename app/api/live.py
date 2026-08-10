"""Low-overhead, file-backed live reporting for dashboard processing jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from time import monotonic
from typing import Any, Mapping

import cv2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobCancelled(RuntimeError):
    """Raised at a frame boundary when dashboard cancellation is requested."""


class LiveReporter:
    """Publish sampled metrics and JPEG previews without retaining source frames."""

    def __init__(
        self,
        directory: Path | None,
        job_id: str | None,
        *,
        total_frames: int | None,
        metric_interval: float = 0.5,
        preview_interval: float = 0.2,
        preview_width: int = 960,
        jpeg_quality: int = 70,
    ) -> None:
        self.enabled = directory is not None and job_id is not None
        self.directory = directory.expanduser().resolve() if directory else None
        self.job_id = job_id
        self.total_frames = total_frames if total_frames and total_frames > 0 else None
        self.metric_interval = max(0.1, metric_interval)
        self.preview_interval = max(0.2, preview_interval)
        self.preview_width = max(320, preview_width)
        self.jpeg_quality = min(90, max(35, jpeg_quality))
        self.started = monotonic()
        self._last_metric = -math.inf
        self._last_preview = -math.inf
        self._preview_reference: str | None = None
        self._sequence = 0
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def elapsed(self) -> float:
        return monotonic() - self.started

    def check_cancelled(self) -> None:
        if self.directory and (self.directory / "cancel.requested").exists():
            raise JobCancelled("Job cancellation was requested by the dashboard.")

    def publish(
        self,
        frame_index: int,
        metrics: Mapping[str, Any],
        *,
        frame: Any | None = None,
        status: str = "running",
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        self.check_cancelled()
        now = monotonic()
        preview_reference = None
        if frame is not None and (force or now - self._last_preview >= self.preview_interval):
            preview_reference = self._write_preview(frame)
            self._preview_reference = preview_reference
            self._last_preview = now
            self.emit(
                "preview_updated",
                frame_index=frame_index,
                status=status,
                metrics=metrics,
                preview_reference=preview_reference,
            )
        if force or now - self._last_metric >= self.metric_interval:
            self._last_metric = now
            self.emit(
                "metrics_updated",
                frame_index=frame_index,
                status=status,
                metrics=metrics,
                preview_reference=preview_reference,
            )
            self.emit(
                "progress_updated",
                frame_index=frame_index,
                status=status,
                metrics=metrics,
                preview_reference=preview_reference,
            )
            self._append_json("metrics.jsonl", self._payload(
                "metrics_updated", frame_index, status, metrics, preview_reference
            ))

    def emit(
        self,
        event_type: str,
        *,
        frame_index: int | None = None,
        status: str,
        metrics: Mapping[str, Any] | None = None,
        preview_reference: str | None = None,
        message: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        payload = self._payload(
            event_type, frame_index, status, metrics or {}, preview_reference, message
        )
        self._append_json("events.jsonl", payload)
        self._write_json("live_state.json", payload)
        return payload

    def _payload(
        self,
        event_type: str,
        frame_index: int | None,
        status: str,
        metrics: Mapping[str, Any],
        preview_reference: str | None,
        message: str | None = None,
    ) -> dict[str, Any]:
        self._sequence += 1
        processed = (frame_index + 1) if frame_index is not None else 0
        progress = (
            min(100.0, processed * 100.0 / self.total_frames)
            if self.total_frames
            else None
        )
        return {
            "id": self._sequence,
            "type": event_type,
            "job_id": self.job_id,
            "timestamp": utc_now(),
            "status": status,
            "frame_index": frame_index,
            "progress": progress,
            "elapsed_seconds": self.elapsed,
            "metrics": _json_safe(dict(metrics)),
            "preview_reference": preview_reference or self._preview_reference,
            "message": message,
        }

    def _write_preview(self, frame: Any) -> str:
        assert self.directory is not None and self.job_id is not None
        height, width = frame.shape[:2]
        preview = frame
        if width > self.preview_width:
            scale = self.preview_width / width
            preview = cv2.resize(
                frame, (self.preview_width, max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(
            ".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("OpenCV could not encode the live preview frame")
        temporary = self.directory / "preview.tmp"
        final = self.directory / "preview.jpg"
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(final)
        return f"/api/v1/jobs/{self.job_id}/preview"

    def _append_json(self, filename: str, payload: Mapping[str, Any]) -> None:
        assert self.directory is not None
        with (self.directory / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
            stream.flush()

    def _write_json(self, filename: str, payload: Mapping[str, Any]) -> None:
        assert self.directory is not None
        destination = self.directory / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(destination)


def video_frame_count(capture: Any, max_frames: int | None) -> int | None:
    raw = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if raw <= 0:
        return max_frames
    return min(raw, max_frames) if max_frames is not None else raw


def processed_frame_count(
    capture: Any, max_frames: int | None, frame_stride: int
) -> int | None:
    """Estimate frames processed after source-frame sampling."""

    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    raw = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled = (raw + frame_stride - 1) // frame_stride if raw > 0 else None
    if max_frames is None:
        return sampled
    return min(sampled, max_frames) if sampled is not None else max_frames


def processing_frame_size(
    width: int, height: int, maximum_width: int | None
) -> tuple[int, int]:
    """Return an aspect-preserving even frame size capped for live processing."""

    if width <= 0 or height <= 0:
        raise ValueError("source frame dimensions must be positive")
    if maximum_width is None or width <= maximum_width:
        return width, height
    if maximum_width <= 0:
        raise ValueError("processing width must be positive")
    scaled_height = max(2, round(height * maximum_width / width))
    # Video encoders are more reliable with even dimensions.
    return maximum_width - (maximum_width % 2), scaled_height - (scaled_height % 2)


def resize_processing_frame(frame: Any, frame_size: tuple[int, int]) -> Any:
    height, width = frame.shape[:2]
    if (width, height) == frame_size:
        return frame
    return cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value
