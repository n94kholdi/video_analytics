"""Persistent local job runner for recorded-video analytics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from concurrent.futures import Future, ThreadPoolExecutor
import json
import mimetypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any

from app.api.presets import ApplicationPreset, get_application
from app.core.config import PROJECT_ROOT


TERMINAL_STATUSES = frozenset({"completed", "failed"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class JobRecord:
    id: str
    application_id: str
    original_filename: str
    camera_id: str
    status: str
    created_at: str
    updated_at: str
    job_directory: str
    input_video: str
    camera_config: str | None = None
    max_frames: int | None = None
    error: str | None = None
    summary: dict[str, Any] | None = None
    artifacts: dict[str, str] = field(default_factory=dict)


class JobManager:
    """Run one process per job while persisting metadata beside its artifacts."""

    def __init__(
        self,
        root: Path,
        *,
        python_executable: str | None = None,
        max_workers: int = 1,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.python_executable = python_executable or sys.executable
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="video-analytics-job"
        )
        self._load_existing()

    def register(
        self,
        *,
        job_id: str,
        application_id: str,
        original_filename: str,
        camera_id: str,
        input_video: Path,
        camera_config: Path | None,
        max_frames: int | None,
    ) -> JobRecord:
        get_application(application_id)
        now = _now()
        record = JobRecord(
            id=job_id,
            application_id=application_id,
            original_filename=original_filename,
            camera_id=camera_id,
            status="queued",
            created_at=now,
            updated_at=now,
            job_directory=str(input_video.parent.resolve()),
            input_video=str(input_video.resolve()),
            camera_config=str(camera_config.resolve()) if camera_config else None,
            max_frames=max_frames,
        )
        with self._lock:
            self._jobs[job_id] = record
            self._save(record)
        return record

    def list(self) -> list[JobRecord]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise KeyError(f"job not found: {job_id}") from exc

    def run(self, job_id: str) -> None:
        record = self.get(job_id)
        preset = get_application(record.application_id)
        self._update(record, status="running", error=None)
        command, expected = self._build_command(record, preset)
        job_dir = Path(record.job_directory)
        log_path = job_dir / "process.log"
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                check=False,
            )
            log_path.write_text(
                f"$ {' '.join(command)}\n\nSTDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}",
                encoding="utf-8",
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(detail[-6000:] or f"process exited with {completed.returncode}")

            artifacts = self._collect_artifacts(record, expected, log_path)
            summary = _last_json_object(completed.stdout)
            self._update(
                record,
                status="completed",
                summary=summary,
                artifacts=artifacts,
                error=None,
            )
        except Exception as exc:
            artifacts = {"process_log": str(log_path)} if log_path.is_file() else {}
            self._update(record, status="failed", error=str(exc), artifacts=artifacts)

    def enqueue(self, job_id: str) -> Future[None]:
        """Submit a job without blocking the HTTP request thread."""

        self.get(job_id)
        return self._executor.submit(self.run, job_id)

    def public_dict(self, record: JobRecord) -> dict[str, Any]:
        data = asdict(record)
        data.pop("job_directory")
        data.pop("input_video")
        data.pop("camera_config")
        data["application"] = get_application(record.application_id).public_dict()
        data["artifacts"] = {
            key: {
                "filename": Path(path).name,
                "media_type": mimetypes.guess_type(path)[0] or "application/octet-stream",
                "url": f"/api/v1/jobs/{record.id}/artifacts/{key}",
            }
            for key, path in record.artifacts.items()
            if Path(path).is_file()
        }
        return data

    def artifact_path(self, job_id: str, key: str) -> Path:
        record = self.get(job_id)
        try:
            path = Path(record.artifacts[key]).resolve()
        except KeyError as exc:
            raise KeyError(f"artifact not found: {key}") from exc
        job_dir = Path(record.job_directory).resolve()
        if not path.is_relative_to(job_dir) or not path.is_file():
            raise KeyError(f"artifact not found: {key}")
        return path

    def _build_command(
        self, record: JobRecord, preset: ApplicationPreset
    ) -> tuple[list[str], dict[str, Path]]:
        job_dir = Path(record.job_directory)
        raw_video = job_dir / "annotated_raw.mp4"
        counts_csv = job_dir / "counts.csv"
        events_jsonl = job_dir / "events.jsonl"
        heatmap_dir = job_dir / "heatmaps"
        command = [
            self.python_executable,
            "-m",
            preset.module,
            record.input_video,
            "--output",
            str(raw_video),
        ]
        expected = {"annotated_video_raw": raw_video}
        if preset.module == "app.analytics.cli":
            command.extend(("--counts-csv", str(counts_csv), "--events-jsonl", str(events_jsonl)))
            expected["counts_csv"] = counts_csv
            expected["events_jsonl"] = events_jsonl
            if "--enable-heatmap" in preset.arguments:
                command.extend(("--heatmap-dir", str(heatmap_dir)))
        # A dashboard camera YAML describes analytics geometry.  The standalone
        # tracking and detection CLIs accept application settings under
        # ``--config`` instead and must not receive ``--camera-config``.
        if record.camera_config and preset.module == "app.analytics.cli":
            command.extend(("--camera-config", record.camera_config))
        if preset.module != "app.detection.cli":
            command.extend(("--camera-id", record.camera_id))
        if record.max_frames is not None:
            command.extend(("--max-frames", str(record.max_frames)))
        command.extend(preset.arguments)
        return command, expected

    def _collect_artifacts(
        self, record: JobRecord, expected: dict[str, Path], log_path: Path
    ) -> dict[str, str]:
        artifacts = {
            key: str(path.resolve()) for key, path in expected.items() if path.is_file()
        }
        raw_video = expected["annotated_video_raw"]
        browser_video = Path(record.job_directory) / "annotated.mp4"
        if raw_video.is_file() and _make_browser_video(raw_video, browser_video):
            artifacts["annotated_video"] = str(browser_video.resolve())
        elif raw_video.is_file():
            artifacts["annotated_video"] = str(raw_video.resolve())
        heatmap_dir = Path(record.job_directory) / "heatmaps"
        if heatmap_dir.is_dir():
            for index, path in enumerate(sorted(p for p in heatmap_dir.rglob("*") if p.is_file()), 1):
                artifacts[f"heatmap_{index}"] = str(path.resolve())
                if path.name.endswith("_occupancy.mp4"):
                    browser_heatmap = Path(record.job_directory) / "heatmap_occupancy.mp4"
                    if _make_browser_video(path, browser_heatmap):
                        artifacts["heatmap_video"] = str(browser_heatmap.resolve())
                    else:
                        artifacts["heatmap_video"] = str(path.resolve())
        artifacts["process_log"] = str(log_path.resolve())
        return artifacts

    def _update(self, record: JobRecord, **changes: Any) -> None:
        with self._lock:
            for name, value in changes.items():
                setattr(record, name, value)
            record.updated_at = _now()
            self._save(record)

    def _save(self, record: JobRecord) -> None:
        path = Path(record.job_directory) / "job.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        temporary.replace(path)

    def _load_existing(self) -> None:
        for metadata in self.root.glob("*/job.json"):
            try:
                values = json.loads(metadata.read_text(encoding="utf-8"))
                record = JobRecord(**values)
                if record.status in {"queued", "running"}:
                    record.status = "failed"
                    record.error = "The analytics service stopped before this job finished."
                    record.updated_at = _now()
                    self._save(record)
                self._jobs[record.id] = record
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue


def _make_browser_video(source: Path, destination: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and destination.is_file()


def _last_json_object(output: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(output):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not output[index + end :].strip():
            return value
    return None
