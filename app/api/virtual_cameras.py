"""Publish uploaded video files into MediaMTX as long-running virtual cameras.

Once a camera has an active publisher here, its RTSP path behaves exactly like
a real camera to every other part of the stack -- the health probe, the
WHEP/HLS live preview, and ``/api/v1/stream-jobs`` all read the same path a
real Larix/IP-camera publisher would use. No other code needs to know a
"virtual" camera exists; the CV pipeline and dashboard never special-case it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import PROJECT_ROOT


CAMERA_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,100}")
ALLOWED_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
MAX_UPLOAD_BYTES = int(os.environ.get("VIDEO_ANALYTICS_MAX_UPLOAD_BYTES", 1_073_741_824))

# Restart-on-crash backs off up to 30s, and gives up after too many crashes in
# a rolling window rather than looping forever against a permanently broken file.
RESTART_WINDOW_SECONDS = 300.0
RESTART_LIMIT = 10
BACKOFF_SECONDS = (1, 2, 4, 8, 16, 30)


def validate_camera_id(value: str) -> str:
    candidate = value.strip()
    if not CAMERA_ID_PATTERN.fullmatch(candidate):
        raise ValueError(
            "camera_id must use 1-100 letters, digits, dots, underscores, or hyphens"
        )
    return candidate


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class VirtualCameraRecord:
    camera_id: str
    original_filename: str
    video_path: str
    push_url: str
    status: str  # "starting" | "running" | "crashed" | "stopped"
    created_at: str
    updated_at: str
    last_error: str | None = None
    restart_count: int = 0

    def public_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("video_path")
        return data


class VirtualCameraManager:
    """Runs one persistent, looping ffmpeg publisher per camera_id."""

    def __init__(
        self,
        root: Path,
        *,
        mediamtx_rtsp_base: str,
        ffmpeg_path: str | None = None,
        max_width: int = 1280,
        fps: int = 25,
        max_concurrent: int = 8,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.mediamtx_rtsp_base = mediamtx_rtsp_base.rstrip("/")
        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg") or "ffmpeg"
        self.max_width = max_width
        self.fps = fps
        self.max_concurrent = max_concurrent
        self._lock = threading.RLock()
        self._records: dict[str, VirtualCameraRecord] = {}
        self._stop_flags: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}

    def push_url_for(self, camera_id: str) -> str:
        return f"{self.mediamtx_rtsp_base}/camera-{camera_id}"

    def get(self, camera_id: str) -> VirtualCameraRecord | None:
        with self._lock:
            return self._records.get(camera_id)

    def list(self) -> list[VirtualCameraRecord]:
        with self._lock:
            return list(self._records.values())

    async def upload(self, camera_id: str, video: UploadFile) -> VirtualCameraRecord:
        camera_id = validate_camera_id(camera_id)
        suffix = Path(video.filename or "").suffix.lower()
        if suffix not in ALLOWED_VIDEO_SUFFIXES:
            raise ValueError("unsupported video file type")
        with self._lock:
            existing = self._records.get(camera_id)
            active = sum(
                1 for record in self._records.values() if record.status in {"starting", "running"}
            )
            if existing is None and active >= self.max_concurrent:
                raise RuntimeError(
                    f"maximum of {self.max_concurrent} concurrent virtual cameras reached"
                )
        self._stop_process(camera_id)
        camera_dir = self.root / camera_id
        camera_dir.mkdir(parents=True, exist_ok=True)
        for stale in camera_dir.glob("input.*"):
            stale.unlink(missing_ok=True)
        video_path = camera_dir / f"input{suffix}"
        await _save_upload(video, video_path, limit=MAX_UPLOAD_BYTES)
        record = VirtualCameraRecord(
            camera_id=camera_id,
            original_filename=video.filename or video_path.name,
            video_path=str(video_path.resolve()),
            push_url=self.push_url_for(camera_id),
            status="starting",
            created_at=_now(),
            updated_at=_now(),
        )
        with self._lock:
            self._records[camera_id] = record
            self._save(record)
        self._start(camera_id)
        return record

    def remove(self, camera_id: str) -> None:
        self._stop_process(camera_id)
        with self._lock:
            self._records.pop(camera_id, None)
        shutil.rmtree(self.root / camera_id, ignore_errors=True)

    def stop_all(self) -> None:
        with self._lock:
            camera_ids = list(self._records.keys())
        for camera_id in camera_ids:
            self._stop_process(camera_id)

    def load_existing(self) -> None:
        """Relaunch every camera whose video file survived a restart.

        Unlike one-shot recorded-video jobs (which are marked failed if the
        service restarts mid-run), a virtual camera is meant to keep behaving
        like an always-online camera across redeploys, so it simply resumes.
        """

        for metadata in self.root.glob("*/record.json"):
            try:
                values = json.loads(metadata.read_text(encoding="utf-8"))
                record = VirtualCameraRecord(**values)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if not Path(record.video_path).is_file():
                continue
            record.status = "starting"
            record.last_error = None
            record.updated_at = _now()
            with self._lock:
                self._records[record.camera_id] = record
                self._save(record)
            self._start(record.camera_id)

    def _start(self, camera_id: str) -> None:
        with self._lock:
            existing_thread = self._threads.get(camera_id)
            if existing_thread is not None and existing_thread.is_alive():
                return
            stop_flag = threading.Event()
            self._stop_flags[camera_id] = stop_flag
            thread = threading.Thread(
                target=self._supervise,
                args=(camera_id, stop_flag),
                name=f"virtual-camera-{camera_id}",
                daemon=True,
            )
            self._threads[camera_id] = thread
        thread.start()

    def _stop_process(self, camera_id: str) -> None:
        with self._lock:
            stop_flag = self._stop_flags.pop(camera_id, None)
            process = self._processes.pop(camera_id, None)
            thread = self._threads.pop(camera_id, None)
        if stop_flag is not None:
            stop_flag.set()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=6)

    def _supervise(self, camera_id: str, stop_flag: threading.Event) -> None:
        attempts: list[float] = []
        while not stop_flag.is_set():
            record = self.get(camera_id)
            if record is None:
                return
            command = self._build_command(Path(record.video_path), record.push_url)
            log_path = self.root / camera_id / "ffmpeg.log"
            try:
                with log_path.open("ab") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    with self._lock:
                        self._processes[camera_id] = process
                    self._update(camera_id, status="running", last_error=None)
                    returncode = process.wait()
                with self._lock:
                    self._processes.pop(camera_id, None)
            except OSError as exc:
                self._update(camera_id, status="crashed", last_error=str(exc))
                return
            if stop_flag.is_set():
                return
            now = time.monotonic()
            attempts.append(now)
            attempts[:] = [moment for moment in attempts if now - moment <= RESTART_WINDOW_SECONDS]
            if len(attempts) > RESTART_LIMIT:
                self._update(
                    camera_id,
                    status="crashed",
                    last_error=f"ffmpeg exited repeatedly (exit code {returncode}); giving up",
                )
                return
            self._update(
                camera_id,
                status="starting",
                last_error=f"ffmpeg exited with code {returncode}, restarting",
                bump_restart=True,
            )
            delay = BACKOFF_SECONDS[min(len(attempts) - 1, len(BACKOFF_SECONDS) - 1)]
            stop_flag.wait(delay)

    def _build_command(self, video_path: Path, push_url: str) -> list[str]:
        return [
            self.ffmpeg_path,
            "-nostdin",
            "-loglevel", "warning",
            "-re",
            "-stream_loop", "-1",
            "-i", str(video_path),
            "-c:v", "libx264",
            "-profile:v", "main",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-vf", f"scale='min({self.max_width},iw)':-2",
            "-r", str(self.fps),
            "-g", str(self.fps * 2),
            "-an",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            push_url,
        ]

    def _update(self, camera_id: str, *, bump_restart: bool = False, **changes: object) -> None:
        with self._lock:
            record = self._records.get(camera_id)
            if record is None:
                return
            for name, value in changes.items():
                setattr(record, name, value)
            if bump_restart:
                record.restart_count += 1
            record.updated_at = _now()
            self._save(record)

    def _save(self, record: VirtualCameraRecord) -> None:
        path = self.root / record.camera_id / "record.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        temporary.replace(path)


async def _save_upload(upload: UploadFile, destination: Path, *, limit: int) -> None:
    written = 0
    with destination.open("wb") as stream:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                destination.unlink(missing_ok=True)
                raise ValueError("uploaded file is too large")
            stream.write(chunk)
    if written == 0:
        destination.unlink(missing_ok=True)
        raise ValueError("uploaded file is empty")


manager = VirtualCameraManager(
    Path(os.environ.get("VIRTUAL_CAMERAS_DIR", PROJECT_ROOT / "output" / "virtual-cameras")),
    mediamtx_rtsp_base=os.environ.get("MEDIAMTX_RTSP_URL", "rtsp://mediamtx:8554"),
    max_width=int(os.environ.get("VIRTUAL_CAMERA_MAX_WIDTH", "1280")),
    fps=int(os.environ.get("VIRTUAL_CAMERA_FPS", "25")),
    max_concurrent=int(os.environ.get("VIRTUAL_CAMERA_MAX_CONCURRENT", "8")),
)

router = APIRouter(prefix="/api/v1/virtual-cameras")


@router.post("/{camera_id}/video", status_code=202)
async def upload_virtual_camera_video(
    camera_id: str, video: Annotated[UploadFile, File(...)]
) -> dict[str, object]:
    try:
        record = await manager.upload(camera_id, video)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        await video.close()
    return {"data": record.public_dict()}


@router.get("/{camera_id}")
def get_virtual_camera(camera_id: str) -> dict[str, object]:
    record = manager.get(camera_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no virtual camera for {camera_id}")
    return {"data": record.public_dict()}


@router.delete("/{camera_id}", status_code=204)
def delete_virtual_camera(camera_id: str) -> None:
    manager.remove(camera_id)
