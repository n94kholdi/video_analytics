"""FastAPI service used by the Tarebar dashboard."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.jobs import JobManager
from app.api.presets import APPLICATIONS, get_application
from app.core.config import PROJECT_ROOT
from app.geometry.config import load_camera_config


JOBS_ROOT = Path(
    os.environ.get("VIDEO_ANALYTICS_JOBS_DIR", PROJECT_ROOT / "output" / "dashboard")
)
MAX_UPLOAD_BYTES = int(os.environ.get("VIDEO_ANALYTICS_MAX_UPLOAD_BYTES", 1_073_741_824))
ALLOWED_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
manager = JobManager(
    JOBS_ROOT,
    max_workers=int(os.environ.get("VIDEO_ANALYTICS_JOB_WORKERS", "1")),
)

app = FastAPI(title="Video Analytics MVP API", version="0.1.0")
origins = [
    item.strip()
    for item in os.environ.get(
        "VIDEO_ANALYTICS_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/applications")
def applications() -> dict[str, list[dict[str, object]]]:
    return {"data": [item.public_dict() for item in APPLICATIONS]}


@app.get("/api/v1/jobs")
def list_jobs() -> dict[str, list[dict[str, object]]]:
    return {"data": [manager.public_dict(item) for item in manager.list()]}


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    try:
        return {"data": manager.public_dict(manager.get(job_id))}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/jobs", status_code=202)
async def create_job(
    video: Annotated[UploadFile, File(...)],
    application_id: Annotated[str, Form(...)],
    camera_id: Annotated[str, Form()] = "uploaded-video",
    max_frames: Annotated[int | None, Form()] = None,
    camera_config: Annotated[UploadFile | None, File()] = None,
) -> dict[str, object]:
    try:
        preset = get_application(application_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if max_frames is not None and max_frames <= 0:
        raise HTTPException(status_code=422, detail="max_frames must be positive")
    camera_id = camera_id.strip()
    if not camera_id or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", camera_id):
        raise HTTPException(
            status_code=422,
            detail="camera_id must use 1-100 letters, digits, dots, underscores, or hyphens",
        )
    video_suffix = Path(video.filename or "").suffix.lower()
    if video_suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=422, detail="unsupported recorded-video file type")
    if preset.requires_camera_config and camera_config is None:
        raise HTTPException(
            status_code=422,
            detail=f"{preset.name} requires a camera YAML configuration",
        )

    job_id = uuid4().hex
    job_directory = manager.root / job_id
    job_directory.mkdir(parents=True)
    video_path = job_directory / f"input{video_suffix}"
    config_path: Path | None = None
    try:
        await _save_upload(video, video_path, limit=MAX_UPLOAD_BYTES)
        if camera_config is not None:
            config_suffix = Path(camera_config.filename or "").suffix.lower()
            if config_suffix not in {".yaml", ".yml"}:
                raise HTTPException(status_code=422, detail="camera configuration must be YAML")
            config_path = job_directory / "camera.yaml"
            await _save_upload(camera_config, config_path, limit=2_000_000)
            try:
                load_camera_config(config_path)
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"invalid camera YAML: {exc}") from exc
        record = manager.register(
            job_id=job_id,
            application_id=application_id,
            original_filename=video.filename or video_path.name,
            camera_id=camera_id,
            input_video=video_path,
            camera_config=config_path,
            max_frames=max_frames,
        )
    except Exception:
        shutil.rmtree(job_directory, ignore_errors=True)
        raise
    finally:
        await video.close()
        if camera_config is not None:
            await camera_config.close()

    manager.enqueue(job_id)
    return {"data": manager.public_dict(record)}


@app.get("/api/v1/jobs/{job_id}/artifacts/{artifact_key}")
def download_artifact(job_id: str, artifact_key: str) -> FileResponse:
    try:
        path = manager.artifact_path(job_id, artifact_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    media_type = None
    if path.suffix.lower() == ".mp4":
        media_type = "video/mp4"
    elif path.suffix.lower() == ".png":
        media_type = "image/png"
    return FileResponse(
        path,
        media_type=media_type,
        filename=path.name,
        content_disposition_type="inline" if media_type else "attachment",
    )


async def _save_upload(upload: UploadFile, destination: Path, *, limit: int) -> None:
    written = 0
    with destination.open("wb") as stream:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                raise HTTPException(status_code=413, detail="uploaded file is too large")
            stream.write(chunk)
    if written == 0:
        raise HTTPException(status_code=422, detail="uploaded file is empty")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=False)
