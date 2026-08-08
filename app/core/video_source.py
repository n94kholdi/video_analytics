"""Validation and naming helpers shared by file and live video sources."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


NETWORK_VIDEO_SCHEMES = frozenset({"rtsp", "rtsps"})


def resolve_video_source(value: str | Path) -> str | Path:
    """Return a validated RTSP URL or an absolute existing local path."""

    raw = str(value)
    parsed = urlparse(raw)
    if parsed.scheme.lower() in NETWORK_VIDEO_SCHEMES:
        if not parsed.hostname or not parsed.path.strip("/"):
            raise ValueError("RTSP source must include a host and stream path")
        return raw

    source = Path(value).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input source does not exist: {source}")
    return source


def video_source_stem(source: str | Path) -> str:
    """Return a filesystem-safe default artifact stem for any source."""

    if isinstance(source, Path):
        return source.stem
    path_name = Path(urlparse(source).path).name
    stem = Path(path_name).stem
    return stem or "live-stream"


def is_network_video_source(source: str | Path) -> bool:
    return isinstance(source, str) and urlparse(source).scheme.lower() in NETWORK_VIDEO_SCHEMES

