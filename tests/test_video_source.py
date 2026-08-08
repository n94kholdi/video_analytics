from pathlib import Path

import pytest

from app.core.video_source import (
    is_network_video_source,
    resolve_video_source,
    video_source_stem,
)


def test_rtsp_source_is_preserved_for_opencv() -> None:
    source = "rtsp://mediamtx:8554/mobile-1"

    assert resolve_video_source(source) == source
    assert is_network_video_source(source)
    assert video_source_stem(source) == "mobile-1"


def test_rtsp_source_requires_a_path() -> None:
    with pytest.raises(ValueError, match="stream path"):
        resolve_video_source("rtsp://mediamtx:8554")


def test_local_source_is_resolved(tmp_path: Path) -> None:
    source = tmp_path / "sample.mp4"
    source.touch()

    assert resolve_video_source(source) == source.resolve()
    assert video_source_stem(source) == "sample"
