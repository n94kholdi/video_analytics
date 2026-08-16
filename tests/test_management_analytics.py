from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
import numpy as np

from app.analytics.cli import _management_spatial_layers
from app.management.api import _authorized, _query
from app.management.models import CameraMinute
from app.management.publisher import MinutePublisher
from app.management.service import _difference_points, _shift_month


def test_camera_minute_accepts_camel_case_and_requires_timezone() -> None:
    value = CameraMinute.model_validate({
        "cameraId": "camera-1", "bucketStart": "2026-08-11T12:00:00Z",
        "sampleCount": 30, "expectedSamples": 30,
    })
    assert value.camera_id == "camera-1"
    assert value.bucket_start.tzinfo is not None

    with pytest.raises(ValidationError, match="timezone"):
        CameraMinute.model_validate({
            "cameraId": "camera-1", "bucketStart": "2026-08-11T12:00:00",
            "sampleCount": 30, "expectedSamples": 30,
        })


def test_ingest_key_comparison_rejects_wrong_service_key(monkeypatch) -> None:
    monkeypatch.setenv("ANALYTICS_INGEST_KEY", "correct-key")
    assert not _authorized("wrong-key", "ANALYTICS_INGEST_KEY")
    assert _authorized("correct-key", "ANALYTICS_INGEST_KEY")


def test_management_query_parses_business_dates_in_configured_timezone(monkeypatch) -> None:
    monkeypatch.setenv("ANALYTICS_TIMEZONE", "Asia/Tehran")
    query = _query("organization", None, "all", "2026-08-11", "2026-08-11", "none", "hour", None, None)
    assert query.from_date.tzinfo is timezone.utc
    assert (query.to_date-query.from_date).total_seconds() == 86_400


def test_difference_grid_preserves_signed_change() -> None:
    difference = _difference_points(
        [{"x": 10.0, "y": 20.0, "value": 7.0, "intensity": 1.0}],
        [{"x": 10.0, "y": 20.0, "value": 10.0, "intensity": 1.0}],
    )
    assert difference == [{"x": 10, "y": 20, "value": -3.0, "intensity": -1.0}]


def test_previous_month_clamps_end_of_month() -> None:
    assert _shift_month(datetime(2024, 3, 31, tzinfo=timezone.utc)) == datetime(2024, 2, 29, tzinfo=timezone.utc)


def test_management_spatial_grid_is_bounded_and_has_all_layers() -> None:
    layers = _management_spatial_layers(np.ones((36, 64)), np.full((36, 64), 120.0))
    assert set(layers) == {"occupancy", "dwell", "traffic", "congestion"}
    assert all(len(points) == 12 for points in layers.values())
    assert layers["dwell"][0]["value"] == 2.0


def test_publisher_is_zero_cost_when_ingestion_is_disabled(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("VIDEO_ANALYTICS_INGEST_URL", raising=False)
    monkeypatch.setenv("ANALYTICS_OUTBOX_DIR", str(tmp_path))
    publisher = MinutePublisher("camera-1")
    publisher.observe({"current_people": 5})
    publisher.close()
    assert list(tmp_path.iterdir()) == []
