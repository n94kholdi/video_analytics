"""Validated ingestion and query contracts for compact analytical facts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(item.capitalize() for item in tail)

class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=_camel, populate_by_name=True, extra="forbid")


class SpatialPoint(ApiModel):
    x: float = Field(ge=0, le=100)
    y: float = Field(ge=0, le=100)
    value: float
    intensity: float = Field(ge=-1, le=1)


class SpatialMinute(ApiModel):
    zone_id: str = Field(default="location", min_length=1, max_length=128)
    zone_name: str = Field(default="کل مکان", min_length=1, max_length=200)
    layer: Literal["occupancy", "dwell", "traffic", "congestion"]
    bucket_seconds: int = Field(default=900, ge=300, le=3600)
    coverage_percent: float | None = Field(default=None, ge=0, le=100)
    points: list[SpatialPoint] = Field(max_length=256)


class QueueMinute(ApiModel):
    queue_id: str = Field(min_length=1, max_length=128)
    queue_name: str = Field(min_length=1, max_length=200)
    sample_count: int = Field(ge=0, le=120)
    length_sum: float = Field(ge=0)
    length_max: int | None = Field(default=None, ge=0)
    length_last: int | None = Field(default=None, ge=0)
    wait_sum_seconds: float = Field(default=0, ge=0)
    wait_sample_count: int = Field(default=0, ge=0, le=1000)
    completed_wait_seconds: list[float] = Field(default_factory=list, max_length=500)
    throughput: int = Field(default=0, ge=0)
    sla_met: int = Field(default=0, ge=0)
    warning_samples: int = Field(default=0, ge=0)
    critical_samples: int = Field(default=0, ge=0)
    movement_speed_sum_mpm: float = Field(default=0, ge=0)
    movement_speed_samples: int = Field(default=0, ge=0)
    physically_calibrated: bool = False


class AnalyticsEventInput(ApiModel):
    event_id: str = Field(min_length=1, max_length=128)
    event_type: str = Field(min_length=1, max_length=128)
    severity: Literal["critical", "warning", "info"] = "info"
    title: str = Field(min_length=1, max_length=300)
    occurred_at: datetime
    metric_key: str | None = Field(default=None, max_length=100)
    value: float | None = None
    payload: dict[str, object] | None = None


class CameraMinute(ApiModel):
    camera_id: str = Field(min_length=1, max_length=128)
    camera_name: str | None = Field(default=None, max_length=200)
    bucket_start: datetime
    sample_count: int = Field(ge=0, le=120)
    expected_samples: int = Field(ge=1, le=120)
    confidence_sum: float = Field(default=0, ge=0, le=120)
    occupancy_sum: float = Field(default=0, ge=0)
    occupancy_max: int | None = Field(default=None, ge=0)
    occupancy_last: int | None = Field(default=None, ge=0)
    entries: int = Field(default=0, ge=0)
    exits: int = Field(default=0, ge=0)
    unique_visitors: int | None = Field(default=None, ge=0)
    physically_calibrated: bool = False
    queues: list[QueueMinute] = Field(default_factory=list, max_length=100)
    spatial: list[SpatialMinute] = Field(default_factory=list, max_length=32)
    events: list[AnalyticsEventInput] = Field(default_factory=list, max_length=500)

    @field_validator("bucket_start")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("bucketStart must include a timezone")
        return value


class IngestBatch(ApiModel):
    observations: list[CameraMinute] = Field(min_length=1, max_length=500)


LocationType = Literal["organization", "field", "market", "booth"]
PlaceType = Literal["all", "field", "market", "booth"]
Comparison = Literal["previous_period", "previous_week", "previous_month", "none"]
Bucket = Literal["hour", "day"]


class AnalyticsQuery(BaseModel):
    location_type: LocationType = "organization"
    location_id: str | None = None
    place_type: PlaceType = "all"
    from_date: datetime
    to_date: datetime
    comparison: Comparison = "previous_period"
    time_from: str | None = None
    time_to: str | None = None
    bucket: Bucket = "hour"
