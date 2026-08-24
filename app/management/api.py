"""FastAPI routes for compact ingestion and management read models."""

from __future__ import annotations

import asyncio
from datetime import datetime, time, timedelta, timezone
import hmac
import os
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status

from app.management.models import AnalyticsQuery, IngestBatch
from app.management.repository import AnalyticsUnavailable, analytics_repository
from app.management.service import management_analytics_service


router = APIRouter(prefix="/api/v1")


def _authorized(provided: str | None, variable: str) -> bool:
    expected = os.environ.get(variable)
    return expected is None or (provided is not None and hmac.compare_digest(provided, expected))


def _query(
    location_type: str, location_id: str | None, place_type: str,
    from_date: str, to_date: str, comparison: str, bucket: str,
    time_from: str | None, time_to: str | None,
) -> AnalyticsQuery:
    try:
        local_zone = ZoneInfo(os.environ.get("ANALYTICS_TIMEZONE", "Asia/Tehran"))
        first = datetime.combine(datetime.strptime(from_date, "%Y-%m-%d").date(), time.min, tzinfo=local_zone).astimezone(timezone.utc)
        last = datetime.combine(datetime.strptime(to_date, "%Y-%m-%d").date() + timedelta(days=1), time.min, tzinfo=local_zone).astimezone(timezone.utc)
        return AnalyticsQuery(location_type=location_type, location_id=location_id, place_type=place_type,
                              from_date=first, to_date=last, comparison=comparison, bucket=bucket,
                              time_from=time_from, time_to=time_to)
    except (ValueError, TypeError, ZoneInfoNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


QueryLocation = Annotated[str, Query(alias="locationType", pattern="^(organization|field|market|booth)$")]
QueryPlace = Annotated[str, Query(alias="placeType", pattern="^(all|field|market|booth)$")]
QueryComparison = Annotated[str, Query(pattern="^(previous_period|previous_week|previous_month|none)$")]
QueryBucket = Annotated[str, Query(pattern="^(hour|day)$")]


def _read_query(
    from_date: Annotated[str, Query(alias="from")],
    to_date: Annotated[str, Query(alias="to")],
    location_type: QueryLocation = "organization",
    location_id: Annotated[str | None, Query(alias="locationId")] = None,
    place_type: QueryPlace = "all",
    comparison: QueryComparison = "previous_period",
    bucket: QueryBucket = "hour",
    time_from: Annotated[str | None, Query(alias="timeFrom")] = None,
    time_to: Annotated[str | None, Query(alias="timeTo")] = None,
) -> AnalyticsQuery:
    if location_type != "organization" and not location_id:
        raise HTTPException(status_code=422, detail="locationId is required")
    return _query(location_type, location_id, place_type, from_date, to_date, comparison, bucket, time_from, time_to)


def _check_read_key(key: Annotated[str | None, Header(alias="X-Analytics-Key")] = None) -> None:
    if not _authorized(key, "ANALYTICS_READ_KEY"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid analytics key")


@router.post("/ingest/minutes", status_code=202)
def ingest_minutes(
    batch: IngestBatch,
    key: Annotated[str | None, Header(alias="X-Analytics-Key")] = None,
) -> dict[str, object]:
    if not _authorized(key, "ANALYTICS_INGEST_KEY"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid ingest key")
    try:
        accepted = analytics_repository.ingest(
            batch.observations,
            float(os.environ.get("ANALYTICS_QUEUE_SLA_SECONDS", "300")),
        )
        return {"data": {"accepted": accepted}}
    except AnalyticsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/management/people-flow", dependencies=[])
def people_flow(query: Annotated[AnalyticsQuery, Depends(_read_query)], _: Annotated[None, Depends(_check_read_key)]) -> dict[str, object]:
    try:
        return {"data": management_analytics_service.people_flow(query)}
    except AnalyticsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/management/queues", dependencies=[])
def queues(query: Annotated[AnalyticsQuery, Depends(_read_query)], _: Annotated[None, Depends(_check_read_key)]) -> dict[str, object]:
    try:
        return {"data": management_analytics_service.queues(query)}
    except AnalyticsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/management/spatial", dependencies=[])
def spatial(query: Annotated[AnalyticsQuery, Depends(_read_query)], _: Annotated[None, Depends(_check_read_key)]) -> dict[str, object]:
    try:
        return {"data": management_analytics_service.spatial(query)}
    except AnalyticsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/management/overview", dependencies=[])
def overview(query: Annotated[AnalyticsQuery, Depends(_read_query)], _: Annotated[None, Depends(_check_read_key)]) -> dict[str, object]:
    try:
        return {"data": management_analytics_service.overview(query)}
    except AnalyticsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def rollup_worker() -> None:
    interval = max(10, int(os.environ.get("ANALYTICS_ROLLUP_INTERVAL_SECONDS", "15")))
    while True:
        try:
            await asyncio.to_thread(analytics_repository.refresh_rollups)
        except AnalyticsUnavailable:
            pass
        await asyncio.sleep(interval)
