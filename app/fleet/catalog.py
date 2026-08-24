"""Load connected cameras and their location/region mappings from PostgreSQL."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from app.fleet.settings import FleetSettings


Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class MappedZone:
    zone_id: str
    name: str
    points: tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class FleetCamera:
    camera_id: str
    name: str
    stream_url: str
    field_id: str | None
    market_id: str | None
    booth_id: str | None
    queues: tuple[MappedZone, ...]
    restricted_zones: tuple[MappedZone, ...]
    signature: str


def rewrite_stream_url(stream_url: str, mediamtx_rtsp_url: str | None) -> str:
    """Map localhost MediaMTX URLs to the Docker-internal RTSP base."""

    parsed = urlparse(stream_url)
    if parsed.scheme.lower() not in {"rtsp", "rtsps"} or not parsed.hostname:
        return stream_url
    is_local_mediamtx = parsed.hostname in {"localhost", "127.0.0.1", "mediamtx"} and (
        parsed.port in {8554, None}
    )
    base = (mediamtx_rtsp_url or "").rstrip("/")
    path = parsed.path.strip("/")
    if is_local_mediamtx and base and path:
        return f"{base}/{path}"
    return stream_url


def parse_polygon(value: Any) -> tuple[Point, ...] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    points: list[Point] = []
    for item in value:
        if isinstance(item, Mapping) and "x" in item and "y" in item:
            try:
                x = float(item["x"])
                y = float(item["y"])
            except (TypeError, ValueError):
                return None
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
            try:
                x = float(item[0])
                y = float(item[1])
            except (TypeError, ValueError):
                return None
        else:
            return None
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return None
        points.append((x, y))
    return tuple(points)


def _zone_id(kind: str, name: str, region_id: str) -> str:
    suffix = region_id[-6:] if region_id else "zone"
    safe_name = "".join(character if character.isalnum() or character in "-_." else "-" for character in name)
    return f"{kind}-{safe_name or kind}-{suffix}"[:100]


def _signature(payload: Mapping[str, Any]) -> str:
    return sha256(dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def cameras_from_rows(
    camera_rows: Sequence[Mapping[str, Any]],
    region_rows: Sequence[Mapping[str, Any]],
    settings: FleetSettings,
) -> tuple[FleetCamera, ...]:
    queues: dict[str, list[MappedZone]] = {}
    restricted: dict[str, list[MappedZone]] = {}
    for row in region_rows:
        camera_id = str(row.get("camera_id") or "")
        region_id = str(row.get("region_id") or "")
        name = str(row.get("name") or "zone")
        region_type = str(row.get("type") or "")
        points = parse_polygon(row.get("polygon"))
        if not camera_id or points is None:
            continue
        zone = MappedZone(_zone_id("queue" if region_type == "QUEUE" else "restricted", name, region_id), name, points)
        if region_type == "QUEUE":
            queues.setdefault(camera_id, []).append(zone)
        elif region_type == "RESTRICTED_AREA":
            restricted.setdefault(camera_id, []).append(zone)

    cameras: list[FleetCamera] = []
    for row in camera_rows:
        camera_id = str(row.get("id") or "").strip()
        stream_url = str(row.get("stream_url") or "").strip()
        if not camera_id or not stream_url:
            continue
        resolved = rewrite_stream_url(stream_url, settings.mediamtx_rtsp_url)
        parsed = urlparse(resolved)
        if parsed.scheme.lower() not in {"rtsp", "rtsps"} or not parsed.hostname or not parsed.path.strip("/"):
            continue
        camera_queues = tuple(queues.get(camera_id, ()))
        camera_restricted = tuple(restricted.get(camera_id, ()))
        cameras.append(
            FleetCamera(
                camera_id=camera_id,
                name=str(row.get("name") or camera_id),
                stream_url=resolved,
                field_id=str(row["field_id"]) if row.get("field_id") else None,
                market_id=str(row["market_id"]) if row.get("market_id") else None,
                booth_id=str(row["booth_id"]) if row.get("booth_id") else None,
                queues=camera_queues,
                restricted_zones=camera_restricted,
                signature=_signature(
                    {
                        "id": camera_id,
                        "stream": resolved,
                        "queues": [(zone.zone_id, zone.points) for zone in camera_queues],
                        "restricted": [(zone.zone_id, zone.points) for zone in camera_restricted],
                    }
                ),
            )
        )
    return tuple(cameras[: settings.max_cameras])


def load_fleet_cameras(settings: FleetSettings) -> tuple[FleetCamera, ...]:
    if not settings.database_url:
        return ()
    try:
        from psycopg import connect
        from psycopg.rows import dict_row
    except ImportError:
        return ()

    with connect(settings.database_url, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, name, "streamUrl" AS stream_url, "fieldId" AS field_id,
                       "marketId" AS market_id, "boothId" AS booth_id
                FROM "Camera"
                WHERE "deletedAt" IS NULL
                  AND "streamUrl" IS NOT NULL
                  AND BTRIM("streamUrl") <> ''
                ORDER BY "createdAt" DESC
                """
            )
            camera_rows = list(cursor.fetchall())
            cursor.execute(
                """
                SELECT cr."cameraId" AS camera_id, r.id AS region_id, r.name, r.type,
                       cr."mainPolygon" AS polygon
                FROM "CameraRegion" cr
                JOIN "Region" r ON r.id = cr."regionId"
                WHERE cr."deletedAt" IS NULL AND r."deletedAt" IS NULL
                """
            )
            region_rows = list(cursor.fetchall())
    return cameras_from_rows(camera_rows, region_rows, settings)
