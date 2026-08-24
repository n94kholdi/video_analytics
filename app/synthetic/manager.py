"""Install and switch generated datasets in a local Tarebar PostgreSQL database."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Iterable, Iterator

from app.management.models import CameraMinute
from app.management.repository import AnalyticsRepository
from app.synthetic.generator import FORMAT_VERSION, PREFIX


class DatasetBundleError(RuntimeError):
    """The selected bundle is incomplete or incompatible."""


class SyntheticDatasetManager:
    def __init__(self, repository: AnalyticsRepository) -> None:
        self.repository = repository

    def activate(self, bundle: Path) -> dict[str, object]:
        manifest = read_manifest(bundle)
        dimensions = _read_json(bundle / str(manifest["dimensionsFile"]))
        self.repository.open()
        try:
            self._clear_synthetic()
            self._load_dimensions(dimensions)
            if manifest["recordKind"] == "camera_minute":
                count = self._load_camera_minutes(bundle / str(manifest["recordFile"]), manifest)
                self._refresh_range(manifest)
            elif manifest["recordKind"] == "precomputed_location_hour":
                count = self._load_location_hours(bundle / str(manifest["recordFile"]), bundle / "queue_hours.jsonl")
                self._load_report_events(bundle / "events.jsonl")
                self._load_report_waits(bundle / "queue_waits.jsonl")
            else:
                raise DatasetBundleError(f"unsupported record kind: {manifest['recordKind']}")
            return {"datasetId": manifest["datasetId"], "profile": manifest["profile"], "recordsLoaded": count,
                    "from": manifest["start"], "to": manifest["end"]}
        except Exception:
            # Never leave a partially installed synthetic hierarchy active.
            self._clear_synthetic()
            raise
        finally:
            self.repository.close()

    def deactivate(self) -> None:
        self.repository.open()
        try:
            self._clear_synthetic()
        finally:
            self.repository.close()

    def active(self) -> str | None:
        self.repository.open()
        try:
            row = self.repository.row(
                'SELECT id FROM "Camera" WHERE id LIKE %s AND "deletedAt" IS NULL ORDER BY id LIMIT 1',
                [f"{PREFIX}%"],
            )
            if not row:
                return None
            camera_id = str(row["id"])
            parts = camera_id.split(":")
            return parts[1] if len(parts) > 2 else None
        finally:
            self.repository.close()

    def _clear_synthetic(self) -> None:
        pattern = f"{PREFIX}%"
        with self.repository.connection() as connection, connection.cursor() as cursor:
            # Compact facts cascade from camera sources; read-model rollups do not.
            cursor.execute("DELETE FROM analytics_location_hour WHERE location_id LIKE %s", (pattern,))
            cursor.execute("DELETE FROM analytics_queue_location_hour WHERE location_id LIKE %s", (pattern,))
            cursor.execute("DELETE FROM analytics_camera_source WHERE camera_id LIKE %s", (pattern,))
            # The generated hierarchy has no UserScope/Region rows. Delete children first.
            cursor.execute('DELETE FROM "Camera" WHERE id LIKE %s', (pattern,))
            cursor.execute('DELETE FROM "Booth" WHERE id LIKE %s', (pattern,))
            cursor.execute('DELETE FROM "Market" WHERE id LIKE %s', (pattern,))
            cursor.execute('DELETE FROM "Field" WHERE id LIKE %s', (pattern,))
            cursor.execute('DELETE FROM "BoothCategory" WHERE id=%s', (f"{PREFIX}category",))
            connection.commit()

    def _load_dimensions(self, value: object) -> None:
        if not isinstance(value, dict):
            raise DatasetBundleError("dimensions.json must contain an object")
        fields = _object_list(value, "fields")
        markets = _object_list(value, "markets")
        booths = _object_list(value, "booths")
        cameras = _object_list(value, "cameras")
        with self.repository.connection() as connection, connection.cursor() as cursor:
            cursor.execute('INSERT INTO "BoothCategory" (id,name,"updatedAt") VALUES (%s,%s,now())', (f"{PREFIX}category", "داده مصنوعی"))
            cursor.executemany('INSERT INTO "Field" (id,name,address,"updatedAt") VALUES (%s,%s,%s,now())',
                               [(item["id"], item["name"], "Synthetic test data") for item in fields])
            cursor.executemany('INSERT INTO "Market" (id,name,address,"fieldId","updatedAt") VALUES (%s,%s,%s,%s,now())',
                               [(item["id"], item["name"], "Synthetic test data", item["fieldId"]) for item in markets])
            cursor.executemany('INSERT INTO "Booth" (id,number,"marketId","categoryId","updatedAt") VALUES (%s,%s,%s,%s,now())',
                               [(item["id"], item["number"], item["marketId"], f"{PREFIX}category") for item in booths])
            cursor.executemany(
                'INSERT INTO "Camera" (id,name,"fieldId","marketId","boothId","updatedAt") VALUES (%s,%s,%s,%s,%s,now())',
                [(item["id"], item["name"], item["fieldId"], item["marketId"], item["boothId"]) for item in cameras],
            )
            cursor.execute("SELECT analytics_sync_camera_sources()")
            cursor.executemany(
                "UPDATE analytics_camera_source SET physically_calibrated=%s WHERE camera_id=%s",
                [(bool(item.get("physicallyCalibrated")), item["id"]) for item in cameras],
            )
            connection.commit()

    def _load_camera_minutes(self, path: Path, manifest: dict[str, object]) -> int:
        count = 0
        batch: list[CameraMinute] = []
        for raw in _json_lines(path):
            batch.append(CameraMinute.model_validate(raw))
            if len(batch) == 500:
                count += self.repository.ingest(batch, 300, sync_sources=False)
                batch.clear()
        if batch:
            count += self.repository.ingest(batch, 300, sync_sources=False)
        return count

    def _refresh_range(self, manifest: dict[str, object]) -> None:
        start = datetime.fromisoformat(str(manifest["start"]))
        end = datetime.fromisoformat(str(manifest["end"]))
        with self.repository.connection() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT analytics_refresh_location_rollups(%s,%s)", (start, end))
            connection.commit()

    def _load_location_hours(self, flow_path: Path, queue_path: Path) -> int:
        count = 0
        with self.repository.connection() as connection, connection.cursor() as cursor:
            flow_batch: list[tuple[object, ...]] = []
            for item in _json_lines(flow_path):
                flow_batch.append((item["locationType"], item["locationId"], item["locationName"], item.get("parentName"),
                                   item["bucketStart"], item["sampleCount"], item["expectedSamples"], item["contributingSources"],
                                   item["expectedSources"], item["confidenceSum"], item["occupancySum"], item["occupancyMax"],
                                   item["entries"], item["exits"], item.get("uniqueVisitors")))
                if len(flow_batch) >= 2000:
                    _insert_flow(cursor, flow_batch)
                    count += len(flow_batch)
                    flow_batch.clear()
            if flow_batch:
                _insert_flow(cursor, flow_batch)
                count += len(flow_batch)
            queue_batch: list[tuple[object, ...]] = []
            for item in _json_lines(queue_path):
                queue_batch.append((item["locationType"], item["locationId"], item["locationName"], item.get("parentName"),
                                    item["queueId"], item["queueName"], item["bucketStart"], item["sampleCount"],
                                    item["lengthSum"], item["lengthMax"], item["waitSumSeconds"], item["waitSampleCount"],
                                    item["throughput"], item["slaMet"], item["warningSamples"], item["criticalSamples"],
                                    item["movementSpeedSumMpm"], item["movementSpeedSamples"], item["calibratedSources"]))
                if len(queue_batch) >= 2000:
                    _insert_queue(cursor, queue_batch)
                    queue_batch.clear()
            if queue_batch:
                _insert_queue(cursor, queue_batch)
            connection.commit()
        return count

    def _load_report_events(self, path: Path) -> None:
        if not path.exists():
            return
        values = list(_json_lines(path))
        if not values:
            return
        with self.repository.connection() as connection, connection.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO analytics_event
                   (event_id,camera_id,event_type,severity,title,metric_key,occurred_at,value,payload)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING""",
                [(item["eventId"], item["cameraId"], item["eventType"], item["severity"], item["title"],
                  item.get("metricKey"), item["occurredAt"], item.get("value"), json.dumps(item.get("payload"))) for item in values],
            )
            connection.commit()

    def _load_report_waits(self, path: Path) -> None:
        if not path.exists():
            return
        values = list(_json_lines(path))
        if not values:
            return
        with self.repository.connection() as connection, connection.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO analytics_queue_wait_event
                   (event_id,camera_id,queue_id,occurred_at,wait_seconds,sla_met)
                   VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                [(item["eventId"], item["cameraId"], item["queueId"], item["occurredAt"],
                  item["waitSeconds"], item["slaMet"]) for item in values],
            )
            connection.commit()


def read_manifest(bundle: Path) -> dict[str, object]:
    manifest = _read_json(bundle / "manifest.json")
    if not isinstance(manifest, dict):
        raise DatasetBundleError("manifest.json must contain an object")
    if manifest.get("formatVersion") != FORMAT_VERSION:
        raise DatasetBundleError("unsupported synthetic dataset format")
    required = ("datasetId", "profile", "recordKind", "recordFile", "dimensionsFile", "start", "end")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise DatasetBundleError(f"manifest is missing: {', '.join(missing)}")
    return manifest


def list_bundles(root: Path) -> list[dict[str, object]]:
    if not root.exists():
        return []
    result = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not (path / "manifest.json").exists():
            continue
        manifest = read_manifest(path)
        result.append({"path": str(path), "datasetId": manifest["datasetId"], "name": manifest.get("name"),
                       "profile": manifest["profile"], "start": manifest["start"], "end": manifest["end"]})
    return result


def _insert_flow(cursor: object, values: list[tuple[object, ...]]) -> None:
    cursor.executemany("""INSERT INTO analytics_location_hour
      (location_type,location_id,location_name,parent_name,bucket_start,sample_count,expected_samples,
       contributing_sources,expected_sources,confidence_sum,occupancy_sum,occupancy_max,entries,exits,unique_visitors)
      VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", values)


def _insert_queue(cursor: object, values: list[tuple[object, ...]]) -> None:
    cursor.executemany("""INSERT INTO analytics_queue_location_hour
      (location_type,location_id,location_name,parent_name,queue_id,queue_name,bucket_start,sample_count,
       length_sum,length_max,wait_sum_seconds,wait_sample_count,throughput,sla_met,warning_samples,
       critical_samples,movement_speed_sum_mpm,movement_speed_samples,calibrated_sources)
      VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", values)


def _json_lines(path: Path) -> Iterator[dict[str, object]]:
    if not path.exists():
        raise DatasetBundleError(f"missing bundle file: {path}")
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DatasetBundleError(f"{path}:{line_number} is not an object")
            yield value


def _read_json(path: Path) -> object:
    if not path.exists():
        raise DatasetBundleError(f"missing bundle file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _object_list(value: dict[str, object], key: str) -> list[dict[str, object]]:
    group = value.get(key)
    if not isinstance(group, list) or not all(isinstance(item, dict) for item in group):
        raise DatasetBundleError(f"dimensions.{key} must be an object array")
    return group
