"""Exercise ingestion, rollups, and all management read models against PostgreSQL."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.management.models import AnalyticsQuery, CameraMinute  # noqa: E402
from app.management.repository import analytics_repository  # noqa: E402
from app.management.service import management_analytics_service  # noqa: E402


CAMERA_ID = "analytics-smoke-camera"
FIELD_ID = "analytics-smoke-field"


def main() -> None:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    analytics_repository.open()
    try:
        _register_source()
        observation = CameraMinute.model_validate({
            "cameraId": CAMERA_ID, "cameraName": "Analytics smoke camera",
            "bucketStart": now.isoformat(), "sampleCount": 30, "expectedSamples": 30,
            "confidenceSum": 28.5, "occupancySum": 180, "occupancyMax": 12,
            "occupancyLast": 8, "entries": 20, "exits": 17,
            "queues": [{"queueId": "service-1", "queueName": "Service 1", "sampleCount": 30,
                        "lengthSum": 90, "lengthMax": 7, "lengthLast": 4,
                        "waitSumSeconds": 3600, "waitSampleCount": 30,
                        "completedWaitSeconds": [120, 240, 420], "throughput": 3, "slaMet": 2,
                        "warningSamples": 4, "criticalSamples": 1,
                        "movementSpeedSumMpm": 36, "movementSpeedSamples": 12,
                        "physicallyCalibrated": True}],
            "spatial": [{"layer": "congestion", "points": [
                {"x": 25, "y": 25, "value": 25, "intensity": .25},
                {"x": 75, "y": 75, "value": 80, "intensity": .8}], "coveragePercent": 100}],
            "events": [{"eventId": "analytics-smoke-event", "eventType": "queue_overflow_started",
                        "severity": "warning", "title": "Smoke warning", "occurredAt": now.isoformat(),
                        "metricKey": "criticalMinutes"}],
        })
        analytics_repository.ingest([observation], 300)
        analytics_repository.refresh_rollups()
        query = AnalyticsQuery(location_type="field", location_id=FIELD_ID, place_type="field",
                               from_date=now-timedelta(hours=1), to_date=now+timedelta(hours=1))
        result = {
            "peopleFlow": management_analytics_service.people_flow(query)["kpis"],
            "queues": management_analytics_service.queues(query)["kpis"],
            "spatialLayers": [item["layer"] for item in management_analytics_service.spatial(query)["layers"]],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        _cleanup()
        analytics_repository.close()


def _register_source() -> None:
    with analytics_repository.connection() as connection, connection.cursor() as cursor:
        cursor.execute("""
          INSERT INTO analytics_camera_source (camera_id,camera_name,field_id,field_name,physically_calibrated)
          VALUES (%s,%s,%s,%s,TRUE)
          ON CONFLICT (camera_id) DO UPDATE SET field_id=EXCLUDED.field_id,field_name=EXCLUDED.field_name
        """, (CAMERA_ID, "Analytics smoke camera", FIELD_ID, "Analytics smoke field"))
        connection.commit()


def _cleanup() -> None:
    with analytics_repository.connection() as connection, connection.cursor() as cursor:
        cursor.execute("DELETE FROM analytics_camera_source WHERE camera_id=%s", (CAMERA_ID,))
        cursor.execute("DELETE FROM analytics_location_hour WHERE location_id=%s", (FIELD_ID,))
        cursor.execute("DELETE FROM analytics_queue_location_hour WHERE location_id=%s", (FIELD_ID,))
        connection.commit()


if __name__ == "__main__":
    main()
