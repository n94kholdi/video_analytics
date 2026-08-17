"""Generate reproducible analytics bundles without rendering synthetic video.

The bundles deliberately use the public ``CameraMinute`` ingestion contract so
the synthetic and CV-derived paths exercise the same persistence code.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import heapq
import html
import json
import math
from pathlib import Path
import random
import re
from typing import Iterable, Iterator, Literal
from zoneinfo import ZoneInfo


Profile = Literal["golden", "report", "scale"]
PREFIX = "synthetic:"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    profile: Profile = "golden"
    name: str | None = None
    seed: int = 20260816
    timezone_name: str = "Asia/Tehran"
    days: int | None = None
    duration_minutes: int | None = None
    fields: int | None = None
    markets: int | None = None
    booths: int | None = None
    cameras: int | None = None
    output_root: Path = Path("outputs/synthetic")

    def resolved(self) -> "DatasetConfig":
        defaults = {
            "golden": {"days": 7, "duration_minutes": None, "fields": 2, "markets": 4, "booths": 12, "cameras": 24},
            "report": {"days": 120, "duration_minutes": None, "fields": 4, "markets": 16, "booths": 96, "cameras": 384},
            "scale": {"days": None, "duration_minutes": 120, "fields": 4, "markets": 20, "booths": 150, "cameras": 2000},
        }[self.profile]
        dataset_id = _safe_id(self.dataset_id)
        return DatasetConfig(
            dataset_id=dataset_id,
            profile=self.profile,
            name=self.name or f"Synthetic {self.profile.title()} — {dataset_id}",
            seed=self.seed,
            timezone_name=self.timezone_name,
            days=self.days if self.days is not None else defaults["days"],
            duration_minutes=self.duration_minutes if self.duration_minutes is not None else defaults["duration_minutes"],
            fields=self.fields if self.fields is not None else defaults["fields"],
            markets=self.markets if self.markets is not None else defaults["markets"],
            booths=self.booths if self.booths is not None else defaults["booths"],
            cameras=self.cameras if self.cameras is not None else defaults["cameras"],
            output_root=self.output_root,
        )


@dataclass
class _QueueState:
    waiting: deque[datetime]
    services: list[tuple[datetime, float]]
    overloaded: bool = False


def generate_dataset(config: DatasetConfig) -> Path:
    """Create a self-contained bundle and return its directory."""
    config = config.resolved()
    _validate(config)
    destination = config.output_root / config.dataset_id
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"dataset already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    dimensions = _dimensions(config)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    if config.profile == "report":
        end = now.replace(minute=0)
        start = end - timedelta(days=int(config.days or 120))
        summary = _generate_report(config, dimensions, start, end, destination)
        record_file = "location_hours.jsonl"
        record_kind = "precomputed_location_hour"
    else:
        minutes = int(config.duration_minutes or int(config.days or 7) * 1440)
        end = now
        start = end - timedelta(minutes=minutes)
        summary = _generate_minutes(config, dimensions, start, end, destination)
        record_file = "camera_minutes.jsonl"
        record_kind = "camera_minute"

    manifest = {
        "formatVersion": FORMAT_VERSION,
        "datasetId": config.dataset_id,
        "syntheticPrefix": f"{PREFIX}{config.dataset_id}:",
        "name": config.name,
        "profile": config.profile,
        "seed": config.seed,
        "timezone": config.timezone_name,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "recordKind": record_kind,
        "recordFile": record_file,
        "dimensionsFile": "dimensions.json",
        "expectedResultsFile": "expected_results.json",
        "summaryFile": "summary.json",
        "analysisFile": "analysis.html",
        "config": _jsonable(asdict(config)),
    }
    _write_json(destination / "manifest.json", manifest)
    _write_json(destination / "dimensions.json", dimensions)
    _write_json(destination / "summary.json", summary)
    _write_json(destination / "expected_results.json", _expected(summary, config))
    (destination / "analysis.html").write_text(_analysis_html(manifest, summary), encoding="utf-8")
    return destination


def _dimensions(config: DatasetConfig) -> dict[str, list[dict[str, object]]]:
    prefix = f"{PREFIX}{config.dataset_id}:"
    fields = [{"id": f"{prefix}field:{index + 1:03d}", "name": f"میدان آزمایشی {index + 1}"}
              for index in range(int(config.fields or 0))]
    markets = []
    for index in range(int(config.markets or 0)):
        field = fields[index % len(fields)]
        markets.append({"id": f"{prefix}market:{index + 1:03d}", "name": f"بازار آزمایشی {index + 1}", "fieldId": field["id"]})
    booths = []
    for index in range(int(config.booths or 0)):
        market = markets[index % len(markets)]
        booths.append({"id": f"{prefix}booth:{index + 1:04d}", "number": f"S{index + 1:04d}", "marketId": market["id"]})
    cameras = []
    for index in range(int(config.cameras or 0)):
        booth = booths[index % len(booths)]
        market = next(item for item in markets if item["id"] == booth["marketId"])
        cameras.append({
            "id": f"{prefix}camera:{index + 1:05d}", "name": f"دوربین مصنوعی {index + 1}",
            "fieldId": market["fieldId"], "marketId": market["id"], "boothId": booth["id"],
            "physicallyCalibrated": index % 4 != 0,
        })
    return {"fields": fields, "markets": markets, "booths": booths, "cameras": cameras}


def _generate_minutes(
    config: DatasetConfig,
    dimensions: dict[str, list[dict[str, object]]],
    start: datetime,
    end: datetime,
    destination: Path,
) -> dict[str, object]:
    rng = random.Random(config.seed)
    zone = ZoneInfo(config.timezone_name)
    cameras = dimensions["cameras"]
    occupancy = {str(camera["id"]): rng.randint(0, 4) for camera in cameras}
    exits_due: dict[str, dict[int, int]] = {str(camera["id"]): defaultdict(int) for camera in cameras}
    queues = {str(camera["id"]): _QueueState(deque(), []) for camera in cameras}
    totals = _summary_template(dimensions)
    hourly: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    heatmap = [[0.0 for _ in range(12)] for _ in range(7)]
    output = destination / "camera_minutes.jsonl"
    record_count = 0
    minute_index = 0
    with output.open("w", encoding="utf-8") as stream:
        timestamp = start
        while timestamp < end:
            local = timestamp.astimezone(zone)
            for camera_index, camera in enumerate(cameras):
                camera_id = str(camera["id"])
                factor = 0.72 + (camera_index % 11) * 0.055
                arrival_rate = _arrival_rate(local, factor, config.profile)
                entries = _poisson(rng, arrival_rate)
                due = exits_due[camera_id].pop(minute_index, 0)
                occupancy[camera_id] = max(0, occupancy[camera_id] + entries - due)
                for _ in range(entries):
                    dwell = max(3, int(rng.lognormvariate(math.log(24), 0.45)))
                    exits_due[camera_id][minute_index + dwell] += 1
                exits = min(due, occupancy[camera_id] + due)
                sample_count = 30
                outage = _is_outage(local, camera_index)
                if outage:
                    sample_count = 0 if camera_index % 3 else 10
                confidence = 0.72 if _is_low_confidence(local, camera_index) else 0.96
                queue = _queue_minute(rng, queues[camera_id], timestamp, entries, bool(camera["physicallyCalibrated"]))
                events = _events(config, camera, timestamp, local, queue, queues[camera_id], rng)
                spatial = _spatial(camera_index, local, occupancy[camera_id], sample_count) if minute_index % 15 == 0 else []
                record = {
                    "cameraId": camera_id, "cameraName": camera["name"], "bucketStart": timestamp.isoformat(),
                    "sampleCount": sample_count, "expectedSamples": 30,
                    "confidenceSum": round(sample_count * confidence, 3),
                    "occupancySum": float(occupancy[camera_id] * sample_count),
                    "occupancyMax": occupancy[camera_id] + (1 if entries else 0), "occupancyLast": occupancy[camera_id],
                    "entries": entries, "exits": exits, "uniqueVisitors": None,
                    "physicallyCalibrated": camera["physicallyCalibrated"],
                    "queues": [queue], "spatial": spatial, "events": events,
                }
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                record_count += 1
                _accumulate(totals, camera, record, queue, events)
                hour_key = local.strftime("%Y-%m-%d %H:00")
                hourly[hour_key]["occupancy_sum"] += occupancy[camera_id]
                hourly[hour_key]["occupancy_samples"] += 1
                hourly[hour_key]["entries"] += entries
                hourly[hour_key]["wait_sum"] += float(queue["waitSumSeconds"])
                hourly[hour_key]["wait_samples"] += int(queue["waitSampleCount"])
                heatmap[local.weekday()][local.hour // 2] += occupancy[camera_id]
            timestamp += timedelta(minutes=1)
            minute_index += 1
    return _finish_summary(config, totals, hourly, heatmap, record_count, start, end)


def _generate_report(
    config: DatasetConfig,
    dimensions: dict[str, list[dict[str, object]]],
    start: datetime,
    end: datetime,
    destination: Path,
) -> dict[str, object]:
    rng = random.Random(config.seed)
    zone = ZoneInfo(config.timezone_name)
    fields = dimensions["fields"]
    markets = dimensions["markets"]
    booths = dimensions["booths"]
    cameras = dimensions["cameras"]
    camera_counts: dict[str, int] = defaultdict(int)
    for camera in cameras:
        for key in ("fieldId", "marketId", "boothId"):
            camera_counts[str(camera[key])] += 1
    locations: list[tuple[str, dict[str, object]]] = [
        *[("field", item) for item in fields], *[("market", item) for item in markets], *[("booth", item) for item in booths],
    ]
    children: dict[str, list[str]] = defaultdict(list)
    for market in markets:
        children[str(market["fieldId"])].append(str(market["id"]))
    for booth in booths:
        children[str(booth["marketId"])].append(str(booth["id"]))
    totals = _summary_template(dimensions)
    hourly: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    heatmap = [[0.0 for _ in range(12)] for _ in range(7)]
    flow_path = destination / "location_hours.jsonl"
    queue_path = destination / "queue_hours.jsonl"
    event_path = destination / "events.jsonl"
    wait_path = destination / "queue_waits.jsonl"
    record_count = 0
    booth_camera = {str(booth["id"]): next(camera for camera in cameras if camera["boothId"] == booth["id"]) for booth in booths}
    with flow_path.open("w", encoding="utf-8") as flow_stream, queue_path.open("w", encoding="utf-8") as queue_stream, event_path.open("w", encoding="utf-8") as event_stream, wait_path.open("w", encoding="utf-8") as wait_stream:
        timestamp = start
        while timestamp < end:
            local = timestamp.astimezone(zone)
            leaf: dict[str, dict[str, float]] = {}
            for index, booth in enumerate(booths):
                growth = 1 + 0.18 * ((timestamp - start).total_seconds() / max(1, (end - start).total_seconds())) * (-1 if index % 5 == 0 else 1)
                occupancy = max(0.0, _arrival_rate(local, 7.5 + index % 7, "report") * 18 * growth + rng.uniform(-2, 2))
                entries = max(0, int(occupancy * rng.uniform(0.7, 1.1)))
                wait = max(20.0, 65 + occupancy * (7.5 if _rush(local) else 3.5) + rng.uniform(-20, 20))
                throughput = max(1, int(entries * min(1.0, 300 / wait)))
                leaf[str(booth["id"])] = {"occupancy": occupancy, "entries": entries, "exits": max(0, entries + rng.randint(-2, 2)), "wait": wait, "throughput": throughput}
            values = dict(leaf)
            for location_type in ("market", "field"):
                source = markets if location_type == "market" else fields
                for location in source:
                    grouped = [values[child] for child in children[str(location["id"])] if child in values]
                    values[str(location["id"])] = {
                        "occupancy": sum(value["occupancy"] for value in grouped),
                        "entries": sum(value["entries"] for value in grouped),
                        "exits": sum(value["exits"] for value in grouped),
                        "wait": sum(value["wait"] * value["throughput"] for value in grouped) / max(1, sum(value["throughput"] for value in grouped)),
                        "throughput": sum(value["throughput"] for value in grouped),
                    }
            for location_type, location in locations:
                location_id = str(location["id"])
                value = values[location_id]
                expected = camera_counts[location_id]
                coverage = 0.7 if _is_report_outage(local, location_id) else 1.0
                contributing = max(1, round(expected * coverage))
                sample_count = contributing * 30 * 60
                expected_samples = expected * 30 * 60
                parent_name = _parent_name(location_type, location, dimensions)
                flow = {
                    "locationType": location_type, "locationId": location_id,
                    "locationName": _location_name(location_type, location), "parentName": parent_name,
                    "bucketStart": timestamp.isoformat(), "sampleCount": sample_count,
                    "expectedSamples": expected_samples, "contributingSources": contributing,
                    "expectedSources": expected, "confidenceSum": sample_count * (0.78 if coverage < 1 else 0.96),
                    # analytics_location_hour stores the sum of per-camera hourly
                    # occupancy averages, despite the historical column name.
                    "occupancySum": round(value["occupancy"], 3),
                    "occupancyMax": math.ceil(value["occupancy"] * 1.22),
                    "entries": int(value["entries"]), "exits": int(value["exits"]), "uniqueVisitors": None,
                }
                flow_stream.write(json.dumps(flow, ensure_ascii=False, separators=(",", ":")) + "\n")
                queue_samples = contributing * 30 * 60
                throughput = int(value["throughput"])
                sla = max(0, min(throughput, round(throughput * (1 if value["wait"] <= 300 else 0.62))))
                queue = {
                    "locationType": location_type, "locationId": location_id,
                    "locationName": flow["locationName"], "parentName": parent_name,
                    "queueId": f"{location_id}:queue:1", "queueName": "صف خدمت مصنوعی",
                    "bucketStart": timestamp.isoformat(), "sampleCount": queue_samples,
                    "lengthSum": round(max(0, value["wait"] / 90) * queue_samples, 3),
                    "lengthMax": max(1, math.ceil(value["wait"] / 45)),
                    "waitSumSeconds": round(value["wait"] * queue_samples, 3), "waitSampleCount": queue_samples,
                    "throughput": throughput, "slaMet": sla,
                    "warningSamples": queue_samples if value["wait"] >= 300 else 0,
                    "criticalSamples": queue_samples if value["wait"] >= 600 else 0,
                    "movementSpeedSumMpm": queue_samples * 4.2, "movementSpeedSamples": queue_samples,
                    "calibratedSources": contributing,
                }
                queue_stream.write(json.dumps(queue, ensure_ascii=False, separators=(",", ":")) + "\n")
                if location_type == "booth":
                    wait_stream.write(json.dumps({
                        "eventId": f"{PREFIX}{config.dataset_id}:wait:{location_id}:{timestamp.strftime('%Y%m%d%H')}",
                        "cameraId": booth_camera[location_id]["id"], "queueId": queue["queueId"],
                        "occurredAt": timestamp.isoformat(), "waitSeconds": round(value["wait"], 2),
                        "slaMet": value["wait"] <= 300,
                    }, ensure_ascii=False, separators=(",", ":")) + "\n")
                record_count += 1
            organization = values[str(fields[0]["id"])].copy()
            organization["occupancy"] = sum(values[str(field["id"])]["occupancy"] for field in fields)
            organization["entries"] = sum(values[str(field["id"])]["entries"] for field in fields)
            organization["wait"] = sum(values[str(field["id"])]["wait"] for field in fields) / len(fields)
            organization["throughput"] = sum(values[str(field["id"])]["throughput"] for field in fields)
            organization_sla = round(organization["throughput"] * (1 if organization["wait"] <= 300 else .62))
            hourly[local.strftime("%Y-%m-%d %H:00")].update({"occupancy_sum": organization["occupancy"], "occupancy_samples": 1, "entries": organization["entries"], "wait_sum": organization["wait"], "wait_samples": 1})
            heatmap[local.weekday()][local.hour // 2] += organization["occupancy"]
            if local.day == 1 and local.hour == 11:
                camera = cameras[(local.month + local.year) % len(cameras)]
                event_stream.write(json.dumps({
                    "eventId": f"{PREFIX}{config.dataset_id}:event:{timestamp.strftime('%Y%m%d%H')}",
                    "cameraId": camera["id"], "eventType": "abnormal_congestion", "severity": "warning",
                    "title": "ازدحام غیرعادی مصنوعی", "metricKey": "occupancy", "occurredAt": timestamp.isoformat(),
                    "value": round(organization["occupancy"], 2), "payload": {"synthetic": True},
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
                totals["events"] += 1
            totals["entries"] += int(organization["entries"])
            totals["occupancySampleSum"] += float(organization["occupancy"])
            totals["occupancySamples"] += 1
            totals["peakOccupancy"] = max(int(totals["peakOccupancy"]), math.ceil(organization["occupancy"] * 1.22))
            totals["waitSampleSum"] += float(organization["wait"])
            totals["waitSamples"] += 1
            totals["throughput"] += int(organization["throughput"])
            totals["slaMet"] += int(organization_sla)
            totals["coverageSamples"] += sum(
                max(1, round(camera_counts[str(field["id"])] * (0.7 if _is_report_outage(local, str(field["id"])) else 1.0))) * 30 * 60
                for field in fields
            )
            totals["expectedSamples"] += len(cameras) * 30 * 60
            timestamp += timedelta(hours=1)
    return _finish_summary(config, totals, hourly, heatmap, record_count, start, end)


def _queue_minute(rng: random.Random, state: _QueueState, timestamp: datetime, arrivals: int, calibrated: bool) -> dict[str, object]:
    completed_waits: list[float] = []
    while state.services and state.services[0][0] <= timestamp:
        _, wait = heapq.heappop(state.services)
        completed_waits.append(wait)
    queue_arrivals = max(0, arrivals + (_poisson(rng, 0.35) if _rush(timestamp) else 0))
    for _ in range(queue_arrivals):
        state.waiting.append(timestamp)
    capacity = 2
    while len(state.services) < capacity and state.waiting:
        arrived = state.waiting.popleft()
        wait = (timestamp - arrived).total_seconds()
        service_seconds = max(30, rng.lognormvariate(math.log(150), 0.32))
        heapq.heappush(state.services, (timestamp + timedelta(seconds=service_seconds), wait))
    waits = [(timestamp - arrived).total_seconds() for arrived in state.waiting]
    current_wait = sum(waits) / len(waits) if waits else 0.0
    length = len(state.waiting)
    samples = 30
    return {
        "queueId": "service-1", "queueName": "صف خدمت مصنوعی", "sampleCount": samples,
        "lengthSum": float(length * samples), "lengthMax": length + (1 if queue_arrivals else 0), "lengthLast": length,
        "waitSumSeconds": current_wait * samples, "waitSampleCount": samples,
        "completedWaitSeconds": [round(value, 2) for value in completed_waits],
        "throughput": len(completed_waits), "slaMet": sum(value <= 300 for value in completed_waits),
        "warningSamples": samples if current_wait >= 300 else 0, "criticalSamples": samples if current_wait >= 600 else 0,
        "movementSpeedSumMpm": 4.2 * samples if calibrated and length else 0,
        "movementSpeedSamples": samples if calibrated and length else 0, "physicallyCalibrated": calibrated,
    }


def _events(config: DatasetConfig, camera: dict[str, object], timestamp: datetime, local: datetime, queue: dict[str, object], state: _QueueState, rng: random.Random) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    overloaded = int(queue["criticalSamples"]) > 0
    if overloaded != state.overloaded:
        state.overloaded = overloaded
        kind = "queue_overflow_started" if overloaded else "queue_overflow_ended"
        events.append({"eventId": f"{PREFIX}{config.dataset_id}:event:{camera['id']}:{timestamp.isoformat()}:{kind}", "eventType": kind,
                       "severity": "warning", "title": "تغییر وضعیت صف مصنوعی", "occurredAt": timestamp.isoformat(),
                       "metricKey": "criticalMinutes", "value": float(queue["lengthLast"]), "payload": {"synthetic": True}})
    if local.hour == 20 and local.minute == 17 and int(str(camera["id"]).rsplit(":", 1)[-1]) % 37 == 0:
        events.append({"eventId": f"{PREFIX}{config.dataset_id}:event:{camera['id']}:{timestamp.isoformat()}:restricted", "eventType": "restricted_area_confirmed",
                       "severity": "critical", "title": "ورود مصنوعی به محدوده ممنوعه", "occurredAt": timestamp.isoformat(),
                       "metricKey": "restrictedEvents", "value": 1, "payload": {"synthetic": True, "confidence": round(rng.uniform(.88, .99), 3)}})
    return events


def _spatial(camera_index: int, local: datetime, occupancy: int, sample_count: int) -> list[dict[str, object]]:
    if sample_count == 0:
        return []
    peak = max(1, occupancy)
    center_x = 25 + (camera_index % 3) * 25
    center_y = 25 + ((camera_index // 3) % 3) * 25
    coverage = round(sample_count * 100 / 30, 2)
    result = []
    for layer, multiplier in (("occupancy", 1.0), ("dwell", 2.5), ("traffic", 1.4), ("congestion", 8.0)):
        value = peak * multiplier * (1.35 if _rush(local) else 1)
        result.append({"zoneId": "location", "zoneName": "کل مکان", "layer": layer, "bucketSeconds": 900,
                       "coveragePercent": coverage, "points": [
                           {"x": center_x, "y": center_y, "value": round(value, 2), "intensity": min(1, round(value / max(1, 20 * multiplier), 3))},
                           {"x": min(100, center_x + 15), "y": min(100, center_y + 10), "value": round(value * .55, 2), "intensity": min(1, round(value * .55 / max(1, 20 * multiplier), 3))},
                       ]})
    return result


def _arrival_rate(local: datetime, factor: float, profile: Profile) -> float:
    hour = local.hour + local.minute / 60
    morning = math.exp(-((hour - 10.5) / 2.2) ** 2)
    evening = 1.25 * math.exp(-((hour - 18.0) / 2.6) ** 2)
    open_factor = 0.04 if hour < 6 or hour >= 23 else 0.18 + morning + evening
    weekend = 1.35 if local.weekday() in (3, 4) else 1.0
    profile_factor = 1.15 if profile == "report" else 1.0
    return max(0.002, factor * open_factor * weekend * profile_factor)


def _rush(value: datetime) -> bool:
    return 9 <= value.hour < 12 or 16 <= value.hour < 20


def _is_outage(local: datetime, camera_index: int) -> bool:
    return local.weekday() == 2 and 13 <= local.hour < 15 and camera_index % 5 == 0


def _is_low_confidence(local: datetime, camera_index: int) -> bool:
    return 18 <= local.hour < 19 and camera_index % 7 == 0


def _is_report_outage(local: datetime, location_id: str) -> bool:
    return local.weekday() == 2 and local.hour in (13, 14) and sum(location_id.encode()) % 5 == 0


def _poisson(rng: random.Random, rate: float) -> int:
    if rate > 30:
        return max(0, round(rng.gauss(rate, math.sqrt(rate))))
    limit = math.exp(-rate)
    product = 1.0
    value = 0
    while product > limit:
        value += 1
        product *= rng.random()
    return value - 1


def _summary_template(dimensions: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    return {"fields": len(dimensions["fields"]), "markets": len(dimensions["markets"]), "booths": len(dimensions["booths"]),
            "cameras": len(dimensions["cameras"]), "entries": 0, "exits": 0, "events": 0,
            "occupancySampleSum": 0.0, "occupancySamples": 0, "peakOccupancy": 0,
            "waitSampleSum": 0.0, "waitSamples": 0, "throughput": 0, "slaMet": 0,
            "coverageSamples": 0, "expectedSamples": 0}


def _accumulate(totals: dict[str, object], camera: dict[str, object], record: dict[str, object], queue: dict[str, object], events: list[dict[str, object]]) -> None:
    totals["entries"] = int(totals["entries"]) + int(record["entries"])
    totals["exits"] = int(totals["exits"]) + int(record["exits"])
    totals["events"] = int(totals["events"]) + len(events)
    totals["occupancySampleSum"] = float(totals["occupancySampleSum"]) + float(record["occupancySum"])
    totals["occupancySamples"] = int(totals["occupancySamples"]) + int(record["sampleCount"])
    totals["peakOccupancy"] = max(int(totals["peakOccupancy"]), int(record["occupancyMax"] or 0))
    totals["waitSampleSum"] = float(totals["waitSampleSum"]) + float(queue["waitSumSeconds"])
    totals["waitSamples"] = int(totals["waitSamples"]) + int(queue["waitSampleCount"])
    totals["throughput"] = int(totals["throughput"]) + int(queue["throughput"])
    totals["slaMet"] = int(totals["slaMet"]) + int(queue["slaMet"])
    totals["coverageSamples"] = int(totals["coverageSamples"]) + int(record["sampleCount"])
    totals["expectedSamples"] = int(totals["expectedSamples"]) + int(record["expectedSamples"])


def _finish_summary(config: DatasetConfig, totals: dict[str, object], hourly: dict[str, dict[str, float]], heatmap: list[list[float]], record_count: int, start: datetime, end: datetime) -> dict[str, object]:
    series = []
    for timestamp, values in sorted(hourly.items()):
        series.append({"timestamp": timestamp,
                       "averageOccupancy": round(values["occupancy_sum"] / max(1, values["occupancy_samples"]), 2),
                       "entries": int(values["entries"]),
                       "averageWaitMinutes": round(values["wait_sum"] / max(1, values["wait_samples"]) / 60, 2)})
    normalized_heatmap = []
    maximum = max((value for row in heatmap for value in row), default=1) or 1
    for day, row in enumerate(heatmap):
        for half_day, value in enumerate(row):
            normalized_heatmap.append({"day": day, "hour": half_day * 2, "value": round(value, 2), "intensity": round(value / maximum, 3)})
    result = dict(totals)
    result.update({
        "profile": config.profile, "recordCount": record_count, "start": start.isoformat(), "end": end.isoformat(),
        "averageOccupancy": round(float(totals["occupancySampleSum"]) / max(1, int(totals["occupancySamples"])), 2),
        "averageWaitMinutes": round(float(totals["waitSampleSum"]) / max(1, int(totals["waitSamples"])) / 60, 2),
        "slaPercent": round(int(totals["slaMet"]) * 100 / max(1, int(totals["throughput"])), 2),
        "coveragePercent": round(int(totals["coverageSamples"]) * 100 / max(1, int(totals["expectedSamples"])), 2) if int(totals["expectedSamples"]) else 100.0,
        "hourly": series, "activityHeatmap": normalized_heatmap,
    })
    return result


def _expected(summary: dict[str, object], config: DatasetConfig) -> dict[str, object]:
    return {"datasetId": config.dataset_id, "seed": config.seed, "tolerance": {"floatingPoint": 0.02, "percent": 0.1},
            "organization": {key: summary[key] for key in ("entries", "averageOccupancy", "peakOccupancy", "averageWaitMinutes", "slaPercent", "coveragePercent", "events")}}


def _analysis_html(manifest: dict[str, object], summary: dict[str, object]) -> str:
    series = list(summary.get("hourly", []))
    sampled = series[::max(1, len(series) // 240)]
    occupancy = [float(item["averageOccupancy"]) for item in sampled]
    waits = [float(item["averageWaitMinutes"]) for item in sampled]
    entries = [float(item["entries"]) for item in sampled]
    cards = [("Records", summary["recordCount"]), ("Cameras", summary["cameras"]), ("Entries", summary["entries"]),
             ("Avg occupancy", summary["averageOccupancy"]), ("Avg wait (min)", summary["averageWaitMinutes"]),
             ("SLA %", summary["slaPercent"]), ("Coverage %", summary["coveragePercent"]), ("Events", summary["events"])]
    card_html = "".join(f"<article><b>{html.escape(str(label))}</b><strong>{html.escape(str(value))}</strong></article>" for label, value in cards)
    return f"""<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\"><title>{html.escape(str(manifest['name']))}</title>
<style>body{{font:14px system-ui;margin:32px;background:#f5f7fb;color:#182033}}h1{{margin-bottom:4px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}article,section{{background:white;border:1px solid #dfe4ee;border-radius:12px;padding:16px;margin:14px 0}}article b{{display:block;color:#667085}}article strong{{display:block;font-size:22px;margin-top:8px}}svg{{width:100%;height:220px}}.legend{{color:#667085}}</style>
<h1>{html.escape(str(manifest['name']))}</h1><p class=\"legend\">Profile: {manifest['profile']} · Seed: {manifest['seed']} · {manifest['start']} → {manifest['end']}</p><div class=\"cards\">{card_html}</div>
<section><h2>Average occupancy</h2>{_svg_line(occupancy, '#2563eb')}</section>
<section><h2>Entries</h2>{_svg_line(entries, '#16a34a')}</section>
<section><h2>Average queue wait (minutes)</h2>{_svg_line(waits, '#dc2626')}</section>
<p class=\"legend\">Open <code>summary.json</code> for exact values and <code>expected_results.json</code> for acceptance checks.</p></html>"""


def _svg_line(values: list[float], color: str) -> str:
    if not values:
        return "<p>No series generated.</p>"
    width, height, pad = 1000, 200, 12
    maximum, minimum = max(values), min(values)
    span = maximum - minimum or 1
    points = []
    for index, value in enumerate(values):
        x = pad + index * (width - 2 * pad) / max(1, len(values) - 1)
        y = pad + (maximum - value) * (height - 2 * pad) / span
        points.append(f"{x:.1f},{y:.1f}")
    return f"<svg viewBox='0 0 {width} {height}' role='img'><polyline fill='none' stroke='{color}' stroke-width='3' points='{' '.join(points)}'/></svg><p class='legend'>min {minimum:.2f} · max {maximum:.2f}</p>"


def _parent_name(location_type: str, location: dict[str, object], dimensions: dict[str, list[dict[str, object]]]) -> str | None:
    if location_type == "field":
        return None
    parent_key = "fieldId" if location_type == "market" else "marketId"
    parent_group = "fields" if location_type == "market" else "markets"
    parent = next(item for item in dimensions[parent_group] if item["id"] == location[parent_key])
    return str(parent["name"])


def _location_name(location_type: str, location: dict[str, object]) -> str:
    return f"غرفه {location['number']}" if location_type == "booth" else str(location["name"])


def _safe_id(value: str) -> str:
    normalized = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,47}", normalized):
        raise ValueError("dataset id must be 2-48 lowercase letters, digits, underscores, or hyphens")
    return normalized


def _validate(config: DatasetConfig) -> None:
    ZoneInfo(config.timezone_name)
    for field in ("fields", "markets", "booths", "cameras"):
        if int(getattr(config, field) or 0) <= 0:
            raise ValueError(f"{field} must be positive")
    if int(config.fields or 0) > int(config.markets or 0) or int(config.markets or 0) > int(config.booths or 0) or int(config.booths or 0) > int(config.cameras or 0):
        raise ValueError("expected fields <= markets <= booths <= cameras")
    if config.profile == "report" and int(config.days or 0) < 1:
        raise ValueError("report days must be positive")
    if config.profile != "report" and int(config.duration_minutes or int(config.days or 0) * 1440) < 1:
        raise ValueError("generation duration must be positive")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
