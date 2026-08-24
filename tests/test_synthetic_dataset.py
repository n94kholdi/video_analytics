from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.management.models import CameraMinute
from app.synthetic.generator import DatasetConfig, generate_dataset
from app.synthetic.manager import list_bundles, read_manifest


def test_golden_bundle_is_deterministic_and_matches_ingestion_contract(tmp_path: Path) -> None:
    first = generate_dataset(DatasetConfig(
        dataset_id="golden-a", profile="golden", seed=17, duration_minutes=30,
        fields=1, markets=1, booths=1, cameras=2, output_root=tmp_path / "one",
    ))
    second = generate_dataset(DatasetConfig(
        dataset_id="golden-a", profile="golden", seed=17, duration_minutes=30,
        fields=1, markets=1, booths=1, cameras=2, output_root=tmp_path / "two",
    ))

    first_records = (first / "camera_minutes.jsonl").read_text(encoding="utf-8").splitlines()
    second_records = (second / "camera_minutes.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(first_records) == 60
    # Generation timestamps may differ only if the minute changes during the test.
    assert [json.loads(row) | {"bucketStart": None} for row in first_records] == [json.loads(row) | {"bucketStart": None} for row in second_records]
    CameraMinute.model_validate_json(first_records[0])

    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert summary["recordCount"] == 60
    assert 0 <= summary["coveragePercent"] <= 100
    assert (first / "analysis.html").read_text(encoding="utf-8").startswith("<!doctype html>")


def test_report_bundle_uses_hourly_rollups(tmp_path: Path) -> None:
    bundle = generate_dataset(DatasetConfig(
        dataset_id="report-a", profile="report", seed=4, days=2,
        fields=1, markets=1, booths=2, cameras=4, output_root=tmp_path,
    ))
    manifest = read_manifest(bundle)
    assert manifest["recordKind"] == "precomputed_location_hour"
    assert (bundle / "camera_minutes.jsonl").exists() is False
    assert len((bundle / "location_hours.jsonl").read_text(encoding="utf-8").splitlines()) == 4 * 48
    assert len((bundle / "queue_waits.jsonl").read_text(encoding="utf-8").splitlines()) == 2 * 48
    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    assert summary["peakOccupancy"] > 0
    assert 0 <= summary["slaPercent"] <= 100


def test_bundle_listing_and_id_validation(tmp_path: Path) -> None:
    generate_dataset(DatasetConfig(
        dataset_id="scale-a", profile="scale", duration_minutes=2,
        fields=1, markets=1, booths=1, cameras=3, output_root=tmp_path,
    ))
    assert [item["datasetId"] for item in list_bundles(tmp_path)] == ["scale-a"]
    with pytest.raises(ValueError, match="dataset id"):
        generate_dataset(DatasetConfig(dataset_id="Unsafe ID!", output_root=tmp_path))
