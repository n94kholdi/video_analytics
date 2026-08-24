"""Command-line interface for synthetic analytics datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

from app.management.repository import AnalyticsRepository
from app.synthetic.generator import DatasetConfig, generate_dataset
from app.synthetic.manager import SyntheticDatasetManager, list_bundles


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and switch Tarebar synthetic analytics datasets")
    parser.add_argument("--root", type=Path, default=Path("outputs/synthetic"), help="bundle directory")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="generate a deterministic bundle")
    generate.add_argument("dataset_id")
    generate.add_argument("--profile", choices=("golden", "report", "scale"), default="golden")
    generate.add_argument("--name")
    generate.add_argument("--seed", type=int, default=20260816)
    generate.add_argument("--days", type=int)
    generate.add_argument("--minutes", type=int)
    generate.add_argument("--fields", type=int)
    generate.add_argument("--markets", type=int)
    generate.add_argument("--booths", type=int)
    generate.add_argument("--cameras", type=int)

    activate = commands.add_parser("activate", help="replace the active synthetic dataset")
    activate.add_argument("dataset_id")
    replay = commands.add_parser("replay", help="send camera-minute records through the HTTP ingestion API")
    replay.add_argument("dataset_id")
    replay.add_argument("--url", default="http://127.0.0.1:8000/api/v1/ingest/minutes")
    replay.add_argument("--key")
    replay.add_argument("--rate", type=float, default=33.0, help="observations per second")
    replay.add_argument("--batch-size", type=int, default=33)
    commands.add_parser("deactivate", help="remove synthetic records from PostgreSQL")
    commands.add_parser("list", help="list generated bundles")
    commands.add_parser("status", help="show the active synthetic dataset")

    args = parser.parse_args()
    if args.command == "generate":
        path = generate_dataset(DatasetConfig(
            dataset_id=args.dataset_id, profile=args.profile, name=args.name, seed=args.seed,
            days=args.days, duration_minutes=args.minutes, fields=args.fields, markets=args.markets,
            booths=args.booths, cameras=args.cameras, output_root=args.root,
        ))
        print(json.dumps({"generated": str(path), "analysis": str(path / "analysis.html")}, ensure_ascii=False, indent=2))
        return
    if args.command == "list":
        print(json.dumps(list_bundles(args.root), ensure_ascii=False, indent=2))
        return
    if args.command == "replay":
        print(json.dumps(_replay(args.root / args.dataset_id, args.url, args.key, args.rate, args.batch_size), ensure_ascii=False, indent=2))
        return

    manager = SyntheticDatasetManager(AnalyticsRepository())
    if args.command == "activate":
        print(json.dumps(manager.activate(args.root / args.dataset_id), ensure_ascii=False, indent=2))
    elif args.command == "deactivate":
        manager.deactivate()
        print(json.dumps({"active": None}))
    elif args.command == "status":
        print(json.dumps({"active": manager.active()}))

def _replay(bundle: Path, url: str, key: str | None, rate: float, batch_size: int) -> dict[str, object]:
    from app.synthetic.manager import read_manifest

    manifest = read_manifest(bundle)
    if manifest["recordKind"] != "camera_minute":
        raise SystemExit("replay requires a golden or scale camera-minute bundle")
    if rate <= 0 or not 1 <= batch_size <= 500:
        raise SystemExit("rate must be positive and batch-size must be between 1 and 500")
    path = bundle / str(manifest["recordFile"])
    accepted = 0
    started = time.monotonic()
    batch: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                accepted += _post(url, key, batch)
                batch.clear()
                target_elapsed = accepted / rate
                delay = target_elapsed - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
        if batch:
            accepted += _post(url, key, batch)
    elapsed = time.monotonic() - started
    return {"datasetId": manifest["datasetId"], "accepted": accepted, "elapsedSeconds": round(elapsed, 2),
            "actualRate": round(accepted / max(elapsed, .001), 2)}


def _post(url: str, key: str | None, observations: list[dict[str, object]]) -> int:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-Analytics-Key"] = key
    request = Request(url, data=json.dumps({"observations": observations}, separators=(",", ":")).encode(), headers=headers, method="POST")
    with urlopen(request, timeout=30) as response:
        body = json.loads(response.read())
    return int(body.get("data", {}).get("accepted", 0))


if __name__ == "__main__":
    main()
