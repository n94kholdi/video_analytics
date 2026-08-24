# Synthetic management analytics datasets

The generator creates structured analytics facts, not rendered video. Downloaded videos still
validate detection/tracking; these bundles validate aggregation, queues, heatmaps, data quality,
reports, and ingestion capacity.

## Profiles

| Profile | Default contents | Intended use |
| --- | --- | --- |
| `golden` | 2 fields, 4 markets, 12 booths, 24 cameras, 7 days of camera-minutes | Exact Phase 2 correctness, hierarchy aggregation, spatial layers, outages, and events |
| `report` | 4 fields, 16 markets, 96 booths, 384 cameras, 120 days of location-hours | Weekly/monthly reports, rankings, changes, and long-term queries without millions of raw rows |
| `scale` | 4 fields, 20 markets, 150 booths, 2,000 cameras, 120 minutes of camera-minutes | Dashboard cardinality and 33/67/100 observation-per-second ingestion tests |

Every bundle is deterministic for a fixed seed and contains:

- `manifest.json` — identity, date range, seed, and profile;
- `dimensions.json` — fields, markets, booths, and cameras;
- `camera_minutes.jsonl` or precomputed `location_hours.jsonl`, `queue_hours.jsonl`, and bounded queue-wait samples;
- `summary.json` — inspectable aggregate values and chart series;
- `expected_results.json` — acceptance values and tolerances;
- `analysis.html` — standalone occupancy, entry, and wait-time diagrams.

Generated bundles are ignored by Git and remain inactive until explicitly activated.

## Generate

Generate bundles as your normal host user from `video_analytics`. The generator uses only the
application's standard Python dependencies. This keeps generated files owned by you; the analytics
container receives the directory through a read-only bind mount.

```bash
cd video_analytics

python3 -m app.synthetic.cli generate demo-golden --profile golden
python3 -m app.synthetic.cli generate demo-report --profile report
python3 -m app.synthetic.cli generate demo-scale --profile scale
python3 -m app.synthetic.cli list
```

Use smaller overrides for a fast trial:

```bash
python3 -m app.synthetic.cli generate quick-check \
  --profile golden --minutes 360 --fields 1 --markets 2 --booths 4 --cameras 8
```

Open `outputs/synthetic/<dataset-id>/analysis.html` directly in a browser before loading anything
into PostgreSQL. Then rebuild/start the dashboard stack so the read-only bind mount is present:

```bash
cd ../Tarebar-Smart-Monitoring-Platform
docker compose up -d --build
```

## Activate, switch, and disable

```bash
# Activate one bundle
docker compose exec video-analytics python -m app.synthetic.cli activate demo-golden

# Show active bundle
docker compose exec video-analytics python -m app.synthetic.cli status

# Switch: activation removes the previous synthetic dataset first
docker compose exec video-analytics python -m app.synthetic.cli activate demo-report

# Disable all synthetic data
docker compose exec video-analytics python -m app.synthetic.cli deactivate
```

Only IDs beginning with the reserved `synthetic:` prefix are removed. Real locations, cameras,
facts, and users are not selected by these operations. Activation failure also cleans partial
synthetic data.

Log in as `ORG_ADMIN`, refresh the dashboard, choose a location whose name starts with
`آزمایشی`, and set the global date filter to the manifest's `start`/`end` dates. The generated
range ends near the generation time, so the default recent-date filter normally works for the
golden and scale profiles. The report profile spans the preceding 120 days.

Managers with field/market scopes will not see generated locations unless explicit synthetic
scopes are added; the harness intentionally does not change user permissions.

## Ingestion-rate test

The scale bundle can be replayed through the real authenticated HTTP endpoint. At 2,000 cameras,
the expected compact rate is roughly 33 camera-minute observations per second.

```bash
# Production-equivalent compact rate
docker compose exec video-analytics python -m app.synthetic.cli replay demo-scale \
  --rate 33 --batch-size 33 --key local-ingest-key-change-me

# 2× and 3× capacity checks
docker compose exec video-analytics python -m app.synthetic.cli replay demo-scale \
  --rate 67 --batch-size 67 --key local-ingest-key-change-me
docker compose exec video-analytics python -m app.synthetic.cli replay demo-scale \
  --rate 100 --batch-size 100 --key local-ingest-key-change-me
```

Run dashboard queries concurrently and observe API latency, PostgreSQL load, rollup freshness,
errors, and accepted counts. Replays are idempotent because camera-minute keys are stable.

## Important limitations

- The report profile intentionally bypasses camera-minute expansion and loads bounded hourly read
  models. Use the golden profile to test minute-to-hour rollup correctness.
- The report profile does not supply detailed spatial playback; use golden for spatial pages.
- Synthetic facts test software behavior, not detector accuracy. Continue evaluating CV accuracy
  with manually annotated downloaded/iPhone videos.
- Unique visitors remain unavailable across multiple cameras because cross-camera identity
  reconciliation is not implemented.
