# Video Analytics MVP

This is the isolated application scaffold for a medium-complexity, video-based
people analytics MVP. Existing computer-vision experiments and model weights
remain outside this directory and are treated as read-only inputs.

Phase 7 adds bounded movement occupancy and dwell heatmaps on top of shared
confirmed tracker foot points. Queue, speed, database, API, and dashboard
behavior remain intentionally unimplemented; event persistence is currently
limited to a shared sink interface and JSONL output.

## Planned scope

The MVP will process recorded video, webcams, and basic RTSP sources through one
shared pipeline:

```text
video source
  -> ONNX person detector
  -> ByteTrack adapter
  -> shared timestamped tracks
  -> camera geometry
  -> analytics
  -> events, metrics, SQLite, annotated video, API, and dashboard
```

Planned analytics include occupancy and directional counts, restricted-area
intrusion, movement and dwell heatmaps, configured queue heuristics, and speed
in pixels/second or metres/second when valid calibration is available.

## Architecture

```text
app/
  core/       settings and shared data contracts
  detection/  ONNX detector adapters
  tracking/   multi-object tracker adapters
  geometry/   zones, lines, and calibration
  analytics/  counting, intrusion, heatmap, queue, and speed modules
  storage/    JSONL, CSV, SQLite, and latest-status output
  api/        FastAPI application
  dashboard/  lightweight dashboard
configs/      YAML application and camera configuration
scripts/      command-line helpers
tests/        unit and integration tests
outputs/      ignored generated artifacts
```

The detector is separated into:

- `app.core.models`: shared immutable `Detection` representation
- `app.detection.preprocessing`: letterbox and BGR-to-RGB tensor conversion
- `app.detection.onnx_detector`: model validation and ONNX Runtime inference
- `app.detection.postprocessing`: output parsing, coordinate restoration, and NMS
- `app.detection.visualization`: optional OpenCV annotation
- `app.detection.cli`: headless image and recorded-video runner

Preprocessing, inference, post-processing, and visualization timing/behavior do
not overlap.

Tracking is separated into:

- `app.tracking.base`: tracker-independent protocol and per-frame result
- `app.tracking.bytetrack`: maintained ByteTrack dependency adapter, person-only
  conversion, lifecycle cleanup, and bounded EMA-smoothed foot-point history
- `app.tracking.visualization`: track IDs, state, foot points, and trajectories
- `app.tracking.cli`: recorded-video detector/tracker runner using source times

`TrackObservation` and `TrajectoryPoint` live in `app.core.models`, so future
analytics do not depend on ByteTrack or Supervision objects. Raw trajectory
positions are bounding-box bottom centers; each sample also retains an
EMA-smoothed position for later speed and queue analytics.

Camera geometry is separated into:

- `app.geometry.config`: validated camera/analytics YAML models, normalized
  zones, directed counting lines, queue service points, and calibration pairs
- `app.geometry.primitives`: inclusive-boundary polygon membership, polygon
  validation, directed line sides, and finite-segment crossing results
- `app.geometry.calibration`: fixed-resolution homography construction and an
  explicit unavailable result when calibration is not configured
- `scripts/configure_camera.py`: optional Tk reference-frame point selector

People counting is separated into:

- `app.analytics.counting`: confirmed-track polygon occupancy, hysteresis-based
  finite-line crossings, per-camera state, cumulative totals, and explicit reset
- `app.analytics.visualization`: composable occupancy and entry/exit overlays
- `app.core.models.Event`: shared event envelope used for `line_crossed` events

Each counting line has a normalized `hysteresis` value. It is interpreted as a
fraction of the frame diagonal, creating a resolution-independent dead band on
both sides of the line. A track must move from one stable side to the other and
cross the configured finite segment before it is counted. Tracks on the line or
moving only within the dead band do not increment totals.

Restricted-area detection is separated into:

- `app.analytics.restricted_area`: foot-point membership, independent
  camera/track/zone state, entry dwell, exit grace, cooldown, and reset
- `app.analytics.restricted_visualization`: named zone status plus pending and
  confirmed intrusion overlays
- `app.storage.EventSink`: the persistence boundary used by analytics
- `app.storage.JsonlEventSink`: append-only persistence for shared `Event`
  envelopes; SQLite remains a later phase

Each restricted zone supports `entry_dwell_seconds`, `exit_grace_seconds`, and
`alert_cooldown_seconds`. Entry and exit lifecycle events remain visible for a
transient crossing, while only a dwell-qualified
`restricted_area_confirmed` event is the cooldown-controlled alert. During a
short missing-observation or outside period, prior state is kept until the exit
grace expires.

Movement heatmaps are separated into:

- `app.analytics.heatmap`: image-grid mapping, optional calibrated ground-grid
  mapping, sample-count occupancy, elapsed-seconds dwell, bounded track state,
  reset/tumbling-window aggregation, and CSV/PNG export
- `HeatmapSnapshot.occupancy`: number of confirmed position samples per cell
- `HeatmapSnapshot.dwell_seconds`: timestamp-derived time assigned to the
  previous confirmed position cell

These are people-movement analytics heatmaps, not detector feature or neural
network activation heatmaps. Long gaps are not treated as dwell: intervals over
`max_sample_gap_seconds` are discarded, and per-track state is evicted after
`track_idle_seconds`. `aggregation_window_seconds` creates constant-memory
tumbling windows; setting it to `null` retains run totals until explicit reset.
Calibration creates a parallel ground-plane grid, using configured
`ground_bounds` or the calibration correspondence extents. Missing calibration
leaves image heatmaps available and reports a clear ground-unavailable reason.
Rendering covers the complete configured heatmap region: zero-value cells use
the low (blue) end of the selected color map and increasingly occupied cells
progress through green/yellow/orange to red. `smoothing_sigma_cells` spreads
each foot-point cell into a readable density region without changing the exact
numeric CSV values. Set the heatmap `region` to the normalized full-frame
rectangle `[[0, 0], [1, 0], [1, 1], [0, 1]]` to color the entire image.

An optional `active_schedule` accepts `start`/`end` in `HH:MM`, weekday names
or integers (`0` is Monday), and an IANA timezone. Schedule evaluation requires
Unix timestamps. Recorded-video source-relative timestamps should leave the
schedule unset unless the caller supplies an epoch time basis.

Normalized `(0, 0)` and `(1, 1)` map to inclusive pixel corners `(0, 0)` and
`(width - 1, height - 1)`. Ground projection is only available through a
validated calibration containing at least four non-degenerate image/ground
correspondences.

## Dependencies

The base dependency set is CPU-capable: NumPy, ONNX Runtime, OpenCV Headless,
PyYAML, and the maintained `trackers` package. API, dashboard, and development
tools are optional dependency groups. ByteTrack does not use BoT-SORT, OSNet,
or PyTorch in this phase. The current `trackers` package itself declares the
regular OpenCV distribution, although this application uses no GUI APIs and
remains headless at runtime.

Python 3.10 or newer is required.

## Current commands

From this directory:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python scripts/check_imports.py
```

Load the default settings:

```bash
python -c "from app.core.config import load_settings; print(load_settings())"
```

Run person detection:

```bash
python -m app.detection.cli input.jpg --output outputs/detected.jpg
python -m app.detection.cli input.mp4 --output outputs/detected.mp4
```

Run person tracking with annotated IDs and trajectories:

```bash
python -m app.tracking.cli input.mp4 --output outputs/tracked.mp4
```

Count confirmed tracked people in every video frame. Without a camera YAML,
the entire image is used as one occupancy zone. The command writes both an
annotated MP4 and a CSV containing one row per frame. `confirmed_humans` is the
visible tracked-person count; polygon columns count only foot points inside each
configured zone:

```bash
python -m app.analytics.cli data/human.mp4 \
  --output outputs/human_counted.mp4 \
  --counts-csv outputs/human_counts.csv
```

To report configured polygon occupancy and line totals instead:

```bash
python -m app.analytics.cli data/human.mp4 \
  --camera-config configs/cameras/example_lobby.yaml \
  --output outputs/human_counted.mp4 \
  --counts-csv outputs/human_counts.csv
```

The configured run also processes restricted zones, draws their current status,
and appends intrusion events to `outputs.events_jsonl`. Override that destination
with `--events-jsonl`; no restricted zones are assumed without a camera YAML.

Heatmap processing is runtime opt-in even when the camera YAML lists the module.
Pass `--enable-heatmap` to produce evolving occupancy and dwell overlay videos,
plus separate final CSV grids, colorized PNGs, and first-frame overlays.
Ground-plane CSVs and PNGs are also produced when calibration exists. Use the
camera YAML's `outputs.heatmap_directory`, allow the default output directory,
or override it explicitly:

```bash
python -m app.analytics.cli data/human.mp4 \
  --camera-config configs/cameras/example_lobby.yaml \
  --enable-heatmap \
  --heatmap-dir outputs/human_heatmaps
```

Without `--enable-heatmap`, no heatmap state is accumulated and no heatmap
files or videos are created. Videos are streamed frame by frame, so enabling
them does not retain an unbounded collection of video frames in memory.

Grid sizes, aggregation window, maximum sample gap, idle-state timeout, color
map, overlay opacity, and rendering smoothing are configured under `heatmap`.
CSV rows follow image
or ground Y and columns follow X. Image-space PNG and overlay dimensions match
the source frame; ground PNG dimensions match the configured ground grid.

Trajectory trails are shown by default. Hide only the trails while continuing
to collect trajectory history for later analytics:

```bash
python -m app.tracking.cli input.mp4 \
  --output outputs/tracked.mp4 \
  --no-trajectories
```

The tracking CLI reports average detection, tracking, and total-frame time.
Tracker activation threshold, lost-track buffer, IoU match threshold, and
history size are configured under `tracker` in `configs/default.yaml`.

For a bounded smoke run:

```bash
python -m app.detection.cli input.mp4 --max-frames 5
```

The CLI prints the selected model/providers, frame and detection counts, and
average preprocessing, inference, post-processing, and total detector time.
CUDA or another available execution provider can be requested first while
retaining CPU fallback:

```bash
python -m app.detection.cli input.mp4 \
  --providers CUDAExecutionProvider CPUExecutionProvider
```

Set `VIDEO_ANALYTICS_CONFIG` to load another YAML file. Individual overrides
are available through `VIDEO_ANALYTICS_LOG_LEVEL`,
`VIDEO_ANALYTICS_OUTPUT_DIR`, `VIDEO_ANALYTICS_DATABASE_PATH`,
`VIDEO_ANALYTICS_DETECTOR_MODEL`, and comma-separated
`VIDEO_ANALYTICS_ONNX_PROVIDERS`.

## Camera geometry configuration

The complete example at `configs/cameras/example_lobby.yaml` includes
occupancy and restricted polygons, a directed line, queue/service geometry,
heatmap settings, and a four-point calibration. Load it independently of the
application settings:

```bash
python -c "from app.geometry import load_camera_config; print(load_camera_config('configs/cameras/example_lobby.yaml'))"
```

To draw geometry on a reference image and save validated YAML:

```bash
python scripts/configure_camera.py data/human.jpg \
  --camera-id lobby_east \
  --name "East lobby" \
  --source data/lobby.mp4 \
  --output configs/cameras/lobby_east.yaml
```

Select a polygon, two line endpoints, an optional queue service point, and at
least four calibration image points. Each calibration click prompts for the
matching ground-plane coordinate. The selector uses optional system Tk; it is
not imported by the headless runtime.

## Supported model

Phase 2 supports only:

```text
All_weights/Weights_final/HumanDetection_light_input_640.onnx
input:  [batch, 3, 640, 640]
output: [batch, 5, 8400]
```

The dynamic batch metadata is accepted, but the frame API intentionally runs a
single frame per call. The output is interpreted as one-class YOLO-style
`center_x, center_y, width, height, confidence` candidates.

`HumanDetection_input_640.onnx` is excluded because it references
`model.onnx.data`, which is missing beside the model in `Weights_final`. A
possible sidecar elsewhere in the experimental repository has not been copied
or assumed to match.

`HumanDetection_server_input_640.onnx` is also excluded in Phase 2. Its primary
`[batch, 300, 6]` output and auxiliary outputs have not yet had their exact box,
score, class, and suppression semantics verified. The light-contract validator
rejects it with a clear shape error instead of guessing.

## Planned commands

The following interfaces are planned and are not available in Phase 7:

```bash
video-analytics api --config configs/default.yaml
video-analytics dashboard --config configs/default.yaml
```

Model paths and detector thresholds are configurable. No weights are copied,
changed, or repaired by this application.
