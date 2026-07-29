# Video Analytics MVP

This is the isolated application scaffold for a medium-complexity, video-based
people analytics MVP. Existing computer-vision experiments and model weights
remain outside this directory and are treated as read-only inputs.

Phase 2 adds CPU-capable ONNX person detection for the verified light model.
Tracking, geometry, analytics, persistence behavior, API routes, and dashboard
behavior are intentionally not implemented yet.

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

## Dependencies

The base dependency set is CPU-capable and headless: NumPy, ONNX Runtime,
OpenCV Headless, and PyYAML. API, dashboard, and development tools are optional
dependency groups. The initial ByteTrack adapter will be selected and isolated
in the tracking phase so the base runtime does not acquire PyTorch or a second
OpenCV distribution.

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

The following interfaces are planned and are not available in Phase 1:

```bash
video-analytics api --config configs/default.yaml
video-analytics dashboard --config configs/default.yaml
```

Model paths and detector thresholds are configurable. No weights are copied,
changed, or repaired by this application.
