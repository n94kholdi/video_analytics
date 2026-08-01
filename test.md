# Phase 4 validation

Run these checks from the project directory:

```bash
cd '/home/nayereh/Documents/Fanap_new/R&D/ComputerVision_models/CV_applications/video_analytics_mvp'
```

## Automated tests

```bash
MPLCONFIGDIR=/tmp/mplconfig \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
rtk ../../.venv/bin/python -m pytest -q
```

Expected: all **68 tests pass**. Phase 4 tests cover normalized coordinate
conversion, inclusive and exclusive polygon boundaries, invalid polygons,
directed line-side transitions, finite line crossings and line-contact
hysteresis, known synthetic homography projection, missing/invalid calibration,
and camera YAML loading/round trips. All Phase 1–3 tests remain included.

## Import and syntax checks

```bash
MPLCONFIGDIR=/tmp/mplconfig rtk ../../.venv/bin/python scripts/check_imports.py
rtk ../../.venv/bin/python -m compileall -q app scripts tests
```

Expected: all application packages import, the default application settings
load, and compilation exits successfully.

## Example camera configuration

```bash
rtk ../../.venv/bin/python -c \
  "from app.geometry import load_camera_config; print(load_camera_config('configs/cameras/example_lobby.yaml').camera_id)"
```

Expected: `lobby_east`.

## Optional manual selector check

Run in a graphical desktop session with system Tk available:

```bash
rtk ../../.venv/bin/python scripts/configure_camera.py data/human.jpg \
  --output /tmp/phase4_camera.yaml
```

Select at least three polygon vertices or two line points. For calibration,
select at least four non-collinear image points and enter matching ground X/Y
coordinates. `Save YAML` must either save a file accepted by
`load_camera_config` or show a specific validation error.

No analytics counts, events, queue classification, speed calculation, storage,
API, or dashboard behavior is part of this phase.
