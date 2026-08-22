# StableTrack (isolated third-party backend)

Paper: *StableTrack: Stabilizing Multi-Object Tracking on Low-Frequency
Detections* (Shelukhan, Mamedov, Kvanchiani), [arXiv:2511.20418](https://arxiv.org/abs/2511.20418).

## Repository and license

No official public repository or license was released with the paper as of
2026-08-19. This directory is an independent, isolated reference implementation
of the published algorithm. Keep application imports out of this package;
adapters live in `app.tracking.stabletrack_adapter`.

## Dependencies

- NumPy (already required by video-analytics)
- OpenCV (already required) for optional CamShift visual tracking
- Optional SciPy for Hungarian assignment; a greedy fallback is used otherwise
- ReID embeddings are supplied by the shared `OsNetReIdentifier` (ONNX)

## Pretrained weights

The paper uses:

- Detector: YOLOX-X (we do **not** replace the project detector)
- Re-ID: DynaMix / SBS50 in the paper; this project reuses the existing OSNet
  ONNX weight configured as `onnx.reid_model`
- Visual tracker: ASMS (not shipped with OpenCV). CamShift is the built-in
  scale-adaptive mean-shift stand-in.

## ReID requirements

StableTrack's first association stage is ReID-gated. When no ReID model is
configured, the adapter still runs BBD- and IoU-gated spatial matching so the
tracker can be compared against ByteTrack without appearance features.

Expected detection format: person `xyxy` boxes with confidence in `[0, 1]` and
`class_id = 0`, the same `Detection` objects produced by the project detector.

## FPS assumptions

The paper evaluates 1/2/4 Hz detection with intermediate frames for Forward and
Backward visual tracking. This deployment processes video at **0.5 FPS**
(~2 s between frames):

- Kalman prediction and Bbox-Based Distance use **real timestamps** (`Δτ` in
  seconds), not a 30 FPS frame counter
- Intermediate frames are used when the caller provides them
- On processed-only streams (fleet / dashboard stride) visual tracking is
  skipped or run last-frame→current-frame as a degraded fallback

`max_age_seconds` defaults to 8 s (four missed 0.5 FPS samples).
