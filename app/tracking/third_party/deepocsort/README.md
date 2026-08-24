# Deep OC-SORT (isolated third-party backend)

Paper: *Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification*
(Maggiolino, Ahmad, Cao, Kitani), [arXiv:2302.11813](https://arxiv.org/abs/2302.11813),
ICIP 2023.

## Repository and license

Official/reference implementation:

- https://github.com/GerardMaggiolino/Deep-OC-SORT
- MIT License

This directory is a detector-agnostic extract of `trackers/ocsort_embedding`
(association, Dynamic Appearance, Adaptive Weighting, OCR, optional CMC). Keep
application imports out of this package; adapters live in
`app.tracking.deepocsort_adapter`.

## Dependencies

- NumPy (already required)
- OpenCV (already required) for optional sparse-flow camera motion compensation
- Shared Hungarian assignment (`app.tracking.third_party.stabletrack.matching`)
- ReID embeddings supplied by the shared `OsNetReIdentifier` (ONNX)

PyTorch, FastReID, YOLOX, and filterpy from the official repo are **not**
required.

## Pretrained weights

The paper uses:

- Detector: YOLOX (we do **not** replace the project detector)
- Re-ID: SBS50 from [fast-reid](https://github.com/JDAI-CV/fast-reid)

This project reuses the existing OSNet ONNX weight configured as
`onnx.reid_model`. GPU is optional: ONNX providers come from `onnx.providers`.

## ReID requirements

Appearance is optional. When no ReID model is configured, the backend still
runs observation-centric IoU + momentum association so it can be compared
against ByteTrack without FastReID/SBS50.

Expected detection format: person `xyxy` boxes with confidence in `[0, 1]` and
`class_id = 0`, the same `Detection` objects produced by the project detector.

## FPS assumptions

Official code assumes consecutive MOT frames (typically 30 FPS) and a unit
Kalman step (`dt = 1` frame). This deployment processes video at **0.5 FPS**
(~2 s between frames):

- Kalman `F` uses elapsed seconds, not a frame counter
- `max_age_seconds` defaults to 8 s (four missed 0.5 FPS samples)
- `delta_t_seconds` defaults to 2 s (one processed frame) for OCM
- `confirmation_hits` defaults to 1 so tracks are usable immediately
- After OCR, a ReID cosine recovery stage can re-identify lost tracks when
  IoU is zero across a 2 s gap
- Sparse-flow CMC is off by default: optical flow is unreliable across 2 s

Camera motion compensation can be enabled with `use_cmc: true` when frames are
available and the camera actually moves.
