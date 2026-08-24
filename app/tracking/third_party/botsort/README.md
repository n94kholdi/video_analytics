# BoT-SORT (isolated third-party backend)

Paper: *BoT-SORT: Robust Associations Multi-Pedestrian Tracking*
(Aharon, Orfaig, Bobrovsky), [arXiv:2206.14651](https://arxiv.org/abs/2206.14651).

## Repository and license

Official/reference implementation:

- https://github.com/NirAharon/BoT-SORT
- MIT License

This directory is a detector-agnostic extract of `tracker/bot_sort.py` and
`tracker/matching.py` (ByteTrack two-stage association, width-height Kalman,
IoU–ReID min-fusion, optional GMC). Keep application imports out of this
package; adapters live in `app.tracking.botsort_adapter`.

## Dependencies

- NumPy (already required)
- OpenCV (already required) for optional sparse-flow camera motion compensation
- Shared Hungarian assignment (`app.tracking.third_party.stabletrack.matching`)
- Timestamp-aware 8-D Kalman (`app.tracking.third_party.deepocsort.kalman`);
  BoT-SORT introduced this `[x, y, w, h, vx, vy, vw, vh]` state, and Deep
  OC-SORT reused it
- Optional GMC via shared `SparseFlowCMC`
- ReID embeddings supplied by the shared `OsNetReIdentifier` (ONNX)

PyTorch, FastReID, YOLOX, and filterpy from the official repo are **not**
required.

## Pretrained weights

The paper uses:

- Detector: YOLOX (we do **not** replace the project detector)
- Re-ID: SBS50 from [fast-reid](https://github.com/JDAI-CV/fast-reid)
  (`MOT17-SBS-S50` / `MOT20-SBS-S50` in the official README)

This project reuses the existing OSNet ONNX weight configured as
`onnx.reid_model`. GPU is optional: ONNX providers come from `onnx.providers`.

## ReID requirements

Appearance is optional. When no ReID model is configured, the backend still
runs ByteTrack-style high/low IoU association so it can be compared against
ByteTrack without FastReID/SBS50.

Expected detection format: person `xyxy` boxes with confidence in `[0, 1]` and
`class_id = 0`, the same `Detection` objects produced by the project detector.

Official BoT-SORT extracts embeddings only for high-score boxes and fuses
cosine distance with IoU by `min(iou_dist, emb_dist / 2)`, gated by
`proximity_thresh` (IoU distance) and `appearance_thresh`.

## GPU requirements

None for the tracker itself. ReID uses ONNX Runtime; `CPUExecutionProvider` is
enough. CUDA is optional through `onnx.providers`.

## FPS assumptions

Official code assumes consecutive MOT frames (typically 30 FPS) and a unit
Kalman step (`dt = 1` frame). `track_buffer` is scaled as
`frame_rate / 30 * 30`, which collapses to **zero frames** at 0.5 FPS.

This deployment processes video at **0.5 FPS** (~2 s between frames):

- Kalman `F` uses elapsed seconds, not a frame counter
- `max_age_seconds` defaults to 8 s (four missed 0.5 FPS samples)
- `confirmation_hits` defaults to 1 so tracks are usable immediately
- `proximity_thresh` defaults to 1.0 (no IoU gate) so ReID can recover when
  boxes no longer overlap across a 2 s gap
- `new_track_threshold` defaults to the high-score threshold so people are
  not dropped after a single 0.4-confidence detection
- Sparse-flow GMC is off by default: optical flow is unreliable across 2 s
- Association uses the last observed box (not the Kalman prediction) when
  elapsed time is ≥ 0.4 s, and a bbox-based-distance recovery stage rematches
  walking people whose boxes no longer overlap

Camera motion compensation can be enabled with `use_cmc: true` when frames are
available and the camera actually moves.
