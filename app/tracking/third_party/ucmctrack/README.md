# UCMCTrack (isolated third-party backend)

Paper: *UCMCTrack: Multi-Object Tracking with Uniform Camera Motion Compensation*
(Yi, Luo, Luo, Huang, Wu, Hu, Hao), AAAI 2024,
[arXiv:2312.08952](https://arxiv.org/abs/2312.08952).

## Repository and license

Official/reference implementation:

- https://github.com/corfyi/UCMCTrack
- MIT License

This directory is a detector-agnostic extract of `tracker/ucmc.py` and
`tracker/kalman.py` (BYTE-style high/low association, ground-plane CV Kalman,
mapped Mahalanobis distance / CMD). Keep application imports out of this
package; adapters live in `app.tracking.ucmctrack_adapter`. Camera files are
loaded by `app.tracking.calibration`, not here.

## Dependencies

- NumPy (already required)
- Shared Hungarian assignment (`app.tracking.third_party.stabletrack.matching`)

filterpy, lap, PyTorch, and YOLOX from the official repo are **not** required.

## Calibration requirements

UCMCTrack maps each box foot-point through camera geometry and associates with
mapped Mahalanobis distance. Geometry is **optional**:

| Mode | Input | Mapping |
| --- | --- | --- |
| Uncalibrated (default) | none | image-plane foot-point (pixels) |
| Intrinsics + extrinsics | official `cam_para` or YAML K, R, t | ground plane at `ground_z` |
| Homography | 3×3 matrix or existing `CalibrationConfig` | image → ground |

Do not hardcode camera values in this package. Register per-camera files in a
catalog directory; unknown `camera_id` values stay uncalibrated.

## Expected detection format

Person `xyxy` boxes with confidence in `[0, 1]` and `class_id = 0`, the same
`Detection` objects produced by the project detector. Appearance/ReID is **not**
part of standard UCMCTrack; the shared OSNet stack is left unused here.

## FPS assumptions

Official code uses `dt = 1 / fps` as a fixed Kalman step and `max_age` in
frames (at 30 FPS, `cdt=30` is about one second). This deployment processes
video at **0.5 FPS** (~2 s between frames):

- Kalman `F` and process noise use elapsed seconds
- A missed person can be recovered for about **1 second**, or one processed
  frame at 0.5 FPS (~2 s), whichever is longer. After that the ID is retired
  and never given to someone else
- `confirmation_hits` defaults to 1 so tracks are usable immediately
- `wx`/`wy`/`vmax` defaults are pixel-space values for uncalibrated cameras;
  metric cameras should override them (about `0.1` / `1.5`) in the camera file
- Association split:
  - **calibrated:** rank with mapped Mahalanobis (MMD, paper). IoU / last
    position only gate still people and short misses, so a newcomer cannot
    inherit a retired ID
  - **uncalibrated:** rank with IoU / last position; MMD is a weak tie-break
    because pixel-space covariance is poorly scaled at 0.5 FPS
- CMC/GMC from the official detector is unused (2 s gaps make it unreliable)

## GPU requirements

None. The tracker is CPU-only motion association.
