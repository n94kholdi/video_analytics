# Computer Vision Systems — Technical Report

**Scope:** every computer-vision capability in the `CV_applications` workspace — the production `video_analytics` pipeline and the standalone perception models in `CV_models/`.
**Audience:** technical reviewers evaluating architecture and methods, not code.
**Status note:** the production pipeline currently wires in only one perception model (a YOLO-style person detector) plus a tracker. Every other capability described in Part II is a self-contained, ONNX-exportable R&D pipeline that is architecturally compatible with the same integration pattern but is not yet mounted into the live pipeline. This is called out per section.

---

## How to read this report

The workspace has two layers with different maturity:

| Layer | What it is | Maturity |
|---|---|---|
| **Part I — Platform** | The `app/` pipeline: detection → tracking → geometry → analytics → storage/API | Production-grade, tested, versioned by "phase" |
| **Part II — Model library** | `CV_models/`: one folder per perception capability (face, pose, plate, PPE...) | R&D-grade, vendored/experimental, ONNX-convertible |

Each model section below is written as a datasheet: **task, architecture, input/output contract, method**, followed by prose explaining *why* that method was chosen.

---

# Part I — Video Analytics Platform

## 1. System architecture

The platform processes recorded video, webcam, or RTSP input through one linear pipeline. Every stage is a separate, independently testable module, and every downstream stage consumes a plain typed data contract rather than a vendor object — so the tracker doesn't know about the detector's tensor format, and analytics don't know about ByteTrack or Supervision internals.

```
video source (file / webcam / RTSP)
        │
        ▼
 ┌─────────────────┐   letterbox resize, BGR→RGB, normalize
 │  ONNX detector   │   single-class person detector, ONNX Runtime
 └─────────────────┘
        │  Detection[] (box, confidence)
        ▼
 ┌─────────────────┐   ByteTrack association + optional OSNet ReID
 │     Tracker      │   Kalman motion model, track lifecycle
 └─────────────────┘
        │  TrackObservation[] (track_id, box, foot-point, timestamp)
        ▼
 ┌─────────────────┐   zones, lines, service points, homography
 │ Camera geometry  │   normalized coordinates, calibration
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐   counting · restricted-area · heatmap
 │    Analytics      │   queueing · speed estimation
 └─────────────────┘
        │  Event[] + per-frame metrics
        ▼
 ┌─────────────────┐
 │ Storage / API /  │  JSONL + SQLite, FastAPI, SSE live dashboard
 │   Dashboard       │
 └─────────────────┘
```

This is a classic **tracking-by-detection** design: a detector finds people every frame; a tracker stitches per-frame detections into persistent identities over time; everything above that (counting, zones, queues, speed) is computed purely from track geometry, not from re-running any neural network. That separation keeps the expensive part (inference) to one detector pass per frame and makes every analytic a cheap, deterministic geometry computation.

## 2. Person detection

| | |
|---|---|
| **Task** | Single-class (person) object detection |
| **Model** | `HumanDetection_light_input_640.onnx` |
| **Input** | `[batch, 3, 640, 640]`, RGB, letterboxed (aspect-preserving pad, not stretched) |
| **Output** | `[batch, 5, 8400]` — YOLO-style dense prediction grid: `center_x, center_y, width, height, confidence` per anchor point across 8400 candidate locations |
| **Runtime** | ONNX Runtime, CPU by default, CUDA (or any other execution provider) selectable with CPU fallback |

**Method.** This is an anchor-free, single-stage YOLO-family detector (the family exercised in `CV_models/HumanDetection/YOLOv8-human`, a from-scratch PyTorch reimplementation trained on COCO + CrowdHuman person-class data). At inference the letterboxed frame is a single forward pass; the 8400 candidates are the flattened output of three detection heads operating on different feature-map strides (stride 8/16/32), so the same network naturally covers near-field (large-box) and far-field (small-box) people without a separate small-object branch. Post-processing restores letterbox padding/scale, filters by a confidence threshold, and applies non-maximum suppression (NMS) to collapse duplicate boxes.

Two variants exist in the model directory but are deliberately **not** used: one is missing its external-weights sidecar file, and the other (`HumanDetection_server_input_640.onnx`) has a `[batch, 300, 6]` output whose box/score/class/NMS semantics were never verified — the platform's model loader validates the tensor contract at startup and rejects anything that doesn't match the known-good shape rather than guessing.

**Design implication.** The pipeline intentionally processes one frame per inference call even though the model declares dynamic batch support — this keeps latency deterministic and avoids batching complexity for a real-time stream.

## 3. Multi-object tracking

| | |
|---|---|
| **Task** | Assign a stable identity to each detected person across frames |
| **Base algorithm** | ByteTrack |
| **Optional appearance model** | OSNet (`osnet_x0_25`), ONNX |
| **Output contract** | `TrackObservation`: track ID, box, EMA-smoothed foot-point, timestamp, lifecycle state |

**Method — ByteTrack.** Most trackers discard low-confidence detections before matching, which loses genuinely-present-but-partially-occluded people. ByteTrack instead matches in two passes: high-confidence detections are matched first to existing tracks by IoU against each track's Kalman-filter motion prediction; **remaining low-confidence detections are then matched against the tracks that are still unmatched** (typically occluded people, whose confidence drops but who are still present). This second pass is what materially reduces identity switches during occlusion compared to a single-threshold tracker. Unmatched tracks are kept alive for a bounded number of frames (a "lost" buffer) before being finalized, so a person who is briefly fully occluded doesn't immediately spawn a new ID on reappearance.

**Method — OSNet re-identification (optional, opt-in).** ByteTrack's motion+IoU matching alone can still swap IDs when two people cross paths or when someone leaves and re-enters the frame after the lost-track buffer expires. OSNet ("Omni-Scale Network") is a small CNN trained specifically for person re-identification: it produces a compact appearance embedding per detected person crop, and identity continuity is reinforced by cosine similarity between embeddings rather than motion alone. It is CPU/GPU inference cost on top of the detector, so it is exposed as an explicit opt-in flag rather than always-on — the report emphasizes it improves *tracker-ID continuity*, not biometric identification, and can still fail when people look alike.

**Data retained per track.** Each observation keeps both the raw bounding-box bottom-center (the "foot point," used as the ground-contact proxy for zone/line geometry) and an exponential-moving-average-smoothed version of the same point, used later for speed and queue-progress estimation so that per-frame detector jitter doesn't get read as movement.

## 4. Camera geometry and calibration

Camera-specific configuration (YAML) defines, in **normalized image coordinates** (`(0,0)`–`(1,1)`, resolution-independent):

- **Zones** — polygons for occupancy counting and restricted areas
- **Directed lines** — finite line segments with a signed crossing direction, for entry/exit counts
- **Queue geometry** — a polygon plus a manually placed service point
- **Calibration correspondences** — ≥4 paired image points ↔ real-world ground-plane points (in metres)

**Method — homography calibration.** Given four or more non-degenerate image↔ground point pairs, the platform fits a planar homography — a 3×3 projective transform that maps image pixels to ground-plane metric coordinates, valid under the standard assumption that all tracked motion happens on one flat ground plane. This is what turns "pixels per second" into "metres per second" and turns a pixel-space zone into an interpretable area. Calibration is optional per-camera: every module (heatmaps, speed) works in image-pixel units without it and gains a parallel metric output when it is configured. Correctness is verifiable, not just assumed — a companion script projects two independently surveyed points through the fitted homography and reports the error against their known real-world distance.

**Line-crossing robustness.** A naive "which side of the line" check flickers when a track sits near the line due to detector jitter. Each line therefore carries a `hysteresis` band expressed as a fraction of the frame diagonal (so it's resolution-independent): a track must be observed stably on one side, then stably on the other, and must cross the finite segment itself — not just its infinite extension — before a crossing counts.

## 5. Analytics suite

All analytics below share a design principle worth stating once: **they read track geometry, not pixels.** No analytic re-invokes the detector or any neural network; each is a deterministic function of `(track_id, foot_point, timestamp)` streams, which is what makes it possible to run several analytics simultaneously at near-zero marginal compute cost.

### 5.1 Occupancy and directional counting
Confirmed tracks (tracks that have survived the tracker's confirmation threshold, filtering out one-frame false positives) are tested for polygon membership using an inclusive-boundary point-in-polygon test. Directional counts use the hysteresis-gated line-crossing logic described above and emit a `line_crossed` event with the crossing direction. The system reports both **live occupancy** (distinct confirmed IDs in the current frame) and **cumulative unique visitors** (distinct confirmed IDs seen since the run started).

### 5.2 Restricted-area intrusion detection
Each restricted zone is evaluated **independently per camera/track**, with a small state machine per (camera, track, zone) triple rather than a single global flag:

- **Entry dwell** — a track must remain inside the zone for a configured duration before an intrusion is *confirmed* (filters people briefly clipping the zone boundary).
- **Exit grace** — a short allowance for missing observations or momentary boundary noise before state is dropped, so a one-frame tracker miss doesn't reset a real, ongoing intrusion.
- **Alert cooldown** — confirmed intrusions re-alert only after a cooldown, preventing alert spam from a person who lingers in the zone.

Entry/exit are visible as lightweight lifecycle events; only the dwell-qualified `restricted_area_confirmed` event is treated as the cooldown-gated alert.

### 5.3 Movement and dwell heatmaps
This is a **people-movement heatmap** (occupancy/dwell aggregated over tracked foot-points), explicitly not a network-activation/saliency heatmap. Two parallel grids are maintained:

- **Occupancy** — a count of confirmed position samples per grid cell.
- **Dwell (seconds)** — timestamp-derived elapsed time attributed to a track's previous confirmed cell, so slow-moving/stationary presence accumulates more weight than a quick pass-through.

An **image-space grid** is always available (mapped over the raw frame); a second **ground-plane grid** becomes available only with valid calibration, using either configured real-world bounds or the extents of the calibration correspondences themselves. Long gaps in observation (beyond a configurable threshold) are excluded from dwell time rather than silently counted as "present," and idle tracks are evicted from memory after a timeout — this keeps memory bounded on long-running streams. Aggregation supports either running totals or fixed-size tumbling time windows. For readability, raw per-cell counts are Gaussian-smoothed into a density surface for the rendered colormap while the underlying CSV export stays exact/unsmoothed. Each snapshot is additionally partitioned into a 3×4 grid of 12 regions, and the three highest-average-occupancy regions are reported and highlighted — a fast way to answer "which part of the space is busiest" without reading a full grid.

### 5.4 Queue analytics
Two independent queue-detection strategies are implemented:

**Configured mode** — a manually drawn polygon plus a service point, with heuristic membership rules: a track becomes a *candidate* only while its foot point is inside the polygon and its smoothed image-speed is below a maximum (i.e., it looks like queueing motion, not just passing through); it becomes a confirmed *member* after a minimum dwell time. `service_completion_radius` (a normalized distance from the service point) determines whether a track leaving the zone is counted as *served* versus simply having left. State tolerates short tracking gaps but reacts immediately to an explicitly observed polygon exit.

**Vertical mode (automatic, default)** — no manual polygon is required. Confirmed people whose bounding-box-center X positions fall within a configurable horizontal distance are grouped into a "row," under the geometric assumption that a real queue in the camera's view tends to line up roughly vertically in the image. Row groupings are recomputed every frame and matched to the previous frame's rows by proximity so IDs and on-screen colors stay stable, and groups smaller than a minimum size are discarded as noise. This is explicitly documented as a deliberately small heuristic — it detects *a vertically-aligned cluster*, not a verified real-world queue.

Both modes report a **raw count** (dwell-qualified active members) and an **exponentially-smoothed count** (`count_smoothing_alpha`-controlled EMA, to stop the displayed number flickering frame to frame), plus overflow events that fire only on a boolean state transition (crossing an `overflow_threshold`) rather than every frame while over threshold.

### 5.5 Speed estimation
Speed is computed over a **bounded timestamp window** (not frame count), so it stays correct under variable frame rate, dropped frames, or a stride-sampled dashboard job. Two independent jump-rejection checks (image-space and, when calibrated, ground-space) discard implausible instantaneous displacements — e.g., a tracker ID switch — instead of letting them corrupt the smoothed velocity. Motion under a minimum-displacement threshold is reported as exactly stationary rather than noisy near-zero values. Two units are always kept distinct in the data model: `speed_pixels_per_second` is always available and always labeled `px/s`; `speed_metres_per_second` exists only when a valid metric homography is present, and the system never silently presents an image-space estimate as if it were physical. Queue analytics reuse the same velocity estimate to report **signed progress-toward-service-point speed** — positive when a queue member is actually advancing, negative when drifting away, zero for purely lateral motion.

## 6. Event model, storage, and live dashboard integration

Every analytic emits through one shared `Event` envelope (so downstream consumers don't need per-analytic parsing logic), persisted append-only to JSONL (SQLite is a planned later phase). A FastAPI service exposes recorded-video jobs and live RTSP jobs through a shared job runner:

- Job lifecycle: create → queue → process → cancel-at-next-frame-boundary
- Live progress via **Server-Sent Events** (`metrics_updated`, `progress_updated`, `warning`, `job_completed`/`failed`/`cancelled`), with stable, reconnectable event IDs
- A throttled MJPEG-style preview endpoint (0.2 s interval, capped-width JPEG) so the dashboard shows near-live frames without accumulating unbounded frame buffers in memory
- A **metric schema** published per application (key/label/type/unit/aggregation/display) so the dashboard can render generic cards/counters/charts without hardcoding per-application knowledge of what a "metric" is

Dashboard jobs additionally downscale processing resolution (capped width, aspect-preserved, never upscaled) and frame-stride-sample the source (process every Nth frame, default 5) purely as a *dashboard responsiveness* trade-off — direct batch CLI runs keep full resolution and stride 1 by default, so offline analytical accuracy is not affected by the live-preview trade-off.

---

# Part II — Perception Model R&D Library

Everything in this part lives under `CV_models/` as an independent, largely self-contained pipeline (own preprocessing, own ONNX export, own demo script). They are grouped below by problem domain rather than by build order, since — unlike Part I's pipeline — there is no execution sequence between them.

## Face domain

### Face detection — RetinaFace

| | |
|---|---|
| **Task** | Face localization + 5-point landmark regression |
| **Architecture** | RetinaFace, MobileNetV2 backbone (`retinaface_mnet_v2`) |
| **Output** | Bounding box + 5 facial landmarks (eyes, nose, mouth corners) per face |

**Method.** RetinaFace is a single-stage, anchor-based multi-task detector: the same feature-pyramid backbone simultaneously predicts a face/no-face classification, a bounding-box regression, and a 5-point landmark regression at every anchor location, trained jointly. The landmark output is not cosmetic — it is what makes downstream face alignment (below) possible without a second landmark model. The MobileNetV2 backbone variant trades some accuracy for a much smaller compute footprint, which is the right trade-off for a stage that runs once per frame before every downstream face task.

### Face recognition — ArcFace embeddings (buffalo_l / IResNet-50)

| | |
|---|---|
| **Task** | Face verification / identification |
| **Architecture** | IResNet-50, ArcFace angular-margin loss, trained on WebFace600K (InsightFace `buffalo_l` pack, `w600k_r50.onnx`) |
| **Input** | 112×112 aligned face crop |
| **Output** | 512-dimensional L2-normalized embedding |

**Method.** The RetinaFace landmarks are used to **align** each detected face to a canonical 5-point reference template (an affine warp to a fixed 112×112 pose) before embedding — this normalizes out in-plane rotation and scale so the embedding network only has to encode identity, not pose. ArcFace's contribution is at training time: instead of a standard softmax classification loss, it adds an angular margin between the true identity's angle and the decision boundary, which pushes embeddings of the same identity to cluster tighter and different identities further apart *on the unit hypersphere* — this is what makes plain cosine similarity between two embeddings a reliable same/different-person signal at inference time, with no classifier network needed for new identities (a new person just needs one enrolled embedding, not retraining). The workspace notes `buffalo_l` (IResNet-50) as a deliberate upgrade over a smaller MobileNetV2-ArcFace baseline specifically for robustness to sunglasses and off-axis viewing angles — a direct accuracy/compute trade-off decision.

### Face occlusion classification

| | |
|---|---|
| **Task** | Binary classification: occluded vs. non-occluded face |
| **Architecture (deployed)** | ConvNeXt-Tiny |
| **Input** | 224×224 RGB crop |
| **Reported accuracy** | 98.67% / F1 0.986 on held-out test split (30-epoch training run) |
| **Pipeline integration** | YOLOv8 human detector → per-person crop → occlusion classifier |

**Method.** Rather than a face-detector-first pipeline, the deployed pipeline runs the person-detector (Part I's family) and classifies occlusion on the *human* box region, trading precise face localization for speed and robustness when a face detector might fail on a partially-covered face. The model family was benchmarked broadly (VGG16/19, DenseNet, ResNet, ConvNeXt at multiple sizes) on a 9,749-image crawled-and-labeled dataset; ConvNeXt-Tiny was selected as the deployed model because it lands at the top of the accuracy table (98.67%) with roughly a third of the parameters of the ConvNeXt-Base/VGG alternatives — a modernized convolutional architecture (patchify stem, large depthwise kernels, LayerNorm, inverted bottlenecks — design ideas ported from vision transformers back into a pure CNN) that gets transformer-competitive accuracy at CNN-level inference cost.

### Face pose-angle estimation

Two distinct methods exist in this workspace for the same underlying problem (head orientation), representing two different points on the accuracy/simplicity trade-off:

**Method A — geometric landmark-ratio estimation (reference implementation).**
Given detected landmarks (MTCNN or RetinaFace), roll is derived from the eye-line angle, yaw from the horizontal asymmetry between eye and nose-tip positions, and pitch from where the nose-tip falls on the vertical span between the eye-line and mouth-line (subdivided into ratio units). This method requires **no training and no additional neural network** — it's pure geometry over already-available landmarks — at the cost of reporting normalized pixel-ratio scores rather than calibrated degrees in its simplest form. A second, angle-calibrated variant of the same geometric idea reports true degrees (±90°) for roll/yaw/pitch, benchmarked against the NIST Face Image Quality Assessment reference methodology.

**Method B — learned regression (production pipeline: `human_facepose_pipeline.py`).**

| | |
|---|---|
| **Upstream** | YOLOv8 human detector locates the person; a face region is cropped by fixed anthropometric ratio of the person box (no separate face detector call) |
| **Architecture** | MobileNetV2 backbone regressor |
| **Output** | 3×3 rotation matrix → decomposed into Euler angles (roll, yaw, pitch) via standard `atan2` extraction |

This is a continuous-rotation-representation approach (in the style of 6DRepNet/WHENet-class models): rather than regressing yaw/pitch/roll angles directly — which have discontinuities at wrap-around points that are hard for a network to learn — the network regresses a rotation matrix (or an equivalent continuous 6D representation), which is then converted to Euler angles analytically as a deterministic post-processing step. Cropping the face by ratio from the *person* box, instead of running a dedicated face detector, is a deliberate latency trade-off: it reuses the person detector that the pipeline already runs, at the cost of a coarser, geometry-assumed face region rather than a tightly localized one.

## Human domain

### Human detection — model family

Three purpose-differentiated variants exist for the same underlying task, each optimized for a different deployment point:

| Variant | Architecture | Target | Notable spec |
|---|---|---|---|
| **YOLOv8-human** | Single-stage, anchor-free, decoupled detection head | General GPU/CPU server inference | Trained on COCO + CrowdHuman person class; family exported to ONNX for `HumanDetection_light_input_640.onnx` used by Part I |
| **UHD (Ultra-lightweight Human Detection)** | Custom compact anchor-based detector (8 anchors), optional Efficient-SE channel attention, IoU-aware branch, optional knowledge distillation | Extreme edge (microcontroller-class) | 64×64 input; variants from 0.13M to 31M parameters; sub-millisecond CPU latency at the small end; dedicated ESP32-S3 / INT8 (`.espdl`) export path |
| **yolov8-jetson** | YOLOv8 (Ultralytics) packaged for edge GPU | NVIDIA Jetson Nano | Flask-served streaming inference, dataset from a purpose-built crowd-detection collection |

**Why three variants.** This is a deliberate spread across the accuracy/latency/footprint trade-off curve rather than duplication: YOLOv8-human is the accuracy-first server-class option (and the one whose family feeds Part I's production detector); UHD exists specifically because "the number of parameters does not correlate to inference speed" for the target use case — for a fixed low input resolution, a purpose-shrunk architecture beats a scaled-down general detector on real CPU/microcontroller latency; the Jetson variant addresses the operational constraints of running on an embedded GPU (packaging, streaming, camera I/O) rather than the model architecture itself.

### Human pose estimation — RTMPose / ViTPose (via `rtmlib`)

| | |
|---|---|
| **Task** | 2D keypoint estimation (body-17 up to whole-body-133: body + feet + hands + face) |
| **Pipeline pattern** | Top-down, two-stage: person detector (YOLOX or RTMDet) → per-person crop → pose estimator |
| **Detector** | YOLOX (multiple sizes, nano→x) or RTMDet |
| **Pose estimator options** | RTMPose (CSPNeXt-based, SimCC coordinate-classification head) · RTMO (one-stage, no separate detector) · ViTPose (ViT transformer backbone, heatmap decoder) |
| **Runtime** | ONNX Runtime / OpenCV DNN / OpenVINO backend, CPU or GPU, no mmpose/mmdet/mmcv dependency |
| **Models present in workspace** | `rtmpose-s_256x192.onnx`, `vitpose-h-apt36k.onnx` (animal-pose variant) |

**Method — top-down two-stage estimation.** A general-purpose detector first finds each person's box; the pose network then runs on the cropped, resized person region rather than the full frame. This standard top-down decomposition trades a second network pass per person for much higher keypoint accuracy than a single whole-frame regression, since the pose network's receptive field and normalization are matched to "one roughly-centered person" rather than an arbitrary scene.

**Method — SimCC (RTMPose's head).** Rather than the classic heatmap regression (predict a 2D Gaussian probability map per keypoint, then find its peak), RTMPose reformulates each keypoint's x and y coordinate as a **classification problem over finely discretized horizontal and vertical position bins** ("SimCC" — simple coordinate classification). This avoids the heatmap approach's resolution/computation trade-off (a finer heatmap costs more compute) and gives RTMPose its favorable speed/accuracy ratio, which is why it is the "lightweight, no heavy dependency" option chosen here over the more classical mmpose stack.

**Method — ViTPose (alternative backbone).** ViTPose instead uses a plain, non-hierarchical Vision Transformer backbone with a lightweight deconvolution decoder head producing conventional heatmaps — architecturally simpler than most CNN pose networks (no task-specific structural priors), and its accuracy scales cleanly with backbone size (S/B/L/H variants). The workspace's `vitpose-h-apt36k.onnx` is trained on animal pose data (APT-36K), indicating the same estimator architecture is reused for animal, not just human, keypoint estimation — one architecture, swappable weights per subject class.

**Method — RTMO (one-stage alternative, listed for completeness).** Where RTMPose requires a separate detector pass per person, RTMO estimates all people's poses in a single forward pass over the whole frame, trading some per-person accuracy for eliminating the detector-crop step entirely — relevant when frame throughput matters more than peak per-person keypoint precision.

### Human pose / action classification

Two independent, purpose-differentiated pipelines cover single-frame and temporal action understanding:

**Pipeline 1 — pose-aware single-frame classification.**

| | |
|---|---|
| **Stage 1** | MediaPipe Pose → 33 body keypoints |
| **Stage 2** | Geometric pose classifier over the keypoints → coarse pose class (sitting / standing / lying) |
| **Stage 3** | 2D CNN appearance classifier (ResNet50 best-performing, MobileNetV3/ResNet18/34 alternatives, any `timm` backbone) → 40-class Stanford40 action label |
| **Reported accuracy / speed** | 88.5% (ResNet50) on Stanford40; ~11 ms end-to-end (~90 FPS) on an RTX 4070-class GPU |

This is a **dual-stream** design: the geometric stage captures coarse body configuration cheaply (keypoints, not pixels), while the CNN stream captures fine-grained appearance cues (object interaction, hand pose detail) that pure skeleton geometry can't see — the two are fused for the final action label rather than relying on either signal alone. The reported ~127× speedup versus an earlier OpenPose+TensorFlow-1.x version of the same idea (1400 ms → 11 ms) is attributed to swapping OpenPose (a heavier, exhaustive part-affinity-field pose method) for MediaPipe's regression-based, mobile-oriented pose model, plus the PyTorch/timm backbone swap.

**Pipeline 2 — temporal video-clip classification.**

| | |
|---|---|
| **Input** | 16-frame clips, 112×112, standard-normalized |
| **Architecture** | 3D CNN: MC3-18 (mixed 3D/2D convolutions) or R3D-18 (fully 3D convolutions) |
| **Pretraining → fine-tuning** | Kinetics-400 → UCF-101 (101 classes) / HMDB51 (51 classes) |
| **Reported accuracy** | MC3-18: 87.05% (UCF-101, exceeding the original paper's 85.0%); R3D-18: 83.80% |

**Method.** Unlike Pipeline 1, this operates on **spatiotemporal volumes** rather than single frames: 3D convolution kernels extend across the time axis as well as height/width, so the network directly learns motion patterns (not just a sequence of independent per-frame appearance guesses). MC3-18 ("Mixed Convolutions") uses full 3D convolutions only in early layers and switches to cheaper 2D convolutions in later layers — based on the empirical finding that temporal reasoning matters most on low-level motion features and less on high-level semantic features — which is why it both outperforms and outruns the fully-3D R3D-18 in this workspace's benchmarks. Both start from Kinetics-400 pretraining (a large-scale video-action dataset) before fine-tuning on the smaller target datasets, standard transfer learning for video, since 3D CNNs are notably data-hungry to train from scratch.

## Vehicle domain

### License plate detection and recognition

Two independent implementations exist, differing in how the character-recognition stage is solved:

**English/general pipeline.**

| Stage | Method |
|---|---|
| Vehicle detection | YOLOv8n, COCO-pretrained (`car` class) |
| Plate detection | YOLOv8, fine-tuned on a dedicated Roboflow license-plate dataset |
| Cross-frame tracking | SORT (Kalman-filter motion prediction + Hungarian-algorithm IoU assignment) |
| Character recognition | EasyOCR (CRNN-family: CNN feature extractor + recurrent sequence model + CTC decoding) |
| Temporal smoothing | Missing-frame interpolation pass over the per-frame detection/plate-text CSV before final visualization |

**Method.** Vehicle detection first constrains the search region before plate detection runs, reducing false positives from plate-shaped clutter elsewhere in the scene. SORT tracks each plate across frames (motion-only, no appearance model — deliberately lightweight compared to Part I's ByteTrack) so that OCR results can be temporally smoothed and missing-frame gaps interpolated rather than trusting any single frame's noisy OCR read.

**Persian plate pipeline (`Persian_Plate_Recognition`).**

| Stage | Method |
|---|---|
| Vehicle + plate detection | YOLOv8s, trained on a purpose-built Persian car/plate dataset |
| Character recognition | YOLOv8n, trained to detect **each Persian character/digit as its own object class** on the cropped plate |
| String assembly | Detected characters sorted by horizontal (x) position to reconstruct reading order |

**Method — detection-based OCR instead of a sequence model.** Rather than a CRNN/sequence-decoding OCR (as in the English pipeline), Persian plate text is recognized by treating **character recognition as object detection**: a second YOLOv8 model is trained with one class per Persian character/digit, and the plate string is simply the detected characters ordered left-to-right. This sidesteps needing a Persian-script-aware sequence-recognition model or training data at the cost of requiring every character shape to be a distinct, well-represented training class — a reasonable trade given the small, fixed alphabet of plate characters (a closed, sign-like symbol set) versus general free-form text.

## Safety / PPE domain

### Mask detection

| | |
|---|---|
| **Architecture** | Anchor-based SSD-style detector (ported from AIZOOTech's FaceMaskDetection design) |
| **Input** | 360×360 RGB |
| **Feature pyramid** | 5 levels, feature-map sizes 45²→4², 1–3 anchor aspect ratios per level |
| **Output classes** | `Mask`, `NoMask` |
| **Post-processing** | Variance-scaled box decoding (offsets relative to each anchor, not absolute coordinates) + custom greedy NMS |

**Method.** This is a single-stage detector in the SSD (Single Shot Detector) family: predefined anchor boxes are tiled across five feature-map resolutions (finer maps catch small/near faces, coarser maps catch large/close faces), and the network predicts a class score plus a box offset relative to each anchor rather than an absolute box — a formulation that constrains the regression problem and is characteristic of anchor-based single-stage detectors predating the anchor-free YOLO designs used elsewhere in this workspace. It directly outputs a per-face Mask/NoMask decision (not a separate face-detection-then-classification pipeline), since mask-wearing is treated here as a detection-time attribute of the face box rather than a downstream classification task.

### Weapon (gun/knife) detection

| | |
|---|---|
| **Architecture** | YOLOv8, single-stage |
| **Classes** | `guns`, `knife` (2-class) |
| **Input** | 640×640, letterboxed |
| **Runtime** | ONNX Runtime, CUDA with CPU fallback |

**Method.** Standard single-stage YOLOv8 decoding: dense center/width/height/confidence predictions per class, NMS-collapsed to final boxes, letterbox-corrected back to source-image coordinates. Framed as a 2-class problem (rather than one binary "weapon" class) so that downstream alerting logic can distinguish firearm versus edged-weapon detections if response protocols differ.

### Helmet detection

| | |
|---|---|
| **Architecture** | YOLOv8, single-stage |
| **Input** | Configurable size (default matches standard YOLOv8 export), letterboxed |
| **Runtime** | ONNX Runtime, CPU |

**Method.** Same YOLOv8 single-stage detection/decoding pattern as the weapon detector — the workspace consistently reuses one detector architecture family (YOLOv8) across every "is this specific object present in this box" PPE/safety task, varying only the training data and class set. This is a deliberate architectural economy: one decode/NMS/letterbox implementation is shared conceptually across mask, weapon, and helmet detection, differing only in weights and class labels.

---

## Cross-cutting technical notes

**Why ONNX everywhere.** Every model in both parts of this report — regardless of original training framework (PyTorch, TensorFlow/MediaPipe-adjacent, or custom) — is converted to ONNX before deployment, and every inference path in the workspace runs through ONNX Runtime rather than the training framework's own runtime. This decouples training-time framework choice from deployment: the production pipeline only needs one inference dependency (ONNX Runtime) regardless of how many different research pipelines feed it models, and the same execution-provider mechanism (CPU ⇄ CUDA ⇄ OpenVINO, with fallback) is available uniformly.

**Detector reuse as an integration pattern.** Several Part II pipelines (face occlusion, face pose, and implicitly others) do not run their own person/face localizer — they reuse the YOLOv8-human detector's output region and crop by a fixed ratio. This is the same "run the expensive detector once, derive many analytics from its output" principle that underlies Part I's entire analytics suite, applied one level down inside individual model pipelines.

**What's integration-ready vs. integrated today.** Part I's live pipeline currently consumes exactly one Part II-family model (the person detector) plus the tracker. Every other Part II capability — face recognition, pose estimation, plate recognition, PPE detection — is ONNX-exported and architecturally pluggable into the same `Detection`/`TrackObservation` contract pattern, but none is currently wired into `app/`. Extending the platform with, say, restricted-area weapon alerts or masked-entry compliance would mean adding a new detector adapter under `app/detection/`, not a new architecture.
