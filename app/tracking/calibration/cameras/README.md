# Per-camera UCMCTrack geometry

UCMCTrack runs **without** these files. Add a camera later by dropping a YAML
or official `cam_para` text file here (or any directory pointed to by
`tracker.camera_geometry_dir`) whose `camera_id` (or filename stem) matches the
runtime camera id.

Supported YAML shapes:

1. Official intrinsic/extrinsic (same quantities as `cam_para` text files)
2. 3×3 homography
3. Existing image↔ground correspondences (`calibration` + `frame_size`)
4. `uncalibrated: true`

See `example.yaml` and `example_cam_para.txt`.
