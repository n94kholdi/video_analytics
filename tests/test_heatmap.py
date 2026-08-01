"""Movement occupancy/dwell heatmap accumulation and export tests."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.analytics.heatmap import (
    HeatmapAccumulator,
    HeatmapGrid,
    HeatmapVideoWriter,
    MovementHeatmaps,
    colorize_heatmap,
    export_heatmap_snapshot,
)
from app.core.models import TrackObservation
from app.analytics.cli import build_parser
from app.geometry.config import CameraConfig


def observation(
    track_id: int,
    timestamp: float,
    point: tuple[float, float],
    *,
    confirmed: bool = True,
    camera_id: str = "cam",
) -> TrackObservation:
    return TrackObservation(
        camera_id=camera_id,
        track_id=track_id,
        timestamp=timestamp,
        frame_index=int(timestamp * 10),
        xyxy=(point[0] - 1, point[1] - 2, point[0] + 1, point[1]),
        foot_point=point,
        detection_confidence=0.9,
        confirmed=confirmed,
        trajectory=(),
    )


def accumulator(**overrides: float | None) -> HeatmapAccumulator:
    options = {
        "max_sample_gap_seconds": 5.0,
        "track_idle_seconds": 10.0,
        **overrides,
    }
    return HeatmapAccumulator(HeatmapGrid((4, 2), (0.0, 0.0, 40.0, 20.0)), **options)


@pytest.mark.parametrize(
    ("point", "cell"),
    [
        ((0.0, 0.0), (0, 0)),
        ((9.99, 9.99), (0, 0)),
        ((10.0, 10.0), (1, 1)),
        ((40.0, 20.0), (1, 3)),
        ((-0.01, 10.0), None),
        ((40.01, 10.0), None),
    ],
)
def test_grid_cell_mapping_and_boundaries(
    point: tuple[float, float], cell: tuple[int, int] | None
) -> None:
    assert HeatmapGrid((4, 2), (0.0, 0.0, 40.0, 20.0)).cell_for(point) == cell


def test_occupancy_counts_samples_but_dwell_uses_elapsed_timestamps() -> None:
    heatmap = accumulator()

    heatmap.update([observation(1, 10.0, (5.0, 5.0))])
    snapshot = heatmap.update([observation(1, 10.75, (15.0, 5.0))])

    assert snapshot.occupancy.sum() == 2
    assert snapshot.occupancy[0, 0] == 1
    assert snapshot.occupancy[0, 1] == 1
    assert snapshot.dwell_seconds.sum() == pytest.approx(0.75)
    assert snapshot.dwell_seconds[0, 0] == pytest.approx(0.75)


def test_multiple_tracks_accumulate_independently_and_unconfirmed_are_ignored() -> None:
    heatmap = accumulator()
    heatmap.update(
        [
            observation(1, 2.0, (5.0, 5.0)),
            observation(2, 2.0, (25.0, 15.0)),
            observation(3, 2.0, (35.0, 15.0), confirmed=False),
        ]
    )
    snapshot = heatmap.update(
        [observation(1, 3.5, (5.0, 5.0)), observation(2, 3.5, (25.0, 15.0))]
    )

    assert snapshot.occupancy.sum() == 4
    assert snapshot.dwell_seconds.sum() == pytest.approx(3.0)
    assert snapshot.occupancy[1, 3] == 0


def test_reset_clears_grids_and_prevents_dwell_bridge() -> None:
    heatmap = accumulator()
    heatmap.update([observation(1, 1.0, (5.0, 5.0))])

    heatmap.reset(timestamp=2.0)
    snapshot = heatmap.update([observation(1, 3.0, (5.0, 5.0))])

    assert snapshot.occupancy.sum() == 1
    assert snapshot.dwell_seconds.sum() == 0.0
    assert snapshot.window_started_at == 2.0


def test_tumbling_window_and_idle_eviction_bound_state() -> None:
    heatmap = accumulator(aggregation_window_seconds=2.0, track_idle_seconds=1.0)
    heatmap.update([observation(1, 0.0, (5.0, 5.0))])
    heatmap.update([], timestamp=1.5)
    assert not heatmap._tracks  # bounded state is part of this phase's contract

    snapshot = heatmap.update([observation(2, 2.0, (15.0, 5.0))])
    assert snapshot.occupancy.sum() == 1
    assert snapshot.window_started_at == 2.0


def camera_mapping(*, calibrated: bool) -> dict[str, object]:
    mapping: dict[str, object] = {
        "camera": {"id": "cam", "name": "Camera", "source": 0},
        "analytics": {"enabled": ["heatmap"]},
        "heatmap": {
            "grid_size": [2, 2],
            "ground_grid_size": [2, 2],
            "max_sample_gap_seconds": 5.0,
        },
    }
    if calibrated:
        mapping["calibration"] = {
            "image_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "ground_points": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "ground_unit": "metres",
        }
    return mapping


def test_missing_calibration_keeps_image_heatmap_and_reports_ground_unavailable() -> None:
    heatmaps = MovementHeatmaps.from_camera_config(
        CameraConfig.from_mapping(camera_mapping(calibrated=False)), (101, 101)
    )

    snapshot = heatmaps.update([observation(1, 0.0, (50.0, 50.0))])

    assert snapshot.image.occupancy.sum() == 1
    assert snapshot.ground is None
    assert snapshot.ground_unavailable_reason == "camera calibration is not configured"


def test_ground_plane_grid_uses_synthetic_calibration() -> None:
    heatmaps = MovementHeatmaps.from_camera_config(
        CameraConfig.from_mapping(camera_mapping(calibrated=True)), (101, 101)
    )

    heatmaps.update([observation(1, 4.0, (50.0, 50.0))])
    snapshot = heatmaps.update([observation(1, 5.25, (50.0, 50.0))])

    assert snapshot.ground is not None
    assert snapshot.ground.unit == "metres"
    assert snapshot.ground.occupancy[1, 1] == 2
    assert snapshot.ground.dwell_seconds[1, 1] == pytest.approx(1.25)


def test_exports_have_deterministic_dimensions(tmp_path: Path) -> None:
    heatmap = accumulator()
    snapshot = heatmap.update([observation(1, 0.0, (5.0, 5.0))])
    reference = np.full((60, 80, 3), 127, dtype=np.uint8)

    paths = export_heatmap_snapshot(
        snapshot,
        tmp_path,
        prefix="cam_image",
        color_map="viridis",
        opacity=0.25,
        output_size=(80, 60),
        reference_frame=reference,
    )

    numeric = np.loadtxt(paths.occupancy_grid, delimiter=",")
    color = cv2.imread(str(paths.occupancy_image))
    overlay = cv2.imread(str(paths.occupancy_overlay))
    assert numeric.shape == (2, 4)
    assert color.shape == (60, 80, 3)
    assert overlay.shape == (60, 80, 3)


def test_colorized_occupancy_and_dwell_exports_remain_distinct() -> None:
    occupancy = np.asarray([[4, 0], [0, 1]], dtype=np.uint64)
    dwell = np.asarray([[0.1, 0], [0, 3.0]], dtype=np.float64)

    assert not np.array_equal(
        colorize_heatmap(occupancy), colorize_heatmap(dwell)
    )


def test_full_density_surface_is_blue_at_zero_and_smooths_hot_regions() -> None:
    empty = np.zeros((9, 9), dtype=np.float64)
    occupied = empty.copy()
    occupied[4, 4] = 10.0

    blue = colorize_heatmap(empty, color_map="jet", smoothing_sigma_cells=1.0)
    density = colorize_heatmap(
        occupied, color_map="jet", smoothing_sigma_cells=1.0
    )

    assert np.all(blue[:, :, 0] > blue[:, :, 2])  # OpenCV images are BGR.
    changed_cells = np.any(density != blue, axis=2)
    assert changed_cells.sum() > 1


def test_heatmap_cli_is_disabled_until_explicitly_enabled() -> None:
    parser = build_parser()

    assert not parser.parse_args(["input.mp4"]).enable_heatmap
    assert parser.parse_args(["input.mp4", "--enable-heatmap"]).enable_heatmap


def test_evolving_heatmap_videos_have_one_frame_per_snapshot(tmp_path: Path) -> None:
    heatmap = HeatmapAccumulator(
        HeatmapGrid((4, 2), (0.0, 0.0, 40.0, 20.0)),
        max_sample_gap_seconds=5.0,
    )
    video = HeatmapVideoWriter(
        tmp_path,
        prefix="cam_image",
        fps=10.0,
        frame_size=(80, 40),
    )
    frame = np.full((40, 80, 3), 80, dtype=np.uint8)
    try:
        for index, point in enumerate(((5.0, 5.0), (15.0, 5.0), (25.0, 15.0))):
            snapshot = heatmap.update([observation(1, index * 0.1, point)])
            video.write(snapshot, frame)
    finally:
        video.close()

    for path in (video.paths.occupancy, video.paths.dwell):
        capture = cv2.VideoCapture(str(path))
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 80
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 40
        capture.release()
