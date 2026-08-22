"""Tracking CLI and trajectory visualization option tests."""

from __future__ import annotations

import numpy as np

from app.core.models import TrackObservation, TrajectoryPoint
from app.tracking.cli import build_parser
from app.tracking.visualization import annotate_tracks


def observation() -> TrackObservation:
    trajectory = (
        TrajectoryPoint(0.0, 0, (10.0, 20.0), (10.0, 20.0)),
        TrajectoryPoint(0.1, 1, (50.0, 20.0), (50.0, 20.0)),
    )
    return TrackObservation(
        camera_id="cam",
        track_id=1,
        timestamp=0.1,
        frame_index=1,
        xyxy=(45.0, 10.0, 55.0, 20.0),
        foot_point=(50.0, 20.0),
        detection_confidence=0.9,
        confirmed=True,
        trajectory=trajectory,
    )


def test_trajectory_display_is_enabled_by_default() -> None:
    args = build_parser().parse_args(["input.mp4"])

    assert args.show_trajectories is True


def test_no_trajectories_cli_option_disables_display() -> None:
    args = build_parser().parse_args(["input.mp4", "--no-trajectories"])

    assert args.show_trajectories is False


def test_reid_is_explicitly_opt_in() -> None:
    assert build_parser().parse_args(["input.mp4"]).enable_reid is False
    assert build_parser().parse_args(["input.mp4", "--enable-reid"]).enable_reid is True


def test_tracker_cli_accepts_registered_types() -> None:
    assert build_parser().parse_args(["input.mp4"]).tracker is None
    assert build_parser().parse_args(["input.mp4", "--tracker", "stabletrack"]).tracker == "stabletrack"
    assert build_parser().parse_args(["input.mp4", "--tracker", "deepocsort"]).tracker == "deepocsort"


def test_trajectory_line_can_be_hidden_without_hiding_track_annotation() -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    item = observation()
    shown = annotate_tracks(frame, [item], show_trajectories=True)
    hidden = annotate_tracks(frame, [item], show_trajectories=False)

    # Mid-trail is outside the box, label, and foot-point marker.
    assert np.any(shown[20, 30] != 0)
    assert np.all(hidden[20, 30] == 0)
    # The bounding box remains visible in both modes.
    assert np.any(hidden[10, 45] != 0)
