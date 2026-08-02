"""Automatic vertical-row queue grouping tests."""

from __future__ import annotations

import numpy as np

from app.analytics import (
    VerticalQueueAnalyzer,
    VerticalQueueConfig,
    annotate_vertical_queues,
    hotter_row_color,
    vertical_row_color,
)
from app.core.models import TrackObservation


def _observation(
    track_id: int,
    center: tuple[float, float],
    timestamp: float = 0.0,
    *,
    confirmed: bool = True,
    speed_pixels: float | None = None,
    speed_metres: float | None = None,
) -> TrackObservation:
    x, y = center
    return TrackObservation(
        camera_id="cam-a",
        track_id=track_id,
        timestamp=timestamp,
        frame_index=int(timestamp * 10),
        xyxy=(x - 4, y - 12, x + 4, y + 12),
        foot_point=(x, y + 12),
        detection_confidence=0.9,
        confirmed=confirmed,
        trajectory=(),
        speed_pixels_per_second=speed_pixels,
        speed_metres_per_second=speed_metres,
    )


def _analyzer(*, distance: float = 0.08, minimum: int = 2) -> VerticalQueueAnalyzer:
    return VerticalQueueAnalyzer(
        (
            VerticalQueueConfig(
                "cam-a",
                (101, 101),
                maximum_center_distance_fraction=distance,
                minimum_people=minimum,
            ),
        )
    )


def test_people_with_nearby_bbox_centers_form_independent_vertical_rows() -> None:
    analyzer = _analyzer(distance=0.08)

    snapshot = analyzer.update(
        "cam-a",
        (
            _observation(1, (20, 25)),
            _observation(2, (23, 65)),
            _observation(3, (70, 30)),
            _observation(4, (75, 70)),
            _observation(5, (50, 50)),
            _observation(6, (21, 45), confirmed=False),
        ),
        timestamp=0,
    )

    assert len(snapshot.rows) == 2
    assert snapshot.rows[0].track_ids == (1, 2)
    assert snapshot.rows[1].track_ids == (3, 4)
    assert snapshot.row_for_track(5) is None
    assert snapshot.row_for_track(6) is None


def test_row_ids_and_colors_remain_stable_during_small_motion() -> None:
    analyzer = _analyzer()
    first = analyzer.update(
        "cam-a",
        (_observation(1, (20, 30)), _observation(2, (23, 70))),
        timestamp=0,
    )
    second = analyzer.update(
        "cam-a",
        (_observation(1, (22, 30), 1), _observation(2, (25, 70), 1)),
        timestamp=1,
    )

    assert first.rows[0].row_id == second.rows[0].row_id
    assert vertical_row_color(first.rows[0].row_id) == vertical_row_color(
        second.rows[0].row_id
    )


def test_each_vertical_queue_exposes_average_member_speed() -> None:
    analyzer = _analyzer()

    snapshot = analyzer.update(
        "cam-a",
        (
            _observation(1, (20, 30), speed_pixels=10, speed_metres=1),
            _observation(2, (23, 70), speed_pixels=20, speed_metres=2),
            _observation(3, (70, 30), speed_pixels=4),
            _observation(4, (73, 70), speed_pixels=8),
        ),
        timestamp=1,
    )

    assert snapshot.rows[0].average_speed_pixels_per_second == 15
    assert snapshot.rows[0].average_speed_metres_per_second == 1.5
    assert snapshot.rows[1].average_speed_pixels_per_second == 6
    assert snapshot.rows[1].average_speed_metres_per_second is None


def test_person_switching_rows_takes_the_destination_row_color() -> None:
    analyzer = _analyzer()
    first_observations = (
        _observation(1, (20, 25)),
        _observation(2, (22, 65)),
        _observation(3, (70, 25)),
        _observation(4, (72, 65)),
    )
    first = analyzer.update("cam-a", first_observations, timestamp=0)
    source_row = first.row_for_track(2)
    destination_row = first.row_for_track(3)
    assert source_row is not None and destination_row is not None

    second_observations = (
        _observation(1, (20, 25), 1),
        _observation(2, (69, 45), 1),
        _observation(3, (70, 25), 1),
        _observation(4, (72, 65), 1),
    )
    second = analyzer.update("cam-a", second_observations, timestamp=1)
    moved_row = second.row_for_track(2)

    assert moved_row is not None
    assert moved_row.row_id == destination_row.row_id
    assert vertical_row_color(moved_row.row_id) != vertical_row_color(
        source_row.row_id
    )


def test_overlay_draws_row_line_and_same_color_member_boxes() -> None:
    analyzer = _analyzer()
    observations = (
        _observation(1, (20, 30)),
        _observation(2, (23, 65)),
    )
    snapshot = analyzer.update("cam-a", observations, timestamp=0)
    color = vertical_row_color(snapshot.rows[0].row_id)

    annotated = annotate_vertical_queues(
        np.zeros((101, 101, 3), dtype=np.uint8), snapshot, observations
    )

    assert tuple(annotated[18, 16]) == color
    assert tuple(annotated[53, 19]) == color
    line_x = int(round(snapshot.rows[0].center_x))
    line_y = 47
    assert tuple(annotated[line_y, line_x]) == color
    hot_color = np.asarray(hotter_row_color(color), dtype=np.uint8)
    line_region = annotated[line_y, line_x - 7 : line_x + 8]
    assert np.any(np.all(line_region == hot_color, axis=1))
    assert tuple(annotated[0, line_x]) == color
    assert tuple(annotated[74, line_x]) == color
    assert tuple(annotated[90, 100]) == (20, 20, 20)


def test_reset_discards_rows_and_restarts_identity_sequence() -> None:
    analyzer = _analyzer()
    observations = (_observation(1, (20, 30)), _observation(2, (23, 60)))
    assert analyzer.update("cam-a", observations, timestamp=0).rows[0].row_id == 1

    analyzer.reset("cam-a")

    assert analyzer.snapshot("cam-a").rows == ()
    assert analyzer.update("cam-a", observations, timestamp=0).rows[0].row_id == 1
