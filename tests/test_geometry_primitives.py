"""Tests for resolution and line/polygon geometry primitives."""

import pytest

from app.geometry.config import NormalizedPoint
from app.geometry.primitives import (
    LineCrossingDirection,
    detect_line_crossing,
    line_side,
    point_in_polygon,
    validate_polygon,
)


def test_normalized_coordinates_map_to_inclusive_frame_pixels() -> None:
    assert NormalizedPoint(0.0, 0.0).to_pixels((1920, 1080)) == (0.0, 0.0)
    assert NormalizedPoint(1.0, 1.0).to_pixels((1920, 1080)) == (1919.0, 1079.0)
    assert NormalizedPoint(0.5, 0.25).to_pixels((101, 201)) == (50.0, 50.0)


def test_polygon_boundary_points_are_configurable() -> None:
    square = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    assert point_in_polygon((0.5, 0.5), square)
    assert point_in_polygon((1.0, 0.5), square)
    assert point_in_polygon((0.0, 0.0), square)
    assert not point_in_polygon((1.0, 0.5), square, include_boundary=False)
    assert not point_in_polygon((1.1, 0.5), square)


@pytest.mark.parametrize(
    "polygon",
    [
        ((0.0, 0.0), (1.0, 0.0)),
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
        ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)),
        ((0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
    ],
)
def test_invalid_polygons_are_rejected(
    polygon: tuple[tuple[float, float], ...]
) -> None:
    with pytest.raises(ValueError):
        validate_polygon(polygon)


def test_directed_line_side_and_crossing_change() -> None:
    start, end = (0.0, 0.0), (10.0, 0.0)
    assert line_side((5.0, -1.0), start, end) == -1
    assert line_side((5.0, 0.0), start, end) == 0
    assert line_side((5.0, 1.0), start, end) == 1

    crossing = detect_line_crossing((5.0, -2.0), (5.0, 2.0), start, end)
    assert crossing.crossed
    assert crossing.direction is LineCrossingDirection.NEGATIVE_TO_POSITIVE
    assert crossing.intersection == pytest.approx((5.0, 0.0))


def test_crossing_requires_finite_segment_and_completed_side_change() -> None:
    line = ((0.0, 0.0), (10.0, 0.0))
    assert not detect_line_crossing((-5.0, -1.0), (-5.0, 1.0), *line).crossed
    assert not detect_line_crossing((5.0, -1.0), (5.0, 0.0), *line).crossed
    assert not detect_line_crossing((2.0, 0.0), (8.0, 0.0), *line).crossed
    resumed = detect_line_crossing(
        (5.0, 0.0), (5.0, 1.0), *line, previous_stable_side=-1
    )
    assert resumed.crossed
    assert resumed.direction is LineCrossingDirection.NEGATIVE_TO_POSITIVE
