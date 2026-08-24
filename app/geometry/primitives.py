"""Resolution-independent, analytics-neutral geometry operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

Point = tuple[float, float]
_EPSILON = 1e-9


def denormalize_point(point: Point, frame_size: tuple[int, int]) -> Point:
    """Map a normalized point onto inclusive pixel coordinates.

    ``frame_size`` is ``(width, height)``. Therefore normalized corners map to
    ``(0, 0)`` and ``(width - 1, height - 1)``.
    """

    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("frame width and height must be positive")
    x, y = point
    if not all(math.isfinite(value) for value in point):
        raise ValueError("normalized coordinates must be finite")
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        raise ValueError("normalized coordinates must be between 0 and 1")
    return (x * (width - 1), y * (height - 1))


def signed_line_side(point: Point, start: Point, end: Point) -> float:
    """Return the signed 2-D cross product relative to a directed line."""

    return ((end[0] - start[0]) * (point[1] - start[1])) - (
        (end[1] - start[1]) * (point[0] - start[0])
    )


def line_side(
    point: Point,
    start: Point,
    end: Point,
    *,
    epsilon: float = _EPSILON,
) -> int:
    """Classify a point as ``-1``, ``0``, or ``1`` beside a directed line."""

    if _distance_squared(start, end) <= epsilon * epsilon:
        raise ValueError("line endpoints must be distinct")
    value = signed_line_side(point, start, end)
    scale = max(1.0, math.sqrt(_distance_squared(start, end)))
    tolerance = epsilon * scale
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def point_in_polygon(
    point: Point,
    polygon: Sequence[Point],
    *,
    include_boundary: bool = True,
    epsilon: float = _EPSILON,
) -> bool:
    """Test polygon membership with explicit, stable boundary handling."""

    validate_polygon(polygon, epsilon=epsilon)
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current, epsilon):
            return include_boundary
        if (current[1] > point[1]) != (previous[1] > point[1]):
            intersection_x = (
                (previous[0] - current[0])
                * (point[1] - current[1])
                / (previous[1] - current[1])
                + current[0]
            )
            if point[0] < intersection_x:
                inside = not inside
        previous = current
    return inside


def polygon_area(polygon: Sequence[Point]) -> float:
    """Return the unsigned shoelace area of a (possibly degenerate) polygon."""

    if len(polygon) < 3:
        return 0.0
    count = len(polygon)
    twice_area = sum(
        polygon[index][0] * polygon[(index + 1) % count][1]
        - polygon[(index + 1) % count][0] * polygon[index][1]
        for index in range(count)
    )
    return abs(twice_area) / 2.0


def clip_polygon_to_bbox(
    polygon: Sequence[Point], bbox: tuple[float, float, float, float]
) -> list[Point]:
    """Sutherland-Hodgman clip of a simple polygon against an axis-aligned box.

    The clip window is convex, so this holds for arbitrary (including
    non-convex) simple subject polygons.
    """

    x1, y1, x2, y2 = bbox
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)

    def clip_edge(points: list[Point], inside, intersect) -> list[Point]:
        if not points:
            return []
        output: list[Point] = []
        previous = points[-1]
        previous_inside = inside(previous)
        for current in points:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersect(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersect(previous, current))
            previous, previous_inside = current, current_inside
        return output

    def intersect_vertical(a: Point, b: Point, x: float) -> Point:
        t = (x - a[0]) / (b[0] - a[0])
        return (x, a[1] + t * (b[1] - a[1]))

    def intersect_horizontal(a: Point, b: Point, y: float) -> Point:
        t = (y - a[1]) / (b[1] - a[1])
        return (a[0] + t * (b[0] - a[0]), y)

    points = list(polygon)
    points = clip_edge(points, lambda p: p[0] >= left, lambda a, b: intersect_vertical(a, b, left))
    points = clip_edge(points, lambda p: p[0] <= right, lambda a, b: intersect_vertical(a, b, right))
    points = clip_edge(points, lambda p: p[1] >= top, lambda a, b: intersect_horizontal(a, b, top))
    points = clip_edge(points, lambda p: p[1] <= bottom, lambda a, b: intersect_horizontal(a, b, bottom))
    return points


def bbox_polygon_coverage_ratio(
    bbox: tuple[float, float, float, float], polygon: Sequence[Point]
) -> float:
    """Return the fraction of a bbox's area that overlaps a polygon.

    This is intersection-over-bbox-area (not intersection-over-union): a
    region polygon is typically much larger than one person's bbox, so
    union-based IoU would rarely be usable as a membership threshold.
    """

    x1, y1, x2, y2 = bbox
    bbox_area = abs(x2 - x1) * abs(y2 - y1)
    if bbox_area <= 0:
        return 0.0
    clipped = clip_polygon_to_bbox(polygon, bbox)
    return polygon_area(clipped) / bbox_area


def validate_polygon(
    polygon: Sequence[Point], *, epsilon: float = _EPSILON
) -> None:
    """Reject short, zero-area, repeated-edge, and self-intersecting polygons."""

    if len(polygon) < 3:
        raise ValueError("polygon must contain at least three points")
    if not all(
        len(point) == 2 and all(math.isfinite(value) for value in point)
        for point in polygon
    ):
        raise ValueError("polygon points must contain two finite coordinates")

    count = len(polygon)
    for index, point in enumerate(polygon):
        if _distance_squared(point, polygon[(index + 1) % count]) <= epsilon**2:
            raise ValueError("polygon must not contain zero-length edges")

    for first in range(count):
        first_next = (first + 1) % count
        for second in range(first + 1, count):
            second_next = (second + 1) % count
            if first in (second, second_next) or first_next in (second, second_next):
                continue
            if _segments_intersect(
                polygon[first],
                polygon[first_next],
                polygon[second],
                polygon[second_next],
                epsilon,
            ):
                raise ValueError("polygon edges must not self-intersect")

    twice_area = sum(
        polygon[index][0] * polygon[(index + 1) % count][1]
        - polygon[(index + 1) % count][0] * polygon[index][1]
        for index in range(count)
    )
    if abs(twice_area) <= epsilon:
        raise ValueError("polygon area must be non-zero")


class LineCrossingDirection(str, Enum):
    """Side transition across a configured directed line."""

    NEGATIVE_TO_POSITIVE = "negative_to_positive"
    POSITIVE_TO_NEGATIVE = "positive_to_negative"


@dataclass(frozen=True, slots=True)
class LineCrossing:
    """Result from one finite line-crossing test."""

    crossed: bool
    previous_side: int
    current_side: int
    direction: LineCrossingDirection | None = None
    intersection: Point | None = None


def detect_line_crossing(
    previous: Point,
    current: Point,
    line_start: Point,
    line_end: Point,
    *,
    previous_stable_side: int | None = None,
    epsilon: float = _EPSILON,
) -> LineCrossing:
    """Detect a completed crossing of a finite line segment.

    Merely touching or moving along the line is not a crossing. When a prior
    sample lies exactly on the line, callers may pass the last non-zero side to
    preserve hysteresis across frames.
    """

    start_side = line_side(previous, line_start, line_end, epsilon=epsilon)
    end_side = line_side(current, line_start, line_end, epsilon=epsilon)
    effective_start = start_side
    if start_side == 0 and previous_stable_side in (-1, 1):
        effective_start = previous_stable_side

    if end_side == 0 or effective_start == 0 or effective_start == end_side:
        return LineCrossing(False, start_side, end_side)
    intersection = _proper_intersection(
        previous, current, line_start, line_end, epsilon
    )
    if intersection is None:
        return LineCrossing(False, start_side, end_side)
    direction = (
        LineCrossingDirection.NEGATIVE_TO_POSITIVE
        if effective_start < end_side
        else LineCrossingDirection.POSITIVE_TO_NEGATIVE
    )
    return LineCrossing(True, start_side, end_side, direction, intersection)


def _proper_intersection(
    a: Point, b: Point, c: Point, d: Point, epsilon: float
) -> Point | None:
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = d[0] - c[0], d[1] - c[1]
    denominator = rx * sy - ry * sx
    if abs(denominator) <= epsilon:
        return None
    qpx, qpy = c[0] - a[0], c[1] - a[1]
    t = (qpx * sy - qpy * sx) / denominator
    u = (qpx * ry - qpy * rx) / denominator
    if -epsilon <= t <= 1.0 + epsilon and -epsilon <= u <= 1.0 + epsilon:
        return (a[0] + t * rx, a[1] + t * ry)
    return None


def _point_on_segment(
    point: Point, start: Point, end: Point, epsilon: float
) -> bool:
    if abs(signed_line_side(point, start, end)) > epsilon * max(
        1.0, math.sqrt(_distance_squared(start, end))
    ):
        return False
    return (
        min(start[0], end[0]) - epsilon
        <= point[0]
        <= max(start[0], end[0]) + epsilon
        and min(start[1], end[1]) - epsilon
        <= point[1]
        <= max(start[1], end[1]) + epsilon
    )


def _segments_intersect(
    a: Point, b: Point, c: Point, d: Point, epsilon: float
) -> bool:
    sides = (
        line_side(c, a, b, epsilon=epsilon),
        line_side(d, a, b, epsilon=epsilon),
        line_side(a, c, d, epsilon=epsilon),
        line_side(b, c, d, epsilon=epsilon),
    )
    if sides[0] * sides[1] < 0 and sides[2] * sides[3] < 0:
        return True
    return any(
        side == 0 and _point_on_segment(point, start, end, epsilon)
        for side, point, start, end in (
            (sides[0], c, a, b),
            (sides[1], d, a, b),
            (sides[2], a, c, d),
            (sides[3], b, c, d),
        )
    )


def _distance_squared(first: Point, second: Point) -> float:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2
