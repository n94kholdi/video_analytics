"""Hungarian assignment and StableTrack similarity measures."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


INF = 1e5


def xyxy_to_xywh(box: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, width, height)


def intersection_over_union(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def iou_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return 1.0 - intersection_over_union(first, second)


def bbox_based_distance(
    predicted_xyxy: Sequence[float],
    detection_xyxy: Sequence[float],
    delta_tau: float,
    *,
    alpha: float = 0.025,
    beta: float = 0.25,
    scale: float = 1.0,
) -> float:
    """Mahalanobis-like distance with a deterministic scale/time covariance.

    ``delta_tau`` is elapsed seconds since the last successful track update.
    Paper constants: ``alpha=0.025``, ``beta=0.25``, ``c=1.0``.
    """

    _px, _py, width, height = xyxy_to_xywh(predicted_xyxy)
    dx, dy, _dw, _dh = xyxy_to_xywh(detection_xyxy)
    clipped = min(max(float(delta_tau), alpha), beta)
    pxx = max((scale * width) ** 2 * clipped, 1e-6)
    pyy = max((scale * height) ** 2 * clipped, 1e-6)
    return float(np.sqrt((dx - _px) ** 2 / pxx + (dy - _py) ** 2 / pyy))


def cosine_similarity(first: NDArray[np.floating] | None, second: NDArray[np.floating] | None) -> float:
    if first is None or second is None:
        return 0.0
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if left.size == 0 or right.size == 0 or left.size != right.size:
        return 0.0
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(left, right) / denom, -1.0, 1.0))


def linear_assignment(cost: NDArray[np.floating]) -> list[tuple[int, int]]:
    """Return min-cost pairs; infinite entries are never matched."""

    if cost.size == 0:
        return []
    finite = np.array(cost, dtype=np.float64, copy=True)
    blocked = ~np.isfinite(finite) | (finite >= INF / 2.0)
    finite[blocked] = INF
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _greedy_assignment(finite, blocked)
    rows, cols = linear_sum_assignment(finite)
    return [
        (int(row), int(col))
        for row, col in zip(rows, cols)
        if not blocked[int(row), int(col)]
    ]


def _greedy_assignment(cost: NDArray[np.floating], blocked: NDArray[np.bool_]) -> list[tuple[int, int]]:
    pairs: list[tuple[float, int, int]] = []
    rows, cols = cost.shape
    for row in range(rows):
        for col in range(cols):
            if blocked[row, col]:
                continue
            pairs.append((float(cost[row, col]), row, col))
    pairs.sort()
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _score, row, col in pairs:
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        matches.append((row, col))
    return matches
