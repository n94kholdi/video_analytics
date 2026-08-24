"""Deep OC-SORT association: IoU, observation-centric momentum, Adaptive Weighting.

Hungarian assignment is reused from the shared matching helper. Algorithm steps
follow ``trackers/ocsort_embedding/association.py`` in GerardMaggiolino/Deep-OC-SORT.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from app.tracking.third_party.stabletrack.matching import linear_assignment


def iou_batch(boxes_a: NDArray[np.floating], boxes_b: NDArray[np.floating]) -> NDArray[np.float64]:
    """Pairwise IoU for ``(N, 4)`` and ``(M, 4)`` xyxy arrays."""

    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    left = np.asarray(boxes_a, dtype=np.float64)[:, :4][:, None, :]
    right = np.asarray(boxes_b, dtype=np.float64)[:, :4][None, :, :]
    x1 = np.maximum(left[..., 0], right[..., 0])
    y1 = np.maximum(left[..., 1], right[..., 1])
    x2 = np.minimum(left[..., 2], right[..., 2])
    y2 = np.minimum(left[..., 3], right[..., 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, left[..., 2] - left[..., 0]) * np.maximum(0.0, left[..., 3] - left[..., 1])
    area_b = np.maximum(0.0, right[..., 2] - right[..., 0]) * np.maximum(0.0, right[..., 3] - right[..., 1])
    union = area_a + area_b - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def speed_direction_batch(
    detections: NDArray[np.floating],
    tracks: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    expanded = np.asarray(tracks, dtype=np.float64)[..., np.newaxis]
    det_x = (detections[:, 0] + detections[:, 2]) / 2.0
    det_y = (detections[:, 1] + detections[:, 3]) / 2.0
    trk_x = (expanded[:, 0] + expanded[:, 2]) / 2.0
    trk_y = (expanded[:, 1] + expanded[:, 3]) / 2.0
    dx = det_x - trk_x
    dy = det_y - trk_y
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    return dy / norm, dx / norm


def compute_aw_max_metric(
    embedding_cost: NDArray[np.floating],
    w_association_emb: float,
    *,
    bottom: float = 0.5,
) -> NDArray[np.float64]:
    """Boost appearance cost when a row/column has a unique high match (paper AW)."""

    cost = np.asarray(embedding_cost, dtype=np.float64)
    weights = np.full_like(cost, float(w_association_emb))
    floor = min(max(float(bottom), 1e-6), 0.999)
    for row in range(cost.shape[0]):
        order = np.argsort(-cost[row])
        if len(order) < 2:
            continue
        best = cost[row, order[0]]
        if best == 0:
            weights[row] *= 0.0
            continue
        ratio = cost[row, order[1]] / best
        weights[row] *= 1.0 - max(ratio - floor, 0.0) / (1.0 - floor)
    for col in range(cost.shape[1]):
        order = np.argsort(-cost[:, col])
        if len(order) < 2:
            continue
        best = cost[order[0], col]
        if best == 0:
            weights[:, col] *= 0.0
            continue
        ratio = cost[order[1], col] / best
        weights[:, col] *= 1.0 - max(ratio - floor, 0.0) / (1.0 - floor)
    return weights * cost


def associate(
    detections: NDArray[np.floating],
    trackers: NDArray[np.floating],
    iou_threshold: float,
    velocities: NDArray[np.floating],
    previous_obs: NDArray[np.floating],
    inertia: float,
    embedding_cost: NDArray[np.floating] | None,
    w_association_emb: float,
    *,
    aw_off: bool = False,
    aw_param: float = 0.5,
    require_iou: bool = True,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """First-stage Deep OC-SORT matching (IoU + OCM + optional AW appearance)."""

    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.arange(len(detections), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    if len(detections) == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.arange(len(trackers), dtype=np.int64),
        )

    direction_y, direction_x = speed_direction_batch(detections, previous_obs)
    inertia_y = np.repeat(velocities[:, 0][:, np.newaxis], direction_y.shape[1], axis=1)
    inertia_x = np.repeat(velocities[:, 1][:, np.newaxis], direction_x.shape[1], axis=1)
    cosine = np.clip(inertia_x * direction_x + inertia_y * direction_y, -1.0, 1.0)
    angle = (np.pi / 2.0 - np.abs(np.arccos(cosine))) / np.pi
    valid = np.ones(previous_obs.shape[0], dtype=np.float64)
    valid[previous_obs[:, 4] < 0] = 0.0
    valid = np.repeat(valid[:, np.newaxis], direction_x.shape[1], axis=1)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    angle_cost = (valid * angle * float(inertia)).T * scores
    iou_matrix = iou_batch(detections, trackers)

    appearance = np.zeros_like(iou_matrix)
    if embedding_cost is not None:
        appearance = np.array(embedding_cost, dtype=np.float64, copy=True)
        if require_iou:
            appearance[iou_matrix <= 0] = 0.0
        if aw_off:
            appearance *= float(w_association_emb)
        else:
            appearance = compute_aw_max_metric(appearance, w_association_emb, bottom=aw_param)

    final_cost = -(iou_matrix + angle_cost + appearance)
    pairs = linear_assignment(final_cost)
    matched: list[tuple[int, int]] = []
    unmatched_dets = set(range(len(detections)))
    unmatched_trks = set(range(len(trackers)))
    gate = float(iou_threshold) if require_iou else -1.0
    for row, col in pairs:
        if iou_matrix[row, col] < gate:
            continue
        matched.append((row, col))
        unmatched_dets.discard(row)
        unmatched_trks.discard(col)
    matches = np.asarray(matched, dtype=np.int64).reshape(-1, 2)
    return (
        matches,
        np.asarray(sorted(unmatched_dets), dtype=np.int64),
        np.asarray(sorted(unmatched_trks), dtype=np.int64),
    )


def cosine_cost(det_embeddings: Sequence[NDArray[np.floating] | None], track_embeddings: Sequence[NDArray[np.floating] | None]) -> NDArray[np.float64] | None:
    """Return detection×track cosine similarity, or ``None`` when appearance is absent."""

    if not det_embeddings or not track_embeddings:
        return None
    if all(item is None for item in det_embeddings) or all(item is None for item in track_embeddings):
        return None
    rows = []
    dim = 0
    for item in list(det_embeddings) + list(track_embeddings):
        if item is not None:
            dim = int(np.asarray(item).size)
            break
    if dim <= 0:
        return None
    for embedding in det_embeddings:
        if embedding is None:
            rows.append(np.zeros(dim, dtype=np.float64))
        else:
            vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
            norm = float(np.linalg.norm(vector))
            rows.append(vector / norm if norm > 1e-12 else np.zeros(dim, dtype=np.float64))
    cols = []
    for embedding in track_embeddings:
        if embedding is None:
            cols.append(np.zeros(dim, dtype=np.float64))
        else:
            vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
            norm = float(np.linalg.norm(vector))
            cols.append(vector / norm if norm > 1e-12 else np.zeros(dim, dtype=np.float64))
    return np.asarray(rows) @ np.asarray(cols).T
