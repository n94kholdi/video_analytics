"""BoT-SORT association: IoU distance, score fusion, and gated ReID min-fusion.

Hungarian assignment is reused from the shared matching helper. Algorithm steps
follow ``tracker/matching.py`` in NirAharon/BoT-SORT.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from app.tracking.third_party.deepocsort.association import cosine_cost, iou_batch
from app.tracking.third_party.stabletrack.matching import bbox_based_distance, linear_assignment


def iou_distance(boxes_a: NDArray[np.floating], boxes_b: NDArray[np.floating]) -> NDArray[np.float64]:
    """Pairwise IoU distance (1 − IoU) for ``(N, 4)`` and ``(M, 4)`` xyxy arrays."""

    return 1.0 - iou_batch(boxes_a, boxes_b)


def fuse_score(cost: NDArray[np.floating], scores: NDArray[np.floating]) -> NDArray[np.float64]:
    """Weight IoU similarity by detection confidence (official ``fuse_score``)."""

    if cost.size == 0:
        return np.asarray(cost, dtype=np.float64)
    iou_sim = 1.0 - np.asarray(cost, dtype=np.float64)
    weights = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    return 1.0 - iou_sim * weights


def fuse_iou_reid(
    iou_dist: NDArray[np.floating],
    appearance_similarity: NDArray[np.floating],
    *,
    proximity_thresh: float,
    appearance_thresh: float,
    iou_for_gate: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Official ``min(iou, emb/2)`` fusion with spatial and appearance gates.

    ``proximity_thresh`` is applied to raw IoU distance, matching
    NirAharon/BoT-SORT (the mask is taken before ``fuse_score``).
    """

    fused = np.asarray(iou_dist, dtype=np.float64, copy=True)
    if appearance_similarity.size == 0:
        return fused
    gate = np.asarray(iou_for_gate if iou_for_gate is not None else fused, dtype=np.float64)
    embedding_distance = (1.0 - np.asarray(appearance_similarity, dtype=np.float64)) / 2.0
    embedding_distance = np.where(embedding_distance > appearance_thresh, 1.0, embedding_distance)
    embedding_distance = np.where(gate > proximity_thresh, 1.0, embedding_distance)
    return np.minimum(fused, embedding_distance)


def gated_assignment(
    cost: NDArray[np.floating],
    thresh: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Min-cost pairs whose cost is at most ``thresh``."""

    rows, cols = cost.shape if cost.ndim == 2 else (0, 0)
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))
    matched: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for row, col in linear_assignment(np.asarray(cost, dtype=np.float64)):
        if float(cost[row, col]) > thresh:
            continue
        matched.append((int(row), int(col)))
        used_rows.add(int(row))
        used_cols.add(int(col))
    unmatched_rows = [index for index in range(rows) if index not in used_rows]
    unmatched_cols = [index for index in range(cols) if index not in used_cols]
    return matched, unmatched_rows, unmatched_cols


def bbd_distance(
    det_boxes: NDArray[np.floating],
    track_boxes: NDArray[np.floating],
    delta_tau: float,
) -> NDArray[np.float64]:
    """Pairwise bbox-based distance used for 0.5 FPS recovery when IoU is zero."""

    n_dets = 0 if det_boxes.size == 0 else len(det_boxes)
    n_tracks = 0 if track_boxes.size == 0 else len(track_boxes)
    cost = np.full((n_dets, n_tracks), np.inf, dtype=np.float64)
    for row in range(n_dets):
        for col in range(n_tracks):
            cost[row, col] = bbox_based_distance(track_boxes[col], det_boxes[row], delta_tau)
    return cost


def associate(
    det_boxes: NDArray[np.floating],
    track_boxes: NDArray[np.floating],
    *,
    match_thresh: float,
    det_scores: NDArray[np.floating] | None = None,
    det_embeddings: Sequence[NDArray[np.floating] | None] | None = None,
    track_embeddings: Sequence[NDArray[np.floating] | None] | None = None,
    proximity_thresh: float = 1.0,
    appearance_thresh: float = 0.25,
    fuse_det_score: bool = True,
    alternate_boxes: NDArray[np.floating] | None = None,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """First-stage BoT-SORT matching (IoU, optional score fuse, optional ReID)."""

    n_dets = 0 if det_boxes.size == 0 else len(det_boxes)
    n_tracks = 0 if track_boxes.size == 0 else len(track_boxes)
    if n_tracks == 0:
        return [], list(range(n_dets)), []
    if n_dets == 0:
        return [], [], list(range(n_tracks))

    cost = iou_distance(det_boxes, track_boxes)
    if alternate_boxes is not None and len(alternate_boxes) == n_tracks:
        cost = np.minimum(cost, iou_distance(det_boxes, alternate_boxes))
    raw_iou = np.asarray(cost, dtype=np.float64)
    if fuse_det_score and det_scores is not None:
        cost = fuse_score(cost, det_scores)
    if det_embeddings is not None and track_embeddings is not None:
        appearance = cosine_cost(det_embeddings, track_embeddings)
        if appearance is not None:
            cost = fuse_iou_reid(
                cost,
                appearance,
                proximity_thresh=proximity_thresh,
                appearance_thresh=appearance_thresh,
                iou_for_gate=raw_iou,
            )
    return gated_assignment(cost, match_thresh)


def associate_center(
    det_boxes: NDArray[np.floating],
    track_boxes: NDArray[np.floating],
    *,
    delta_tau: float,
    match_thresh: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match leftover detections to leftover tracks by bbox-based distance."""

    n_dets = 0 if det_boxes.size == 0 else len(det_boxes)
    n_tracks = 0 if track_boxes.size == 0 else len(track_boxes)
    if n_tracks == 0:
        return [], list(range(n_dets)), []
    if n_dets == 0:
        return [], [], list(range(n_tracks))
    return gated_assignment(bbd_distance(det_boxes, track_boxes, delta_tau), match_thresh)
