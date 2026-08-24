"""CLEAR and HOTA metrics over timestamped xyxy tracks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from app.core.models import TrackObservation
from app.tracking.third_party.stabletrack.matching import intersection_over_union, linear_assignment


@dataclass(frozen=True, slots=True)
class GroundTruthBox:
    frame_index: int
    track_id: int
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class MotMetrics:
    hota: float
    idf1: float
    mota: float
    id_switches: int
    fragmentation: int
    det_a: float
    ass_a: float
    precision: float
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_tracks(
    ground_truth: Sequence[GroundTruthBox],
    hypotheses: Sequence[TrackObservation] | Sequence[tuple[int, int, tuple[float, float, float, float]]],
    *,
    iou_threshold: float = 0.5,
) -> MotMetrics:
    gt_frames = _index_ground_truth(ground_truth)
    hyp_frames = _index_hypotheses(hypotheses)
    frames = sorted(set(gt_frames) | set(hyp_frames))
    matches_by_frame, tp, fp, fn = _match_sequence(gt_frames, hyp_frames, frames, iou_threshold)
    id_switches = _count_id_switches(matches_by_frame)
    fragmentation = _count_fragmentation(gt_frames, matches_by_frame)
    gt_count = sum(len(items) for items in gt_frames.values())
    mota = 1.0 - (fn + fp + id_switches) / gt_count if gt_count else 0.0
    idf1 = _identity_f1(matches_by_frame, gt_count, sum(len(items) for items in hyp_frames.values()))
    hota, det_a, ass_a = _hota(gt_frames, hyp_frames, frames)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return MotMetrics(
        hota=hota,
        idf1=idf1,
        mota=mota,
        id_switches=id_switches,
        fragmentation=fragmentation,
        det_a=det_a,
        ass_a=ass_a,
        precision=precision,
        recall=recall,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def _index_ground_truth(items: Sequence[GroundTruthBox]) -> dict[int, list[GroundTruthBox]]:
    frames: dict[int, list[GroundTruthBox]] = defaultdict(list)
    for item in items:
        frames[item.frame_index].append(item)
    return frames


def _index_hypotheses(
    items: Sequence[TrackObservation] | Sequence[tuple[int, int, tuple[float, float, float, float]]],
) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
    frames: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    for item in items:
        if isinstance(item, TrackObservation):
            if not item.confirmed:
                continue
            frames[item.frame_index].append((item.track_id, item.xyxy))
        else:
            frame_index, track_id, xyxy = item
            frames[int(frame_index)].append((int(track_id), tuple(float(v) for v in xyxy)))
    return frames


def _match_sequence(
    gt_frames: Mapping[int, Sequence[GroundTruthBox]],
    hyp_frames: Mapping[int, Sequence[tuple[int, tuple[float, float, float, float]]]],
    frames: Sequence[int],
    iou_threshold: float,
) -> tuple[dict[int, list[tuple[int, int]]], int, int, int]:
    matches_by_frame: dict[int, list[tuple[int, int]]] = {}
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for frame in frames:
        gt_items = list(gt_frames.get(frame, ()))
        hyp_items = list(hyp_frames.get(frame, ()))
        pairs = _match_frame(gt_items, hyp_items, iou_threshold)
        matches_by_frame[frame] = [(gt_items[row].track_id, hyp_items[col][0]) for row, col in pairs]
        true_positives += len(pairs)
        false_negatives += len(gt_items) - len(pairs)
        false_positives += len(hyp_items) - len(pairs)
    return matches_by_frame, true_positives, false_positives, false_negatives


def _match_frame(
    gt_items: Sequence[GroundTruthBox],
    hyp_items: Sequence[tuple[int, tuple[float, float, float, float]]],
    iou_threshold: float,
) -> list[tuple[int, int]]:
    if not gt_items or not hyp_items:
        return []
    cost = np.ones((len(gt_items), len(hyp_items)), dtype=np.float64)
    for row, gt_item in enumerate(gt_items):
        for col, (_track_id, box) in enumerate(hyp_items):
            overlap = intersection_over_union(gt_item.xyxy, box)
            cost[row, col] = 1.0 - overlap if overlap >= iou_threshold else 1e5
    return linear_assignment(cost)


def _count_id_switches(matches_by_frame: Mapping[int, Sequence[tuple[int, int]]]) -> int:
    previous: dict[int, int] = {}
    switches = 0
    for frame in sorted(matches_by_frame):
        current = {gt_id: hyp_id for gt_id, hyp_id in matches_by_frame[frame]}
        for gt_id, hyp_id in current.items():
            last = previous.get(gt_id)
            if last is not None and last != hyp_id:
                switches += 1
        previous.update(current)
    return switches


def _count_fragmentation(
    gt_frames: Mapping[int, Sequence[GroundTruthBox]],
    matches_by_frame: Mapping[int, Sequence[tuple[int, int]]],
) -> int:
    presence: dict[int, list[tuple[int, bool]]] = defaultdict(list)
    for frame in sorted(set(gt_frames) | set(matches_by_frame)):
        matched = {gt_id for gt_id, _hyp in matches_by_frame.get(frame, ())}
        for item in gt_frames.get(frame, ()):
            presence[item.track_id].append((frame, item.track_id in matched))
    fragments = 0
    for samples in presence.values():
        was_tracked = False
        gap = False
        for _frame, matched in samples:
            if matched:
                if was_tracked and gap:
                    fragments += 1
                was_tracked = True
                gap = False
            elif was_tracked:
                gap = True
    return fragments


def _identity_f1(
    matches_by_frame: Mapping[int, Sequence[tuple[int, int]]],
    gt_count: int,
    hyp_count: int,
) -> float:
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for matches in matches_by_frame.values():
        for pair in matches:
            pair_counts[pair] += 1
    assigned: dict[int, int] = {}
    used_hyp: set[int] = set()
    for (gt_id, hyp_id), count in sorted(pair_counts.items(), key=lambda item: item[1], reverse=True):
        if gt_id in assigned or hyp_id in used_hyp:
            continue
        assigned[gt_id] = hyp_id
        used_hyp.add(hyp_id)
    idtp = sum(count for (gt_id, hyp_id), count in pair_counts.items() if assigned.get(gt_id) == hyp_id)
    idfp = hyp_count - idtp
    idfn = gt_count - idtp
    denom = 2 * idtp + idfp + idfn
    return (2 * idtp) / denom if denom else 0.0


def _hota(
    gt_frames: Mapping[int, Sequence[GroundTruthBox]],
    hyp_frames: Mapping[int, Sequence[tuple[int, tuple[float, float, float, float]]]],
    frames: Sequence[int],
) -> tuple[float, float, float]:
    alphas = [round(value, 2) for value in np.arange(0.05, 0.99, 0.05)]
    scores: list[float] = []
    det_scores: list[float] = []
    ass_scores: list[float] = []
    for alpha in alphas:
        matches_by_frame, tp, fp, fn = _match_sequence(gt_frames, hyp_frames, frames, alpha)
        det_a = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        pair_tp: dict[tuple[int, int], int] = defaultdict(int)
        gt_totals: dict[int, int] = defaultdict(int)
        hyp_totals: dict[int, int] = defaultdict(int)
        for frame in frames:
            for item in gt_frames.get(frame, ()):
                gt_totals[item.track_id] += 1
            for hyp_id, _box in hyp_frames.get(frame, ()):
                hyp_totals[hyp_id] += 1
            for pair in matches_by_frame[frame]:
                pair_tp[pair] += 1
        if not pair_tp:
            ass_a = 0.0
        else:
            ass_terms = []
            for pair, tpa in pair_tp.items():
                gt_id, hyp_id = pair
                fna = gt_totals[gt_id] - tpa
                fpa = hyp_totals[hyp_id] - tpa
                ass_terms.append(tpa / (tpa + fna + fpa) if (tpa + fna + fpa) else 0.0)
            ass_a = float(np.mean(ass_terms))
        scores.append(float(np.sqrt(det_a * ass_a)))
        det_scores.append(det_a)
        ass_scores.append(ass_a)
    return float(np.mean(scores)), float(np.mean(det_scores)), float(np.mean(ass_scores))
