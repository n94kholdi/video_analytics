"""Light-model output parsing and NMS unit tests."""

import numpy as np
import pytest

from app.detection.postprocessing import (
    DetectorOutputError,
    non_max_suppression,
    parse_light_output,
)
from app.detection.preprocessing import LetterboxTransform


@pytest.fixture
def identity_transform() -> LetterboxTransform:
    return LetterboxTransform(
        original_height=640,
        original_width=640,
        input_height=640,
        input_width=640,
        resized_height=640,
        resized_width=640,
        pad_top=0,
        pad_left=0,
    )


def test_output_transposition_confidence_filtering_and_nms(
    identity_transform: LetterboxTransform,
) -> None:
    candidates = np.asarray(
        [
            [100.0, 100.0, 40.0, 40.0, 0.90],
            [102.0, 102.0, 40.0, 40.0, 0.80],
            [300.0, 300.0, 20.0, 20.0, 0.20],
        ],
        dtype=np.float32,
    )
    raw = candidates.T[np.newaxis]

    detections = parse_light_output(
        raw,
        identity_transform,
        confidence_threshold=0.5,
        iou_threshold=0.5,
    )

    assert len(detections) == 1
    np.testing.assert_allclose(detections[0].xyxy, (80.0, 80.0, 120.0, 120.0))
    assert detections[0].confidence == pytest.approx(0.9)


def test_candidate_first_output_orientation_is_supported(
    identity_transform: LetterboxTransform,
) -> None:
    raw = np.asarray(
        [[[50.0, 60.0, 20.0, 40.0, 0.75]]],
        dtype=np.float32,
    )

    detections = parse_light_output(
        raw,
        identity_transform,
        confidence_threshold=0.5,
        iou_threshold=0.5,
    )

    assert len(detections) == 1
    np.testing.assert_allclose(detections[0].xyxy, (40.0, 40.0, 60.0, 80.0))


def test_nms_retains_non_overlapping_boxes() -> None:
    boxes = np.asarray(
        [[0, 0, 10, 10], [1, 1, 11, 11], [30, 30, 40, 40]],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)

    kept = non_max_suppression(boxes, scores, 0.5)

    assert kept.tolist() == [0, 2]


@pytest.mark.parametrize(
    "raw",
    [
        np.empty((1, 5, 0), dtype=np.float32),
        np.empty((1, 0, 5), dtype=np.float32),
    ],
)
def test_empty_outputs_return_no_detections(
    raw: np.ndarray,
    identity_transform: LetterboxTransform,
) -> None:
    assert (
        parse_light_output(
            raw,
            identity_transform,
            confidence_threshold=0.5,
            iou_threshold=0.5,
        )
        == ()
    )


def test_unexpected_runtime_output_shape_is_rejected(
    identity_transform: LetterboxTransform,
) -> None:
    with pytest.raises(DetectorOutputError, match="five values"):
        parse_light_output(
            np.zeros((1, 6, 100), dtype=np.float32),
            identity_transform,
            confidence_threshold=0.5,
            iou_threshold=0.5,
        )

