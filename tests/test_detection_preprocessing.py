"""Letterbox and coordinate-restoration unit tests."""

import numpy as np

from app.detection.preprocessing import letterbox, preprocess_bgr


def test_letterbox_preserves_aspect_ratio_and_centers_padding() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    padded, transform = letterbox(frame, (640, 640))

    assert padded.shape == (640, 640, 3)
    assert transform.resized_width == 640
    assert transform.resized_height == 320
    assert transform.pad_left == 0
    assert transform.pad_top == 160
    assert np.all(padded[:160] == 114)


def test_coordinate_restoration_reverses_letterbox() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    _, transform = letterbox(frame, (640, 640))
    model_box = np.asarray([[32.0, 192.0, 608.0, 448.0]], dtype=np.float32)

    restored = transform.restore_boxes(model_box)

    np.testing.assert_allclose(
        restored,
        np.asarray([[10.0, 10.0, 190.0, 90.0]], dtype=np.float32),
    )


def test_preprocess_converts_bgr_to_normalized_rgb_nchw() -> None:
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    frame[0, 0] = (0, 128, 255)

    processed = preprocess_bgr(frame)

    assert processed.tensor.shape == (1, 3, 640, 640)
    assert processed.tensor.dtype == np.float32
    np.testing.assert_allclose(
        processed.tensor[0, :, 0, 0],
        np.asarray([1.0, 128.0 / 255.0, 0.0], dtype=np.float32),
    )

