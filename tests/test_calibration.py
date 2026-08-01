"""Synthetic homography and unavailable-calibration tests."""

import pytest

from app.geometry.calibration import ImageToGroundProjector
from app.geometry.config import CalibrationConfig, CameraConfigError, NormalizedPoint


def rectangle_calibration() -> CalibrationConfig:
    return CalibrationConfig(
        image_points=(
            NormalizedPoint(0.0, 0.0),
            NormalizedPoint(1.0, 0.0),
            NormalizedPoint(1.0, 1.0),
            NormalizedPoint(0.0, 1.0),
        ),
        ground_points=((10.0, 20.0), (30.0, 20.0), (30.0, 60.0), (10.0, 60.0)),
    )


def test_known_synthetic_homography_projection() -> None:
    projector = ImageToGroundProjector.from_calibration(
        rectangle_calibration(), (101, 201)
    )

    result = projector.project((50.0, 100.0))

    assert result.available
    assert result.point == pytest.approx((20.0, 40.0))
    assert result.unit == "metres"


def test_missing_calibration_returns_clear_unavailable_result() -> None:
    projector = ImageToGroundProjector.from_calibration(None, (640, 480))

    result = projector.project((100.0, 200.0))

    assert not projector.available
    assert not result.available
    assert result.point is None
    assert result.reason == "camera calibration is not configured"


@pytest.mark.parametrize(
    ("image_points", "ground_points", "message"),
    [
        (
            (NormalizedPoint(0, 0), NormalizedPoint(1, 0), NormalizedPoint(0, 1)),
            ((0, 0), (1, 0), (0, 1)),
            "at least four",
        ),
        (
            (
                NormalizedPoint(0, 0),
                NormalizedPoint(0.25, 0),
                NormalizedPoint(0.5, 0),
                NormalizedPoint(1, 0),
            ),
            ((0, 0), (1, 0), (2, 0), (3, 0)),
            "degenerate",
        ),
    ],
)
def test_invalid_calibration_is_rejected(
    image_points: tuple[NormalizedPoint, ...],
    ground_points: tuple[tuple[float, float], ...],
    message: str,
) -> None:
    with pytest.raises(CameraConfigError, match=message):
        CalibrationConfig(image_points, ground_points)
