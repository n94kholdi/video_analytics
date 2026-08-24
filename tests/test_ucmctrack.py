"""UCMCTrack association, timestamp, calibration catalog, and adapter-isolation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.models import Detection
from app.geometry.config import CalibrationConfig, NormalizedPoint
from app.tracking.calibration.camera import (
    CameraGeometry,
    CameraGeometryCatalog,
    GroundPlaneMapper,
    geometry_from_calibration,
    load_camera_geometry,
)
from app.tracking.ucmctrack_adapter import UCMCTrackAdapter
from app.tracking.third_party.ucmctrack.kalman import GroundKalman
from app.tracking.third_party.ucmctrack.tracker import UCMCTrack, UCMCTrackConfig


def person(x: float, *, y: float = 10.0, confidence: float = 0.9) -> Detection:
    return Detection((x, y, x + 20.0, y + 40.0), confidence)


def test_kalman_prediction_scales_with_elapsed_seconds_not_frame_count() -> None:
    measurement = np.array([10.0, 20.0], dtype=np.float64)
    covariance = np.eye(2, dtype=np.float64)
    filter_ = GroundKalman(measurement, covariance, wx=20.0, wy=20.0, vmax=250.0)
    filter_.mean[1] = 5.0
    one_frame = GroundKalman(measurement, covariance, wx=20.0, wy=20.0, vmax=250.0)
    one_frame.mean[1] = 5.0
    filter_.predict(2.0)
    one_frame.predict(1.0 / 30.0)

    assert filter_.mean[0] > one_frame.mean[0]
    assert filter_.position()[0] > 10.0


def test_uncalibrated_mapper_uses_box_foot_point() -> None:
    mapper = GroundPlaneMapper.uncalibrated()
    mapped = mapper.map_box((10.0, 20.0, 30.0, 80.0))

    assert mapped is not None
    assert mapped.calibrated is False
    assert mapped.xy == pytest.approx((20.0, 80.0))
    assert mapped.covariance[0, 0] > 0


def test_ucmctrack_reuses_id_across_two_second_gap() -> None:
    tracker = UCMCTrackAdapter(frame_rate=0.5, confirmation_frames=1)
    first = tracker.update([person(0.0)], camera_id="cam", timestamp=10.0, frame_index=0)
    second = tracker.update([person(4.0)], camera_id="cam", timestamp=12.0, frame_index=1)

    assert first.observations[0].track_id == second.observations[0].track_id
    assert second.observations[0].timestamp == pytest.approx(12.0)


def test_ucmctrack_keeps_id_when_person_walks_beyond_iou() -> None:
    tracker = UCMCTrackAdapter(frame_rate=0.5, confirmation_frames=1)
    first = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    second = tracker.update([person(30.0)], camera_id="cam", timestamp=2.0, frame_index=1)
    third = tracker.update([person(60.0)], camera_id="cam", timestamp=4.0, frame_index=2)

    assert second.observations[0].track_id == first.observations[0].track_id
    assert third.observations[0].track_id == first.observations[0].track_id
    assert tracker.retained_track_count == 1


def test_two_separated_people_keep_distinct_ids() -> None:
    tracker = UCMCTrackAdapter(frame_rate=0.5, confirmation_frames=1)
    first = tracker.update([person(0.0), person(200.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    second = tracker.update([person(25.0), person(225.0)], camera_id="cam", timestamp=2.0, frame_index=1)

    assert {item.track_id for item in first.observations} == {item.track_id for item in second.observations}
    assert len(second.observations) == 2


def test_small_jitter_does_not_change_id() -> None:
    tracker = UCMCTrackAdapter(frame_rate=0.5, confirmation_frames=1)
    offsets = (0.0, 2.0, -1.0, 3.0, 1.0, 2.5)
    ids: list[int] = []
    for index, offset in enumerate(offsets):
        result = tracker.update(
            [person(offset)],
            camera_id="cam",
            timestamp=float(index * 2),
            frame_index=index,
        )
        ids.append(result.observations[0].track_id)

    assert ids == [ids[0]] * len(ids)
    assert tracker.retained_track_count == 1


def test_nearby_still_people_keep_their_own_ids() -> None:
    tracker = UCMCTrackAdapter(frame_rate=0.5, confirmation_frames=1)
    first = tracker.update(
        [person(0.0), person(45.0)],
        camera_id="cam",
        timestamp=0.0,
        frame_index=0,
    )
    left_id = next(item.track_id for item in first.observations if item.xyxy[0] < 20)
    right_id = next(item.track_id for item in first.observations if item.xyxy[0] > 20)
    second = tracker.update(
        [person(2.0), person(47.0)],
        camera_id="cam",
        timestamp=2.0,
        frame_index=1,
    )
    third = tracker.update(
        [person(1.0), person(46.0)],
        camera_id="cam",
        timestamp=4.0,
        frame_index=2,
    )

    assert next(item.track_id for item in second.observations if item.xyxy[0] < 20) == left_id
    assert next(item.track_id for item in second.observations if item.xyxy[0] > 20) == right_id
    assert next(item.track_id for item in third.observations if item.xyxy[0] < 20) == left_id
    assert next(item.track_id for item in third.observations if item.xyxy[0] > 20) == right_id


def test_ucmctrack_expires_after_max_age_seconds() -> None:
    tracker = UCMCTrackAdapter(frame_rate=0.5, confirmation_frames=1, max_age_seconds=2.0)
    created = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    tracker.update([], camera_id="cam", timestamp=2.0, frame_index=1)
    expired = tracker.update([], camera_id="cam", timestamp=4.1, frame_index=2)

    assert created.observations[0].track_id in expired.expired_track_ids
    assert tracker.retained_track_count == 0


def test_missed_frame_recovers_same_id() -> None:
    tracker = UCMCTrackAdapter(frame_rate=0.5, confirmation_frames=1)
    created = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    missed = tracker.update([], camera_id="cam", timestamp=2.0, frame_index=1)
    recovered = tracker.update([person(3.0)], camera_id="cam", timestamp=4.0, frame_index=2)

    assert missed.observations == ()
    assert recovered.observations[0].track_id == created.observations[0].track_id
    assert recovered.expired_track_ids == ()


def test_person_who_left_does_not_donate_id_to_a_newcomer() -> None:
    tracker = UCMCTrackAdapter(frame_rate=0.5, confirmation_frames=1)
    created = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    tracker.update([], camera_id="cam", timestamp=2.0, frame_index=1)
    left = tracker.update([], camera_id="cam", timestamp=4.1, frame_index=2)
    newcomer = tracker.update([person(0.0)], camera_id="cam", timestamp=6.1, frame_index=3)

    old_id = created.observations[0].track_id
    assert old_id in left.expired_track_ids
    assert newcomer.observations[0].track_id != old_id
    assert newcomer.observations[0].track_id == old_id + 1


def test_low_score_second_association_keeps_identity() -> None:
    tracker = UCMCTrackAdapter(
        frame_rate=0.5,
        confirmation_frames=1,
        activation_threshold=0.4,
        track_low_threshold=0.1,
        max_age_seconds=8.0,
    )
    created = tracker.update([person(0.0, confidence=0.9)], camera_id="cam", timestamp=0.0, frame_index=0)
    recovered = tracker.update([person(2.0, confidence=0.2)], camera_id="cam", timestamp=2.0, frame_index=1)

    assert recovered.observations[0].track_id == created.observations[0].track_id


def test_unknown_camera_stays_uncalibrated() -> None:
    catalog = CameraGeometryCatalog()
    mapper = catalog.mapper_for("future_camera")

    assert "future_camera" not in catalog
    assert mapper.calibrated is False


def test_catalog_register_makes_a_new_camera_available() -> None:
    catalog = CameraGeometryCatalog()
    homography = np.array([[0.02, 0.0, 1.0], [0.0, 0.03, 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    catalog.register(
        CameraGeometry(
            "lobby_east",
            GroundPlaneMapper.from_homography(homography),
            wx=0.1,
            wy=0.1,
            vmax=1.5,
        )
    )

    assert "lobby_east" in catalog
    assert catalog.mapper_for("lobby_east").calibrated is True
    wx, wy, vmax, _gate = catalog.parameters_for(
        "lobby_east", wx=20.0, wy=20.0, vmax=250.0, assignment_threshold=15.0
    )
    assert (wx, wy, vmax) == (0.1, 0.1, 1.5)
    assert catalog.mapper_for("other_cam").calibrated is False


def _calibrated_adapter() -> UCMCTrackAdapter:
    homography = np.array([[0.02, 0.0, 1.0], [0.0, 0.03, 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    catalog = CameraGeometryCatalog()
    catalog.register(
        CameraGeometry(
            "cam",
            GroundPlaneMapper.from_homography(homography),
            wx=0.1,
            wy=0.1,
            vmax=1.5,
        )
    )
    return UCMCTrackAdapter(frame_rate=0.5, confirmation_frames=1, camera_catalog=catalog)


def test_calibrated_camera_keeps_id_when_person_walks_beyond_iou() -> None:
    tracker = _calibrated_adapter()
    first = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    second = tracker.update([person(30.0)], camera_id="cam", timestamp=2.0, frame_index=1)
    third = tracker.update([person(60.0)], camera_id="cam", timestamp=4.0, frame_index=2)

    assert second.observations[0].track_id == first.observations[0].track_id
    assert third.observations[0].track_id == first.observations[0].track_id


def test_calibrated_camera_recovers_id_after_a_short_miss() -> None:
    tracker = _calibrated_adapter()
    created = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    missed = tracker.update([], camera_id="cam", timestamp=2.0, frame_index=1)
    recovered = tracker.update([person(3.0)], camera_id="cam", timestamp=4.0, frame_index=2)

    assert missed.observations == ()
    assert recovered.observations[0].track_id == created.observations[0].track_id


def test_calibrated_camera_does_not_give_a_left_person_id_to_a_newcomer() -> None:
    tracker = _calibrated_adapter()
    created = tracker.update([person(0.0)], camera_id="cam", timestamp=0.0, frame_index=0)
    tracker.update([], camera_id="cam", timestamp=2.0, frame_index=1)
    left = tracker.update([], camera_id="cam", timestamp=4.1, frame_index=2)
    newcomer = tracker.update([person(0.0)], camera_id="cam", timestamp=6.1, frame_index=3)

    old_id = created.observations[0].track_id
    assert old_id in left.expired_track_ids
    assert newcomer.observations[0].track_id != old_id


def test_homography_mapper_differs_from_image_plane() -> None:
    homography = np.array([[0.02, 0.0, 1.0], [0.0, 0.03, 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    mapped = GroundPlaneMapper.from_homography(homography).map_box((10.0, 20.0, 30.0, 80.0))
    pixels = GroundPlaneMapper.uncalibrated().map_box((10.0, 20.0, 30.0, 80.0))

    assert mapped is not None and pixels is not None
    assert mapped.calibrated is True
    assert mapped.xy[0] != pytest.approx(pixels.xy[0])


def test_geometry_from_existing_homography_calibration() -> None:
    calibration = CalibrationConfig(
        image_points=(
            NormalizedPoint(0.0, 0.0),
            NormalizedPoint(1.0, 0.0),
            NormalizedPoint(1.0, 1.0),
            NormalizedPoint(0.0, 1.0),
        ),
        ground_points=((10.0, 20.0), (30.0, 20.0), (30.0, 60.0), (10.0, 60.0)),
    )
    geometry = geometry_from_calibration(calibration, (101, 201), camera_id="lobby")
    mapped = geometry.mapper.map_box((50.0, 0.0, 50.0, 100.0))

    assert geometry.calibrated is True
    assert mapped is not None
    assert mapped.xy == pytest.approx((20.0, 40.0), abs=1e-4)


def test_yaml_and_cam_para_load_for_future_cameras(tmp_path: Path) -> None:
    yaml_path = tmp_path / "door_north.yaml"
    yaml_path.write_text(
        "camera_id: door_north\n"
        "homography:\n"
        "  - [0.02, 0.0, 1.0]\n"
        "  - [0.0, 0.03, 2.0]\n"
        "  - [0.0, 0.0, 1.0]\n",
        encoding="utf-8",
    )
    para_path = tmp_path / "seq_01.txt"
    para_path.write_text(
        "RotationMatrices\n"
        "0.00000 -1.00000 0.00000\n"
        "-0.05234 0.00000 -0.99863\n"
        "0.99863 0.00000 -0.05234\n"
        "\n"
        "TranslationVectors\n"
        "0 1391 3968\n"
        "\n"
        "IntrinsicMatrix\n"
        "1213 0 960\n"
        "0 1213 540\n"
        "0 0 1\n",
        encoding="utf-8",
    )
    loaded_yaml = load_camera_geometry(yaml_path)
    loaded_para = load_camera_geometry(para_path)
    catalog = CameraGeometryCatalog.from_directory(tmp_path)

    assert loaded_yaml.camera_id == "door_north"
    assert loaded_yaml.calibrated is True
    assert loaded_para.calibrated is True
    assert catalog.mapper_for("door_north").calibrated is True
    assert catalog.mapper_for("seq_01").calibrated is True


def test_backend_does_not_import_application_adapter() -> None:
    backend = UCMCTrack(UCMCTrackConfig(activation_threshold=0.4, confirmation_hits=1))
    boxes = np.asarray([[0.0, 10.0, 20.0, 50.0]], dtype=np.float32)
    scores = np.asarray([0.9], dtype=np.float32)
    outputs = backend.update(
        boxes=boxes,
        scores=scores,
        class_ids=None,
        timestamp=0.0,
        mapper=GroundPlaneMapper.uncalibrated(),
    )

    assert outputs[0].track_id == 1
    assert backend.last_reid_ms == 0.0
