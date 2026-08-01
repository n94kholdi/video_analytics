"""Phase 5 synthetic people-counting tests."""

from __future__ import annotations

import numpy as np

from app.analytics import (
    CameraCountingConfig,
    PeopleCounter,
    annotate_people_counts,
)
from app.core.models import TrackObservation
from app.geometry.config import CountingLine, NormalizedPoint, PolygonZone


def _zone(zone_id: str, left: float = 0.2, right: float = 0.8) -> PolygonZone:
    return PolygonZone(
        zone_id,
        (
            NormalizedPoint(left, 0.2),
            NormalizedPoint(right, 0.2),
            NormalizedPoint(right, 0.8),
            NormalizedPoint(left, 0.8),
        ),
    )


def _line(line_id: str = "door", hysteresis: float = 0.03) -> CountingLine:
    return CountingLine(
        line_id,
        NormalizedPoint(0.2, 0.5),
        NormalizedPoint(0.8, 0.5),
        hysteresis=hysteresis,
    )


def _observation(
    track_id: int,
    point: tuple[float, float],
    *,
    camera_id: str = "cam-a",
    timestamp: float = 1.0,
    confirmed: bool = True,
) -> TrackObservation:
    x, y = point
    return TrackObservation(
        camera_id=camera_id,
        track_id=track_id,
        timestamp=timestamp,
        frame_index=int(timestamp * 10),
        xyxy=(x - 2, y - 10, x + 2, y),
        foot_point=point,
        detection_confidence=0.9,
        confirmed=confirmed,
        trajectory=(),
    )


def _counter(
    *,
    zones: tuple[PolygonZone, ...] = (),
    lines: tuple[CountingLine, ...] = (),
) -> PeopleCounter:
    return PeopleCounter((CameraCountingConfig("cam-a", (101, 101), zones, lines),))


def test_entering_and_leaving_polygon_updates_current_occupancy() -> None:
    counter = _counter(zones=(_zone("floor"),))

    outside = counter.update("cam-a", [_observation(1, (10, 50))])
    inside = counter.update("cam-a", [_observation(1, (50, 50), timestamp=2)])
    left = counter.update("cam-a", [_observation(1, (90, 50), timestamp=3)])

    assert outside.snapshot.occupancy_for("floor") == 0
    assert inside.snapshot.occupancy_for("floor") == 1
    assert left.snapshot.occupancy_for("floor") == 0


def test_both_crossing_directions_emit_events_and_update_totals() -> None:
    counter = _counter(lines=(_line(),))
    counter.update("cam-a", [_observation(7, (50, 40))])

    entered = counter.update("cam-a", [_observation(7, (50, 60), timestamp=2)])
    exited = counter.update("cam-a", [_observation(7, (50, 40), timestamp=3)])

    assert entered.events[0].event_type == "line_crossed"
    assert entered.events[0].camera_id == "cam-a"
    assert entered.events[0].track_id == 7
    assert entered.events[0].line_id == "door"
    assert entered.events[0].timestamp == 2
    assert entered.events[0].payload == {
        "direction": "entry",
        "side_transition": "negative_to_positive",
    }
    assert exited.events[0].payload["direction"] == "exit"
    assert exited.snapshot.line_for("door").entries == 1
    assert exited.snapshot.line_for("door").exits == 1


def test_standing_on_line_and_oscillation_inside_hysteresis_do_not_count() -> None:
    counter = _counter(lines=(_line(hysteresis=0.05),))

    for timestamp, y in enumerate((40, 48, 52, 49, 51), start=1):
        result = counter.update(
            "cam-a", [_observation(1, (50, y), timestamp=float(timestamp))]
        )
        assert result.events == ()
    crossed = counter.update("cam-a", [_observation(1, (50, 60), timestamp=6)])
    for timestamp, y in enumerate((52, 48, 50, 51), start=7):
        result = counter.update(
            "cam-a", [_observation(1, (50, y), timestamp=float(timestamp))]
        )
        assert result.events == ()
    assert crossed.snapshot.cumulative_entries == 1
    assert result.snapshot.cumulative_entries == 1
    assert result.snapshot.cumulative_exits == 0


def test_standing_exactly_on_a_line_never_initializes_a_crossing() -> None:
    counter = _counter(lines=(_line(hysteresis=0.0),))

    for timestamp in range(1, 5):
        result = counter.update(
            "cam-a", [_observation(1, (50, 50), timestamp=float(timestamp))]
        )

    assert result.events == ()
    assert result.snapshot.cumulative_entries == 0


def test_multiple_tracks_count_independently_and_unconfirmed_tracks_are_ignored() -> None:
    counter = _counter(zones=(_zone("floor"),), lines=(_line(),))
    counter.update(
        "cam-a",
        [
            _observation(1, (40, 40)),
            _observation(2, (60, 40)),
            _observation(3, (50, 60), confirmed=False),
        ],
    )
    result = counter.update(
        "cam-a",
        [
            _observation(1, (40, 60), timestamp=2),
            _observation(2, (60, 60), timestamp=2),
            _observation(3, (50, 40), timestamp=2, confirmed=False),
        ],
    )

    assert result.snapshot.occupancy_for("floor") == 2
    assert len(result.events) == 2
    assert result.snapshot.cumulative_entries == 2


def test_track_disappearance_clears_occupancy_and_breaks_crossing_history() -> None:
    counter = _counter(zones=(_zone("floor"),), lines=(_line(),))
    counter.update("cam-a", [_observation(1, (50, 40))])

    missing = counter.update("cam-a", [], timestamp=2)
    resumed = counter.update("cam-a", [_observation(1, (50, 60), timestamp=3)])

    assert missing.snapshot.occupancy_for("floor") == 0
    assert resumed.events == ()
    assert resumed.snapshot.cumulative_entries == 0


def test_explicit_state_reset_starts_a_new_processing_run() -> None:
    counter = _counter(lines=(_line(),))
    counter.update("cam-a", [_observation(1, (50, 40))])
    counter.update("cam-a", [_observation(1, (50, 60), timestamp=2)])

    counter.reset()

    assert counter.snapshot("cam-a").cumulative_entries == 0
    assert counter.update("cam-a", [_observation(1, (50, 40), timestamp=3)]).events == ()


def test_cameras_and_overlapping_zones_keep_independent_state() -> None:
    configs = (
        CameraCountingConfig(
            "cam-a",
            (101, 101),
            (_zone("left", 0.1, 0.6), _zone("right", 0.4, 0.9)),
            (_line(),),
        ),
        CameraCountingConfig(
            "cam-b", (101, 101), (_zone("other"),), (_line("other-door"),)
        ),
    )
    counter = PeopleCounter(configs)

    a = counter.update("cam-a", [_observation(1, (50, 50))])
    b = counter.update("cam-b", [_observation(1, (10, 10), camera_id="cam-b")])

    assert a.snapshot.occupancy_for("left") == 1
    assert a.snapshot.occupancy_for("right") == 1
    assert b.snapshot.occupancy_for("other") == 0
    assert counter.snapshot("cam-a").current_occupancy == 2
    assert counter.snapshot("cam-b").current_occupancy == 0


def test_counter_overlay_returns_an_annotated_copy() -> None:
    counter = _counter(zones=(_zone("floor"),), lines=(_line(),))
    snapshot = counter.update("cam-a", [_observation(1, (50, 50))]).snapshot
    frame = np.zeros((120, 500, 3), dtype=np.uint8)

    annotated = annotate_people_counts(frame, snapshot)

    assert np.count_nonzero(frame) == 0
    assert np.count_nonzero(annotated) > 0
