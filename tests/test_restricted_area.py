"""Phase 6 synthetic restricted-area tests."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from app.analytics import (
    CameraRestrictedAreaConfig,
    IntrusionState,
    RestrictedAreaDetector,
    annotate_restricted_areas,
)
from app.core.models import Event, TrackObservation
from app.geometry.config import ActiveSchedule, NormalizedPoint, RestrictedZone
from app.storage import JsonlEventSink


def _zone(
    zone_id: str,
    *,
    left: float = 0.2,
    right: float = 0.8,
    dwell: float = 1.0,
    grace: float = 1.0,
    cooldown: float = 10.0,
    schedule: ActiveSchedule | None = None,
) -> RestrictedZone:
    return RestrictedZone(
        zone_id,
        (
            NormalizedPoint(left, 0.2),
            NormalizedPoint(right, 0.2),
            NormalizedPoint(right, 0.8),
            NormalizedPoint(left, 0.8),
        ),
        dwell,
        grace,
        cooldown,
        schedule,
    )


def _observation(
    track_id: int,
    point: tuple[float, float],
    timestamp: float,
    *,
    camera_id: str = "cam-a",
    confirmed: bool = True,
) -> TrackObservation:
    x, y = point
    return TrackObservation(
        camera_id=camera_id,
        track_id=track_id,
        timestamp=timestamp,
        frame_index=int(timestamp * 10),
        xyxy=(x - 3, y - 12, x + 3, y),
        foot_point=point,
        detection_confidence=0.9,
        confirmed=confirmed,
        trajectory=(),
    )


def _detector(*zones: RestrictedZone, sink: object | None = None) -> RestrictedAreaDetector:
    return RestrictedAreaDetector(
        (CameraRestrictedAreaConfig("cam-a", (101, 101), zones),),
        event_sink=sink,  # type: ignore[arg-type]
    )


def _event_types(events: tuple[Event, ...]) -> tuple[str, ...]:
    return tuple(event.event_type for event in events)


def test_transient_entry_never_emits_confirmed_alert() -> None:
    detector = _detector(_zone("secure", dwell=2, grace=0.5))

    entered = detector.update("cam-a", [_observation(1, (50, 50), 0)], timestamp=0)
    pending_exit = detector.update("cam-a", [_observation(1, (10, 50), 0.2)], timestamp=0.2)
    exited = detector.update("cam-a", [_observation(1, (10, 50), 0.7)], timestamp=0.7)

    assert _event_types(entered.events) == ("restricted_area_entered",)
    assert pending_exit.events == ()
    assert _event_types(exited.events) == ("restricted_area_exited",)
    assert exited.events[0].payload == {"confirmed": False, "duration_seconds": 0.7}
    assert exited.snapshot.state_for("secure", 1) is IntrusionState.EXITED
    assert exited.snapshot.zone_for("secure").current_tracks == 0
    assert exited.snapshot.zone_for("secure").cumulative_entries == 1
    assert exited.snapshot.zone_for("secure").cumulative_exits == 1


def test_confirmed_intrusion_uses_foot_point_and_dwell_time() -> None:
    detector = _detector(_zone("secure", dwell=1))
    # The box overlaps the zone, but only its bottom-center foot point controls membership.
    outside = detector.update("cam-a", [_observation(1, (10, 50), 0)], timestamp=0)
    detector.update("cam-a", [_observation(1, (50, 50), 1)], timestamp=1)
    confirmed = detector.update("cam-a", [_observation(1, (50, 50), 2)], timestamp=2)

    assert outside.events == ()
    assert _event_types(confirmed.events) == ("restricted_area_confirmed",)
    assert confirmed.events[0].zone_id == "secure"
    assert confirmed.events[0].track_id == 1
    assert confirmed.events[0].payload == {"dwell_seconds": 1}
    assert confirmed.snapshot.zone_for("secure").confirmed_tracks == 1


def test_exit_grace_period_delays_exit_event_and_status_change() -> None:
    detector = _detector(_zone("secure", dwell=0, grace=1))
    detector.update("cam-a", [_observation(1, (50, 50), 0)], timestamp=0)

    pending = detector.update("cam-a", [_observation(1, (10, 50), 1)], timestamp=1)
    still_pending = detector.update("cam-a", [], timestamp=1.9)
    exited = detector.update("cam-a", [], timestamp=2)

    assert pending.events == ()
    assert still_pending.snapshot.state_for("secure", 1) is IntrusionState.CONFIRMED
    assert _event_types(exited.events) == ("restricted_area_exited",)


def test_reentry_starts_a_new_lifecycle() -> None:
    detector = _detector(_zone("secure", dwell=0, grace=0, cooldown=0))
    first = detector.update("cam-a", [_observation(4, (50, 50), 0)], timestamp=0)
    detector.update("cam-a", [_observation(4, (10, 50), 1)], timestamp=1)
    second = detector.update("cam-a", [_observation(4, (50, 50), 2)], timestamp=2)

    assert _event_types(first.events) == (
        "restricted_area_entered",
        "restricted_area_confirmed",
    )
    assert _event_types(second.events) == (
        "restricted_area_entered",
        "restricted_area_confirmed",
    )


def test_alert_cooldown_suppresses_then_delays_repeat_confirmation() -> None:
    detector = _detector(_zone("secure", dwell=1, grace=0, cooldown=10))
    detector.update("cam-a", [_observation(1, (50, 50), 0)], timestamp=0)
    detector.update("cam-a", [_observation(1, (50, 50), 1)], timestamp=1)
    detector.update("cam-a", [_observation(1, (10, 50), 2)], timestamp=2)
    detector.update("cam-a", [_observation(1, (50, 50), 3)], timestamp=3)

    suppressed = detector.update("cam-a", [_observation(1, (50, 50), 4)], timestamp=4)
    delayed = detector.update("cam-a", [_observation(1, (50, 50), 11)], timestamp=11)

    assert suppressed.events == ()
    assert suppressed.snapshot.state_for("secure", 1) is IntrusionState.CONFIRMED
    assert _event_types(delayed.events) == ("restricted_area_confirmed",)


def test_short_tracking_gap_preserves_entry_and_dwell_state() -> None:
    detector = _detector(_zone("secure", dwell=1, grace=1))
    detector.update("cam-a", [_observation(1, (50, 50), 0)], timestamp=0)
    missing = detector.update("cam-a", [], timestamp=0.5)
    resumed = detector.update("cam-a", [_observation(1, (50, 50), 1)], timestamp=1)

    assert missing.events == ()
    assert missing.snapshot.state_for("secure", 1) is IntrusionState.ENTERED
    assert _event_types(resumed.events) == ("restricted_area_confirmed",)


def test_multiple_tracks_and_overlapping_named_zones_are_independent() -> None:
    detector = _detector(
        _zone("left", left=0.1, right=0.6, dwell=0),
        _zone("right", left=0.4, right=0.9, dwell=0),
    )
    result = detector.update(
        "cam-a",
        [_observation(1, (50, 50), 0), _observation(2, (80, 50), 0)],
        timestamp=0,
    )

    assert len(result.events) == 6
    assert result.snapshot.zone_for("left").confirmed_tracks == 1
    assert result.snapshot.zone_for("right").confirmed_tracks == 2
    assert result.snapshot.zone_for("left").current_tracks == 1
    assert result.snapshot.zone_for("right").cumulative_entries == 2
    assert result.snapshot.state_for("left", 2) is IntrusionState.OUTSIDE


def test_unconfirmed_tracks_are_ignored_and_reset_clears_all_state() -> None:
    detector = _detector(_zone("secure", dwell=0))
    ignored = detector.update(
        "cam-a", [_observation(1, (50, 50), 0, confirmed=False)], timestamp=0
    )
    detector.update("cam-a", [_observation(1, (50, 50), 1)], timestamp=1)

    detector.reset("cam-a")

    assert ignored.events == ()
    assert detector.snapshot("cam-a").tracks == ()
    assert detector.snapshot("cam-a").zone_for("secure").confirmed_tracks == 0
    assert detector.snapshot("cam-a").zone_for("secure").cumulative_entries == 0


def test_active_schedule_disables_detection_outside_configured_time() -> None:
    # Unix epoch is Thursday 00:00 UTC.
    inactive_schedule = ActiveSchedule(60, 120, (3,), "UTC")
    detector = _detector(_zone("secure", dwell=0, schedule=inactive_schedule))

    result = detector.update("cam-a", [_observation(1, (50, 50), 0)], timestamp=0)

    assert result.events == ()
    assert not result.snapshot.zone_for("secure").active


def test_overnight_active_schedule_uses_previous_configured_weekday() -> None:
    schedule = ActiveSchedule(22 * 60, 6 * 60, (0,), "UTC")
    monday_late = datetime(2026, 8, 3, 23, 0, tzinfo=timezone.utc).timestamp()
    tuesday_early = datetime(2026, 8, 4, 2, 0, tzinfo=timezone.utc).timestamp()
    tuesday_late = datetime(2026, 8, 4, 7, 0, tzinfo=timezone.utc).timestamp()

    assert schedule.is_active(monday_late)
    assert schedule.is_active(tuesday_early)
    assert not schedule.is_active(tuesday_late)


def test_events_persist_through_shared_sink_and_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "events" / "restricted.jsonl"
    detector = _detector(_zone("secure", dwell=0), sink=JsonlEventSink(path))

    result = detector.update("cam-a", [_observation(1, (50, 50), 0)], timestamp=0)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == len(result.events) == 2
    assert rows[0]["event_type"] == "restricted_area_entered"
    assert rows[1]["event_type"] == "restricted_area_confirmed"
    assert rows[1]["zone_id"] == "secure"


def test_restricted_overlay_returns_an_annotated_copy() -> None:
    config = CameraRestrictedAreaConfig("cam-a", (101, 101), (_zone("secure", dwell=0),))
    detector = RestrictedAreaDetector((config,))
    observation = _observation(1, (50, 50), 0)
    snapshot = detector.update("cam-a", [observation], timestamp=0).snapshot
    frame = np.zeros((101, 101, 3), dtype=np.uint8)

    annotated = annotate_restricted_areas(frame, config, snapshot, [observation])

    assert np.count_nonzero(frame) == 0
    assert np.count_nonzero(annotated) > 0
