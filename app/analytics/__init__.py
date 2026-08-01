"""People counting and restricted-area analytics on shared observations."""

from app.analytics.counting import (
    CameraCountingConfig,
    CountingResult,
    CountingSnapshot,
    LineCount,
    OccupancyCount,
    PeopleCounter,
)
from app.analytics.visualization import annotate_people_counts
from app.analytics.restricted_area import (
    CameraRestrictedAreaConfig,
    IntrusionState,
    IntrusionTrackStatus,
    RestrictedAreaDetector,
    RestrictedAreaResult,
    RestrictedAreaSnapshot,
    RestrictedZoneStatus,
)
from app.analytics.restricted_visualization import annotate_restricted_areas

__all__ = [
    "CameraCountingConfig",
    "CameraRestrictedAreaConfig",
    "CountingResult",
    "CountingSnapshot",
    "LineCount",
    "IntrusionState",
    "IntrusionTrackStatus",
    "OccupancyCount",
    "PeopleCounter",
    "RestrictedAreaDetector",
    "RestrictedAreaResult",
    "RestrictedAreaSnapshot",
    "RestrictedZoneStatus",
    "annotate_people_counts",
    "annotate_restricted_areas",
]
