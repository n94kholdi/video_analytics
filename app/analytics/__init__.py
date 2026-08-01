"""People-counting analytics built on shared tracker observations."""

from app.analytics.counting import (
    CameraCountingConfig,
    CountingResult,
    CountingSnapshot,
    LineCount,
    OccupancyCount,
    PeopleCounter,
)
from app.analytics.visualization import annotate_people_counts

__all__ = [
    "CameraCountingConfig",
    "CountingResult",
    "CountingSnapshot",
    "LineCount",
    "OccupancyCount",
    "PeopleCounter",
    "annotate_people_counts",
]
