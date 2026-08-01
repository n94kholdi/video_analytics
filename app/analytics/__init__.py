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
from app.analytics.heatmap import (
    HeatmapAccumulator,
    HeatmapExportPaths,
    HeatmapGrid,
    HeatmapSnapshot,
    HeatmapVideoPaths,
    HeatmapVideoWriter,
    MovementHeatmaps,
    MovementHeatmapSnapshot,
    colorize_heatmap,
    export_heatmap_snapshot,
    export_numeric_grid,
    overlay_heatmap,
)

__all__ = [
    "CameraCountingConfig",
    "CameraRestrictedAreaConfig",
    "CountingResult",
    "CountingSnapshot",
    "HeatmapAccumulator",
    "HeatmapExportPaths",
    "HeatmapGrid",
    "HeatmapSnapshot",
    "HeatmapVideoPaths",
    "HeatmapVideoWriter",
    "LineCount",
    "IntrusionState",
    "IntrusionTrackStatus",
    "OccupancyCount",
    "MovementHeatmaps",
    "MovementHeatmapSnapshot",
    "PeopleCounter",
    "RestrictedAreaDetector",
    "RestrictedAreaResult",
    "RestrictedAreaSnapshot",
    "RestrictedZoneStatus",
    "annotate_people_counts",
    "annotate_restricted_areas",
    "colorize_heatmap",
    "export_heatmap_snapshot",
    "export_numeric_grid",
    "overlay_heatmap",
]
