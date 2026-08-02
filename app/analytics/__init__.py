"""People counting, restricted-area, heatmap, and queue analytics."""

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
from app.analytics.queue import (
    CameraQueueConfig,
    QueueAnalyzer,
    QueueResult,
    QueueSnapshot,
    QueueStatus,
    QueueTrackState,
    QueueTrackStatus,
)
from app.analytics.queue_visualization import annotate_queues
from app.analytics.vertical_queue import (
    VerticalQueueAnalyzer,
    VerticalQueueConfig,
    VerticalQueueRow,
    VerticalQueueSnapshot,
)
from app.analytics.vertical_queue_visualization import (
    annotate_vertical_queues,
    hotter_row_color,
    vertical_row_color,
)

__all__ = [
    "CameraCountingConfig",
    "CameraRestrictedAreaConfig",
    "CameraQueueConfig",
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
    "QueueAnalyzer",
    "QueueResult",
    "QueueSnapshot",
    "QueueStatus",
    "QueueTrackState",
    "QueueTrackStatus",
    "RestrictedAreaDetector",
    "RestrictedAreaResult",
    "RestrictedAreaSnapshot",
    "RestrictedZoneStatus",
    "VerticalQueueAnalyzer",
    "VerticalQueueConfig",
    "VerticalQueueRow",
    "VerticalQueueSnapshot",
    "annotate_people_counts",
    "annotate_queues",
    "annotate_restricted_areas",
    "annotate_vertical_queues",
    "colorize_heatmap",
    "export_heatmap_snapshot",
    "export_numeric_grid",
    "hotter_row_color",
    "overlay_heatmap",
    "vertical_row_color",
]
