"""Recorded-video applications exposed by the HTTP API."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    key: str
    label: str
    value_type: str = "number"
    unit: str | None = None
    aggregation: str = "current"
    display: str = "card"
    availability: str = "both"


@dataclass(frozen=True, slots=True)
class ApplicationPreset:
    """A safe, named mapping to one of the existing CLI pipelines."""

    application_id: str
    name: str
    description: str
    module: str
    arguments: tuple[str, ...] = ()
    requires_camera_config: bool = False
    metrics: tuple[MetricDefinition, ...] = ()

    def public_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("module")
        data.pop("arguments")
        data["id"] = data.pop("application_id")
        data["metric_schema"] = data.pop("metrics")
        return data


PROCESSING_METRICS = (
    MetricDefinition("processing_fps", "Processing FPS", unit="fps", aggregation="average", display="chart"),
    MetricDefinition("frame_count", "Processed frames", aggregation="total", display="counter"),
    MetricDefinition("progress", "Progress", unit="%", display="status", availability="live"),
    MetricDefinition("elapsed_seconds", "Elapsed time", unit="s", display="counter"),
)
DETECTION_METRICS = (
    MetricDefinition("current_people", "Current people", display="counter"),
    MetricDefinition("total_detections", "Total detections", aggregation="total", display="counter"),
) + PROCESSING_METRICS
TRACKING_METRICS = (
    MetricDefinition("current_people", "Current people", display="counter"),
    MetricDefinition("total_unique_people", "Total unique people", aggregation="total", display="counter"),
    MetricDefinition("active_tracks", "Active tracks", display="status"),
    MetricDefinition("lost_tracks", "Lost tracks", aggregation="total", display="counter"),
) + PROCESSING_METRICS
COUNTING_METRICS = TRACKING_METRICS + (
    MetricDefinition("entry_count", "Entries", aggregation="total", display="counter"),
    MetricDefinition("exit_count", "Exits", aggregation="total", display="counter"),
    MetricDefinition("zone_occupancy", "Area occupancy", value_type="table", display="table"),
)
RESTRICTED_METRICS = (
    MetricDefinition("restricted_occupancy", "Currently inside restricted area", display="status"),
    MetricDefinition("restricted_entries", "Restricted-area entries", aggregation="total", display="counter"),
    MetricDefinition("restricted_exits", "Restricted-area exits", aggregation="total", display="counter"),
    MetricDefinition("restricted_violations", "Confirmed restricted-area alerts", aggregation="total", display="counter"),
)
QUEUE_METRICS = COUNTING_METRICS + (
    MetricDefinition("queue_length", "Detected people queue", unit="people", display="chart"),
    MetricDefinition("queue_wait_seconds", "Queue waiting time", unit="s", aggregation="average", display="chart"),
    MetricDefinition("queue_speed", "Queue movement speed", unit="px/s", aggregation="average", display="chart"),
    MetricDefinition("average_person_speed", "Average person speed", unit="px/s", aggregation="average", display="chart"),
)
CROWDED_REGION_METRIC = MetricDefinition(
    "top_crowded_regions",
    "Top crowded regions",
    value_type="table",
    display="table",
)
HEATMAP_METRICS = COUNTING_METRICS + (CROWDED_REGION_METRIC,)


APPLICATIONS = (
    ApplicationPreset(
        "detection",
        "Human detection",
        "Detect people and draw bounding boxes.",
        "app.detection.cli",
        metrics=DETECTION_METRICS,
    ),
    ApplicationPreset(
        "tracking",
        "People tracking",
        "Detect and track people with IDs and trajectories.",
        "app.tracking.cli",
        metrics=TRACKING_METRICS,
    ),
    ApplicationPreset(
        "people_counting",
        "People counting",
        "Track people, report visible occupancy, and count configured line crossings.",
        "app.analytics.cli",
        metrics=COUNTING_METRICS,
    ),
    ApplicationPreset(
        "restricted_area",
        "Restricted-area monitoring",
        "Detect entries, confirmed intrusions, and exits for configured restricted zones.",
        "app.analytics.cli",
        ("--enable-restricted-area",),
        True,
        TRACKING_METRICS + RESTRICTED_METRICS,
    ),
    ApplicationPreset(
        "heatmap",
        "Movement and dwell heatmaps",
        "Generate annotated video, occupancy heatmaps, and dwell-time heatmaps.",
        "app.analytics.cli",
        ("--enable-heatmap",),
        metrics=HEATMAP_METRICS,
    ),
    ApplicationPreset(
        "vertical_queue",
        "Automatic vertical queues",
        "Group vertically aligned people and estimate queue counts and movement speed.",
        "app.analytics.cli",
        ("--enable-queue", "--queue-mode", "vertical"),
        metrics=QUEUE_METRICS,
    ),
    ApplicationPreset(
        "configured_queue",
        "Detected people queue",
        "Measure queue length, overflow, waiting time, and progress in configured polygons.",
        "app.analytics.cli",
        ("--enable-queue", "--queue-mode", "configured"),
        True,
        QUEUE_METRICS,
    ),
    ApplicationPreset(
        "full_analytics",
        "Combined configured analytics",
        "Run counting, restricted areas, heatmaps, configured queues, and speed together.",
        "app.analytics.cli",
        (
            "--enable-restricted-area",
            "--enable-heatmap",
            "--enable-queue",
            "--queue-mode",
            "configured",
        ),
        True,
        COUNTING_METRICS
        + RESTRICTED_METRICS
        + tuple(
            metric
            for metric in QUEUE_METRICS
            if metric.key.startswith("queue_")
            or metric.key == "average_person_speed"
        )
        + (CROWDED_REGION_METRIC,),
    ),
)

APPLICATION_BY_ID = {item.application_id: item for item in APPLICATIONS}


def get_application(application_id: str) -> ApplicationPreset:
    try:
        return APPLICATION_BY_ID[application_id]
    except KeyError as exc:
        raise ValueError(f"unknown application: {application_id}") from exc
