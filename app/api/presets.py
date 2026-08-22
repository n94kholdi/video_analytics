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
TRACKING_METRICS = (
    MetricDefinition("current_people", "Current people", display="counter"),
    MetricDefinition("active_tracker", "Active tracker", value_type="string", display="status"),
) + PROCESSING_METRICS
COUNTING_METRICS = TRACKING_METRICS + (
    MetricDefinition("zone_occupancy", "Area occupancy", value_type="table", display="table"),
)
RESTRICTED_METRICS = (
    MetricDefinition("restricted_occupancy", "Currently inside restricted area", display="status"),
    MetricDefinition("restricted_entries", "Restricted-area entries", aggregation="total", display="counter"),
    MetricDefinition("restricted_exits", "Restricted-area exits", aggregation="total", display="counter"),
    MetricDefinition("restricted_violations", "Confirmed restricted-area alerts", aggregation="total", display="counter"),
)
QUEUE_STATUS_METRICS = (
    MetricDefinition("queue_length", "People in the queue", unit="people", display="chart"),
    MetricDefinition("queue_wait_seconds", "Queue waiting time", unit="s", aggregation="average", display="chart"),
    MetricDefinition("queue_speed", "Queue movement speed", unit="px/s", aggregation="average", display="chart"),
    MetricDefinition("queue_details", "People and speed per configured queue", value_type="table", display="table"),
    MetricDefinition("average_person_speed", "Average person speed", unit="px/s", aggregation="average", display="chart"),
)
QUEUE_METRICS = COUNTING_METRICS + QUEUE_STATUS_METRICS
CROWDED_REGION_METRIC = MetricDefinition(
    "top_crowded_regions",
    "Top crowded regions",
    value_type="table",
    display="table",
)
HEATMAP_METRICS = COUNTING_METRICS + (CROWDED_REGION_METRIC,)


APPLICATIONS = (
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
        "Queue line monitoring",
        "Count people inside a user-drawn queue polygon and estimate queue movement speed.",
        "app.analytics.cli",
        ("--enable-queue", "--queue-mode", "configured"),
        True,
        TRACKING_METRICS + QUEUE_STATUS_METRICS,
    ),
)

APPLICATION_BY_ID = {item.application_id: item for item in APPLICATIONS}


def get_application(application_id: str) -> ApplicationPreset:
    try:
        return APPLICATION_BY_ID[application_id]
    except KeyError as exc:
        raise ValueError(f"unknown application: {application_id}") from exc
