"""Recorded-video applications exposed by the HTTP API."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ApplicationPreset:
    """A safe, named mapping to one of the existing CLI pipelines."""

    application_id: str
    name: str
    description: str
    module: str
    arguments: tuple[str, ...] = ()
    requires_camera_config: bool = False

    def public_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("module")
        data.pop("arguments")
        data["id"] = data.pop("application_id")
        return data


APPLICATIONS = (
    ApplicationPreset(
        "detection",
        "Human detection",
        "Detect people and draw bounding boxes.",
        "app.detection.cli",
    ),
    ApplicationPreset(
        "tracking",
        "People tracking",
        "Detect and track people with IDs and trajectories.",
        "app.tracking.cli",
    ),
    ApplicationPreset(
        "people_counting",
        "People counting",
        "Track people, report visible occupancy, and count configured line crossings.",
        "app.analytics.cli",
    ),
    ApplicationPreset(
        "restricted_area",
        "Restricted-area monitoring",
        "Detect entries, confirmed intrusions, and exits for configured restricted zones.",
        "app.analytics.cli",
        ("--enable-restricted-area",),
        True,
    ),
    ApplicationPreset(
        "heatmap",
        "Movement and dwell heatmaps",
        "Generate annotated video, occupancy heatmaps, and dwell-time heatmaps.",
        "app.analytics.cli",
        ("--enable-heatmap",),
    ),
    ApplicationPreset(
        "vertical_queue",
        "Automatic vertical queues",
        "Group vertically aligned people and estimate queue counts and movement speed.",
        "app.analytics.cli",
        ("--enable-queue", "--queue-mode", "vertical"),
    ),
    ApplicationPreset(
        "configured_queue",
        "Configured queue monitoring",
        "Measure queue length, overflow, waiting time, and progress in configured polygons.",
        "app.analytics.cli",
        ("--enable-queue", "--queue-mode", "configured"),
        True,
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
    ),
)

APPLICATION_BY_ID = {item.application_id: item for item in APPLICATIONS}


def get_application(application_id: str) -> ApplicationPreset:
    try:
        return APPLICATION_BY_ID[application_id]
    except KeyError as exc:
        raise ValueError(f"unknown application: {application_id}") from exc
