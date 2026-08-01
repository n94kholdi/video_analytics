"""Bounded movement occupancy and timestamp-based dwell heatmaps.

These are analytics heatmaps built from confirmed track foot points. They are
unrelated to neural-network activation or feature-map visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from app.core.models import TrackObservation
from app.geometry.calibration import ImageToGroundProjector
from app.geometry.config import CameraConfig, HeatmapConfig
from app.geometry.primitives import Point, point_in_polygon


@dataclass(frozen=True, slots=True)
class HeatmapGrid:
    """A rectangular row-major grid with an optional inclusion polygon."""

    size: tuple[int, int]
    bounds: tuple[float, float, float, float]
    region: tuple[Point, ...] | None = None

    def __post_init__(self) -> None:
        width, height = self.size
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or width <= 0
            or height <= 0
        ):
            raise ValueError("heatmap grid width and height must be positive")
        min_x, min_y, max_x, max_y = self.bounds
        if not all(math.isfinite(value) for value in self.bounds):
            raise ValueError("heatmap grid bounds must be finite")
        if max_x <= min_x or max_y <= min_y:
            raise ValueError("heatmap grid maximum bounds must exceed minimum bounds")

    @property
    def shape(self) -> tuple[int, int]:
        """Return NumPy's ``(rows, columns)`` shape."""

        return (self.size[1], self.size[0])

    def cell_for(self, point: Point) -> tuple[int, int] | None:
        """Map a point to ``(row, column)``, including maximum boundaries."""

        if not all(math.isfinite(value) for value in point):
            return None
        min_x, min_y, max_x, max_y = self.bounds
        x, y = point
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return None
        if self.region is not None and not point_in_polygon(
            point, self.region, include_boundary=True
        ):
            return None
        width, height = self.size
        scaled_x = (x - min_x) / (max_x - min_x) * width
        scaled_y = (y - min_y) / (max_y - min_y) * height
        # Homography arithmetic can place an exact cell edge a few ulps below
        # its mathematical value. Snap only values extremely close to an edge.
        if math.isclose(scaled_x, round(scaled_x), rel_tol=0.0, abs_tol=1e-9):
            scaled_x = float(round(scaled_x))
        if math.isclose(scaled_y, round(scaled_y), rel_tol=0.0, abs_tol=1e-9):
            scaled_y = float(round(scaled_y))
        column = min(width - 1, int(scaled_x))
        row = min(height - 1, int(scaled_y))
        return (row, column)


@dataclass(frozen=True, slots=True)
class HeatmapSnapshot:
    """Independent occupancy-sample and dwell-seconds grid snapshots."""

    occupancy: np.ndarray
    dwell_seconds: np.ndarray
    coordinate_space: str
    unit: str
    window_started_at: float | None
    last_timestamp: float | None


@dataclass(frozen=True, slots=True)
class MovementHeatmapSnapshot:
    """Image heatmaps and an optional calibrated ground-plane counterpart."""

    image: HeatmapSnapshot
    ground: HeatmapSnapshot | None
    ground_unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class HeatmapExportPaths:
    """Files written for one occupancy/dwell snapshot pair."""

    occupancy_grid: Path
    dwell_grid: Path
    occupancy_image: Path
    dwell_image: Path
    occupancy_overlay: Path | None = None
    dwell_overlay: Path | None = None


@dataclass(frozen=True, slots=True)
class HeatmapVideoPaths:
    """Paths for evolving occupancy and dwell overlay videos."""

    occupancy: Path
    dwell: Path


@dataclass(slots=True)
class _TrackSample:
    timestamp: float
    cell: tuple[int, int] | None


class HeatmapAccumulator:
    """Accumulate two fixed-size grids and bounded per-track timestamp state.

    Dwell time between two confirmed samples is assigned to the previous
    sample's cell. Intervals larger than ``max_sample_gap_seconds`` are not
    inferred. A configured aggregation period is a tumbling window; otherwise
    callers define run boundaries with :meth:`reset`.
    """

    def __init__(
        self,
        grid: HeatmapGrid,
        *,
        coordinate_space: str = "image",
        unit: str = "pixels",
        aggregation_window_seconds: float | None = None,
        max_sample_gap_seconds: float = 2.0,
        track_idle_seconds: float = 30.0,
    ) -> None:
        for value, name in (
            (aggregation_window_seconds, "aggregation_window_seconds"),
            (max_sample_gap_seconds, "max_sample_gap_seconds"),
            (track_idle_seconds, "track_idle_seconds"),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive")
        self.grid = grid
        self.coordinate_space = coordinate_space
        self.unit = unit
        self.aggregation_window_seconds = aggregation_window_seconds
        self.max_sample_gap_seconds = max_sample_gap_seconds
        self.track_idle_seconds = track_idle_seconds
        self._occupancy = np.zeros(grid.shape, dtype=np.uint64)
        self._dwell = np.zeros(grid.shape, dtype=np.float64)
        self._tracks: dict[int, _TrackSample] = {}
        self._window_started_at: float | None = None
        self._last_timestamp: float | None = None

    def update(
        self,
        observations: Sequence[TrackObservation],
        *,
        timestamp: float | None = None,
        point_getter: Callable[[TrackObservation], Point | None] | None = None,
    ) -> HeatmapSnapshot:
        """Consume confirmed observations, using each observation's timestamp."""

        getter = point_getter or (lambda observation: observation.foot_point)
        update_time = self._resolve_update_time(observations, timestamp)
        if update_time is not None:
            self._roll_window(update_time)
            self._evict_idle_tracks(update_time)

        # A tracker should emit one row per ID. Last-write deduplication makes
        # accidental duplicates unable to inflate occupancy within one update.
        unique = {observation.track_id: observation for observation in observations}
        for track_id, observation in unique.items():
            if not observation.confirmed:
                self._tracks.pop(track_id, None)
                continue
            sample_time = observation.timestamp
            if not math.isfinite(sample_time):
                continue
            self._roll_window(sample_time)
            point = getter(observation)
            cell = self.grid.cell_for(point) if point is not None else None
            if cell is not None:
                self._occupancy[cell] += 1

            previous = self._tracks.get(track_id)
            if previous is not None:
                elapsed = sample_time - previous.timestamp
                if (
                    0.0 < elapsed <= self.max_sample_gap_seconds
                    and previous.cell is not None
                ):
                    self._dwell[previous.cell] += elapsed
            if previous is None or sample_time >= previous.timestamp:
                self._tracks[track_id] = _TrackSample(sample_time, cell)
            self._last_timestamp = (
                sample_time
                if self._last_timestamp is None
                else max(self._last_timestamp, sample_time)
            )
        return self.snapshot()

    def reset(self, *, timestamp: float | None = None) -> None:
        """Clear both grids and all per-track history at an explicit boundary."""

        if timestamp is not None and not math.isfinite(timestamp):
            raise ValueError("reset timestamp must be finite")
        self._occupancy.fill(0)
        self._dwell.fill(0.0)
        self._tracks.clear()
        self._window_started_at = timestamp
        self._last_timestamp = timestamp

    def snapshot(self) -> HeatmapSnapshot:
        """Return copies so callers cannot mutate live analytics state."""

        return HeatmapSnapshot(
            self._occupancy.copy(),
            self._dwell.copy(),
            self.coordinate_space,
            self.unit,
            self._window_started_at,
            self._last_timestamp,
        )

    def _resolve_update_time(
        self,
        observations: Sequence[TrackObservation],
        timestamp: float | None,
    ) -> float | None:
        if timestamp is not None:
            if not math.isfinite(timestamp):
                raise ValueError("heatmap update timestamp must be finite")
            return timestamp
        times = [item.timestamp for item in observations if math.isfinite(item.timestamp)]
        return max(times, default=None)

    def _roll_window(self, timestamp: float) -> None:
        if self._window_started_at is None:
            self._window_started_at = timestamp
            return
        window = self.aggregation_window_seconds
        if window is not None and timestamp >= self._window_started_at + window:
            # A tumbling window is deliberately constant-memory. Aligning to
            # the current sample also handles arbitrarily large time jumps.
            self.reset(timestamp=timestamp)

    def _evict_idle_tracks(self, timestamp: float) -> None:
        expired = [
            track_id
            for track_id, sample in self._tracks.items()
            if timestamp - sample.timestamp > self.track_idle_seconds
        ]
        for track_id in expired:
            del self._tracks[track_id]


class MovementHeatmaps:
    """Camera-scoped image heatmaps plus optional calibrated ground heatmaps."""

    def __init__(
        self,
        camera_id: str,
        image: HeatmapAccumulator,
        *,
        ground: HeatmapAccumulator | None = None,
        projector: ImageToGroundProjector | None = None,
        ground_unavailable_reason: str | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.image = image
        self.ground = ground
        self.projector = projector
        self.ground_unavailable_reason = ground_unavailable_reason

    @classmethod
    def for_image(
        cls,
        camera_id: str,
        frame_size: tuple[int, int],
        *,
        settings: HeatmapConfig | None = None,
    ) -> "MovementHeatmaps":
        """Create an image-only heatmap when no camera YAML is supplied."""

        width, height = frame_size
        if width < 2 or height < 2:
            raise ValueError("heatmap frame width and height must be at least two")
        configured = settings or HeatmapConfig()
        region = (
            tuple(point.to_pixels(frame_size) for point in configured.region)
            if configured.region is not None
            else None
        )
        image = HeatmapAccumulator(
            HeatmapGrid(
                configured.grid_size,
                (0.0, 0.0, float(width - 1), float(height - 1)),
                region,
            ),
            **_accumulator_options(configured),
        )
        return cls(
            camera_id,
            image,
            ground_unavailable_reason="camera calibration is not configured",
        )

    @classmethod
    def from_camera_config(
        cls, config: CameraConfig, frame_size: tuple[int, int]
    ) -> "MovementHeatmaps":
        """Create image and, when calibrated, ground-plane accumulators."""

        width, height = frame_size
        if width < 2 or height < 2:
            raise ValueError("heatmap frame width and height must be at least two")
        settings = config.heatmap
        image_region = (
            tuple(point.to_pixels(frame_size) for point in settings.region)
            if settings.region is not None
            else None
        )
        image = HeatmapAccumulator(
            HeatmapGrid(
                settings.grid_size,
                (0.0, 0.0, float(width - 1), float(height - 1)),
                image_region,
            ),
            **_accumulator_options(settings),
        )
        projector = ImageToGroundProjector.from_calibration(
            config.calibration, frame_size
        )
        if config.calibration is None or not projector.available:
            return cls(
                config.camera_id,
                image,
                projector=projector,
                ground_unavailable_reason=projector.unavailable_reason,
            )

        ground_bounds = settings.ground_bounds or _calibration_bounds(config)
        ground = HeatmapAccumulator(
            HeatmapGrid(settings.ground_grid_size or settings.grid_size, ground_bounds),
            coordinate_space="ground",
            unit=config.calibration.ground_unit,
            **_accumulator_options(settings),
        )
        return cls(config.camera_id, image, ground=ground, projector=projector)

    def update(
        self,
        observations: Sequence[TrackObservation],
        *,
        timestamp: float | None = None,
    ) -> MovementHeatmapSnapshot:
        """Update both coordinate spaces from the same confirmed foot points."""

        camera_observations = tuple(
            item for item in observations if item.camera_id == self.camera_id
        )
        self.image.update(camera_observations, timestamp=timestamp)
        if self.ground is not None and self.projector is not None:
            self.ground.update(
                camera_observations,
                timestamp=timestamp,
                point_getter=self._project_foot_point,
            )
        return self.snapshot()

    def reset(self, *, timestamp: float | None = None) -> None:
        self.image.reset(timestamp=timestamp)
        if self.ground is not None:
            self.ground.reset(timestamp=timestamp)

    def snapshot(self) -> MovementHeatmapSnapshot:
        return MovementHeatmapSnapshot(
            self.image.snapshot(),
            self.ground.snapshot() if self.ground is not None else None,
            self.ground_unavailable_reason,
        )

    def _project_foot_point(self, observation: TrackObservation) -> Point | None:
        assert self.projector is not None
        projected = self.projector.project(observation.foot_point)
        return projected.point if projected.available else None


class HeatmapVideoWriter:
    """Stream evolving occupancy and dwell overlays to bounded MP4 outputs."""

    def __init__(
        self,
        directory: str | Path,
        *,
        prefix: str,
        fps: float,
        frame_size: tuple[int, int],
        color_map: str = "jet",
        opacity: float = 0.55,
        smoothing_sigma_cells: float = 1.0,
    ) -> None:
        width, height = frame_size
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("heatmap video FPS must be positive")
        if width <= 0 or height <= 0:
            raise ValueError("heatmap video dimensions must be positive")
        if not math.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
            raise ValueError("heatmap overlay opacity must be between 0 and 1")
        if not math.isfinite(smoothing_sigma_cells) or smoothing_sigma_cells < 0:
            raise ValueError("heatmap smoothing sigma must be non-negative")
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        self.paths = HeatmapVideoPaths(
            destination / f"{prefix}_occupancy.mp4",
            destination / f"{prefix}_dwell_seconds.mp4",
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._occupancy = cv2.VideoWriter(
            str(self.paths.occupancy), fourcc, fps, (width, height)
        )
        self._dwell = cv2.VideoWriter(
            str(self.paths.dwell), fourcc, fps, (width, height)
        )
        if not self._occupancy.isOpened() or not self._dwell.isOpened():
            self.close()
            raise OSError(f"OpenCV could not create heatmap videos in {destination}")
        self.frame_size = frame_size
        self.color_map = color_map
        self.opacity = opacity
        self.smoothing_sigma_cells = smoothing_sigma_cells

    def write(
        self,
        snapshot: HeatmapSnapshot,
        frame: np.ndarray,
        *,
        counted_points: Sequence[Point] = (),
    ) -> None:
        """Render one state, optionally marking the exact counted foot points."""

        expected_width, expected_height = self.frame_size
        if frame.shape[:2] != (expected_height, expected_width):
            raise ValueError("heatmap video frame dimensions changed during processing")
        occupancy = colorize_heatmap(
            snapshot.occupancy,
            color_map=self.color_map,
            output_size=self.frame_size,
            smoothing_sigma_cells=self.smoothing_sigma_cells,
        )
        dwell = colorize_heatmap(
            snapshot.dwell_seconds,
            color_map=self.color_map,
            output_size=self.frame_size,
            smoothing_sigma_cells=self.smoothing_sigma_cells,
        )
        occupancy_overlay = overlay_heatmap(frame, occupancy, opacity=self.opacity)
        dwell_overlay = overlay_heatmap(frame, dwell, opacity=self.opacity)
        _draw_heatmap_label(
            occupancy_overlay,
            "OCCUPANCY HEATMAP - blue=0, red=most samples "
            f"(cell max {int(snapshot.occupancy.max(initial=0))}); "
            "white dots=counted foot points",
        )
        _draw_heatmap_label(
            dwell_overlay,
            "DWELL HEATMAP - blue=0, red=most time "
            f"(cell max {snapshot.dwell_seconds.max(initial=0.0):.2f}s)",
        )
        for point in counted_points:
            center = (int(round(point[0])), int(round(point[1])))
            cv2.circle(occupancy_overlay, center, 7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(dwell_overlay, center, 7, (255, 255, 255), 2, cv2.LINE_AA)
        self._occupancy.write(occupancy_overlay)
        self._dwell.write(dwell_overlay)

    def close(self) -> None:
        """Finalize both MP4 containers."""

        self._occupancy.release()
        self._dwell.release()


def export_numeric_grid(grid: np.ndarray, path: str | Path) -> Path:
    """Export a two-dimensional numeric grid as CSV or NumPy ``.npy``."""

    if grid.ndim != 2:
        raise ValueError("heatmap numeric export requires a two-dimensional grid")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".csv":
        format_string = "%d" if np.issubdtype(grid.dtype, np.integer) else "%.9g"
        np.savetxt(output, grid, delimiter=",", fmt=format_string)
    elif suffix == ".npy":
        np.save(output, grid)
    else:
        raise ValueError("numeric heatmap path must end in .csv or .npy")
    return output


def colorize_heatmap(
    grid: np.ndarray,
    *,
    color_map: str = "jet",
    output_size: tuple[int, int] | None = None,
    smoothing_sigma_cells: float = 1.0,
) -> np.ndarray:
    """Render a full blue-to-red density surface from a numeric grid.

    Smoothing affects visualization only; exported numeric grids retain exact
    per-cell sample counts and dwell seconds.
    """

    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("heatmap image export requires a non-empty 2-D grid")
    if not math.isfinite(smoothing_sigma_cells) or smoothing_sigma_cells < 0:
        raise ValueError("heatmap smoothing sigma must be non-negative")
    finite = np.nan_to_num(grid.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    finite = np.maximum(finite, 0.0)
    if smoothing_sigma_cells > 0:
        finite = cv2.GaussianBlur(
            finite,
            (0, 0),
            sigmaX=smoothing_sigma_cells,
            sigmaY=smoothing_sigma_cells,
            borderType=cv2.BORDER_CONSTANT,
        )
    maximum = float(finite.max(initial=0.0))
    intensities = (
        np.rint(finite / maximum * 255.0).astype(np.uint8)
        if maximum > 0.0
        else np.zeros(grid.shape, dtype=np.uint8)
    )
    colored = cv2.applyColorMap(intensities, _color_map_code(color_map))
    if output_size is not None:
        width, height = output_size
        if width <= 0 or height <= 0:
            raise ValueError("heatmap output dimensions must be positive")
        colored = cv2.resize(colored, (width, height), interpolation=cv2.INTER_LINEAR)
    return colored


def overlay_heatmap(
    reference_frame: np.ndarray,
    colorized: np.ndarray,
    *,
    opacity: float = 0.55,
) -> np.ndarray:
    """Blend a colorized analytics heatmap over a BGR reference frame."""

    if reference_frame.ndim != 3 or reference_frame.shape[2] != 3:
        raise ValueError("heatmap overlay reference must be a BGR image")
    if not math.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
        raise ValueError("heatmap overlay opacity must be between 0 and 1")
    height, width = reference_frame.shape[:2]
    resized = cv2.resize(colorized, (width, height), interpolation=cv2.INTER_LINEAR)
    blended = cv2.addWeighted(reference_frame, 1.0 - opacity, resized, opacity, 0.0)
    return blended


def export_heatmap_snapshot(
    snapshot: HeatmapSnapshot,
    directory: str | Path,
    *,
    prefix: str,
    color_map: str = "jet",
    opacity: float = 0.55,
    smoothing_sigma_cells: float = 1.0,
    output_size: tuple[int, int] | None = None,
    reference_frame: np.ndarray | None = None,
) -> HeatmapExportPaths:
    """Write numeric grids, color images, and optional reference overlays."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    occupancy_grid = export_numeric_grid(
        snapshot.occupancy, destination / f"{prefix}_occupancy.csv"
    )
    dwell_grid = export_numeric_grid(
        snapshot.dwell_seconds, destination / f"{prefix}_dwell_seconds.csv"
    )
    occupancy_color = colorize_heatmap(
        snapshot.occupancy,
        color_map=color_map,
        output_size=output_size,
        smoothing_sigma_cells=smoothing_sigma_cells,
    )
    dwell_color = colorize_heatmap(
        snapshot.dwell_seconds,
        color_map=color_map,
        output_size=output_size,
        smoothing_sigma_cells=smoothing_sigma_cells,
    )
    occupancy_image = destination / f"{prefix}_occupancy.png"
    dwell_image = destination / f"{prefix}_dwell_seconds.png"
    _write_image(occupancy_image, occupancy_color)
    _write_image(dwell_image, dwell_color)
    occupancy_overlay = None
    dwell_overlay = None
    if reference_frame is not None:
        occupancy_overlay = destination / f"{prefix}_occupancy_overlay.png"
        dwell_overlay = destination / f"{prefix}_dwell_seconds_overlay.png"
        _write_image(
            occupancy_overlay,
            overlay_heatmap(reference_frame, occupancy_color, opacity=opacity),
        )
        _write_image(
            dwell_overlay,
            overlay_heatmap(reference_frame, dwell_color, opacity=opacity),
        )
    return HeatmapExportPaths(
        occupancy_grid,
        dwell_grid,
        occupancy_image,
        dwell_image,
        occupancy_overlay,
        dwell_overlay,
    )


def _accumulator_options(settings: HeatmapConfig) -> dict[str, float | None]:
    return {
        "aggregation_window_seconds": settings.aggregation_window_seconds,
        "max_sample_gap_seconds": settings.max_sample_gap_seconds,
        "track_idle_seconds": settings.track_idle_seconds,
    }


def _calibration_bounds(config: CameraConfig) -> tuple[float, float, float, float]:
    assert config.calibration is not None
    xs = [point[0] for point in config.calibration.ground_points]
    ys = [point[1] for point in config.calibration.ground_points]
    bounds = (min(xs), min(ys), max(xs), max(ys))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("calibration ground points do not span a two-dimensional grid")
    return bounds


def _color_map_code(name: str) -> int:
    normalized = name.strip().lower().replace("-", "_")
    supported = {
        "autumn": cv2.COLORMAP_AUTUMN,
        "bone": cv2.COLORMAP_BONE,
        "hot": cv2.COLORMAP_HOT,
        "inferno": cv2.COLORMAP_INFERNO,
        "jet": cv2.COLORMAP_JET,
        "magma": cv2.COLORMAP_MAGMA,
        "parula": cv2.COLORMAP_PARULA,
        "plasma": cv2.COLORMAP_PLASMA,
        "turbo": cv2.COLORMAP_TURBO,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }
    try:
        return supported[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported heatmap color map {name!r}; choose one of "
            + ", ".join(sorted(supported))
        ) from exc


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"OpenCV could not write heatmap image: {path}")


def _draw_heatmap_label(image: np.ndarray, label: str) -> None:
    scale = max(0.6, min(image.shape[:2]) / 900.0)
    thickness = max(1, round(scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    cv2.rectangle(
        image,
        (12, 12),
        (28 + text_width, 28 + text_height + baseline),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        label,
        (20, 22 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
