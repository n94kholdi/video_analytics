#!/usr/bin/env python3
"""Click-based utility for creating a validated camera YAML file."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
import sys
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.geometry.config import (  # noqa: E402
    CameraConfig,
    CameraConfigError,
    dump_camera_config,
)


class ConfigurationEditor:
    """Small Tk canvas editor; Tk is needed only when this script is run."""

    def __init__(self, root: Any, image: Any, args: argparse.Namespace) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.args = args
        height, width = image.shape[:2]
        scale = min(1200 / width, 760 / height, 1.0)
        self.width = max(1, round(width * scale))
        self.height = max(1, round(height * scale))
        preview = cv2.resize(image, (self.width, self.height))
        ok, encoded = cv2.imencode(".png", preview)
        if not ok:
            raise RuntimeError("could not encode the reference frame")
        self.photo = tk.PhotoImage(data=base64.b64encode(encoded).decode("ascii"))

        self.mode = tk.StringVar(value="polygon")
        self.polygon_kind = tk.StringVar(value="occupancy")
        self.status = tk.StringVar(value="Click polygon vertices; Save validates all geometry.")
        self.points: dict[str, list[tuple[float, float]]] = {
            "polygon": [],
            "line": [],
            "service": [],
            "calibration": [],
        }
        self.ground_points: list[tuple[float, float]] = []

        controls = ttk.Frame(root, padding=6)
        controls.pack(fill="x")
        for mode in ("polygon", "line", "service", "calibration"):
            ttk.Radiobutton(
                controls, text=mode.title(), variable=self.mode, value=mode
            ).pack(side="left")
        ttk.Label(controls, text="Polygon use:").pack(side="left", padx=(16, 2))
        ttk.Combobox(
            controls,
            textvariable=self.polygon_kind,
            values=("occupancy", "restricted", "queue", "heatmap"),
            state="readonly",
            width=11,
        ).pack(side="left")
        ttk.Button(controls, text="Undo", command=self.undo).pack(side="left", padx=8)
        ttk.Button(controls, text="Clear mode", command=self.clear_mode).pack(side="left")
        ttk.Button(controls, text="Save YAML", command=self.save).pack(side="right")

        self.canvas = tk.Canvas(root, width=self.width, height=self.height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.bind("<Button-1>", self.click)
        ttk.Label(root, textvariable=self.status, padding=6).pack(fill="x")

    def click(self, event: Any) -> None:
        from tkinter import simpledialog

        point = (
            min(1.0, max(0.0, event.x / max(1, self.width - 1))),
            min(1.0, max(0.0, event.y / max(1, self.height - 1))),
        )
        mode = self.mode.get()
        if mode == "line" and len(self.points[mode]) == 2:
            self.points[mode].clear()
        if mode == "service":
            self.points[mode] = [point]
        else:
            self.points[mode].append(point)
        if mode == "calibration":
            x = simpledialog.askfloat("Ground X", "Ground-plane X (metres):")
            y = simpledialog.askfloat("Ground Y", "Ground-plane Y (metres):")
            if x is None or y is None:
                self.points[mode].pop()
                return
            self.ground_points.append((x, y))
        self.redraw()

    def undo(self) -> None:
        mode = self.mode.get()
        if self.points[mode]:
            self.points[mode].pop()
            if mode == "calibration":
                self.ground_points.pop()
        self.redraw()

    def clear_mode(self) -> None:
        mode = self.mode.get()
        self.points[mode].clear()
        if mode == "calibration":
            self.ground_points.clear()
        self.redraw()

    def redraw(self) -> None:
        colors = {
            "polygon": "#00ff66",
            "line": "#ffcc00",
            "service": "#ff55ff",
            "calibration": "#00ccff",
        }
        self.canvas.delete("overlay")
        for mode, points in self.points.items():
            pixels = [(x * (self.width - 1), y * (self.height - 1)) for x, y in points]
            if len(pixels) >= 2:
                flat = [coordinate for point in pixels for coordinate in point]
                self.canvas.create_line(
                    *flat,
                    fill=colors[mode],
                    width=2,
                    tags="overlay",
                )
            if mode == "polygon" and len(pixels) >= 3:
                self.canvas.create_line(
                    pixels[-1][0],
                    pixels[-1][1],
                    pixels[0][0],
                    pixels[0][1],
                    fill=colors[mode],
                    width=2,
                    tags="overlay",
                )
            for index, (x, y) in enumerate(pixels, start=1):
                self.canvas.create_oval(
                    x - 4, y - 4, x + 4, y + 4, fill=colors[mode], tags="overlay"
                )
                self.canvas.create_text(
                    x + 8,
                    y - 8,
                    text=f"{mode[0].upper()}{index}",
                    fill=colors[mode],
                    anchor="w",
                    tags="overlay",
                )
        counts = ", ".join(f"{name}: {len(value)}" for name, value in self.points.items())
        self.status.set(counts)

    def save(self) -> None:
        from tkinter import messagebox

        try:
            config = CameraConfig.from_mapping(self._mapping())
            dump_camera_config(config, self.args.output)
        except CameraConfigError as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return
        messagebox.showinfo("Saved", f"Configuration saved to {self.args.output}")
        self.status.set(f"Saved validated YAML: {self.args.output}")

    def _mapping(self) -> dict[str, Any]:
        analytics: dict[str, Any] = {
            "enabled": [],
            "occupancy_zones": [],
            "restricted_zones": [],
            "counting_lines": [],
            "queues": [],
        }
        polygon = [list(point) for point in self.points["polygon"]]
        kind = self.polygon_kind.get()
        if polygon:
            if kind == "occupancy":
                analytics["enabled"].append("occupancy")
                analytics["occupancy_zones"] = [{"id": "occupancy_1", "points": polygon}]
            elif kind == "restricted":
                analytics["enabled"].append("restricted_area")
                analytics["restricted_zones"] = [{"id": "restricted_1", "points": polygon}]
            elif kind == "queue":
                analytics["enabled"].append("queue")
                analytics["queues"] = [{
                    "id": "queue_1",
                    "polygon": polygon,
                    "service_point": {
                        "point": list(self.points["service"][0])
                        if self.points["service"]
                        else None,
                        "label": "service",
                    },
                    "overflow_threshold": self.args.queue_overflow,
                }]
        if len(self.points["line"]) == 2:
            analytics["enabled"].append("line_counting")
            analytics["counting_lines"] = [{
                "id": "entrance_1",
                "start": list(self.points["line"][0]),
                "end": list(self.points["line"][1]),
                "positive_label": "entry",
                "negative_label": "exit",
            }]
        mapping: dict[str, Any] = {
            "camera": {"id": self.args.camera_id, "name": self.args.name, "source": self.args.source},
            "analytics": analytics,
            "heatmap": {
                "region": polygon if kind == "heatmap" and polygon else None,
                "grid_size": [64, 36],
            },
        }
        if kind == "heatmap" and polygon:
            analytics["enabled"].append("heatmap")
        if self.points["calibration"]:
            mapping["calibration"] = {
                "image_points": [list(point) for point in self.points["calibration"]],
                "ground_points": [list(point) for point in self.ground_points],
                "ground_unit": "metres",
            }
        return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_frame", type=Path)
    parser.add_argument("--output", type=Path, default=Path("configs/cameras/camera.yaml"))
    parser.add_argument("--camera-id", default="camera_1")
    parser.add_argument("--name", default="Camera 1")
    parser.add_argument("--source", default="input.mp4")
    parser.add_argument("--queue-overflow", type=int, default=5)
    args = parser.parse_args()
    image = cv2.imread(str(args.reference_frame))
    if image is None:
        parser.error(f"could not read reference frame: {args.reference_frame}")
    try:
        import tkinter as tk
    except ImportError as exc:
        parser.error(f"Tk is required for this optional utility: {exc}")
    root = tk.Tk()
    root.title("Video analytics camera configuration")
    ConfigurationEditor(root, image, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
