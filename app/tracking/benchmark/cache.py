"""Serialize detector outputs once so every tracker sees the same boxes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator

from app.core.models import Detection


@dataclass(frozen=True, slots=True)
class CachedDetection:
    frame_index: int
    timestamp: float
    source_index: int
    detections: tuple[Detection, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "source_index": self.source_index,
            "detections": [
                {
                    "xyxy": list(item.xyxy),
                    "confidence": item.confidence,
                    "class_id": item.class_id,
                    "class_name": item.class_name,
                }
                for item in self.detections
            ],
        }

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "CachedDetection":
        detections = tuple(
            Detection(
                xyxy=tuple(float(coord) for coord in item["xyxy"]),  # type: ignore[arg-type,index]
                confidence=float(item["confidence"]),
                class_id=int(item.get("class_id", 0)),
                class_name=item.get("class_name", "person"),  # type: ignore[arg-type]
            )
            for item in values.get("detections", [])  # type: ignore[union-attr]
        )
        return cls(
            frame_index=int(values["frame_index"]),
            timestamp=float(values["timestamp"]),
            source_index=int(values.get("source_index", values["frame_index"])),
            detections=detections,
        )


class DetectionCache:
    """JSONL cache of detector outputs keyed by processed-frame index."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, frames: Iterable[CachedDetection]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as stream:
            for item in frames:
                stream.write(json.dumps(item.to_mapping(), separators=(",", ":")) + "\n")

    def read(self) -> list[CachedDetection]:
        if not self.path.is_file():
            raise FileNotFoundError(f"detection cache not found: {self.path}")
        frames: list[CachedDetection] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                frames.append(CachedDetection.from_mapping(json.loads(line)))
        return frames

    def __iter__(self) -> Iterator[CachedDetection]:
        yield from self.read()
