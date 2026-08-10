"""Small shared event persistence boundary used by analytics modules."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Protocol, Sequence

from app.core.models import Event


class EventSink(Protocol):
    """Persistence interface; database implementations can be added later."""

    def persist(self, events: Sequence[Event]) -> None:
        """Persist an ordered event batch."""


class JsonlEventSink:
    """Append shared events as one JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def persist(self, events: Sequence[Event]) -> None:
        if not events:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            for event in events:
                stream.write(json.dumps(asdict(event), sort_keys=True) + "\n")


__all__ = ["EventSink", "JsonlEventSink"]
