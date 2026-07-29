"""Verify that every Phase 1 package boundary is importable."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PACKAGES = (
    "app",
    "app.core",
    "app.detection",
    "app.tracking",
    "app.geometry",
    "app.analytics",
    "app.storage",
    "app.api",
    "app.dashboard",
)


def main() -> None:
    """Import Phase 1 packages and load the default configuration."""

    for package in PACKAGES:
        importlib.import_module(package)

    from app.core.config import load_settings

    settings = load_settings()
    print(f"Imported {len(PACKAGES)} packages.")
    print(f"Loaded configuration for {settings.name!r}.")


if __name__ == "__main__":
    main()

