"""Package-boundary import smoke tests."""

import importlib

import pytest


@pytest.mark.parametrize(
    "package",
    [
        "app",
        "app.core",
        "app.detection",
        "app.tracking",
        "app.geometry",
        "app.analytics",
        "app.storage",
        "app.api",
        "app.management",
        "app.tracking.benchmark",
    ],
)
def test_package_imports(package: str) -> None:
    assert importlib.import_module(package) is not None

