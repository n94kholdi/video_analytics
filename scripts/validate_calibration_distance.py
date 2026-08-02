"""Check a camera homography against one independently measured distance."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.analytics.speed import validate_known_ground_distance  # noqa: E402
from app.geometry.calibration import ImageToGroundProjector  # noqa: E402
from app.geometry.config import load_camera_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera_config", type=Path)
    parser.add_argument("--frame-size", nargs=2, type=int, required=True, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--first", nargs=2, type=float, required=True, metavar=("X", "Y"))
    parser.add_argument("--second", nargs=2, type=float, required=True, metavar=("X", "Y"))
    parser.add_argument("--known-metres", type=float, required=True)
    args = parser.parse_args()

    config = load_camera_config(args.camera_config)
    projector = ImageToGroundProjector.from_calibration(
        config.calibration, tuple(args.frame_size)
    )
    check = validate_known_ground_distance(
        projector, tuple(args.first), tuple(args.second), args.known_metres
    )
    print(f"projected: {check.projected_distance_metres:.3f} m")
    print(f"known:     {check.known_distance_metres:.3f} m")
    print(f"error:     {check.absolute_error_metres:.3f} m ({check.relative_error:.1%})")


if __name__ == "__main__":
    main()
