#!/usr/bin/env python3
"""Render an oblique ORBIS descent over the Jungfrau region."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import forge3d as f3d


JUNGFRAU_LON = 7.910
JUNGFRAU_LAT = 46.494
START_ALTITUDE_M = 408_000.0


def descent_waypoints(
    frame_count: int,
    *,
    end_altitude_m: float,
    heading_deg: float = 225.0,
    start_pitch_deg: float = 70.0,
    end_pitch_deg: float = 85.0,
) -> list[tuple[float, float, float, float, float]]:
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    if not 0.0 <= end_altitude_m < START_ALTITUDE_M:
        raise ValueError("end_altitude_m must be in [0, 408000)")
    start = math.log(START_ALTITUDE_M + 1.0)
    end = math.log(end_altitude_m + 1.0)
    waypoints = []
    for index in range(frame_count):
        t = index / (frame_count - 1)
        altitude = math.exp(start + (end - start) * t) - 1.0
        pitch = start_pitch_deg + (end_pitch_deg - start_pitch_deg) * t
        waypoints.append((JUNGFRAU_LON, JUNGFRAU_LAT, altitude, heading_deg, pitch))
    waypoints[0] = (JUNGFRAU_LON, JUNGFRAU_LAT, START_ALTITUDE_M, heading_deg, start_pitch_deg)
    waypoints[-1] = (JUNGFRAU_LON, JUNGFRAU_LAT, end_altitude_m, heading_deg, end_pitch_deg)
    return waypoints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dem",
        type=Path,
        default=Path(__file__).parents[1] / "assets" / "tif" / "switzerland_dem.tif",
    )
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--end-altitude-m", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=Path("orbis-swiss-alps-final.png"))
    args = parser.parse_args()

    scene = f3d.GlobeScene(args.dem, JUNGFRAU_LON, JUNGFRAU_LAT, "Jungfrau")
    metrics = scene.scripted_descent(
        descent_waypoints(args.frames, end_altitude_m=args.end_altitude_m)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.snapshot().save(args.output)
    print(metrics.as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
