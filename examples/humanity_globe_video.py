#!/usr/bin/env python3
"""Recreate the Humanity Globe population-density video with forge3d helpers.

Reference concept and original rayshader/rayrender recipe:
https://gist.github.com/tylermorganwall/3ee1c6e2a5dff19aca7836c05cbbf9ac

Data source family: GPW-v4 population density, revision 11, 2020.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples" / "out" / "humanity_globe"
DEFAULT_CACHE_DIR = REPO_ROOT / "examples" / ".cache" / "humanity_globe"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "humanity_globe_forge3d.mp4"
DEFAULT_PREVIEW = DEFAULT_OUTPUT_DIR / "humanity_globe_preview.png"
DEFAULT_GPW_15MIN = REPO_ROOT / "data" / "gpw_v4_population_density_rev11_2020_15_min.tif"
DEFAULT_SIZE = 720
DEFAULT_FPS = 25
DEFAULT_DURATION = 28.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a forge3d recreation of the Humanity Globe population-density MP4."
    )
    parser.add_argument("--gpw-tif", type=Path, default=None, help="Explicit GPW-v4 2020 population-density GeoTIFF.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--frames-only", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def frame_count(args: argparse.Namespace) -> int:
    if args.frames is not None:
        return max(1, int(args.frames))
    return max(1, int(round(float(args.fps) * float(args.duration))))


def main() -> int:
    parse_args()
    raise SystemExit("Rendering is implemented in later tasks.")


if __name__ == "__main__":
    raise SystemExit(main())
