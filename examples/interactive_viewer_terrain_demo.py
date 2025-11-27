"""Interactive viewer demo built on top of the terrain_demo configuration.

This example shows how to:

- Reuse the existing terrain_demo CLI to configure a terrain scene.
- Launch the forge3d interactive viewer from Python.
- Pass optional snapshot parameters to capture a high-resolution PNG.

Run from the repository root::

    python examples/interactive_viewer_terrain_demo.py --width 1600 --height 900 \
        --snapshot-path snapshot.png --snapshot-width 2560 --snapshot-height 1440

You can then continue to drive the viewer via the terminal as documented in
``docs/interactive_viewer.rst`` (e.g. ``:obj``, ``:gltf``, ``:snapshot``, ``:p5``, etc.).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import forge3d as f3d

# We deliberately reuse the terrain_demo CLI surface so that users familiar with
# the headless demo can lift the same arguments into an interactive session.
import terrain_demo


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the forge3d interactive viewer for a terrain demo scene.",
    )

    # Mirror a subset of terrain_demo flags that are most relevant for quick
    # experimentation. The full terrain_demo CLI remains available separately.
    parser.add_argument(
        "--dem",
        type=Path,
        default=terrain_demo.DEFAULT_DEM,
        help="Path to a GeoTIFF DEM (same default as terrain_demo).",
    )
    parser.add_argument(
        "--hdr",
        type=Path,
        default=terrain_demo.DEFAULT_HDR,
        help="Path to an environment HDR image (same default as terrain_demo).",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=terrain_demo.DEFAULT_SIZE,
        metavar=("WIDTH", "HEIGHT"),
        help="Initial viewer window size in pixels.",
    )

    # Exposure is consumed by terrain_demo._build_renderer_config; we forward it
    # so that lighting behavior matches the headless demo by default.
    parser.add_argument(
        "--exposure",
        type=float,
        default=1.0,
        help="ACES exposure multiplier (forwarded to terrain_demo).",
    )

    # Interactive-viewer specific flags.
    parser.add_argument(
        "--title",
        type=str,
        default="forge3d Terrain Interactive Viewer",
        help="Window title for the interactive viewer.",
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        help=(
            "Optional path for an automatic snapshot shortly after startup. "
            "Behaves like an initial :snapshot <path> command."
        ),
    )
    parser.add_argument(
        "--snapshot-width",
        type=int,
        help=(
            "Optional snapshot width override in pixels. Must be used together "
            "with --snapshot-height."
        ),
    )
    parser.add_argument(
        "--snapshot-height",
        type=int,
        help=(
            "Optional snapshot height override in pixels. Must be used together "
            "with --snapshot-width."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Create the terrain renderer configuration using the same helper as
    # examples/terrain_demo.py. This ensures the GPU-side resources and
    # lighting setup match the headless demo, while we hand control over to
    # the interactive viewer for exploration.
    _ = terrain_demo._build_renderer_config(args)

    width, height = args.size

    f3d.open_viewer(
        width=int(width),
        height=int(height),
        title=str(args.title),
        snapshot_path=str(args.snapshot_path) if args.snapshot_path is not None else None,
        snapshot_width=args.snapshot_width,
        snapshot_height=args.snapshot_height,
        # You can pre-populate additional settings via initial commands. For
        # example, enable GTAO and volumetric fog at startup:
        initial_commands=[
            ":gi gtao on",  # enable GTAO
            ":fog on",      # enable fog
        ],
    )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
