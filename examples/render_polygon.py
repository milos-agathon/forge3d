from __future__ import annotations
import os

# Speed tip: force Metal backend on macOS to avoid backend probing stall on first run
# Set both legacy and current env vars for wgpu
os.environ.setdefault("WGPU_BACKENDS", "METAL")
os.environ.setdefault("WGPU_BACKEND", "metal")

import argparse
import numpy as np
from forge3d import render_polygons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render polygons from geometry or GIS files")
    parser.add_argument("--width", type=int, default=1000, help="Output width in pixels")
    parser.add_argument("--height", type=int, default=700, help="Output height in pixels")
    parser.add_argument("--quick", action="store_true", help="Use a smaller size (800x600) for speed")
    parser.add_argument("--polygons", type=str, default=None, help="Path to a vector file (.shp, .geojson, .gpkg, .gdb)")
    parser.add_argument("--layer", type=str, default=None, help="Optional layer name for layered datasets (.gdb, .gpkg)")
    parser.add_argument("--output", type=str, default="reports/render_polygon.png", help="Output PNG path")
    parser.add_argument("--viewer", action="store_true", help="Open interactive viewer (press F12 to export screenshot)")
    parser.add_argument("path", nargs="?", help="Optional positional vector path (.shp, .geojson, .gpkg, .gdb); same as --polygons")
    args = parser.parse_args()

    # Output image size (smaller defaults for faster demo)
    if args.quick:
        W, H = 800, 600
    else:
        W, H = int(args.width), int(args.height)

    # Build polygon input: either GIS file path (directly supported) or synthetic example
    # Prefer explicit --polygons over positional path if both provided
    src_path = args.polygons if args.polygons else args.path
    if src_path:
        poly_arg = {"path": src_path}
        if args.layer:
            poly_arg["layer"] = args.layer
    else:
        # Synthetic world-geometry: one filled polygon with a rectangular hole
        exterior = np.array([
            [-8.0, -4.0],
            [28.0, -4.0],
            [28.0, 24.0],
            [-8.0, 24.0],
        ], dtype=np.float64)
        hole = np.array([
            [4.0, 2.0],
            [12.0, 2.0],
            [12.0, 10.0],
            [4.0, 10.0],
        ], dtype=np.float64)
        poly_arg = {"exterior": exterior, "holes": [hole]}

    # Render
    img = render_polygons(
        polygons=poly_arg,
        size=(W, H),
        fill_rgba=(0.20, 0.50, 0.90, 1.0),
        stroke_rgba=(0.00, 0.00, 0.00, 1.0),
        stroke_width=1.5,
        show_in_viewer=bool(args.viewer),
    )

    # Save PNG only when not using viewer mode
    if not args.viewer:
        try:
            from PIL import Image
            out_path = args.output
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            Image.fromarray(img).save(out_path)
            print(f"Saved {out_path}")
        except Exception as e:
            print("(Optional) Pillow not available; skipping PNG save:", e)
