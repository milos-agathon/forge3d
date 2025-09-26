# examples/f1_extrude_demo.py
# Demonstrates polygon extrusion using the geometry helper API
# Exists to showcase F1 output counts and UV scaling
# RELEVANT FILES:src/geometry/extrude.rs,python/forge3d/geometry.py,tests/test_f1_extrude.py,tests/test_f15_validate.py

from __future__ import annotations

import argparse
from pathlib import Path as _Path

import numpy as np

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d.geometry as geometry  # noqa: E402


def create_polygon() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.2, 0.6],
            [0.6, 1.4],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Extrude a polygon and report mesh stats")
    parser.add_argument("--height", type=float, default=2.0, help="Extrusion height")
    parser.add_argument(
        "--cap-uv-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to cap UVs",
    )
    args = parser.parse_args()

    polygon = create_polygon()
    mesh = geometry.extrude_polygon(polygon, height=args.height, cap_uv_scale=args.cap_uv_scale)

    print("forge3d F1 extrusion demo")
    print(f"height: {args.height:.2f}")
    print(f"vertex count: {mesh.vertex_count}")
    print(f"triangle count: {mesh.triangle_count}")
    bbox_min = mesh.positions.min(axis=0)
    bbox_max = mesh.positions.max(axis=0)
    print(f"bounds: min={bbox_min}, max={bbox_max}")

    out_path = _Path("f1_extrusion.npy")
    np.save(out_path, mesh.positions)
    print(f"saved vertex positions to {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
