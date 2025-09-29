# examples/adaptive_subdivision.py
# Demonstrates adaptive Loop subdivision based on edge length and curvature thresholds.

from __future__ import annotations

import numpy as np

from forge3d.geometry import primitive_mesh, subdivide_adaptive
from forge3d.io import save_obj


def main() -> None:
    # Start from a coarse sphere
    mesh = primitive_mesh("sphere", rings=8, radial_segments=8)
    print(f"Original: V={mesh.vertex_count} T={mesh.triangle_count}")

    # Refine using edge length and curvature constraints
    refined = subdivide_adaptive(
        mesh,
        edge_length_limit=0.05,     # world units
        curvature_threshold=0.6,    # radians
        max_levels=3,
        preserve_boundary=True,
    )

    print(f"Refined:  V={refined.vertex_count} T={refined.triangle_count}")

    # Optionally export to OBJ for inspection
    try:
        save_obj(refined, "adaptive_subdivided.obj")
        print("Saved: adaptive_subdivided.obj")
    except Exception as e:
        print(f"OBJ export skipped/failed: {e}")


if __name__ == "__main__":
    main()
