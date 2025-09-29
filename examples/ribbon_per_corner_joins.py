# examples/ribbon_per_corner_joins.py
# Demonstrates per-corner ribbon join styles and miter limiting.

from __future__ import annotations

import numpy as np

from forge3d.geometry import generate_ribbon
from forge3d.io import save_obj


def main() -> None:
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [1.5, 1.0, 0.0],
            [2.5, 1.0, 0.0],
            [2.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    # Per-vertex join styles: 0=miter, 1=bevel, 2=round
    join_styles = np.array([0, 1, 2, 0, 0], dtype=np.uint8)
    rib = generate_ribbon(path, width_start=0.2, width_end=0.2, join_style="miter", miter_limit=6.0, join_styles=join_styles)

    print(f"Ribbon: V={rib.vertex_count} T={rib.triangle_count}")
    try:
        save_obj(rib, "ribbon_per_corner.obj")
        print("Saved: ribbon_per_corner.obj")
    except Exception as e:
        print(f"OBJ export skipped/failed: {e}")


if __name__ == "__main__":
    main()
