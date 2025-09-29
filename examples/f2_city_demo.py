# examples/f2_city_demo.py
# Minimal city demo: extrude a handful of OSM-like building footprints and export OBJ.

from __future__ import annotations

import numpy as np

from forge3d.io import import_osm_buildings_extrude, save_obj


def main() -> None:
    # Synthetic footprints (XY), heights in meters
    blocks = []
    rng = np.random.default_rng(42)

    # Grid of rectangles with varied sizes/heights
    for gx in range(3):
        for gy in range(2):
            x0, y0 = gx * 4.0, gy * 3.0
            w = 1.5 + 0.7 * rng.random()
            h = 1.2 + 0.6 * rng.random()
            rect = np.array(
                [
                    [x0 + 0.0, y0 + 0.0],
                    [x0 + w,   y0 + 0.0],
                    [x0 + w,   y0 + h  ],
                    [x0 + 0.0, y0 + h  ],
                ],
                dtype=np.float32,
            )
            height = float(rng.integers(8, 22))  # 8–21 meters
            blocks.append({"coords": rect, "height": height})

    # Extrude and export
    mesh = import_osm_buildings_extrude(blocks, default_height=10.0)
    print(f"Buildings: V={mesh.vertex_count} T={mesh.triangle_count}")
    try:
        save_obj(mesh, "city_demo_buildings.obj")
        print("Saved: city_demo_buildings.obj")
    except Exception as e:
        print(f"OBJ export skipped/failed: {e}")


if __name__ == "__main__":
    main()
