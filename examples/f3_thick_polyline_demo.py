# examples/f3_thick_polyline_demo.py
# Demonstrates thick 3D polylines with miter/bevel/round joins and pixel-to-world width mapping.

from __future__ import annotations

import math
import numpy as np

from forge3d.geometry import generate_thick_polyline
from forge3d.io import save_obj


def pixel_to_world_width(pixel_width: float, z: float, fov_y_deg: float, image_height_px: int) -> float:
    # World height at distance z: 2*z*tan(fov/2). Pixel world size ~ world_height / image_height
    f = math.tan(math.radians(float(fov_y_deg)) * 0.5)
    world_h = 2.0 * float(z) * float(f)
    return float(pixel_width) * (world_h / float(image_height_px))


def main() -> None:
    # Simple L-shaped path in XY plane
    path = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32)

    # Map 3px on screen at z=4.0 with 45deg FOV and 1080p viewport to a world-space width
    width_world = pixel_to_world_width(3.0, z=4.0, fov_y_deg=45.0, image_height_px=1080)

    # Generate ribbons with different join styles
    bevel = generate_thick_polyline(path, width_world=width_world, depth_offset=0.002, join_style="bevel")
    miter = generate_thick_polyline(path, width_world=width_world, depth_offset=0.002, join_style="miter", miter_limit=6.0)
    roundj = generate_thick_polyline(path, width_world=width_world, depth_offset=0.002, join_style="round")

    # Save OBJs
    save_obj(bevel, "polyline_bevel.obj")
    save_obj(miter, "polyline_miter.obj")
    save_obj(roundj, "polyline_round.obj")
    print("Saved: polyline_bevel.obj, polyline_miter.obj, polyline_round.obj")


if __name__ == "__main__":
    main()
