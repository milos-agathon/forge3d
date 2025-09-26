from _import_shim import ensure_repo_import
ensure_repo_import()

import numpy as np
from pathlib import Path

try:
    import forge3d as f3d
except Exception:
    print("forge3d extension not available; running in pure-Python fallback.")


def main():
    # Heightmap: smooth slope
    H, W = 80, 120
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    hm = 0.5 * (y + x)

    s = f3d.Scene(640, 480, grid=64, colormap="terrain")
    s.set_height_from_r32f(hm)

    # Build a synthetic external water mask (ellipse)
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H // 2, W // 2
    ry, rx = H * 0.25, W * 0.35
    mask = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0

    # Use the external mask (C1 API)
    s.enable_water_surface()
    s.set_water_surface_height(0.5)
    s.set_water_mask(mask)

    # Water appearance (C2)
    s.set_water_depth_colors((0.15, 0.65, 1.0), (0.02, 0.15, 0.3))
    s.set_water_tint(0.05, 0.3, 0.9, 0.2)
    s.set_water_alpha(0.55)

    # Shoreline foam (C3)
    s.enable_shoreline_foam()
    s.set_shoreline_foam_params(width_px=3, intensity=0.75, noise_scale=22.0)

    s.set_camera_look_at((3.2, 2.0, 3.2), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 45.0, 0.1, 100.0)

    out = Path("ex_water_mask_external_foam.png")
    s.render_png(out)
    print("Wrote", out)


if __name__ == "__main__":
    main()
