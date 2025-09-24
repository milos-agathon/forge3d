from _import_shim import ensure_repo_import
ensure_repo_import()

import numpy as np
from pathlib import Path

try:
    import forge3d as f3d
except Exception as e:
    print("forge3d extension not available; running in pure-Python fallback.")


def main():
    # Create a synthetic heightmap: gentle slope with a low basin
    H, W = 64, 64
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    hm = 0.6 * (y + x) / 2.0
    hm[20:30, 20:44] *= 0.25  # shallow basin region

    s = f3d.Scene(512, 384, grid=64, colormap="terrain")
    s.set_height_from_r32f(hm)

    # Enable water surface and detect water from DEM
    s.enable_water_surface()
    s.set_water_surface_height(0.35)
    water_mask = s.detect_water_from_dem(method='flat', smooth_iters=1)
    print("Detected water mask:", water_mask.shape, "coverage=", float(water_mask.mean()))

    # Configure depth-aware water coloration and shoreline foam
    s.set_water_depth_colors((0.1, 0.6, 1.0), (0.0, 0.12, 0.25))
    s.set_water_tint(0.0, 0.2, 0.9, 0.25)
    s.set_water_alpha(0.5)
    s.enable_shoreline_foam()
    s.set_shoreline_foam_params(width_px=2, intensity=0.7, noise_scale=18.0)

    # Camera
    s.set_camera_look_at((3.0, 2.0, 3.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 45.0, 0.1, 100.0)

    # Render and write PNG
    out = Path("ex_water_detection_foam.png")
    s.render_png(out)
    print("Wrote", out)


if __name__ == "__main__":
    main()
