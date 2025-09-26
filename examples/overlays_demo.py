from _import_shim import ensure_repo_import
ensure_repo_import()

import numpy as np
from pathlib import Path

try:
    import forge3d as f3d
except Exception:
    print("forge3d extension not available; running in pure-Python fallback.")


def make_heightmap(h=64, w=96):
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    hm = 0.4 * (np.sin(2*np.pi*x) * np.cos(2*np.pi*y) * 0.5 + 0.5) + 0.1 * y
    return hm


def main():
    s = f3d.Scene(640, 400, grid=64, colormap="terrain")
    hm = make_heightmap()
    s.set_height_from_r32f(hm)
    s.set_camera_look_at((3.0, 2.0, 3.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 45.0, 0.1, 100.0)

    # D4: Drape raster overlay (simple stripes)
    H, W = 120, 200
    ov = np.zeros((H, W, 3), dtype=np.uint8)
    ov[:, ::8, 0] = 255
    ov[::8, :, 1] = 255
    s.set_raster_overlay(ov, alpha=0.35, offset_xy=(30, 40), scale=1.5)

    # D5: Altitude overlay
    s.enable_altitude_overlay(alpha=0.25)

    # D8: Hillshade overlay
    s.enable_shadow_overlay(azimuth_deg=300.0, altitude_deg=35.0, strength=0.6, blend='multiply')

    # D6/D7: Contours
    s.generate_contours(interval=0.05, smooth=1)
    s.enable_contours_overlay(color=(0,0,0), width_px=1)

    # D2: Compass, D3: Scale bar
    s.enable_compass_rose(position='top_right', size_px=64, color=(255,255,255), bg_alpha=0.2)
    s.enable_scale_bar(position='bottom_left', max_width_px=200, color=(255,255,255))

    # D1/D10: Text overlays + title bar
    s.add_text_overlay("Sample Label", x=40, y=60, size_px=18, color=(255,255,0), anchor='top_left')
    s.set_title_bar("Forge3D Overlays Demo", height_px=32, bg_rgba=(0,0,0,128), color=(255,255,255))

    out = Path("ex_overlays_demo.png")
    s.render_png(out)
    print("Wrote", out)


if __name__ == "__main__":
    main()
