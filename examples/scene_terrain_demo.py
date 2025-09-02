from _import_shim import ensure_repo_import
ensure_repo_import()

try:
    import forge3d as f3d
except Exception:
    print("forge3d extension not available; skipping demo.")
    import sys; sys.exit(0)

from pathlib import Path
import numpy as np

def main():
    H, W = 64, 64
    height = (np.linspace(0,1,W, dtype=np.float32)[None,:] * np.ones((H,1), np.float32)).copy(order="C")
    s = f3d.Scene(256, 256, grid=128, colormap="viridis")
    s.set_height_from_r32f(height)
    s.set_camera_look_at((3,2,3), (0,0,0), (0,1,0), 45.0, 0.1, 100.0)
    s.render_png(Path("ex_terrain.png"))
    rgba = s.render_rgba()
    print("terrain.png written, rgba shape:", rgba.shape)

if __name__ == "__main__":
    main()