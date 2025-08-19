import numpy as np
import vulkan_forge as vf

def main():
    H, W = 32, 32
    h32 = (np.linspace(0,1,W, dtype=np.float32)[None,:] * np.ones((H,1), np.float32)).copy(order="C")
    r = vf.Renderer(128, 128)
    r.add_terrain(h32, spacing=(1.0, 1.0), exaggeration=1.0, colormap="viridis")
    mn, mx, mean, std = r.terrain_stats()
    r.normalize_terrain("zscore")
    mn2, mx2, *_ = r.terrain_stats()
    r.set_height_range(mn2, mx2)
    print("Original stats:", mn, mx)
    print("After zscore normalization:", mn2, mx2)

if __name__ == "__main__":
    main()