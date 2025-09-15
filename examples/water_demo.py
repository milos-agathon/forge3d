# examples/water_demo.py
# Minimal demo that shades a small grid with the water helper and writes PNG.
# This file exists to showcase A6 usage and output an artifact into out/.
# RELEVANT FILES:python/forge3d/pbr.py,README.md,docs/api/water.md

import os
import numpy as np
from forge3d.water import water_shade
from forge3d import numpy_to_png


def main():
    H, W = 64, 64
    n = np.dstack([np.zeros((H, W), np.float32), np.zeros((H, W), np.float32), np.ones((H, W), np.float32)])
    v = np.dstack([np.zeros((H, W), np.float32), np.zeros((H, W), np.float32), -np.ones((H, W), np.float32)])
    # Sweep light dir over grid
    xv, yv = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    l = np.dstack([0.5 + 0.5 * xv, 0.8 * np.ones_like(xv), 0.2 + 0.2 * yv]).astype(np.float32)
    base = np.array([0.1, 0.2, 0.8], np.float32)
    rgb = water_shade(n, v, l, base, ior=1.33, absorption=(0.0, 0.05, 0.1), roughness=0.08, thickness=1.0)
    # Tonemap and write PNG
    ldr = (rgb / (1.0 + rgb)).clip(0, 1)
    img = (ldr * 255 + 0.5).astype(np.uint8)
    out_dir = os.path.join("out")
    os.makedirs(out_dir, exist_ok=True)
    numpy_to_png(os.path.join(out_dir, "water_demo.png"), img)


if __name__ == "__main__":
    main()
