import os
from typing import Iterable

# Hint: force Metal backend on macOS to avoid backend probing stall on first run
os.environ.setdefault("WGPU_BACKENDS", "METAL")
os.environ.setdefault("WGPU_BACKEND", "metal")

import numpy as np
from forge3d import render_polygons, render_overlay


def compute_transform(rings: Iterable[np.ndarray], size: tuple[int, int]) -> tuple[float, float, float, float]:
    W, H = int(size[0]), int(size[1])
    min_x = float("inf"); min_y = float("inf"); max_x = float("-inf"); max_y = float("-inf")
    for arr in rings:
        a = np.asarray(arr, dtype=np.float64)
        if a.size == 0:
            continue
        min_x = min(min_x, float(np.min(a[:, 0])))
        min_y = min(min_y, float(np.min(a[:, 1])))
        max_x = max(max_x, float(np.max(a[:, 0])))
        max_y = max(max_y, float(np.max(a[:, 1])))
    dx = max(max_x - min_x, 1e-6)
    dy = max(max_y - min_y, 1e-6)
    d = max(dx, dy)
    cx, cy = 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)
    sx, sy = W / d, H / d
    tx, ty = W * 0.5 - sx * cx, H * 0.5 - sy * cy
    return sx, sy, tx, ty


if __name__ == "__main__":
    # Output size and synthetic map geometry
    W, H = 1200, 900
    exterior = np.array([
        [-8.0, -4.0],
        [28.0, -4.0],
        [28.0, 24.0],
        [-8.0, 24.0],
    ], dtype=np.float64)
    hole = np.array([
        [4.0, 2.0],
        [12.0, 2.0],
        [12.0, 10.0],
        [4.0, 10.0],
    ], dtype=np.float64)

    # Shared world->pixel transform
    xf = compute_transform([exterior, hole], (W, H))

    # 1) Render the filled base map
    base = render_polygons(
        polygons=[{"exterior": exterior, "holes": [hole]}],
        size=(W, H),
        fill_rgba=(0.90, 0.92, 0.95, 1.0),   # light fill for map
        stroke_rgba=(0.20, 0.20, 0.20, 1.0),
        stroke_width=1.25,
        transform=xf,
    )

    # 2) Build synthetic overlays: roads (polylines) and POIs (points)
    # Roads: a couple of line segments
    road1 = np.array([[-6.0, -2.0], [26.0, 22.0]], dtype=np.float64)
    road2 = np.array([[-6.0, 12.0], [26.0, -2.0]], dtype=np.float64)
    roads = [road1, road2]

    # POIs: a few points scattered around
    pois = np.array([
        [0.0, 0.0],
        [10.0, 6.0],
        [18.0, 14.0],
    ], dtype=np.float64)

    # 3) Overlay using the same transform
    composed = render_overlay(
        base_rgba=base,
        polygons=[{"exterior": exterior, "holes": [hole]}],  # draw outlines too
        polylines=roads,
        points=pois,
        stroke_rgba=(0.05, 0.05, 0.05, 1.0),
        stroke_width=2.0,                  # thicker road lines
        point_rgba=(0.85, 0.20, 0.20, 1.0),
        point_size=10.0,
        transform=xf,
    )

    # Save PNG (if Pillow is available)
    try:
        from PIL import Image

        os.makedirs("reports", exist_ok=True)
        Image.fromarray(composed).save("reports/render_overlay.png")
        print("Saved reports/render_overlay.png")
    except Exception as e:
        print("(Optional) Pillow not available; skipping PNG save:", e)