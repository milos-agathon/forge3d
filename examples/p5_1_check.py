#!/usr/bin/env python3
"""
P5.1 Acceptance Check (examples/p5_1_check.py)

Validates artifacts produced by the viewer/exporter under reports/p5_1/:
  - ao_cornell_off_on.png
  - ao_buffers_grid.png
  - ao_params_sweep.png
Also validates metrics and thresholds and writes p5_1_PASS.txt on success.

Usage:
  python examples/p5_1_check.py
"""
import os
import sys
import json
from typing import Tuple

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install pillow", file=sys.stderr)
    sys.exit(2)

import numpy as np

REPORT_DIR = os.path.join("reports", "p5_1")
PASS_FILE = os.path.join(REPORT_DIR, "p5_1_PASS.txt")
META_PATH = os.path.join(REPORT_DIR, "p5_1_meta.json")


def rgb_to_luma(img_rgba: np.ndarray) -> np.ndarray:
    rgb = img_rgba[..., :3].astype(np.float32) / 255.0
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def load_rgba(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)


def tile_from_grid(img: np.ndarray, cols: int, rows: int, c: int, r: int) -> np.ndarray:
    H, W, _ = img.shape
    tw = W // cols
    th = H // rows
    x0 = c * tw
    y0 = r * th
    return img[y0:y0+th, x0:x0+tw, :]


def compute_metrics_from_images() -> dict:
    out = {}
    grid_path = os.path.join(REPORT_DIR, "ao_buffers_grid.png")
    if not os.path.exists(grid_path):
        return out
    img = load_rgba(grid_path)
    # Our grid layout is 5x2: [raw, blur_h, blur_v, temporal, final] x [SSAO, GTAO]
    # Use GTAO row (r=1) and temporal column (c=3) for final AO metrics
    final_tile = tile_from_grid(img, cols=5, rows=2, c=3, r=1)
    l_final = rgb_to_luma(final_tile)
    H, W = l_final.shape
    rx = max(1, W // 10)
    ry = max(1, H // 10)
    cx0 = W//2 - rx//2
    cy0 = H//2 - ry//2
    center_ao_mean = float(l_final[cy0:cy0+ry, cx0:cx0+rx].mean())
    corner_ao_mean = float(l_final[0:ry, 0:rx].mean())
    out["center_ao_mean"] = center_ao_mean
    out["corner_ao_mean"] = corner_ao_mean

    # Blur gradient reduction: compare stddev between raw and blur_v tiles on same row (GTAO)
    raw_tile = tile_from_grid(img, cols=5, rows=2, c=0, r=1)
    blur_tile = tile_from_grid(img, cols=5, rows=2, c=2, r=1)
    # Use a 10% wide vertical edge strip (left edge) to approximate gradient region
    strip_w = max(2, W // 10)
    raw_strip = rgb_to_luma(raw_tile)[:, :strip_w]
    blur_strip = rgb_to_luma(blur_tile)[:, :strip_w]
    s_raw = float(raw_strip.std())
    s_blur = float(blur_strip.std())
    if s_raw > 1e-6:
        blur_gradient_reduction = max(0.0, (s_raw - s_blur) / s_raw * 100.0)
    else:
        blur_gradient_reduction = 0.0
    out["blur_gradient_reduction"] = blur_gradient_reduction

    # Specular preservation: compare cornell split (right half vs left half)
    cornell = os.path.join(REPORT_DIR, "ao_cornell_off_on.png")
    if os.path.exists(cornell):
        corn = load_rgba(cornell)
        Hc, Wc, _ = corn.shape
        left = corn[:, :Wc//2, :]
        right = corn[:, Wc//2:, :]
        # Use top-1% brightest pixels by luma in OFF (left)
        lum_left = rgb_to_luma(left)
        lum_right = rgb_to_luma(right)
        k = max(1, int(lum_left.size * 0.01))
        thresh = np.partition(lum_left.flatten(), -k)[-k]
        mask = lum_left >= thresh
        delta = np.abs(lum_right[mask] - lum_left[mask]).mean()
        out["specular_preservation_delta"] = float(delta)

    return out


def main() -> int:
    os.makedirs(REPORT_DIR, exist_ok=True)
    # Prefer meta if present, otherwise compute from images
    meta = {}
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    calc = compute_metrics_from_images()

    # Merge: image-derived values take precedence
    corner = calc.get("corner_ao_mean") or meta.get("corner_ao_mean")
    center = calc.get("center_ao_mean") or meta.get("center_ao_mean")
    blur_red = calc.get("blur_gradient_reduction") or meta.get("blur_gradient_reduction")
    spec_delta = calc.get("specular_preservation_delta") or meta.get("specular_preservation_delta")

    # Thresholds
    ok_center = (center is not None) and (center <= 0.75)
    ok_corner = (corner is not None) and (corner <= 0.55) and ((center - corner) >= 0.10)
    ok_blur = (blur_red is not None) and (blur_red >= 30.0)
    ok_spec = (spec_delta is not None) and (spec_delta < 0.01)

    all_ok = bool(ok_center and ok_corner and ok_blur and ok_spec)

    # Emit PASS/FAIL with metrics
    lines = [
        f"center_ao_mean={center:.4f}" if center is not None else "center_ao_mean=NA",
        f"corner_ao_mean={corner:.4f}" if corner is not None else "corner_ao_mean=NA",
        f"blur_gradient_reduction={blur_red:.2f}%" if blur_red is not None else "blur_gradient_reduction=NA",
        f"specular_preservation_delta={spec_delta:.4f}" if spec_delta is not None else "specular_preservation_delta=NA",
        f"RESULT={'PASS' if all_ok else 'FAIL'}",
    ]
    with open(PASS_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
