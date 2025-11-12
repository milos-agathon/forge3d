#!/usr/bin/env python3
"""
P5.1 Acceptance Check for SSAO/GTAO artifacts.

Validates three artifacts produced by the viewer:
  - reports/p5_1/ao_cornell_off_on.png
  - reports/p5_1/ao_buffers_grid.png
  - reports/p5_1/ao_params_sweep.png

Usage:
  python scripts/check_p5_1.py

Exit code 0 on PASS, 1 on FAIL. Prints a short summary report.
"""
import os
import sys
from dataclasses import dataclass
from typing import Tuple

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install pillow", file=sys.stderr)
    sys.exit(2)

import numpy as np

REPORT_DIR = os.path.join("reports", "p5_1")


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def load_rgba(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    return np.asarray(img, dtype=np.uint8)


def rgb_to_luma(img_rgba: np.ndarray) -> np.ndarray:
    # img_rgba: HxWx4 uint8
    rgb = img_rgba[..., :3].astype(np.float32) / 255.0
    # Rec. 709 luminance
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return luma


def check_cornell_split(path: str) -> CheckResult:
    name = "Cornell OFF/ON Split"
    if not os.path.exists(path):
        return CheckResult(name, False, f"missing file: {path}")
    img = load_rgba(path)
    H, W, _ = img.shape
    if W % 2 != 0:
        return CheckResult(name, False, f"unexpected width {W}, expected even (side-by-side)")
    left = img[:, : W // 2, :]
    right = img[:, W // 2 :, :]
    yl = rgb_to_luma(left).mean()
    yr = rgb_to_luma(right).mean()
    # Expect AO ON (right) to be darker by at least relative 3%
    passed = yr <= yl * 0.97
    details = f"mean_luma_off={yl:.4f}, mean_luma_on={yr:.4f}, ratio={yr/yl:.3f} (<=0.970 required)"
    return CheckResult(name, passed, details)


def tile_from_grid(img: np.ndarray, cols: int, rows: int, c: int, r: int) -> np.ndarray:
    H, W, _ = img.shape
    tw = W // cols
    th = H // rows
    x0 = c * tw
    y0 = r * th
    return img[y0 : y0 + th, x0 : x0 + tw, :]


def check_ao_grid(path: str) -> CheckResult:
    name = "AO Buffers Grid (raw/blur/resolved x SSAO/GTAO)"
    if not os.path.exists(path):
        return CheckResult(name, False, f"missing file: {path}")
    img = load_rgba(path)
    # Layout produced is 3x2 (cols: raw, blur, resolved; rows: SSAO, GTAO)
    cols, rows = 3, 2
    # For each row, expect: std(raw) > std(blur) by >= 5%, and std(resolved) <= std(raw)
    row_ok = []
    for r in range(rows):
        raw = rgb_to_luma(tile_from_grid(img, cols, rows, 0, r))
        blur = rgb_to_luma(tile_from_grid(img, cols, rows, 1, r))
        res = rgb_to_luma(tile_from_grid(img, cols, rows, 2, r))
        s_raw = float(raw.std())
        s_blur = float(blur.std())
        s_res = float(res.std())
        raw_vs_blur = (s_raw / max(1e-6, s_blur)) if s_blur > 0 else np.inf
        cond_blur = s_raw >= s_blur * 1.05  # 5% noise reduction
        cond_res = s_res <= s_raw + 1e-5
        row_ok.append(cond_blur and cond_res)
    passed = all(row_ok)
    details = f"rows_ok={row_ok} (expect all True)"
    return CheckResult(name, passed, details)


def check_param_sweep(path: str) -> CheckResult:
    name = "AO Parameter Sweep (radius x intensity)"
    if not os.path.exists(path):
        return CheckResult(name, False, f"missing file: {path}")
    img = load_rgba(path)
    # Layout produced is 3x3 (rows: radii 0.25,0.5,1.0; cols: intensity 0.5,1.0,1.5)
    cols, rows = 3, 3
    # Expect monotonic non-increasing mean luma along intensity (across columns) for each row
    # and weakly non-increasing down rows as radius grows.
    lumas = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            tile = tile_from_grid(img, cols, rows, c, r)
            lumas[r, c] = rgb_to_luma(tile).mean()
    # Check columns monotonic for each row
    row_monotonic = []
    for r in range(rows):
        ok = all(lumas[r, c + 1] <= lumas[r, c] + 1e-3 for c in range(cols - 1))
        row_monotonic.append(ok)
    # Check rows monotonic for each column (as radius grows)
    col_monotonic = []
    for c in range(cols):
        ok = all(lumas[r + 1, c] <= lumas[r, c] + 2e-3 for r in range(rows - 1))
        col_monotonic.append(ok)
    passed = all(row_monotonic) and all(col_monotonic)
    details = (
        f"row_monotonic={row_monotonic}, col_monotonic={col_monotonic}, means=\n" +
        "\n".join(["  " + " ".join(f"{v:.3f}" for v in lumas[r]) for r in range(rows)])
    )
    return CheckResult(name, passed, details)


def main() -> int:
    cornell = os.path.join(REPORT_DIR, "ao_cornell_off_on.png")
    grid = os.path.join(REPORT_DIR, "ao_buffers_grid.png")
    sweep = os.path.join(REPORT_DIR, "ao_params_sweep.png")

    checks = [
        check_cornell_split(cornell),
        check_ao_grid(grid),
        check_param_sweep(sweep),
    ]

    all_ok = all(c.passed for c in checks)
    print("\n=== P5.1 Acceptance Report ===")
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"- {c.name}: {status}\n    {c.details}")
    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
