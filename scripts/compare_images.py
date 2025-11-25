#!/usr/bin/env python3
"""Small helper to compare two images numerically.

Usage
-----

    python scripts/compare_images.py ref.png test.png

The script prints per-channel means/stds, mean luminance, and simple
error metrics (MSE, MAE, max error) for quick sanity checks when
iterating on terrain_demo outputs against a reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow") from exc


def _load_rgba(path: Path) -> np.ndarray:
    img = Image.open(str(path)).convert("RGBA")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def _channel_stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    rgb = arr[..., :3]
    mean = rgb.mean(axis=(0, 1))  # R,G,B
    std = rgb.std(axis=(0, 1))
    luma = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    ).mean()
    return mean, std, float(luma)


def _error_metrics(ref: np.ndarray, test: np.ndarray) -> dict[str, float]:
    if ref.shape != test.shape:
        raise SystemExit(
            f"Image shapes differ: ref={ref.shape}, test={test.shape} (resize before comparing)"
        )
    diff = test - ref
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    max_err = float(np.max(np.abs(diff)))
    return {"mse": mse, "mae": mae, "max": max_err}


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "Usage: python scripts/compare_images.py ref.png test.png",
            file=sys.stderr,
        )
        return 1

    ref_path = Path(argv[1])
    test_path = Path(argv[2])

    if not ref_path.exists():
        raise SystemExit(f"Reference image not found: {ref_path}")
    if not test_path.exists():
        raise SystemExit(f"Test image not found: {test_path}")

    ref = _load_rgba(ref_path)
    test = _load_rgba(test_path)

    ref_mean, ref_std, ref_luma = _channel_stats(ref)
    test_mean, test_std, test_luma = _channel_stats(test)
    errs = _error_metrics(ref, test)

    print(f"ref:  {ref_path}  shape={ref.shape}")
    print(f"  mean RGB: {ref_mean.round(4)}  std RGB: {ref_std.round(4)}  mean luma: {ref_luma:.4f}")
    print(f"test: {test_path}  shape={test.shape}")
    print(f"  mean RGB: {test_mean.round(4)}  std RGB: {test_std.round(4)}  mean luma: {test_luma:.4f}")
    print("errors (test - ref) on [0,1] range:")
    print(f"  mse: {errs['mse']:.6f}  mae: {errs['mae']:.6f}  max: {errs['max']:.6f}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
