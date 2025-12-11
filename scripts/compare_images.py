#!/usr/bin/env python3
"""Compare two images numerically with SSIM and error metrics.

Usage
-----
    python scripts/compare_images.py ref.png test.png [--ssim] [--diff diff.png] [--json out.json]

Options:
    --ssim          Compute structural similarity index (requires scipy)
    --diff PATH     Save difference image to PATH
    --json PATH     Save metrics as JSON to PATH

The script prints per-channel means/stds, mean luminance, and error metrics
(MSE, MAE, max error, SSIM) for terrain render validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def _compute_ssim(ref: np.ndarray, test: np.ndarray) -> float:
    """Compute SSIM between two images (grayscale luminance comparison)."""
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        print("Warning: scipy not available, SSIM will use simplified formula", file=sys.stderr)
        return _compute_ssim_simple(ref, test)

    # Convert to grayscale luminance
    ref_y = 0.2126 * ref[..., 0] + 0.7152 * ref[..., 1] + 0.0722 * ref[..., 2]
    test_y = 0.2126 * test[..., 0] + 0.7152 * test[..., 1] + 0.0722 * test[..., 2]

    # SSIM constants (for [0,1] range)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    win_size = 11

    # Local means
    mu_x = uniform_filter(ref_y, size=win_size, mode='reflect')
    mu_y = uniform_filter(test_y, size=win_size, mode='reflect')

    # Local variances and covariance
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = uniform_filter(ref_y ** 2, size=win_size, mode='reflect') - mu_x_sq
    sigma_y_sq = uniform_filter(test_y ** 2, size=win_size, mode='reflect') - mu_y_sq
    sigma_xy = uniform_filter(ref_y * test_y, size=win_size, mode='reflect') - mu_xy

    # SSIM formula
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

    return float(np.mean(ssim_map))


def _compute_ssim_simple(ref: np.ndarray, test: np.ndarray) -> float:
    """Simplified global SSIM without local windowing (fallback)."""
    ref_y = 0.2126 * ref[..., 0] + 0.7152 * ref[..., 1] + 0.0722 * ref[..., 2]
    test_y = 0.2126 * test[..., 0] + 0.7152 * test[..., 1] + 0.0722 * test[..., 2]

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = np.mean(ref_y)
    mu_y = np.mean(test_y)
    sigma_x = np.std(ref_y)
    sigma_y = np.std(test_y)
    sigma_xy = np.mean((ref_y - mu_x) * (test_y - mu_y))

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1) * (sigma_x**2 + sigma_y**2 + C2))
    return float(ssim)


def _save_diff_image(ref: np.ndarray, test: np.ndarray, path: Path) -> None:
    """Save a difference image (amplified for visibility)."""
    diff = np.abs(test - ref)
    # Amplify difference 5x for visibility, clamp to [0,1]
    diff_amplified = np.clip(diff * 5.0, 0.0, 1.0)
    # Set alpha to 1.0
    diff_amplified[..., 3] = 1.0
    diff_uint8 = (diff_amplified * 255).astype(np.uint8)
    Image.fromarray(diff_uint8, mode="RGBA").save(str(path))


def _md5_file(path: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two images numerically.")
    parser.add_argument("ref", type=Path, help="Reference image path")
    parser.add_argument("test", type=Path, help="Test image path")
    parser.add_argument("--ssim", action="store_true", help="Compute SSIM metric")
    parser.add_argument("--diff", type=Path, help="Save difference image to this path")
    parser.add_argument("--json", type=Path, dest="json_out", help="Save metrics as JSON")
    args = parser.parse_args(argv[1:] if argv else None)

    if not args.ref.exists():
        raise SystemExit(f"Reference image not found: {args.ref}")
    if not args.test.exists():
        raise SystemExit(f"Test image not found: {args.test}")

    ref = _load_rgba(args.ref)
    test = _load_rgba(args.test)

    ref_mean, ref_std, ref_luma = _channel_stats(ref)
    test_mean, test_std, test_luma = _channel_stats(test)
    errs = _error_metrics(ref, test)

    print(f"ref:  {args.ref}  shape={ref.shape}")
    print(f"  mean RGB: {ref_mean.round(4)}  std RGB: {ref_std.round(4)}  mean luma: {ref_luma:.4f}")
    print(f"test: {args.test}  shape={test.shape}")
    print(f"  mean RGB: {test_mean.round(4)}  std RGB: {test_std.round(4)}  mean luma: {test_luma:.4f}")
    print("errors (test - ref) on [0,1] range:")
    print(f"  mse: {errs['mse']:.6f}  mae: {errs['mae']:.6f}  max: {errs['max']:.6f}")

    ssim_val = None
    if args.ssim:
        ssim_val = _compute_ssim(ref, test)
        print(f"  ssim: {ssim_val:.6f}")

    if args.diff:
        _save_diff_image(ref, test, args.diff)
        print(f"Saved difference image to: {args.diff}")

    if args.json_out:
        metrics = {
            "ref_path": str(args.ref),
            "test_path": str(args.test),
            "ref_md5": _md5_file(args.ref),
            "test_md5": _md5_file(args.test),
            "ref_shape": list(ref.shape),
            "test_shape": list(test.shape),
            "ref_mean_rgb": ref_mean.tolist(),
            "test_mean_rgb": test_mean.tolist(),
            "ref_luma": ref_luma,
            "test_luma": test_luma,
            "mse": errs["mse"],
            "mae": errs["mae"],
            "max_error": errs["max"],
        }
        if ssim_val is not None:
            metrics["ssim"] = ssim_val
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics JSON to: {args.json_out}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
