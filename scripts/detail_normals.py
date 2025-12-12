#!/usr/bin/env python
"""Generate detail normal map from DEM for P6 Gradient Match.

Computes high-frequency detail from a DEM by subtracting a Gaussian-blurred
version, then converts the residual heightmap into a tangent-space normal map.

Usage:
    python tools/detail_normals.py --dem assets/Gore_Range_Albers_1m.tif \\
        --sigma-px 3.0 --output assets/generated/detail_normal.png

Output format:
    - Tangent-space normal map (RGB8)
    - Channel mapping: R=X, G=Y, B=Z (OpenGL convention)
    - Encoding: [0,255] maps to [-1,1] per channel
    - Neutral normal (0,0,1) encodes as RGB (128,128,255)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))


def load_dem(path: Path) -> np.ndarray:
    """Load DEM from GeoTIFF, return float32 2D array."""
    try:
        import rasterio
    except ImportError:
        raise ImportError(
            "rasterio required for DEM loading. "
            "Install with: pip install rasterio"
        )
    with rasterio.open(path) as src:
        data = src.read(1, masked=False).astype(np.float32)
    return data


def gaussian_blur(arr: np.ndarray, sigma_px: float) -> np.ndarray:
    """Apply Gaussian blur with given sigma in pixels."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr, sigma=sigma_px, mode="reflect")


def compute_detail_residual(height: np.ndarray, sigma_px: float) -> np.ndarray:
    """Compute detail = original - blur(original, sigma)."""
    blurred = gaussian_blur(height, sigma_px)
    return height - blurred


def detail_to_normal_map(
    detail: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Convert detail heightmap to tangent-space normal map.
    
    Args:
        detail: 2D float array of height residuals
        scale: Vertical scale factor for normal strength
    
    Returns:
        RGB uint8 array (H, W, 3) with tangent-space normals
        Channel mapping: R=X, G=Y, B=Z (OpenGL convention)
    """
    h, w = detail.shape
    
    # Compute gradients using Sobel-like finite differences
    # Central differences for interior, one-sided at edges
    dx = np.zeros_like(detail)
    dy = np.zeros_like(detail)
    
    # Central differences for X gradient
    dx[:, 1:-1] = (detail[:, 2:] - detail[:, :-2]) * 0.5
    dx[:, 0] = detail[:, 1] - detail[:, 0]
    dx[:, -1] = detail[:, -1] - detail[:, -2]
    
    # Central differences for Y gradient
    dy[1:-1, :] = (detail[2:, :] - detail[:-2, :]) * 0.5
    dy[0, :] = detail[1, :] - detail[0, :]
    dy[-1, :] = detail[-1, :] - detail[-2, :]
    
    # Scale gradients
    dx *= scale
    dy *= scale
    
    # Construct normal vectors: n = normalize(-dx, -dy, 1)
    # For a heightfield z=f(x,y), the normal is (-dz/dx, -dz/dy, 1) normalized
    nx = -dx
    ny = -dy
    nz = np.ones_like(detail)
    
    # Normalize
    length = np.sqrt(nx*nx + ny*ny + nz*nz)
    length = np.maximum(length, 1e-8)
    nx /= length
    ny /= length
    nz /= length
    
    # Encode to [0, 255]: n * 0.5 + 0.5 -> [0, 1] -> [0, 255]
    r = ((nx * 0.5 + 0.5) * 255.0).astype(np.uint8)
    g = ((ny * 0.5 + 0.5) * 255.0).astype(np.uint8)
    b = ((nz * 0.5 + 0.5) * 255.0).astype(np.uint8)
    
    return np.stack([r, g, b], axis=-1)


def save_normal_map(normal_map: np.ndarray, path: Path) -> None:
    """Save normal map as PNG."""
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normal_map, mode="RGB").save(str(path))


def compute_file_hash(path: Path) -> str:
    """Compute MD5 hash of file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate detail normal map from DEM for P6 Gradient Match"
    )
    parser.add_argument(
        "--dem",
        type=Path,
        required=True,
        help="Path to input DEM (GeoTIFF)",
    )
    parser.add_argument(
        "--sigma-px",
        type=float,
        default=3.0,
        help="Gaussian blur sigma in pixels (default: 3.0)",
    )
    parser.add_argument(
        "--normal-scale",
        type=float,
        default=1.0,
        help="Normal strength scale factor (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: assets/generated/detail_normal_{sigma}.png)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (alternative to --output; filename auto-generated)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file if it exists",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Output metadata JSON path",
    )
    
    args = parser.parse_args(argv)
    
    # Resolve paths
    dem_path = args.dem
    if not dem_path.is_absolute():
        dem_path = PROJECT_ROOT / dem_path
    
    if not dem_path.exists():
        print(f"ERROR: DEM file not found: {dem_path}")
        return 1
    
    # Determine output path
    out_name = f"detail_normal_sigma{args.sigma_px:.1f}.png"
    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    elif args.outdir is not None:
        outdir = args.outdir
        if not outdir.is_absolute():
            outdir = PROJECT_ROOT / outdir
        outdir.mkdir(parents=True, exist_ok=True)
        output_path = outdir / out_name
    else:
        output_path = PROJECT_ROOT / "assets" / "generated" / out_name
    
    # Check overwrite
    if output_path.exists() and not args.overwrite:
        print(f"ERROR: Output file exists: {output_path}")
        print("Use --overwrite to replace it.")
        return 1
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading DEM: {dem_path}")
    height = load_dem(dem_path)
    print(f"  Shape: {height.shape}")
    print(f"  Range: [{height.min():.2f}, {height.max():.2f}]")
    
    print(f"Computing detail residual (sigma={args.sigma_px} px)...")
    detail = compute_detail_residual(height, args.sigma_px)
    print(f"  Detail range: [{detail.min():.4f}, {detail.max():.4f}]")
    
    print(f"Converting to normal map (scale={args.normal_scale})...")
    normal_map = detail_to_normal_map(detail, scale=args.normal_scale)
    
    print(f"Saving: {output_path}")
    save_normal_map(normal_map, output_path)
    
    # Save metadata
    meta = {
        "dem_path": str(dem_path),
        "dem_hash": compute_file_hash(dem_path),
        "sigma_px": args.sigma_px,
        "normal_scale": args.normal_scale,
        "output_path": str(output_path),
        "output_hash": compute_file_hash(output_path),
        "height_shape": list(height.shape),
        "detail_range": [float(detail.min()), float(detail.max())],
    }
    
    meta_path = args.meta
    if meta_path is None:
        meta_path = output_path.with_suffix(".json")
    else:
        if not meta_path.is_absolute():
            meta_path = PROJECT_ROOT / meta_path
    
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {meta_path}")
    
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
