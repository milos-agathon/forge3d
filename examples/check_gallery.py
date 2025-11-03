import sys
from pathlib import Path
import argparse
import numpy as np

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # Pillow not available; we'll fall back to .npy loader

parser = argparse.ArgumentParser(description="Check lighting gallery metrics")
parser.add_argument("image", nargs="?", default="reports/lighting_gallery.png", help="Path to gallery mosaic PNG (default: reports/lighting_gallery.png)")
parser.add_argument("--tile-size", type=int, default=None, help="Optional square tile size; if omitted, infer from image dimensions")
args = parser.parse_args()

img_path = Path(args.image)

def _load_rgba(path: Path) -> np.ndarray:
    p = Path(path)
    # PNG path via Pillow (if available)
    if Image is not None:
        try:
            im = np.array(Image.open(p).convert("RGBA"), dtype=np.float32) / 255.0
            return im
        except Exception:
            pass
    # Fallback: if path is .npy or a .npy sibling exists, load it
    npy_path = p if p.suffix.lower() == ".npy" else p.with_suffix(".npy")
    if npy_path.exists():
        arr = np.load(npy_path)
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr, np.ones_like(arr)], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            alpha = np.ones((arr.shape[0], arr.shape[1], 1), dtype=arr.dtype)
            arr = np.concatenate([arr, alpha], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            pass
        else:
            raise RuntimeError(f"Unsupported array shape from {npy_path}: {arr.shape}")
        arr = arr.astype(np.float32)
        # Normalize if in 0..255 range
        if arr.max(initial=0) > 1.0:
            arr = arr / 255.0
        return arr
    raise RuntimeError(f"Could not load image {p} (Pillow not installed and no NPY fallback found)")

im = _load_rgba(img_path)
H, W, _ = im.shape
rows, cols = 2, 4
# Infer tile dimensions if not provided
if args.tile_size is None:
    tile_w = W // cols
    tile_h = H // rows
else:
    tile_w = tile_h = int(args.tile_size)

def luminance(rgba):
    rgb = rgba[..., :3]
    return 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]

def tile(r,c):
    y0, x0 = r*tile_h, c*tile_w
    return im[y0:y0+tile_h, x0:x0+tile_w, :]

def object_mask(rgba):
    # white background -> mask pixels not near white
    rgb = rgba[..., :3]
    return np.any(np.abs(rgb - 1.0) > 0.02, axis=-1)

# Tiles
labels = [
    "Lambert x1.0","Lambert x1.3","Phong","Blinn-Phong",
    "IBL rot 0°","IBL rot 90°","IBL rot 180°","IBL rot 270°"
]
tiles = [tile(r, c) for r in range(rows) for c in range(cols)]
masks = [object_mask(t) for t in tiles]
lums = [luminance(t)[m] for t,m in zip(tiles, masks)]

# 1) Lambert brightness ratio
lambert_ratio = lums[1].mean() / max(1e-6, lums[0].mean())

# 2) Specular area (fraction above a high luminance threshold) for Phong vs Blinn
thr = 0.85
spec_area_phong = float((lums[2] > thr).mean())
spec_area_blinn = float((lums[3] > thr).mean())

# 3) IBL rotation differences: mean absolute difference between adjacent rotations
def mean_abs_diff(a, b, ma, mb):
    m = ma & mb
    lum_a = luminance(a)[m]
    lum_b = luminance(b)[m]
    return float(np.mean(np.abs(lum_a - lum_b))) if m.any() else 0.0

ibl_diffs = [
    mean_abs_diff(tiles[4], tiles[5], masks[4], masks[5]),
    mean_abs_diff(tiles[5], tiles[6], masks[5], masks[6]),
    mean_abs_diff(tiles[6], tiles[7], masks[6], masks[7]),
]

print("Lambert x1.3 / x1.0 mean luminance ratio:", f"{lambert_ratio:.3f}")
print("Spec area fraction (Phong):", f"{spec_area_phong:.4f}")
print("Spec area fraction (Blinn-Phong):", f"{spec_area_blinn:.4f}")
print("IBL mean abs diffs (0->90, 90->180, 180->270):",
      ", ".join(f"{d:.4f}" for d in ibl_diffs))