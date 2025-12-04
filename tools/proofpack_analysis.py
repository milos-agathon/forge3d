from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def load_png_rgba(path: str) -> np.ndarray:
    """Return HxWx4 uint8 array. Raise on failure."""

    png_path = Path(path)
    if not png_path.is_file():
        raise FileNotFoundError(f"PNG not found: {png_path}")
    with Image.open(png_path) as img:
        return np.array(img.convert("RGBA"), dtype=np.uint8)


def luma_01(rgba_u8: np.ndarray) -> np.ndarray:
    """Compute Rec.709 luma in [0,1] from uint8 RGBA array."""

    return np.tensordot(rgba_u8[..., :3].astype(np.float32), LUMA_WEIGHTS, axes=([-1], [0])) / 255.0


def sha256_file(path: str) -> str:
    """Return hex sha256 digest for the file contents."""

    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_nonuniform_metrics(luma: np.ndarray) -> Dict[str, float]:
    """Return mean, p05, p95, and unique bin count for [0,1] luma array."""

    return {
        "mean": float(np.mean(luma)),
        "p05": float(np.percentile(luma, 5)),
        "p95": float(np.percentile(luma, 95)),
        "unique_bins_256": int(np.count_nonzero(np.histogram(luma.flatten(), bins=256, range=(0.0, 1.0))[0])),
    }


def ssim_approx(a: np.ndarray, b: np.ndarray) -> float:
    """Deterministic global SSIM approximation (channel-wise average)."""

    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c1 = 6.5025
    c2 = 58.5225
    mu1 = a.mean(axis=(0, 1))
    mu2 = b.mean(axis=(0, 1))
    sigma1 = ((a - mu1) ** 2).mean(axis=(0, 1))
    sigma2 = ((b - mu2) ** 2).mean(axis=(0, 1))
    sigma12 = ((a - mu1) * (b - mu2)).mean(axis=(0, 1))
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    return float(np.mean(ssim))


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference in [0,1] for uint8 arrays."""

    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def decode_normal(rgb_u8: np.ndarray) -> np.ndarray:
    """Input HxWx3 uint8 -> float32 HxWx3 in [-1,1]."""

    return rgb_u8.astype(np.float32) / 255.0 * 2.0 - 1.0


def normalize_vec3(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize vectors with epsilon clamp to avoid divide-by-zero."""

    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return v / norm


def angle_error_deg(n_ref: np.ndarray, n_test: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return float32 HxW degrees (0..180), invalid=nan."""

    ref_n = normalize_vec3(n_ref)
    test_n = normalize_vec3(n_test)
    dot = np.sum(ref_n * test_n, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.degrees(np.arccos(dot))
    theta = theta.astype(np.float32)
    theta = np.where(valid, theta, np.nan)
    return theta


def band_masks_from_lod(lod_01: np.ndarray) -> Dict[str, np.ndarray]:
    """Return boolean masks: near, mid, far."""

    return {
        "near": lod_01 < 0.33,
        "mid": (lod_01 >= 0.33) & (lod_01 <= 0.66),
        "far": lod_01 > 0.66,
    }


def write_json(path: str, obj: dict) -> None:
    """Write indent=2, sort_keys=True, utf-8, newline."""

    def _default(o: object) -> object:
        try:
            import numpy as _np  # type: ignore
        except Exception:
            _np = None  # type: ignore
        if _np is not None and isinstance(o, _np.generic):
            return o.item()
        if hasattr(o, "__fspath__"):
            return str(o)
        return str(o)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_default)
        f.write("\n")

