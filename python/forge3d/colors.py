"""Color conversion utilities for forge3d viewers and renderers."""

from __future__ import annotations

from typing import List, Tuple
import math

import numpy as np


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple (0-255 range).
    
    Args:
        hex_color: Color in hex format, e.g. '#FF5500' or 'FF5500'
        
    Returns:
        Tuple of (R, G, B) values in 0-255 range
    """
    hex_color = hex_color.lstrip('#')
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> List[float]:
    """Convert hex color to RGBA list (0.0-1.0 range).
    
    Args:
        hex_color: Color in hex format, e.g. '#FF5500' or 'FF5500' or '#FF5500AA'
        alpha: Alpha value (0.0-1.0), used if hex doesn't include alpha
        
    Returns:
        List of [R, G, B, A] values in 0.0-1.0 range
        
    Raises:
        ValueError: if hex color format is invalid
    """
    hex_color = hex_color.lstrip('#')
    
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return [r, g, b, alpha]
    elif len(hex_color) == 8:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        a = int(hex_color[6:8], 16) / 255.0
        return [r, g, b, a]
    else:
        raise ValueError(f"Invalid hex color: {hex_color}")


def rgb_to_normalized(rgb: Tuple[int, int, int]) -> List[float]:
    """Convert RGB (0-255) to normalized (0.0-1.0) values.
    
    Args:
        rgb: Tuple of (R, G, B) values in 0-255 range
        
    Returns:
        List of [R, G, B] values in 0.0-1.0 range
    """
    return [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]


_REC709_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
_F64_MAX = np.finfo(np.float64).max


def environment_mood_tint(
    environment: "np.ndarray",
    *,
    horizon_fraction: float = 10 / 64,
    max_gain: float = 1.25,
) -> "np.ndarray":
    """Derive an opt-in horizon mood tint from a linear-RGB environment."""
    if not np.isfinite(horizon_fraction) or not (0.0 < float(horizon_fraction) <= 1.0):
        raise ValueError(f"horizon_fraction must be in (0, 1], got {horizon_fraction!r}")
    if not np.isfinite(max_gain) or float(max_gain) < 1.0:
        raise ValueError(f"max_gain must be >= 1, got {max_gain!r}")

    env = np.asarray(environment)
    if env.ndim != 3 or env.shape[2] < 3:
        raise ValueError(f"environment must be (H, W, >=3), got shape {env.shape}")
    if env.shape[0] == 0 or env.shape[1] == 0:
        raise ValueError(f"environment must have non-empty H and W, got shape {env.shape}")
    if not np.isfinite(env).all():
        raise ValueError("environment contains non-finite samples")

    out_dtype = env.dtype if np.issubdtype(env.dtype, np.floating) else np.float64
    h = int(env.shape[0])
    band_height = max(1, min(h, round(float(horizon_fraction) * h)))
    start = (h - band_height) // 2
    band = env[start:start + band_height, :, :3].astype(np.float64)
    mean_rgb = band.mean(axis=(0, 1))
    lum = float(mean_rgb @ _REC709_WEIGHTS)
    if lum <= 1e-12:
        return np.ones(3, dtype=out_dtype)
    return np.clip(mean_rgb / lum, 1.0 / float(max_gain), float(max_gain)).astype(out_dtype)


def apply_luminance_preserving_tint(
    image: "np.ndarray",
    tint: "np.ndarray",
    *,
    strength: float = 0.0,
) -> "np.ndarray":
    """Apply a luminance-preserving RGB tint to a completed RGB(A) image."""
    if not np.isfinite(strength) or not (0.0 <= float(strength) <= 1.0):
        raise ValueError(f"strength must be in [0, 1], got {strength!r}")

    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"image must be (H, W, 3) or (H, W, 4), got shape {img.shape}")
    if not np.isfinite(img).all():
        raise ValueError("image contains non-finite samples")

    tint_arr = np.asarray(tint, dtype=np.float64)
    if tint_arr.shape != (3,):
        raise ValueError(f"tint must have exactly three components, got shape {tint_arr.shape}")
    if not np.isfinite(tint_arr).all():
        raise ValueError("tint contains non-finite values")

    if float(strength) == 0.0 or np.array_equal(tint_arr, np.ones(3, dtype=np.float64)):
        return img.copy()

    in_dtype = img.dtype
    if np.issubdtype(in_dtype, np.integer):
        info = np.iinfo(in_dtype)
        lo, hi = float(info.min), float(info.max)
    else:
        finfo = np.finfo(in_dtype)
        lo, hi = float(finfo.min), float(finfo.max)

    rgb = img[..., :3].astype(np.float64)
    graded = _apply_tint_float64(rgb, tint_arr, float(strength), lo, hi)

    out = img.copy()
    if np.issubdtype(in_dtype, np.integer):
        out[..., :3] = _round_saturating_int(graded, in_dtype)
    else:
        out[..., :3] = graded.astype(in_dtype)
    return out


def _apply_tint_float64(rgb: "np.ndarray", tint: "np.ndarray", strength: float, lo: float, hi: float) -> "np.ndarray":
    delta = strength * (tint - 1.0)
    mix = 1.0 + delta
    with np.errstate(over="ignore", invalid="ignore"):
        literal = rgb * mix
        literal += ((rgb @ _REC709_WEIGHTS) - (literal @ _REC709_WEIGHTS))[..., None]
    if np.isfinite(literal).all():
        return np.clip(literal, lo, hi)

    coeff = np.tile(-_REC709_WEIGHTS * delta, (3, 1)).T
    for channel in range(3):
        coeff[channel, channel] += 1.0 + delta[channel]

    peak_rgb = float(np.abs(rgb).max()) if rgb.size else 0.0
    peak_coeff = float(np.abs(coeff).max())
    threshold = _F64_MAX / 4.0
    safe = (
        peak_rgb == 0.0
        or peak_coeff == 0.0
        or (peak_rgb <= 1.0 and peak_coeff <= threshold)
        or (peak_rgb > 1.0 and peak_coeff <= threshold / peak_rgb)
    )
    scale = 1.0 if safe else peak_rgb
    work = rgb / scale
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = work @ coeff
    bad = ~np.isfinite(scaled)
    if bad.any():
        flat_work = work.reshape(-1, 3)
        flat_scaled = scaled.reshape(-1, 3)
        bad_rows, bad_cols = np.nonzero(bad.reshape(-1, 3))
        for row, col in zip(bad_rows, bad_cols):
            flat_scaled[row, col] = _finite_sum3(float(flat_work[row, 0] * coeff[0, col]),
                                                 float(flat_work[row, 1] * coeff[1, col]),
                                                 float(flat_work[row, 2] * coeff[2, col]))

    scaled = np.clip(scaled, lo / scale, hi / scale)
    with np.errstate(over="ignore"):
        out = scaled * scale
    return np.clip(out, lo, hi)


def _finite_sum3(a: float, b: float, c: float) -> float:
    m = max(abs(a), abs(b), abs(c))
    if m == 0.0:
        return 0.0
    n = a / m + b / m + c / m
    if n == 0.0:
        return 0.0
    if abs(n) > _F64_MAX / m:
        return math.copysign(math.inf, n)
    return n * m


def _round_saturating_int(values: "np.ndarray", dtype: "np.dtype") -> "np.ndarray":
    dtype = np.dtype(dtype)
    info = np.iinfo(dtype)
    rounded = np.rint(values)
    out = np.empty(rounded.shape, dtype=dtype)
    low = rounded <= float(info.min)
    high = rounded >= float(info.max)
    mid = ~(low | high)
    out[low] = info.min
    out[high] = info.max
    out[mid] = rounded[mid].astype(dtype)
    return out
