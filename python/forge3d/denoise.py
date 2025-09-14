# python/forge3d/denoise.py
# Edge-aware A-trous denoiser guided by albedo/normal/depth; pure NumPy implementation.
# Provides a minimal, deterministic denoiser for Workstream A5 acceptance and tests.
# RELEVANT FILES:python/forge3d/path_tracing.py,tests/test_a5_denoise.py,docs/api/denoise.md

from __future__ import annotations

from typing import Optional
import numpy as np


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(eps, n)


def atrous_denoise(
    color: np.ndarray,
    *,
    albedo: Optional[np.ndarray] = None,
    normal: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
    iterations: int = 3,
    sigma_color: float = 0.1,
    sigma_albedo: float = 0.2,
    sigma_normal: float = 0.3,
    sigma_depth: float = 0.5,
) -> np.ndarray:
    """A-trous edge-aware denoiser guided by auxiliary features.

    Args:
        color: float32 array (H, W, 3) in linear space.
        albedo: optional float32 (H, W, 3), guides edges in material/color.
        normal: optional float32 (H, W, 3), expected unit or near-unit.
        depth: optional float32 (H, W) linear depth.
        iterations: number of a-trous passes; each doubles step size.
        sigma_*: edge thresholds for bilateral weights.

    Returns:
        Denoised color, same shape/type as input (float32).
    """
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("color must be (H, W, 3)")
    h, w, _ = color.shape
    c = color.astype(np.float32, copy=True)

    if albedo is not None:
        if albedo.shape != (h, w, 3):
            raise ValueError("albedo must match color shape (H, W, 3)")
        a = albedo.astype(np.float32, copy=False)
    else:
        a = c

    if normal is not None:
        if normal.shape != (h, w, 3):
            raise ValueError("normal must match color shape (H, W, 3)")
        n = _normalize(normal.astype(np.float32, copy=False))
    else:
        n = None

    if depth is not None:
        if depth.shape != (h, w):
            raise ValueError("depth must be (H, W)")
        d = depth.astype(np.float32, copy=False)
    else:
        d = None

    # 5-tap a-trous kernel (B3-spline): [1, 4, 6, 4, 1] normalized
    k = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    k /= k.sum()

    step = 1
    out = c.copy()
    for _ in range(int(max(1, iterations))):
        accum = np.zeros_like(out)
        wsum = np.zeros((h, w, 1), dtype=np.float32)

        # Evaluate separable a-trous with bilateral weights using 5x5 neighborhood with holes.
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                wy = k[dy + 2]
                wx = k[dx + 2]
                base_w = wy * wx
                if base_w == 0.0:
                    continue
                oy = dy * step
                ox = dx * step

                src = _shift(out, oy, ox)
                w_ij = np.full((h, w, 1), base_w, dtype=np.float32)

                # Guidance difference (albedo preferred)
                ga = _shift(a, oy, ox)
                dc = ga - a
                w_ij *= np.exp(- (np.sum(dc * dc, axis=-1, keepdims=True)) / (2.0 * (sigma_color ** 2) + 1e-8))

                # Optional extra albedo term (disabled if same as guidance)
                if albedo is not None and sigma_albedo > 0:
                    da = ga - a
                    w_ij *= np.exp(- (np.sum(da * da, axis=-1, keepdims=True)) / (2.0 * (sigma_albedo ** 2) + 1e-8))

                # Normal angular difference
                if n is not None:
                    nn = _shift(n, oy, ox)
                    cos_ang = np.clip(np.sum(nn * n, axis=-1, keepdims=True), -1.0, 1.0)
                    ang = np.arccos(cos_ang)
                    w_ij *= np.exp(- (ang * ang) / (2.0 * (sigma_normal ** 2) + 1e-8))

                # Depth difference
                if d is not None:
                    dd = _shift(d[..., None], oy, ox) - d[..., None]
                    w_ij *= np.exp(- (dd * dd) / (2.0 * (sigma_depth ** 2) + 1e-8))

                accum += src * w_ij
                wsum += w_ij

        out = accum / np.maximum(wsum, 1e-8)
        step *= 2

    return out.astype(np.float32, copy=False)


def _shift(x: np.ndarray, oy: int, ox: int) -> np.ndarray:
    """Shift array with zero padding.

    Works for 2D (H,W) or 3D (H,W,C) arrays.
    """
    if x.ndim == 2:
        h, w = x.shape
        out = np.zeros_like(x)
        ys = slice(max(0, oy), min(h, h + oy))
        xs = slice(max(0, ox), min(w, w + ox))
        yd = slice(max(0, -oy), max(0, -oy) + (ys.stop - ys.start))
        xd = slice(max(0, -ox), max(0, -ox) + (xs.stop - xs.start))
        out[yd, xd] = x[ys, xs]
        return out
    elif x.ndim == 3:
        h, w, c = x.shape
        out = np.zeros_like(x)
        ys = slice(max(0, oy), min(h, h + oy))
        xs = slice(max(0, ox), min(w, w + ox))
        yd = slice(max(0, -oy), max(0, -oy) + (ys.stop - ys.start))
        xd = slice(max(0, -ox), max(0, -ox) + (xs.stop - xs.start))
        out[yd, xd, :] = x[ys, xs, :]
        return out
    else:
        raise ValueError("_shift expects 2D or 3D array")
