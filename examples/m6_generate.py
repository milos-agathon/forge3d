"""M6: BRDF tile CPU/GPU validation harness.

This module provides a small helper to compare the CPU/legacy BRDF tile
path against the BRDF tile renderer exposed via forge3d, and to report
summary RMS metrics for tests/test_m6_validation.py.

RELEVANT FILES:
- tests/test_m6_validation.py
- python/forge3d/__init__.py (render_brdf_tile*, render_brdf_tile_full)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

import forge3d as f3d


@dataclass
class ValidationResult:
    sample_rows: int
    sample_cols: int
    sample_count: int
    rms: np.ndarray  # shape (H, W)
    percentile_999: float


def _generate_brdf_tiles(
    brdf: str,
    roughness: float,
    tile_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render BRDF tiles using legacy and full paths.

    Parameters
    ----------
    brdf : str
        BRDF model name (e.g., "ggx").
    roughness : float
        Material roughness in [0, 1].
    tile_size : int
        Tile size in pixels (square).

    Returns
    -------
    base : np.ndarray
        Result from render_brdf_tile (legacy path).
    full : np.ndarray
        Result from render_brdf_tile_full (current path).
    """
    base = f3d.render_brdf_tile(brdf, float(roughness), int(tile_size), int(tile_size), False)
    full = f3d.render_brdf_tile_full(brdf, float(roughness), int(tile_size), int(tile_size))
    return base.astype(np.float32), full.astype(np.float32)


def run_validation(
    *,
    tile_size: int = 256,
    roughness: float = 0.5,
    samples_per_axis: int = 32,
    eval_scale: int = 2,
) -> ValidationResult:
    """Run M6 BRDF CPU/GPU validation and return summarized metrics.

    The implementation follows the expectations in tests/test_m6_validation.py:

    - Produces a grid of sample_rows x sample_cols evaluation points
      (here derived from ``samples_per_axis``).
    - Computes per-pixel RMS between the legacy and "full" BRDF tile paths.
    - Reports the 99.9th percentile of RMS values.

    Parameters
    ----------
    tile_size : int
        Tile size passed through to BRDF tile renderers.
    roughness : float
        BRDF roughness value in [0, 1].
    samples_per_axis : int
        Number of samples along one axis for evaluation grid.
    eval_scale : int
        Evaluation scale factor (currently informational; kept
        to match test signature).
    """
    # Render a pair of tiles to compare.
    base, full = _generate_brdf_tiles("ggx", float(roughness), int(tile_size))

    # Ensure shapes match and convert to float32 for RMS.
    if base.shape != full.shape:
        raise RuntimeError(f"BRDF tiles shape mismatch: {base.shape} vs {full.shape}")

    # Compute per-pixel squared error in linear space.
    diff = (base[..., :3] / 255.0) - (full[..., :3] / 255.0)
    sq_err = np.sum(diff * diff, axis=-1)  # (H, W)

    # Derive sample grid from samples_per_axis; we simply downsample
    # the RMS field onto a coarse grid for reporting.
    h, w = sq_err.shape
    rows = cols = int(samples_per_axis)
    step_y = max(h // rows, 1)
    step_x = max(w // cols, 1)

    # Sample RMS on a coarse grid.
    rms_samples = []
    for iy in range(rows):
        y = min(iy * step_y, h - 1)
        for ix in range(cols):
            x = min(ix * step_x, w - 1)
            rms_samples.append(sq_err[y, x])
    rms_samples = np.asarray(rms_samples, dtype=np.float32)

    # Full-resolution RMS map for debugging/inspection.
    rms_full = np.sqrt(sq_err)

    # 99.9th percentile of RMS over the full tile.
    percentile_999 = float(np.percentile(rms_full, 99.9))

    return ValidationResult(
        sample_rows=rows,
        sample_cols=cols,
        sample_count=rows * cols,
        rms=rms_full,
        percentile_999=percentile_999,
    )
