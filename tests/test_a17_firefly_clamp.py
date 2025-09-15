# tests/test_a17_firefly_clamp.py
# Verifies luminance-based firefly clamp reduces outliers with minimal bias.
# This exists to enforce A17 acceptance criteria in a deterministic CPU stub.
# RELEVANT FILES:python/forge3d/path_tracing.py,python/forge3d/path_tracing.pyi,README.md

import numpy as np
import pytest

import forge3d


def _uint8_luminance(rgb8: np.ndarray) -> np.ndarray:
    r = rgb8[..., 0].astype(np.float32)
    g = rgb8[..., 1].astype(np.float32)
    b = rgb8[..., 2].astype(np.float32)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_firefly_clamp_reduces_outliers_with_small_bias():
    # Deterministic dimensions and seed
    W, H = 96, 96
    seed = 123

    tracer = forge3d.PathTracer(W, H, max_bounces=1, seed=seed)

    # Baseline noisy render (frames>1 to create brighter speckles)
    img_no = tracer.render_rgba(W, H, seed=seed, frames=3)
    lum_no = _uint8_luminance(img_no[..., :3])

    # Apply luminance clamp in normalized units (0..1); use 0.6 as a safe cap
    img_cl = tracer.render_rgba(W, H, seed=seed, frames=3, luminance_clamp=0.6)
    lum_cl = _uint8_luminance(img_cl[..., :3])

    # Define outliers as very high luminance pixels
    thr = 240.0  # on 0..255 scale
    out_no = int(np.count_nonzero(lum_no >= thr))
    out_cl = int(np.count_nonzero(lum_cl >= thr))

    # Ensure there are outliers without clamp to make the test meaningful
    assert out_no >= 10, f"Not enough outliers in baseline: {out_no}"

    # A17 Acceptance: Max outliers ↓ ≥10× (i.e., at most 10% remain)
    assert out_cl * 10 <= out_no, f"Outliers not reduced ≥10×: {out_no} -> {out_cl}"

    # Minimal bias: mean luminance should not shift dramatically (<15%)
    mean_no = float(np.mean(lum_no))
    mean_cl = float(np.mean(lum_cl))
    # Relative difference, guard divide by zero
    if mean_no > 0:
        rel = abs(mean_cl - mean_no) / mean_no
        assert rel <= 0.15, f"Mean luminance changed too much: {rel:.3f} (>15%)"
    else:
        # If baseline mean is zero (unlikely), clamped should also be near zero
        assert mean_cl < 1.0
