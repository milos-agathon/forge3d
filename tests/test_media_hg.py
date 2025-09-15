# tests/test_media_hg.py
# Minimal tests for Henyey–Greenstein phase and sampling (A11)
# Validate normalization and pdf consistency
# RELEVANT FILES:python/forge3d/lighting.py,src/shaders/lighting_media.wgsl

import math
import numpy as np
import pytest


def test_hg_phase_normalization_mc():
    import forge3d.lighting as lighting

    rng = np.random.default_rng(123)
    for g in (-0.8, -0.2, 0.0, 0.4, 0.8):
        n = 50_000
        # Uniform directions on sphere
        u1 = rng.random(n, dtype=np.float32)
        u2 = rng.random(n, dtype=np.float32)
        z = 1.0 - 2.0 * u1
        # Monte Carlo integral of phase over sphere should be ~1
        p = lighting.hg_phase(z, g)
        integral = (4.0 * math.pi) * np.mean(p)
        assert np.isfinite(integral)
        assert abs(integral - 1.0) < 0.02


def test_sample_hg_pdf_consistency():
    import forge3d.lighting as lighting

    rng = np.random.default_rng(456)
    u1 = rng.random(8192, dtype=np.float32)
    u2 = rng.random(8192, dtype=np.float32)
    g = 0.6
    dirs, pdf = lighting.sample_hg(u1, u2, g)
    # PDF must be positive and finite
    assert np.all(np.isfinite(pdf)) and np.all(pdf > 0)
    # Cos(theta) is z in local frame
    cos_theta = dirs[..., 2]
    pdf2 = lighting.hg_phase(cos_theta, g)
    mae = float(np.mean(np.abs(pdf - pdf2)))
    assert mae < 1e-3

