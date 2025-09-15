# tests/test_water_bsdf.py
# Unit tests for A6 dielectric water helpers: Fresnel and Beer–Lambert.
# This file exists to validate monotonicity and attenuation behavior deterministically.
# RELEVANT FILES:python/forge3d/pbr.py,README.md,docs/api/water.md

import numpy as np
import pytest

from forge3d import path_tracing as pt  # noqa: F401 (import signals module presence)
from forge3d import __version__  # noqa: F401
from forge3d.water import fresnel_schlick, beer_lambert_transmittance, water_shade


def test_fresnel_monotonic_with_angle():
    ior = 1.33
    angles = np.linspace(0.0, 1.0, 11)
    vals = fresnel_schlick(angles, ior)
    assert np.all(np.diff(vals) >= -1e-6)
    assert 0.0 <= vals[0] <= vals[-1] <= 1.0


def test_beer_lambert_attenuates():
    a = np.array([0.0, 0.1, 0.5], dtype=np.float32)
    t0 = beer_lambert_transmittance(a, 0.0)
    t1 = beer_lambert_transmittance(a, 1.0)
    t2 = beer_lambert_transmittance(a, 2.0)
    assert np.allclose(t0, 1.0)
    assert np.all(t1 <= t0 + 1e-6)
    assert np.all(t2 <= t1 + 1e-6)


def test_water_shade_basic_blend():
    n = np.array([[[0, 0, 1]]], dtype=np.float32)
    v = np.array([[[0, 0, -1]]], dtype=np.float32)
    l = np.array([[[0.5, 0.8, 0.2]]], dtype=np.float32)
    base = np.array([0.1, 0.2, 0.8], dtype=np.float32)
    col = water_shade(n, v, l, base, ior=1.33, absorption=(0.0, 0.05, 0.1), roughness=0.1, thickness=1.0)
    assert col.shape == (1, 1, 3)
    # More absorption should reduce blue channel
    col2 = water_shade(n, v, l, base, ior=1.33, absorption=(0.0, 0.05, 1.0), roughness=0.1, thickness=1.0)
    assert col2[..., 2] <= col[..., 2] + 1e-6
