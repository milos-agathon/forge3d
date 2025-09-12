# tests/test_mpl_norms.py
# Tests for LogNorm/PowerNorm/BoundaryNorm adapters parity with Matplotlib.
# This exists to validate Workstream R3 acceptance criteria.
# RELEVANT FILES:python/forge3d/adapters/mpl_cmap.py,examples/mpl_norms_demo.py,docs/integration/matplotlib.md

import numpy as np
import pytest


@pytest.mark.skipif(
    __import__('importlib').util.find_spec('matplotlib') is None,
    reason="Matplotlib not available"
)
def test_lognorm_parity():
    import matplotlib.colors as mcolors
    from forge3d.adapters.mpl_cmap import LogNormAdapter

    rng = np.random.default_rng(42)
    a = np.abs(rng.lognormal(mean=0.0, sigma=1.0, size=(128, 128)).astype(np.float64)) + 1e-6
    vmin, vmax = float(a.min()), float(a.max())

    ours = LogNormAdapter(vmin=vmin, vmax=vmax, clip=True)(a)
    ref = mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=True)(a)
    max_abs_diff = float(np.max(np.abs(ours - ref)))
    assert max_abs_diff <= 1e-7


@pytest.mark.skipif(
    __import__('importlib').util.find_spec('matplotlib') is None,
    reason="Matplotlib not available"
)
def test_powernorm_parity():
    import matplotlib.colors as mcolors
    from forge3d.adapters.mpl_cmap import PowerNormAdapter

    rng = np.random.default_rng(7)
    a = rng.random((128, 128), dtype=np.float64)
    vmin, vmax = 0.0, 1.0

    ours = PowerNormAdapter(gamma=0.8, vmin=vmin, vmax=vmax, clip=False)(a)
    ref = mcolors.PowerNorm(gamma=0.8, vmin=vmin, vmax=vmax, clip=False)(a)
    max_abs_diff = float(np.max(np.abs(ours - ref)))
    assert max_abs_diff <= 1e-7


@pytest.mark.skipif(
    __import__('importlib').util.find_spec('matplotlib') is None,
    reason="Matplotlib not available"
)
def test_boundarynorm_parity():
    import matplotlib.colors as mcolors
    from forge3d.adapters.mpl_cmap import BoundaryNormAdapter

    boundaries = [0.0, 0.25, 0.5, 0.75, 1.0]
    ncolors = 4
    rng = np.random.default_rng(11)
    a = rng.random((64, 64), dtype=np.float64)

    ours = BoundaryNormAdapter(boundaries=boundaries, ncolors=ncolors, clip=False)(a)
    ref = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=ncolors, clip=False)(a)
    max_abs_diff = float(np.max(np.abs(ours - ref)))
    assert max_abs_diff <= 1e-7

